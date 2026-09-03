"""Runner tests, centred on the conversation-isolation guarantee.

FireRouter caches routing decisions per conversation. If two runs share one,
part of the second run is served by the first run's cached route, and the arm
labels on the resulting data are lies. These tests read what the transport
actually received rather than trusting the runner's own bookkeeping, because
the failure mode is precisely the runner believing it did the right thing.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from downgrade.arms import ALL_ARMS, BASELINE_DIRECT, ROUTING_HEADER, arm_by_name
from downgrade.client import FireworksClient
from downgrade.models import SweepConfig
from downgrade.runner import AgentLoop, ConversationGuard, ConversationReuseError
from downgrade.suite.spec import AnswerCheck, TaskSpec
from downgrade.tools import FunctionTool, ToolDef, ToolError, ToolRegistry
from downgrade.transport import FakeTransport, TransportResponse

PRIMARY = "accounts/fireworks/models/kimi-k3"
CHEAP = "accounts/fireworks/models/glm-5p2"


def config(**kw: Any) -> SweepConfig:
    defaults: dict[str, Any] = {
        "primary_model": PRIMARY,
        "secondary_model": CHEAP,
        "short_slugs": True,
        "max_steps": 6,
    }
    defaults.update(kw)
    return SweepConfig(**defaults)


def task(task_id: str = "t1", **kw: Any) -> TaskSpec:
    defaults: dict[str, Any] = {
        "task_id": task_id,
        "family": "tabular_aggregation",
        "prompt": f"Prompt for {task_id}",
        "answer_check": AnswerCheck(kind="numeric", value=42.0),
    }
    defaults.update(kw)
    return TaskSpec(**defaults)


def chunk(
    *,
    model: str = CHEAP,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {
        "id": "cmpl-1",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def usage_chunk(model: str = CHEAP, prompt: int = 100, completion: int = 20) -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "model": model,
        "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def answer_response(text: str = "42", model: str = CHEAP) -> TransportResponse:
    """A turn that returns a final answer and calls no tools."""
    return TransportResponse(
        status_code=200,
        headers={"x-ratelimit-remaining": "99"},
        chunks=[
            chunk(model=model, content=text),
            chunk(model=model, finish="stop"),
            usage_chunk(model),
        ],
    )


def tool_response(name: str, arguments: str, model: str = CHEAP) -> TransportResponse:
    """A turn that emits one tool call, split across deltas like the real API."""
    return TransportResponse(
        status_code=200,
        headers={},
        chunks=[
            chunk(
                model=model,
                tool_calls=[
                    {"index": 0, "id": "call_1", "function": {"name": name, "arguments": ""}}
                ],
            ),
            chunk(
                model=model,
                tool_calls=[{"index": 0, "function": {"arguments": arguments}}],
            ),
            chunk(model=model, finish="tool_calls"),
            usage_chunk(model),
        ],
    )


def make_registry() -> ToolRegistry:
    def query_table(concept: str) -> dict[str, Any]:
        return {"concept": concept, "value": 42.0}

    def boom() -> None:
        raise ToolError("upstream unavailable")

    return ToolRegistry(
        [
            FunctionTool(
                definition=ToolDef(
                    name="query_table",
                    description="Look up a concept",
                    parameters={
                        "type": "object",
                        "properties": {"concept": {"type": "string"}},
                        "required": ["concept"],
                    },
                ),
                func=query_table,
            ),
            FunctionTool(
                definition=ToolDef(name="boom", description="Always fails", parameters={}),
                func=boom,
            ),
        ]
    )


def make_loop(
    responses: list[TransportResponse], cfg: SweepConfig | None = None
) -> tuple[AgentLoop, FakeTransport]:
    transport = FakeTransport(responses)
    client = FireworksClient(transport, api_key="test-key")
    loop = AgentLoop(client, make_registry(), cfg or config())
    return loop, transport


class TestConversationIsolation:
    def test_run_accepts_no_history_parameter(self) -> None:
        """The guarantee is the signature. Adding `messages` or `session`
        here is the change that would silently break the experiment."""
        params = set(inspect.signature(AgentLoop.run).parameters)
        assert params == {"self", "task", "arm", "replicate"}
        forbidden = {"messages", "history", "session", "conversation", "conversation_id"}
        assert not (params & forbidden)

    def test_every_run_gets_a_distinct_conversation(self) -> None:
        loop, _ = make_loop([answer_response() for _ in range(6)])
        ids = {
            loop.run(task(f"t{i}"), arm, replicate=r).conversation_id
            for i, arm in enumerate([BASELINE_DIRECT, arm_by_name("pref_5")])
            for r in range(3)
        }
        assert len(ids) == 6

    def test_no_run_inherits_the_previous_run_message_history(self) -> None:
        """The real failure mode: run two carrying run one's turns."""
        loop, transport = make_loop(
            [
                tool_response("query_table", '{"concept": "Assets"}'),
                answer_response(),
                answer_response(),
            ]
        )
        loop.run(task("t1"), BASELINE_DIRECT)
        first_run_calls = transport.call_count
        loop.run(task("t2"), BASELINE_DIRECT)

        opening = transport.payloads()[first_run_calls]
        assert [m["role"] for m in opening["messages"]] == ["system", "user"]
        assert opening["messages"][1]["content"] == "Prompt for t2"
        serialised = str(opening["messages"])
        assert "t1" not in serialised
        assert "Assets" not in serialised

    def test_every_request_in_a_run_carries_the_routing_header(self) -> None:
        """Not just the first. A run whose later turns lost the header would
        be served at the default preference of 3 without saying so."""
        loop, transport = make_loop(
            [
                tool_response("query_table", '{"concept": "Assets"}'),
                tool_response("query_table", '{"concept": "Liabilities"}'),
                answer_response(),
            ]
        )
        loop.run(task(), arm_by_name("pref_5"))
        assert transport.call_count == 3
        assert all(r.headers.get(ROUTING_HEADER) == "5" for r in transport.requests)

    def test_the_direct_arm_sends_no_routing_header(self) -> None:
        """It is not routed at all; sending a preference would misdescribe it."""
        loop, transport = make_loop([answer_response()])
        loop.run(task(), BASELINE_DIRECT)
        assert ROUTING_HEADER not in transport.requests[0].headers

    def test_the_direct_arm_names_the_primary_model_not_the_router(self) -> None:
        loop, transport = make_loop([answer_response()])
        loop.run(task(), BASELINE_DIRECT)
        assert transport.payloads()[0]["model"] == PRIMARY

    def test_router_arms_send_the_pair_string(self) -> None:
        loop, transport = make_loop([answer_response()])
        loop.run(task(), arm_by_name("pref_3"))
        assert transport.payloads()[0]["model"] == "firerouter/kimi-k3/glm-5p2"

    def test_guard_rejects_a_reused_conversation(self) -> None:
        guard = ConversationGuard()
        guard.claim("conv-1")
        with pytest.raises(ConversationReuseError, match="already used"):
            guard.claim("conv-1")

    def test_guard_counts_distinct_conversations(self) -> None:
        loop, _ = make_loop([answer_response() for _ in range(4)])
        for i in range(4):
            loop.run(task(f"t{i}"), BASELINE_DIRECT)
        assert loop.guard.count == 4


class TestSamplingSettings:
    def test_temperature_and_seed_are_sent_on_every_call(self) -> None:
        cfg = config(temperature=0.0, seed=7)
        loop, transport = make_loop(
            [tool_response("query_table", '{"concept": "Assets"}'), answer_response()], cfg
        )
        loop.run(task(), arm_by_name("pref_2"))
        assert all(p["temperature"] == 0.0 and p["seed"] == 7 for p in transport.payloads())

    def test_every_arm_uses_identical_sampling_settings(self) -> None:
        """If temperature differs between arms the sweep measures sampling."""
        cfg = config()
        loop, transport = make_loop([answer_response() for _ in ALL_ARMS], cfg)
        for arm in ALL_ARMS:
            loop.run(task(), arm)
        temperatures = {p["temperature"] for p in transport.payloads()}
        seeds = {p.get("seed") for p in transport.payloads()}
        assert temperatures == {cfg.temperature}
        assert seeds == {cfg.seed}

    def test_usage_is_requested_in_the_stream(self) -> None:
        """Without stream_options Fireworks omits usage entirely."""
        loop, transport = make_loop([answer_response()])
        loop.run(task(), BASELINE_DIRECT)
        assert transport.payloads()[0]["stream_options"] == {"include_usage": True}

    def test_trajectory_records_the_config_fingerprint(self) -> None:
        cfg = config()
        loop, _ = make_loop([answer_response()], cfg)
        assert loop.run(task(), BASELINE_DIRECT).config_fingerprint == cfg.fingerprint


class TestRunOutcomes:
    def test_a_plain_answer_completes(self) -> None:
        loop, _ = make_loop([answer_response("42")])
        traj = loop.run(task(), BASELINE_DIRECT)
        assert traj.status == "completed"
        assert traj.final_answer == "42"
        assert len(traj.steps) == 1

    def test_a_tool_call_is_executed_and_fed_back(self) -> None:
        loop, transport = make_loop(
            [tool_response("query_table", '{"concept": "Assets"}'), answer_response()]
        )
        traj = loop.run(task(), BASELINE_DIRECT)
        assert traj.status == "completed"
        assert traj.tool_names() == ["query_table"]
        assert traj.tool_results()[0].content == {"concept": "Assets", "value": 42.0}
        roles = [m["role"] for m in transport.payloads()[1]["messages"]]
        assert roles == ["system", "user", "assistant", "tool"]

    def test_a_failing_tool_becomes_an_observation_not_a_crash(self) -> None:
        """The model may recover; whether it did is what retry detection reads."""
        loop, transport = make_loop([tool_response("boom", "{}"), answer_response()])
        traj = loop.run(task(), BASELINE_DIRECT)
        assert traj.status == "completed"
        assert traj.tool_results()[0].ok is False
        assert "upstream unavailable" in str(transport.payloads()[1]["messages"][-1]["content"])

    def test_an_unknown_tool_does_not_end_the_run(self) -> None:
        loop, _ = make_loop([tool_response("hallucinated", "{}"), answer_response()])
        traj = loop.run(task(), BASELINE_DIRECT)
        assert traj.status == "completed"
        assert traj.tool_results()[0].ok is False
        assert "No such tool" in (traj.tool_results()[0].error or "")

    def test_a_transport_error_yields_an_errored_trajectory(self) -> None:
        """Never raises: a dropped run would unbalance the arm's replicates."""
        loop, _ = make_loop([TransportResponse(status_code=0, error="ConnectError: refused")])
        traj = loop.run(task(), BASELINE_DIRECT)
        assert traj.status == "errored"
        assert "refused" in (traj.error or "")
        assert len(traj.steps) == 1

    def test_an_http_error_is_recorded_not_raised(self) -> None:
        loop, _ = make_loop([TransportResponse(status_code=429, error="rate limited")])
        traj = loop.run(task(), BASELINE_DIRECT)
        assert traj.status == "errored"

    def test_exhausting_the_step_budget_is_no_termination(self) -> None:
        """Distinct from a crash: the model kept working and never concluded."""
        cfg = config(max_steps=3)
        loop, transport = make_loop(
            [tool_response("query_table", '{"concept": "Assets"}') for _ in range(3)], cfg
        )
        traj = loop.run(task(), BASELINE_DIRECT, replicate=2)
        assert traj.status == "no_termination"
        assert traj.final_answer is None
        assert transport.call_count == 3

    def test_a_task_can_lower_the_step_budget(self) -> None:
        loop, transport = make_loop(
            [tool_response("query_table", '{"concept": "Assets"}') for _ in range(2)],
            config(max_steps=9),
        )
        loop.run(task(max_steps=2), BASELINE_DIRECT)
        assert transport.call_count == 2

    def test_replicate_and_arm_are_recorded_on_the_trajectory(self) -> None:
        loop, _ = make_loop([answer_response()])
        traj = loop.run(task("t9"), arm_by_name("pref_4"), replicate=3)
        assert (traj.task_id, traj.arm, traj.replicate) == ("t9", "pref_4", 3)
        assert traj.run_id.startswith("t9:pref_4:3:")

    def test_wall_time_is_recorded(self) -> None:
        loop, _ = make_loop([answer_response()])
        assert loop.run(task(), BASELINE_DIRECT).wall_ms >= 0


class TestRouteCapture:
    def test_served_model_is_read_from_the_response(self) -> None:
        loop, _ = make_loop([answer_response(model=CHEAP)])
        route = loop.run(task(), arm_by_name("pref_5")).steps[0].route
        assert route.served_model == CHEAP
        assert route.source == "response_model"

    def test_a_downgrade_is_flagged_against_the_primary(self) -> None:
        loop, _ = make_loop([answer_response(model=CHEAP)])
        assert loop.run(task(), arm_by_name("pref_5")).steps[0].route.downgraded is True

    def test_being_served_the_primary_is_not_a_downgrade(self) -> None:
        loop, _ = make_loop([answer_response(model=PRIMARY)])
        assert loop.run(task(), arm_by_name("pref_1")).steps[0].route.downgraded is False

    def test_route_drift_within_a_run_is_visible(self) -> None:
        """Routing caches per conversation, but the cache can still turn over."""
        loop, _ = make_loop(
            [
                tool_response("query_table", '{"concept": "Assets"}', model=PRIMARY),
                answer_response(model=CHEAP),
            ]
        )
        traj = loop.run(task(), arm_by_name("pref_3"))
        assert traj.served_models == [PRIMARY, CHEAP]
        assert traj.route_stability is False

    def test_usage_accumulates_across_turns(self) -> None:
        loop, _ = make_loop(
            [tool_response("query_table", '{"concept": "Assets"}'), answer_response()]
        )
        totals = loop.run(task(), BASELINE_DIRECT).totals
        assert (totals.input_tokens, totals.output_tokens) == (200, 40)
        assert totals.complete

    def test_a_stream_without_usage_is_marked_incomplete(self) -> None:
        truncated = TransportResponse(
            status_code=200, chunks=[chunk(content="42"), chunk(finish="stop")]
        )
        loop, _ = make_loop([truncated])
        totals = loop.run(task(), BASELINE_DIRECT).totals
        assert totals.input_tokens == 0
        assert not totals.complete

    def test_response_headers_are_captured(self) -> None:
        loop, _ = make_loop([answer_response()])
        headers = loop.run(task(), BASELINE_DIRECT).steps[0].route.response_headers
        assert headers["x-ratelimit-remaining"] == "99"

    def test_preference_is_recorded_on_the_route(self) -> None:
        loop, _ = make_loop([answer_response()])
        assert loop.run(task(), arm_by_name("pref_4")).steps[0].route.preference == 4
