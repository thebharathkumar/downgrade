"""Client tests for the three details of this API that are easy to get wrong.

Usage arrives only in the final chunk. Tool calls arrive as deltas that must
be reassembled by index. The served model is the only per-request route
attribution the API offers.
"""

from __future__ import annotations

from typing import Any

from downgrade.client import FireworksClient
from downgrade.transport import FakeTransport, TransportResponse

CHEAP = "accounts/fireworks/models/glm-5p2"
PRIMARY = "accounts/fireworks/models/kimi-k3"


def call(response: TransportResponse, **kw: Any) -> Any:
    client = FireworksClient(FakeTransport([response]), api_key="k")
    defaults: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "firerouter/kimi-k3/glm-5p2",
        "turn_index": 0,
        "temperature": 0.0,
        "max_tokens": 256,
        "primary_model": PRIMARY,
    }
    defaults.update(kw)
    return client.chat(**defaults)


def delta_chunk(tool_calls: list[dict[str, Any]], model: str = CHEAP) -> dict[str, Any]:
    return {"id": "c", "model": model, "choices": [{"delta": {"tool_calls": tool_calls}}]}


class TestToolCallReassembly:
    def test_arguments_split_across_chunks_are_joined(self) -> None:
        result = call(
            TransportResponse(
                status_code=200,
                chunks=[
                    delta_chunk([{"index": 0, "id": "c1", "function": {"name": "lookup"}}]),
                    delta_chunk([{"index": 0, "function": {"arguments": '{"conc'}}]),
                    delta_chunk([{"index": 0, "function": {"arguments": 'ept": "Assets"}'}}]),
                ],
            )
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"concept": "Assets"}

    def test_parallel_tool_calls_are_kept_in_index_order(self) -> None:
        result = call(
            TransportResponse(
                status_code=200,
                chunks=[
                    delta_chunk(
                        [
                            {
                                "index": 1,
                                "id": "b",
                                "function": {"name": "second", "arguments": "{}"},
                            },
                            {
                                "index": 0,
                                "id": "a",
                                "function": {"name": "first", "arguments": "{}"},
                            },
                        ]
                    )
                ],
            )
        )
        assert [c.name for c in result.tool_calls] == ["first", "second"]

    def test_malformed_arguments_are_preserved_not_dropped(self) -> None:
        """A vanished call is what missing_tool_call looks for, so a parse bug
        must never be able to masquerade as a routing finding."""
        result = call(
            TransportResponse(
                status_code=200,
                chunks=[
                    delta_chunk(
                        [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "lookup", "arguments": "{not json"},
                            }
                        ]
                    )
                ],
            )
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"__unparsed__": "{not json"}

    def test_non_object_arguments_are_wrapped(self) -> None:
        result = call(
            TransportResponse(
                status_code=200,
                chunks=[
                    delta_chunk(
                        [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "lookup", "arguments": "[1,2]"},
                            }
                        ]
                    )
                ],
            )
        )
        assert result.tool_calls[0].arguments == {"__value__": [1, 2]}

    def test_a_delta_with_no_name_is_not_a_call(self) -> None:
        result = call(
            TransportResponse(
                status_code=200,
                chunks=[delta_chunk([{"index": 0, "function": {"arguments": "{}"}}])],
            )
        )
        assert result.tool_calls == []

    def test_a_call_without_an_id_gets_a_synthetic_one(self) -> None:
        result = call(
            TransportResponse(
                status_code=200,
                chunks=[
                    delta_chunk([{"index": 0, "function": {"name": "lookup", "arguments": "{}"}}])
                ],
            )
        )
        assert result.tool_calls[0].call_id == "call_0"


class TestRouteAttribution:
    def test_falls_back_to_the_response_model(self) -> None:
        result = call(TransportResponse(status_code=200, chunks=[{"id": "c", "model": CHEAP}]))
        assert result.route.served_model == CHEAP
        assert result.route.source == "response_model"

    def test_prefers_an_explicit_header_when_one_exists(self) -> None:
        """No routing header is documented; if one appears it should win, and
        `source` must say which answered."""
        result = call(
            TransportResponse(
                status_code=200,
                headers={"X-Fireworks-Served-Model": PRIMARY},
                chunks=[{"id": "c", "model": CHEAP}],
            )
        )
        assert result.route.served_model == PRIMARY
        assert result.route.source == "header"

    def test_unknown_when_nothing_attributes_the_route(self) -> None:
        result = call(TransportResponse(status_code=200, chunks=[{"id": "c"}]))
        assert result.route.served_model is None
        assert result.route.source == "unknown"
        assert result.route.downgraded is None

    def test_downgrade_compares_on_slug_not_full_path(self) -> None:
        result = call(
            TransportResponse(status_code=200, chunks=[{"id": "c", "model": "kimi-k3"}]),
            primary_model=PRIMARY,
        )
        assert result.route.downgraded is False


class TestUsageAndErrors:
    def test_usage_is_read_from_the_last_chunk_that_has_it(self) -> None:
        """Some servers send a trailing empty chunk after the usage one."""
        result = call(
            TransportResponse(
                status_code=200,
                chunks=[
                    {"id": "c", "model": CHEAP, "choices": []},
                    {"id": "c", "usage": {"prompt_tokens": 11, "completion_tokens": 3}},
                    {"id": "c", "choices": []},
                ],
            )
        )
        assert (result.route.usage.input_tokens, result.route.usage.output_tokens) == (11, 3)
        assert result.route.usage.complete

    def test_a_stream_without_usage_is_incomplete_not_zero(self) -> None:
        result = call(TransportResponse(status_code=200, chunks=[{"id": "c", "model": CHEAP}]))
        assert result.route.usage.input_tokens == 0
        assert not result.route.usage.complete

    def test_an_error_response_still_produces_a_route(self) -> None:
        """The runner needs somewhere to record what was attempted."""
        result = call(TransportResponse(status_code=429, error="rate limited"))
        assert not result.ok
        assert result.error == "rate limited"
        assert result.route is not None
        assert result.route.preference is None

    def test_finish_reason_is_captured(self) -> None:
        result = call(
            TransportResponse(
                status_code=200,
                chunks=[{"id": "c", "model": CHEAP, "choices": [{"finish_reason": "stop"}]}],
            )
        )
        assert result.route.finish_reason == "stop"

    def test_the_api_key_becomes_a_bearer_header(self) -> None:
        transport = FakeTransport([TransportResponse(status_code=200)])
        FireworksClient(transport, api_key="secret").chat(
            messages=[], model="m", turn_index=0, temperature=0.0, max_tokens=8
        )
        assert transport.requests[0].headers["Authorization"] == "Bearer secret"

    def test_no_authorization_header_without_a_key(self) -> None:
        transport = FakeTransport([TransportResponse(status_code=200)])
        FireworksClient(transport).chat(
            messages=[], model="m", turn_index=0, temperature=0.0, max_tokens=8
        )
        assert "Authorization" not in transport.requests[0].headers

    def test_tools_are_only_sent_when_present(self) -> None:
        transport = FakeTransport([TransportResponse(status_code=200)])
        FireworksClient(transport).chat(
            messages=[], model="m", turn_index=0, temperature=0.0, max_tokens=8, tools=[]
        )
        assert "tools" not in transport.requests[0].payload
