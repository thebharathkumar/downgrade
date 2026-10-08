"""Hand-built trajectories for classifier, statistics and report tests.

Building trajectories directly, rather than through the agent loop and a
scripted transport, keeps each test about the one behaviour it names: the
classifier only ever reads Trajectory objects, so that is what these make.
"""

from __future__ import annotations

from typing import Any

from downgrade.models import (
    RouteObservation,
    Step,
    SweepConfig,
    ToolCall,
    ToolResult,
    Trajectory,
    Usage,
)
from downgrade.suite.spec import AnswerCheck, Constraint, TaskSpec

PRIMARY = "accounts/fireworks/models/kimi-k3"
CHEAP = "accounts/fireworks/models/glm-5p2"

Call = tuple[str, dict[str, Any], Any]


def config(**kw: Any) -> SweepConfig:
    defaults: dict[str, Any] = {
        "primary_model": PRIMARY,
        "secondary_model": CHEAP,
        "short_slugs": True,
        "replicates": 5,
    }
    defaults.update(kw)
    return SweepConfig(**defaults)


def task(task_id: str = "equity", **kw: Any) -> TaskSpec:
    defaults: dict[str, Any] = {
        "task_id": task_id,
        "family": "tabular_aggregation",
        "prompt": "Report combined equity for ACME and BOREAS.",
        "answer_check": AnswerCheck(kind="numeric", value=1000.0),
        "constraints": [
            Constraint(id="uses_lookup", predicate="tool_called", args={"name": "get_figure"}),
            Constraint(id="short", predicate="answer_max_words", args={"n": 30}),
        ],
    }
    defaults.update(kw)
    return TaskSpec(**defaults)


GOOD_CALLS: list[Call] = [
    ("get_figure", {"company": "ACME", "concept": "assets"}, {"value": 1200.0}),
    ("get_figure", {"company": "ACME", "concept": "liabilities"}, {"value": 800.0}),
    ("get_figure", {"company": "BOREAS", "concept": "assets"}, {"value": 900.0}),
    ("get_figure", {"company": "BOREAS", "concept": "liabilities"}, {"value": 300.0}),
]


def trajectory(
    *,
    task_id: str = "equity",
    arm: str = "baseline_direct",
    replicate: int = 0,
    calls: list[Call] | None = None,
    answer: str | None = "Combined equity is 1000.",
    status: str = "completed",
    served: str = PRIMARY,
    fingerprint: str | None = None,
    error: str | None = None,
    failed: set[int] | None = None,
) -> Trajectory:
    """One run: each call in its own step, then a final answer step."""
    cfg_fp = fingerprint or config().fingerprint
    steps: list[Step] = []
    for i, (name, args, result) in enumerate(GOOD_CALLS if calls is None else calls):
        call = ToolCall(call_id=f"c{i}", name=name, arguments=args)
        ok = i not in (failed or set())
        steps.append(
            Step(
                index=i,
                tool_calls=[call],
                tool_results=[
                    ToolResult(
                        call_id=call.call_id,
                        name=name,
                        content=result if ok else None,
                        ok=ok,
                        error=None if ok else "boom",
                    )
                ],
                route=_route(i, served),
            )
        )
    if status == "completed":
        steps.append(Step(index=len(steps), thought=answer or "", route=_route(len(steps), served)))
    return Trajectory(
        run_id=f"{task_id}:{arm}:{replicate}",
        task_id=task_id,
        arm=arm,
        replicate=replicate,
        conversation_id=f"conv-{task_id}-{arm}-{replicate}",
        config_fingerprint=cfg_fp,
        steps=steps,
        final_answer=answer if status == "completed" else None,
        status=status,  # type: ignore[arg-type]
        error=error,
    )


def _route(turn: int, served: str) -> RouteObservation:
    return RouteObservation(
        turn_index=turn,
        requested_model="firerouter/kimi-k3/glm-5p2",
        served_model=served,
        downgraded=served != PRIMARY,
        source="response_model",
        temperature=0.0,
        seed=20260903,
        usage=Usage(input_tokens=100, output_tokens=10, complete=True),
    )


def arm_runs(arm: str, n: int = 5, task_id: str = "equity", **kw: Any) -> list[Trajectory]:
    return [trajectory(task_id=task_id, arm=arm, replicate=r, **kw) for r in range(n)]
