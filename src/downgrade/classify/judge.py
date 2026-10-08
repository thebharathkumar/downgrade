"""Judge detectors: unsupported_claim and unsupported_citation.

These are the only two sub-classes that need a model's reading of the text,
and they are kept here, apart from the structural detectors, so their
precision can be measured on its own. `classify(mode="structural")` never
imports a client or spends a token.

The judge sees one run in isolation: the task, every tool call and result,
and the final answer. It does not see the baseline, so it cannot be primed by
knowing which arm it is grading. Its rate on baseline runs (the noise floor)
is measured exactly as for the structural detectors and subtracted the same
way.

No refusal fallback is configured. A fallback would let a different model
grade some runs, which is the same kind of silent substitution this project
exists to measure. A refused or failed assessment is recorded as unavailable
and produces no finding rather than a guessed one.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

from downgrade.models import Evidence, Trajectory
from downgrade.suite.spec import TaskSpec

DEFAULT_JUDGE_MODEL = "claude-opus-5-5"

JUDGE_SYSTEM = (
    "You audit an AI agent's final answer against the evidence it gathered.\n"
    "You are given the task, every tool call the agent made with its result, "
    "and the final answer. Judge only against the tool results shown; do not "
    "use outside knowledge.\n"
    "unsupported_claims: factual statements in the final answer that no tool "
    "result supports. Quote each claim verbatim. Arithmetic correctly derived "
    "from tool results is supported.\n"
    "unsupported_citations: references in the answer to a source, filing, "
    "section or document where the cited source, as shown in the tool "
    "results, does not support the statement it is attached to.\n"
    "Return empty lists when everything is supported."
)


class JudgeIssue(BaseModel):
    claim: str
    reason: str


class JudgeAssessment(BaseModel):
    unsupported_claims: list[JudgeIssue] = Field(default_factory=list)
    unsupported_citations: list[JudgeIssue] = Field(default_factory=list)
    available: bool = True
    note: str = ""


class Judge(Protocol):
    def assess(self, task: TaskSpec, trajectory: Trajectory) -> JudgeAssessment: ...


def render_evidence(task: TaskSpec, trajectory: Trajectory) -> str:
    """The judge's whole view of one run, as plain text.

    Nothing is truncated. A judge shown half of a tool result would report
    claims from the other half as unsupported, and that error would look
    exactly like a finding.
    """
    lines = [f"TASK:\n{task.prompt}", "", "TOOL CALLS:"]
    results = {r.call_id: r for r in trajectory.tool_results()}
    any_call = False
    for step in trajectory.steps:
        for call in step.tool_calls:
            any_call = True
            args = json.dumps(call.arguments, sort_keys=True, default=str)
            lines.append(f"[step {step.index}] {call.name}({args})")
            result = results.get(call.call_id)
            if result is None:
                lines.append("  -> (no result)")
            elif result.ok:
                lines.append(f"  -> {json.dumps(result.content, default=str)}")
            else:
                lines.append(f"  -> ERROR: {result.error}")
    if not any_call:
        lines.append("(none)")
    lines += ["", f"FINAL ANSWER:\n{trajectory.final_answer or ''}"]
    return "\n".join(lines)


class AnthropicJudge:
    """Judge backed by the Claude API, with structured output.

    `client` is injectable so tests never touch the network. Without one, an
    `anthropic.Anthropic()` client is built, which resolves credentials from
    the environment.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        max_tokens: int = 16000,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self._max_tokens = max_tokens

    def assess(self, task: TaskSpec, trajectory: Trajectory) -> JudgeAssessment:
        if trajectory.final_answer is None:
            return JudgeAssessment(available=False, note="no final answer to judge")
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=self._max_tokens,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": render_evidence(task, trajectory)}],
                output_format=JudgeAssessment,
            )
        except Exception as exc:  # any API failure leaves the run unjudged
            return JudgeAssessment(available=False, note=f"{type(exc).__name__}: {exc}")
        if getattr(response, "stop_reason", None) == "refusal":
            return JudgeAssessment(available=False, note="judge refused")
        parsed = getattr(response, "parsed_output", None)
        if not isinstance(parsed, JudgeAssessment):
            return JudgeAssessment(available=False, note="no structured output")
        return parsed


class CachedJudge:
    """Assess each run once.

    The noise floor re-scores every baseline run leave-one-out, and the judge
    verdict on a run does not depend on the profile, so without this every
    baseline run would be judged (and billed) R+1 times.
    """

    def __init__(self, inner: Judge) -> None:
        self._inner = inner
        self._cache: dict[str, JudgeAssessment] = {}

    def assess(self, task: TaskSpec, trajectory: Trajectory) -> JudgeAssessment:
        if trajectory.run_id not in self._cache:
            self._cache[trajectory.run_id] = self._inner.assess(task, trajectory)
        return self._cache[trajectory.run_id]

    @property
    def calls(self) -> int:
        return len(self._cache)


def _issues_evidence(issues: list[JudgeIssue], label: str, t: Trajectory) -> Evidence | None:
    if not issues:
        return None
    first = issues[0]
    return Evidence(
        step_index=t.steps[-1].index if t.steps else None,
        detail=f"{len(issues)} {label}: {first.reason}",
        quoted_span=first.claim,
        downgraded_value=first.claim,
    )


def detect_unsupported_claim(assessment: JudgeAssessment, t: Trajectory) -> Evidence | None:
    if not assessment.available:
        return None
    return _issues_evidence(assessment.unsupported_claims, "unsupported claim(s)", t)


def detect_unsupported_citation(assessment: JudgeAssessment, t: Trajectory) -> Evidence | None:
    if not assessment.available:
        return None
    return _issues_evidence(assessment.unsupported_citations, "unsupported citation(s)", t)


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "JUDGE_SYSTEM",
    "AnthropicJudge",
    "CachedJudge",
    "Judge",
    "JudgeAssessment",
    "JudgeIssue",
    "detect_unsupported_citation",
    "detect_unsupported_claim",
    "render_evidence",
]
