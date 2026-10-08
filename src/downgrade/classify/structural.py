"""Structural detectors: decided by set difference, counting and provenance.

Each detector takes the task, the baseline profile and one trajectory, and
returns Evidence when its sub-class applies or None when it does not. None of
them calls a model. The two judge detectors live in `judge.py` so that their
precision can be measured separately from these.
"""

from __future__ import annotations

from collections import Counter

from downgrade.classify.answer import check_answer, has_expectation
from downgrade.classify.constraints import check_constraints
from downgrade.classify.profile import observed_values
from downgrade.classify.values import extract_numbers, is_grounded
from downgrade.models import BaselineProfile, Evidence, Trajectory
from downgrade.suite.spec import TaskSpec


def _last_step(trajectory: Trajectory) -> int | None:
    return trajectory.steps[-1].index if trajectory.steps else None


def _of(rate: float, profile: BaselineProfile) -> str:
    n = profile.replicate_count
    return f"{round(rate * n)}/{n} baseline runs"


def detect_run_error(task: TaskSpec, p: BaselineProfile, t: Trajectory) -> Evidence | None:
    if t.status != "errored":
        return None
    return Evidence(
        step_index=_last_step(t),
        detail=f"run errored: {t.error or 'unknown error'}",
        baseline_reference=f"baseline completed in {_of(p.correct_rate, p)} correctly",
    )


def detect_no_termination(task: TaskSpec, p: BaselineProfile, t: Trajectory) -> Evidence | None:
    if t.status != "no_termination":
        return None
    return Evidence(
        step_index=_last_step(t),
        detail=f"no final answer after {len(t.steps)} step(s); the step budget ran out",
        baseline_reference=f"baseline median {p.median_tool_calls:g} tool call(s)",
    )


def detect_wrong_answer(task: TaskSpec, p: BaselineProfile, t: Trajectory) -> Evidence | None:
    if t.status != "completed" or not has_expectation(task.answer_check):
        return None
    if check_answer(task.answer_check, t.final_answer):
        return None
    return Evidence(
        step_index=_last_step(t),
        detail=f"final answer fails the {task.answer_check.kind} check",
        baseline_reference=f"baseline correct in {_of(p.correct_rate, p)}",
        downgraded_value=t.final_answer,
        quoted_span=str(task.answer_check.value or task.answer_check.pattern),
    )


def detect_missing_tool_call(task: TaskSpec, p: BaselineProfile, t: Trajectory) -> Evidence | None:
    """A call reliable in baseline that this run never made.

    Signatures first, so a call made with narrower arguments counts as
    missing. When baseline's arguments vary too much for any one signature to
    be reliable, fall back to tool names: a tool baseline always reaches for
    and this run never touched.
    """
    made = set(t.tool_signatures())
    missing = sorted(
        sig
        for sig, rate in p.tool_signature_rates.items()
        if p.is_reliable(p.tool_signature_rates, sig) and sig not in made
    )
    names = set(t.tool_names())
    missing_names = sorted(
        name
        for name in p.tool_name_rates
        if p.is_reliable(p.tool_name_rates, name) and name not in names
    )
    if not missing and not missing_names:
        return None
    key = missing[0] if missing else missing_names[0]
    rates = p.tool_signature_rates if missing else p.tool_name_rates
    listed = missing or missing_names
    return Evidence(
        detail=f"{len(listed)} call(s) reliable in baseline are absent: {', '.join(listed)}",
        baseline_reference=f"{key} made in {_of(rates[key], p)}",
        downgraded_value="absent",
    )


def detect_dropped_constraint(task: TaskSpec, p: BaselineProfile, t: Trajectory) -> Evidence | None:
    for result in check_constraints(task, t):
        if result.satisfied:
            continue
        if not p.is_reliable(p.constraint_satisfaction_rates, result.constraint_id):
            continue
        rate = p.constraint_satisfaction_rates[result.constraint_id]
        return Evidence(
            step_index=_last_step(t),
            detail=f"constraint '{result.constraint_id}' ({result.predicate}) failed",
            baseline_reference=f"satisfied in {_of(rate, p)}",
            downgraded_value=result.detail,
            constraint_id=result.constraint_id,
        )
    return None


def detect_redundant_retry(task: TaskSpec, p: BaselineProfile, t: Trajectory) -> Evidence | None:
    """The same call repeated more often than baseline needed it.

    Compared against the baseline MEDIAN count for that signature, not zero:
    a task where baseline legitimately polls the same call twice does not
    make a downgraded run that also polls twice redundant.
    """
    counts = Counter(t.tool_signatures())
    worst: tuple[str, int, float] | None = None
    for signature, count in counts.items():
        expected = p.call_count_by_signature.get(signature, 0.0)
        if count >= 2 and count > max(1.0, expected) and (worst is None or count > worst[1]):
            worst = (signature, count, expected)
    if worst is None:
        return None
    signature, count, expected = worst
    first_repeat = None
    seen = 0
    for step in t.steps:
        seen += sum(1 for c in step.tool_calls if c.signature() == signature)
        if seen >= 2:
            first_repeat = step.index
            break
    return Evidence(
        step_index=first_repeat,
        detail=f"{signature} called {count} times",
        baseline_reference=f"baseline median {expected:g} call(s)",
        downgraded_value=str(count),
    )


def detect_fabricated_value(task: TaskSpec, p: BaselineProfile, t: Trajectory) -> Evidence | None:
    """A number in the answer that no tool call in this run produced."""
    observed = observed_values(t) | {float(v) for v in p.grounded_values}
    ungrounded = [
        token
        for token in extract_numbers(t.final_answer)
        if not token.is_small_integer and not is_grounded(token, observed)
    ]
    if not ungrounded:
        return None
    return Evidence(
        step_index=_last_step(t),
        detail=(
            f"{len(ungrounded)} value(s) in the answer match no tool result in this run: "
            + ", ".join(token.text for token in ungrounded)
        ),
        baseline_reference=f"{len(observed)} value(s) observed or grounded",
        downgraded_value=ungrounded[0].text,
        quoted_span=ungrounded[0].text,
    )


__all__ = [
    "detect_dropped_constraint",
    "detect_fabricated_value",
    "detect_missing_tool_call",
    "detect_no_termination",
    "detect_redundant_retry",
    "detect_run_error",
    "detect_wrong_answer",
]
