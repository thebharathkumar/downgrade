"""Building a BaselineProfile from the control arm's replicates.

The profile is the reference every downgraded run is scored against. It is a
summary of a SET of runs: rates rather than a single run's choices, so a
detector can ask whether a behaviour was reliable in baseline before treating
its absence as a regression.
"""

from __future__ import annotations

import statistics
from collections import Counter

from downgrade.classify.answer import check_answer
from downgrade.classify.constraints import check_constraints
from downgrade.classify.values import canonical, collect_values, extract_numbers
from downgrade.models import BaselineProfile, Trajectory
from downgrade.suite.spec import TaskSpec

# Argument names that identify a document a tool read. Collected so the
# profile records which sources baseline consulted.
DOCUMENT_ARGUMENTS = ("doc_id", "document", "document_id", "path", "accession", "filing")


def build_profile(
    task: TaskSpec,
    baseline: list[Trajectory],
    *,
    baseline_arm: str,
    reliability_threshold: float,
) -> BaselineProfile:
    """Summarise what the baseline arm does on `task` across its replicates.

    `grounded_values` holds numbers a downgraded run may state without having
    looked them up itself: figures in the task prompt, and figures that appear
    in baseline's own final answers reliably (derived values such as a sum).
    Values baseline's tools returned are deliberately NOT included. A
    downgraded run that states a figure it never retrieved got it from
    somewhere other than evidence, even if baseline happened to retrieve it.
    """
    n = len(baseline)
    profile = BaselineProfile(
        task_id=task.task_id,
        baseline_arm=baseline_arm,
        replicate_count=n,
        reliability_threshold=reliability_threshold,
    )
    if n == 0:
        return profile

    signature_runs: Counter[str] = Counter()
    name_runs: Counter[str] = Counter()
    per_run_counts: list[Counter[str]] = []
    constraint_hits: Counter[str] = Counter()
    answer_number_runs: Counter[str] = Counter()
    cited: set[str] = set()
    correct = 0

    for trajectory in baseline:
        signatures = trajectory.tool_signatures()
        counts = Counter(signatures)
        per_run_counts.append(counts)
        signature_runs.update(set(signatures))
        name_runs.update(set(trajectory.tool_names()))

        for result in check_constraints(task, trajectory):
            if result.satisfied:
                constraint_hits[result.constraint_id] += 1

        answer_number_runs.update(
            {canonical(t.value) for t in extract_numbers(trajectory.final_answer)}
        )
        for step in trajectory.steps:
            for call in step.tool_calls:
                for key in DOCUMENT_ARGUMENTS:
                    if key in call.arguments:
                        cited.add(str(call.arguments[key]))

        if trajectory.status == "completed" and check_answer(
            task.answer_check, trajectory.final_answer
        ):
            correct += 1

    all_signatures = set(signature_runs)
    profile.tool_signature_rates = {s: signature_runs[s] / n for s in all_signatures}
    profile.tool_name_rates = {name: count / n for name, count in name_runs.items()}
    profile.call_count_by_signature = {
        s: float(statistics.median(c.get(s, 0) for c in per_run_counts)) for s in all_signatures
    }
    totals = [float(sum(c.values())) for c in per_run_counts]
    profile.median_tool_calls = float(statistics.median(totals))
    profile.tool_call_count_mean = statistics.fmean(totals)
    profile.tool_call_count_stdev = statistics.pstdev(totals)
    profile.constraint_satisfaction_rates = {
        c.id: constraint_hits[c.id] / n for c in task.constraints
    }

    grounded = {canonical(t.value) for t in extract_numbers(task.prompt)}
    grounded.update(
        value for value, runs in answer_number_runs.items() if runs / n >= reliability_threshold
    )
    profile.grounded_values = grounded
    profile.cited_documents = cited
    profile.correct_rate = correct / n

    answers = [(t.final_answer or "").strip() for t in baseline]
    profile.answers = answers
    profile.distinct_answers = len(set(answers))
    profile.modal_answer_rate = Counter(answers).most_common(1)[0][1] / n
    return profile


def observed_values(trajectory: Trajectory) -> set[float]:
    """Every number any successful tool call in this run returned."""
    values: set[float] = set()
    for result in trajectory.tool_results():
        if result.ok:
            values |= collect_values(result.content)
    return values


__all__ = ["DOCUMENT_ARGUMENTS", "build_profile", "observed_values"]
