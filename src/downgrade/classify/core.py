"""The classifier: one finding per run, by a documented priority ladder.

Every downgraded run is scored against the baseline profile and receives at
most ONE finding: the highest-priority sub-class that applies. Loud kinds
come first, then the silent ones in the order the README lists. Returning
the first match rather than every match keeps per-kind rates a partition of
the runs, so "fraction of runs with any finding" is the sum of the per-kind
rates and no run is counted twice in the statistics layer.

Comparisons whose trajectories carry a different config fingerprint from the
sweep config are refused. That comparison would measure the settings drift,
not the routing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from downgrade.arms import DEFAULT_BASELINE_ARM
from downgrade.classify.constraints import validate_constraints
from downgrade.classify.judge import (
    CachedJudge,
    Judge,
    JudgeAssessment,
    detect_unsupported_citation,
    detect_unsupported_claim,
)
from downgrade.classify.profile import build_profile
from downgrade.classify.structural import (
    detect_dropped_constraint,
    detect_fabricated_value,
    detect_missing_tool_call,
    detect_no_termination,
    detect_redundant_retry,
    detect_run_error,
    detect_wrong_answer,
)
from downgrade.models import (
    DETECTOR_FOR_KIND,
    LOUD_KINDS,
    SEVERITY_RANK,
    ArmComparison,
    BaselineProfile,
    ClassifyMode,
    Evidence,
    Finding,
    FindingKind,
    FindingRate,
    NoiseFloor,
    SweepConfig,
    Trajectory,
)
from downgrade.suite.spec import TaskSpec

StructuralFn = Callable[[TaskSpec, BaselineProfile, Trajectory], Evidence | None]
JudgeFn = Callable[[JudgeAssessment, Trajectory], Evidence | None]

STRUCTURAL: dict[FindingKind, StructuralFn] = {
    FindingKind.WRONG_ANSWER: detect_wrong_answer,
    FindingKind.RUN_ERROR: detect_run_error,
    FindingKind.NO_TERMINATION: detect_no_termination,
    FindingKind.MISSING_TOOL_CALL: detect_missing_tool_call,
    FindingKind.DROPPED_CONSTRAINT: detect_dropped_constraint,
    FindingKind.REDUNDANT_RETRY: detect_redundant_retry,
    FindingKind.FABRICATED_VALUE: detect_fabricated_value,
}

JUDGED: dict[FindingKind, JudgeFn] = {
    FindingKind.UNSUPPORTED_CLAIM: detect_unsupported_claim,
    FindingKind.UNSUPPORTED_CITATION: detect_unsupported_citation,
}

LADDER: tuple[FindingKind, ...] = tuple(sorted(SEVERITY_RANK, key=SEVERITY_RANK.__getitem__))


class FingerprintMismatchError(ValueError):
    """Trajectories were produced under different settings than the config."""


def kinds_for_mode(mode: ClassifyMode) -> list[FindingKind]:
    """The kinds a mode can produce, in ladder order.

    Loud kinds are structural and always present: a judge-only run still has
    to know a run errored before asking a model about its claims.
    """
    return [
        kind
        for kind in LADDER
        if kind in LOUD_KINDS or mode == "both" or DETECTOR_FOR_KIND[kind] == mode
    ]


def check_fingerprints(config: SweepConfig, trajectories: Iterable[Trajectory]) -> None:
    expected = config.fingerprint
    drifted = sorted({t.run_id for t in trajectories if t.config_fingerprint != expected})
    if drifted:
        raise FingerprintMismatchError(
            f"{len(drifted)} trajectory(ies) were produced under a different config "
            f"(expected fingerprint {expected}), e.g. {drifted[0]}. Comparing them "
            "would measure the settings drift rather than the routing."
        )


def classify_run(
    task: TaskSpec,
    profile: BaselineProfile,
    trajectory: Trajectory,
    *,
    mode: ClassifyMode = "structural",
    judge: Judge | None = None,
) -> Finding | None:
    """Walk the ladder and return the first finding that applies, if any."""
    if mode != "structural" and judge is None:
        raise ValueError(f"mode '{mode}' needs a judge; pass one or use mode='structural'")

    assessment: JudgeAssessment | None = None
    for kind in kinds_for_mode(mode):
        if kind in STRUCTURAL:
            evidence = STRUCTURAL[kind](task, profile, trajectory)
        else:
            if assessment is None:
                assert judge is not None
                assessment = judge.assess(task, trajectory)
            evidence = JUDGED[kind](assessment, trajectory)
        if evidence is not None:
            return Finding(
                kind=kind,
                detector=DETECTOR_FOR_KIND[kind],
                run_id=trajectory.run_id,
                task_id=trajectory.task_id,
                arm=trajectory.arm,
                replicate=trajectory.replicate,
                evidence=evidence,
            )
    return None


def _rates(findings: list[Finding], replicate_count: int, mode: ClassifyMode) -> list[FindingRate]:
    return [
        FindingRate(
            kind=kind,
            detector=DETECTOR_FOR_KIND[kind],
            occurrences=sum(1 for f in findings if f.kind == kind),
            replicate_count=replicate_count,
            findings=[f for f in findings if f.kind == kind],
        )
        for kind in kinds_for_mode(mode)
    ]


def _by_replicate(trajectories: Iterable[Trajectory]) -> list[Trajectory]:
    return sorted(trajectories, key=lambda t: (t.replicate, t.run_id))


def compute_noise_floor(
    task: TaskSpec,
    baseline: list[Trajectory],
    *,
    config: SweepConfig,
    mode: ClassifyMode = "structural",
    judge: Judge | None = None,
    baseline_arm: str = DEFAULT_BASELINE_ARM,
) -> NoiseFloor:
    """Leave-one-out: score each baseline run against a profile of the rest.

    Every finding this produces is a false positive by construction. With
    fewer than two baseline runs there is no "rest", and the floor is empty
    rather than invented.
    """
    runs = _by_replicate(baseline)
    findings: list[Finding] = []
    if len(runs) >= 2:
        for i, held_out in enumerate(runs):
            rest = runs[:i] + runs[i + 1 :]
            profile = build_profile(
                task,
                rest,
                baseline_arm=baseline_arm,
                reliability_threshold=config.reliability_threshold,
            )
            finding = classify_run(task, profile, held_out, mode=mode, judge=judge)
            if finding is not None:
                findings.append(finding)
    count = len(runs) if len(runs) >= 2 else 0
    return NoiseFloor(
        task_id=task.task_id,
        baseline_arm=baseline_arm,
        replicate_count=count,
        finding_rates=_rates(findings, count, mode),
    )


def classify(
    task: TaskSpec,
    baseline: list[Trajectory],
    downgraded: list[Trajectory],
    *,
    config: SweepConfig,
    mode: ClassifyMode = "structural",
    judge: Judge | None = None,
    noise_floor: NoiseFloor | None = None,
    baseline_arm: str = DEFAULT_BASELINE_ARM,
) -> ArmComparison:
    """Score one downgraded arm against the baseline arm, for one task."""
    validate_constraints(task)
    check_fingerprints(config, [*baseline, *downgraded])
    arms = {t.arm for t in downgraded}
    if len(arms) > 1:
        raise ValueError(f"downgraded trajectories span several arms: {sorted(arms)}")
    stray = {t.task_id for t in [*baseline, *downgraded]} - {task.task_id}
    if stray:
        raise ValueError(f"trajectories for other tasks passed to '{task.task_id}': {stray}")

    base_runs = _by_replicate(baseline)
    down_runs = _by_replicate(downgraded)
    profile = build_profile(
        task,
        base_runs,
        baseline_arm=baseline_arm,
        reliability_threshold=config.reliability_threshold,
    )
    if noise_floor is None:
        noise_floor = compute_noise_floor(
            task, base_runs, config=config, mode=mode, judge=judge, baseline_arm=baseline_arm
        )

    findings = [
        f
        for t in down_runs
        if (f := classify_run(task, profile, t, mode=mode, judge=judge)) is not None
    ]
    routed = [s.route.downgraded for t in down_runs for s in t.steps]
    attributed = [r for r in routed if r is not None]
    down_profile_correct = build_profile(
        task, down_runs, baseline_arm=baseline_arm, reliability_threshold=1.0
    ).correct_rate

    return ArmComparison(
        task_id=task.task_id,
        baseline_arm=baseline_arm,
        downgraded_arm=next(iter(arms)) if arms else "",
        baseline_run_ids=[t.run_id for t in base_runs],
        downgraded_run_ids=[t.run_id for t in down_runs],
        mode=mode,
        profile=profile,
        finding_rates=_rates(findings, len(down_runs), mode),
        noise_floor=noise_floor,
        baseline_correct_rate=profile.correct_rate,
        downgraded_correct_rate=down_profile_correct,
        route_downgrade_rate=(sum(attributed) / len(attributed)) if attributed else 0.0,
    )


def classify_sweep(
    tasks: dict[str, TaskSpec],
    trajectories: list[Trajectory],
    *,
    config: SweepConfig,
    mode: ClassifyMode = "structural",
    judge: Judge | None = None,
    baseline_arm: str = DEFAULT_BASELINE_ARM,
) -> list[ArmComparison]:
    """Classify every (task, downgraded arm) pair in a sweep.

    The noise floor is computed once per task and shared by every arm, and
    the judge is wrapped in a per-run cache, so each run is judged at most
    once however many comparisons it takes part in.
    """
    check_fingerprints(config, trajectories)
    unknown = sorted({t.task_id for t in trajectories} - set(tasks))
    if unknown:
        raise KeyError(f"no task spec for: {', '.join(unknown)}")
    cached: Judge | None = CachedJudge(judge) if judge is not None else None

    comparisons: list[ArmComparison] = []
    for task_id in sorted({t.task_id for t in trajectories}):
        task = tasks[task_id]
        runs = [t for t in trajectories if t.task_id == task_id]
        baseline = [t for t in runs if t.arm == baseline_arm]
        if not baseline:
            raise ValueError(f"task '{task_id}' has no '{baseline_arm}' runs to compare against")
        floor = compute_noise_floor(
            task, baseline, config=config, mode=mode, judge=cached, baseline_arm=baseline_arm
        )
        for arm in sorted({t.arm for t in runs} - {baseline_arm}):
            comparisons.append(
                classify(
                    task,
                    baseline,
                    [t for t in runs if t.arm == arm],
                    config=config,
                    mode=mode,
                    judge=cached,
                    noise_floor=floor,
                    baseline_arm=baseline_arm,
                )
            )
    return comparisons


__all__ = [
    "JUDGED",
    "LADDER",
    "STRUCTURAL",
    "FingerprintMismatchError",
    "check_fingerprints",
    "classify",
    "classify_run",
    "classify_sweep",
    "compute_noise_floor",
    "kinds_for_mode",
]
