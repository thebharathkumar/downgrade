"""From ArmComparisons to tested claims.

The unit of inference is one (downgraded arm, finding kind) pair, pooled
across every task in the suite. Per-task replicate counts are small by design
(5 by default), and a per-task test at R=5 cannot reject anything below a
huge effect, so pooling is where the power is. Per-task numbers stay in the
report as description, not as tests.

The null rate for each kind is the baseline's own leave-one-out finding rate,
pooled across tasks and replaced by the upper end of its Wilson interval. The
null is therefore "this arm draws findings no more often than baseline
compared with itself, allowing for how imprecisely that floor is known".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, Field, computed_field

from downgrade.arms import ARMS_BY_NAME, PREFERENCE_LABELS
from downgrade.classify.core import kinds_for_mode
from downgrade.models import (
    DETECTOR_FOR_KIND,
    LOUD_KINDS,
    ArmComparison,
    ClassifyMode,
    Detector,
    Finding,
    FindingKind,
    SweepConfig,
)
from downgrade.stats.backtest import (
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    transition_counts,
)
from downgrade.stats.multiple import benjamini_hochberg
from downgrade.stats.sequential import SPRTDecision, anytime_pvalue, sprt, wilson_upper

# Level at which a backtest is read as failing. Fixed here, not configurable,
# because it only decides whether a warning is printed next to an arm.
BACKTEST_ALPHA = 0.05


class FloorEstimate(BaseModel):
    """Baseline-against-itself finding rate for one kind, pooled over tasks."""

    kind: FindingKind
    detector: Detector
    flagged: int
    runs: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rate(self) -> float:
        return self.flagged / self.runs if self.runs else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def null_rate(self) -> float:
        """The rate the test assumes: the Wilson upper bound of `rate`."""
        return wilson_upper(self.flagged, self.runs)


class Hypothesis(BaseModel):
    """H0: this arm draws `kind` findings no more often than the floor."""

    arm: str
    kind: FindingKind
    detector: Detector
    runs: int
    flagged: int
    floor_rate: float
    null_rate: float
    p_value: float
    p_adjusted: float = 1.0
    rejected: bool = False
    sprt_decision: SPRTDecision = "continue"
    sprt_runs: int = 0
    exchangeable: bool = True

    @computed_field  # type: ignore[prop-decorator]
    @property
    def observed_rate(self) -> float:
        return self.flagged / self.runs if self.runs else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def excess(self) -> float:
        return max(0.0, self.observed_rate - self.floor_rate)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_loud(self) -> bool:
        return self.kind in LOUD_KINDS


class Backtest(BaseModel):
    """Kupiec and Christoffersen on one arm's any-finding sequence."""

    arm: str
    runs: int
    violations: int
    expected_rate: float
    kupiec_statistic: float
    kupiec_p: float
    independence_statistic: float
    independence_p: float
    coverage_statistic: float
    coverage_p: float
    rate_after_violation: float = 0.0
    rate_after_clean: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def violation_rate(self) -> float:
        return self.violations / self.runs if self.runs else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clustered(self) -> bool:
        """Dependence in the direction that matters: findings follow findings.

        The independence test is two-sided, so it also rejects violations
        that alternate too regularly. That is dependence too, but not the
        warm-cache or drift pattern this flag warns about.
        """
        return (
            self.independence_p <= BACKTEST_ALPHA
            and self.rate_after_violation > self.rate_after_clean
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coverage_ok(self) -> bool:
        return self.kupiec_p > BACKTEST_ALPHA


class ArmSummary(BaseModel):
    arm: str
    preference: int | None
    label: str
    tasks: int
    runs: int
    route_downgrade_rate: float
    baseline_correct_rate: float
    correct_rate: float
    loud_runs: int
    silent_runs: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def loud_rate(self) -> float:
        return self.loud_runs / self.runs if self.runs else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def silent_rate(self) -> float:
        return self.silent_runs / self.runs if self.runs else 0.0


class SweepAnalysis(BaseModel):
    config: SweepConfig
    mode: ClassifyMode
    task_ids: list[str] = Field(default_factory=list)
    floors: list[FloorEstimate] = Field(default_factory=list)
    arms: list[ArmSummary] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    backtests: list[Backtest] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def family_size(self) -> int:
        return len(self.hypotheses)

    @property
    def discoveries(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.rejected]

    @property
    def silent_discoveries(self) -> list[Hypothesis]:
        return [h for h in self.discoveries if not h.is_loud]

    def backtest_for(self, arm: str) -> Backtest | None:
        return next((b for b in self.backtests if b.arm == arm), None)

    def floor_for(self, kind: FindingKind) -> FloorEstimate | None:
        return next((f for f in self.floors if f.kind == kind), None)


def _replicate_major(per_task: list[list[str]]) -> list[str]:
    """Interleave tasks: replicate 0 of each task, then replicate 1, ..."""
    out: list[str] = []
    depth = max((len(ids) for ids in per_task), default=0)
    for r in range(depth):
        for ids in per_task:
            if r < len(ids):
                out.append(ids[r])
    return out


def _kind_by_run(findings: Iterable[Finding]) -> dict[str, FindingKind]:
    return {f.run_id: f.kind for f in findings}


def analyze(comparisons: list[ArmComparison], config: SweepConfig) -> SweepAnalysis:
    """Turn classified comparisons into floors, tests and backtests."""
    if not comparisons:
        raise ValueError("no comparisons to analyse")
    modes = {c.mode for c in comparisons}
    if len(modes) != 1:
        raise ValueError(f"comparisons mix classify modes: {sorted(modes)}")
    mode: ClassifyMode = next(iter(modes))
    kinds = kinds_for_mode(mode)

    by_task: dict[str, ArmComparison] = {}
    by_arm: dict[str, list[ArmComparison]] = defaultdict(list)
    for comparison in sorted(comparisons, key=lambda c: (c.task_id, c.downgraded_arm)):
        by_task.setdefault(comparison.task_id, comparison)
        by_arm[comparison.downgraded_arm].append(comparison)
    task_ids = sorted(by_task)

    # Floors: one noise floor per task, shared by every arm of that task.
    floor_findings: list[Finding] = []
    floor_runs = 0
    baseline_sequences: list[list[str]] = []
    for task_id in task_ids:
        first = by_task[task_id]
        if first.noise_floor is None or first.noise_floor.replicate_count == 0:
            continue
        floor_runs += first.noise_floor.replicate_count
        baseline_sequences.append(first.baseline_run_ids)
        for rate in first.noise_floor.finding_rates:
            floor_findings.extend(rate.findings)
    floors = [
        FloorEstimate(
            kind=kind,
            detector=DETECTOR_FOR_KIND[kind],
            flagged=sum(1 for f in floor_findings if f.kind == kind),
            runs=floor_runs,
        )
        for kind in kinds
    ]
    floor_by_kind = {f.kind: f for f in floors}
    any_floor_rate = len(floor_findings) / floor_runs if floor_runs else 0.0

    backtests: list[Backtest] = []
    if floor_runs:
        baseline_flags = _kind_by_run(floor_findings)
        sequence = [r in baseline_flags for r in _replicate_major(baseline_sequences)]
        backtests.append(_backtest(comparisons[0].baseline_arm, sequence, any_floor_rate))

    hypotheses: list[Hypothesis] = []
    summaries: list[ArmSummary] = []
    all_findings: list[Finding] = []
    for arm, arm_comparisons in sorted(by_arm.items()):
        arm_findings = [f for c in arm_comparisons for f in c.all_findings]
        all_findings.extend(arm_findings)
        kind_of = _kind_by_run(arm_findings)
        order = _replicate_major([c.downgraded_run_ids for c in arm_comparisons])

        backtest = _backtest(arm, [r in kind_of for r in order], any_floor_rate)
        backtests.append(backtest)

        for kind in kinds:
            flags = [kind_of.get(r) == kind for r in order]
            floor = floor_by_kind[kind]
            null = floor.null_rate
            decision = sprt(
                flags,
                null,
                min(0.999, null + config.min_detectable_lift),
                alpha=config.sprt_alpha,
                beta=config.sprt_beta,
            )
            hypotheses.append(
                Hypothesis(
                    arm=arm,
                    kind=kind,
                    detector=DETECTOR_FOR_KIND[kind],
                    runs=len(flags),
                    flagged=sum(flags),
                    floor_rate=floor.rate,
                    null_rate=null,
                    p_value=anytime_pvalue(flags, null),
                    sprt_decision=decision.decision,
                    sprt_runs=decision.observations_used,
                    exchangeable=not backtest.clustered,
                )
            )

        spec = ARMS_BY_NAME.get(arm)
        preference = spec.preference if spec is not None else None
        runs = len(order)
        summaries.append(
            ArmSummary(
                arm=arm,
                preference=preference,
                label=PREFERENCE_LABELS.get(preference or 0, "unknown"),
                tasks=len(arm_comparisons),
                runs=runs,
                route_downgrade_rate=_weighted(arm_comparisons, "route_downgrade_rate"),
                baseline_correct_rate=_weighted(arm_comparisons, "baseline_correct_rate"),
                correct_rate=_weighted(arm_comparisons, "downgraded_correct_rate"),
                loud_runs=sum(1 for f in arm_findings if f.is_loud),
                silent_runs=sum(1 for f in arm_findings if not f.is_loud),
            )
        )

    fdr = benjamini_hochberg([h.p_value for h in hypotheses], q=config.fdr_q)
    for hypothesis, adjusted, rejected in zip(hypotheses, fdr.adjusted, fdr.rejected, strict=True):
        hypothesis.p_adjusted = adjusted
        hypothesis.rejected = rejected

    summaries.sort(key=lambda s: (s.preference is None, s.preference or 0, s.arm))
    return SweepAnalysis(
        config=config,
        mode=mode,
        task_ids=task_ids,
        floors=floors,
        arms=summaries,
        hypotheses=hypotheses,
        backtests=backtests,
        findings=sorted(all_findings, key=lambda f: (f.arm, f.severity, f.task_id, f.replicate)),
    )


def _weighted(comparisons: list[ArmComparison], field: str) -> float:
    """Mean of a per-task rate, weighted by that task's downgraded runs."""
    total = sum(len(c.downgraded_run_ids) for c in comparisons)
    if total == 0:
        return 0.0
    weighted = sum(float(getattr(c, field)) * len(c.downgraded_run_ids) for c in comparisons)
    return weighted / total


def _backtest(arm: str, violations: list[bool], expected: float) -> Backtest:
    pof = kupiec_pof(violations, expected)
    ind = christoffersen_independence(violations)
    cc = conditional_coverage(violations, expected)
    n00, n01, n10, n11 = transition_counts(violations)
    return Backtest(
        arm=arm,
        runs=len(violations),
        violations=sum(violations),
        expected_rate=expected,
        kupiec_statistic=pof.statistic,
        kupiec_p=pof.p_value,
        independence_statistic=ind.statistic,
        independence_p=ind.p_value,
        coverage_statistic=cc.statistic,
        coverage_p=cc.p_value,
        rate_after_violation=n11 / (n10 + n11) if (n10 + n11) else 0.0,
        rate_after_clean=n01 / (n00 + n01) if (n00 + n01) else 0.0,
    )


__all__ = [
    "BACKTEST_ALPHA",
    "ArmSummary",
    "Backtest",
    "FloorEstimate",
    "Hypothesis",
    "SweepAnalysis",
    "analyze",
]
