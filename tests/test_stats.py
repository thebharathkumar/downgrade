"""Statistics layer tests.

The checks that matter most are the validity ones: under the null, the
anytime p-value must reject no more often than its level even when the
sample is inspected after every run, and BH must match the textbook
procedure. A test with the wrong size would make every finding in the report
an artefact.
"""

from __future__ import annotations

import math
import random

import pytest
from _builders import CHEAP, GOOD_CALLS, arm_runs, config, task, trajectory

from downgrade.classify import classify_sweep
from downgrade.models import FindingKind
from downgrade.stats import (
    analyze,
    anytime_pvalue,
    benjamini_hochberg,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
    sprt,
    wilson_upper,
)
from downgrade.stats.backtest import chi2_sf, transition_counts


class TestWilson:
    def test_zero_successes_still_has_a_positive_bound(self) -> None:
        assert 0.4 < wilson_upper(0, 5) < 0.45
        assert wilson_upper(0, 120) < 0.035

    def test_bound_is_above_the_estimate(self) -> None:
        assert wilson_upper(3, 10) > 0.3

    def test_no_trials_is_uninformative(self) -> None:
        assert wilson_upper(0, 0) == 1.0


class TestAnytimePValue:
    def test_no_data_and_no_flags(self) -> None:
        assert anytime_pvalue([], 0.1) == 1.0
        assert anytime_pvalue([False] * 50, 0.1) == 1.0

    def test_many_flags_reject(self) -> None:
        assert anytime_pvalue([True] * 20, 0.05) < 1e-6

    def test_degenerate_null_rates(self) -> None:
        assert anytime_pvalue([True], 0.0) == 0.0
        assert anytime_pvalue([False], 0.0) == 1.0
        assert anytime_pvalue([True], 1.0) == 1.0

    def test_more_evidence_never_hurts_the_running_max(self) -> None:
        flags = [True, True, True, False]
        assert anytime_pvalue(flags, 0.1) == anytime_pvalue(flags[:3], 0.1)

    @pytest.mark.parametrize("p0", [0.05, 0.2])
    def test_type_one_error_holds_under_continuous_monitoring(self, p0: float) -> None:
        rng = random.Random(7)
        alpha, trials, n = 0.05, 400, 60
        rejections = 0
        for _ in range(trials):
            flags = [rng.random() < p0 for _ in range(n)]
            if anytime_pvalue(flags, p0) <= alpha:
                rejections += 1
        assert rejections / trials <= alpha

    def test_holds_when_the_true_rate_is_below_the_null(self) -> None:
        rng = random.Random(11)
        rejections = sum(
            anytime_pvalue([rng.random() < 0.02 for _ in range(60)], 0.1) <= 0.05
            for _ in range(300)
        )
        assert rejections == 0

    def test_has_power_against_a_real_lift(self) -> None:
        rng = random.Random(3)
        hits = sum(
            anytime_pvalue([rng.random() < 0.5 for _ in range(60)], 0.05) <= 0.05
            for _ in range(100)
        )
        assert hits >= 95


class TestSPRT:
    def test_rejects_on_a_run_of_flags(self) -> None:
        result = sprt([True] * 10, 0.05, 0.3)
        assert result.decision == "reject_null"
        assert result.observations_used < 10

    def test_accepts_on_a_run_of_clean_results(self) -> None:
        assert sprt([False] * 40, 0.05, 0.3).decision == "accept_null"

    def test_continues_when_undecided(self) -> None:
        result = sprt([False, True], 0.05, 0.3)
        assert result.decision == "continue" and result.observations_used == 2
        assert result.lower < result.log_likelihood_ratio < result.upper


class TestBenjaminiHochberg:
    def test_matches_the_textbook_example(self) -> None:
        pvalues = [0.01, 0.04, 0.03, 0.005, 0.5]
        result = benjamini_hochberg(pvalues, q=0.05)
        assert result.adjusted == pytest.approx([0.025, 0.05, 0.05, 0.025, 0.5])
        assert result.rejected == [True, True, True, True, False]
        assert result.discoveries == 4

    def test_by_is_more_conservative(self) -> None:
        pvalues = [0.01, 0.02, 0.03]
        bh = benjamini_hochberg(pvalues, method="bh")
        by = benjamini_hochberg(pvalues, method="by")
        assert all(b >= a for a, b in zip(bh.adjusted, by.adjusted, strict=True))

    def test_adjusted_values_are_monotone_in_p(self) -> None:
        pvalues = [0.2, 0.001, 0.04, 0.03, 0.9]
        adjusted = benjamini_hochberg(pvalues).adjusted
        ranked = [adjusted[i] for i in sorted(range(5), key=pvalues.__getitem__)]
        assert ranked == sorted(ranked)

    def test_empty_family(self) -> None:
        assert benjamini_hochberg([]).discoveries == 0


class TestBacktests:
    def test_chi2_tails(self) -> None:
        assert chi2_sf(3.841459, 1) == pytest.approx(0.05, abs=1e-5)
        assert chi2_sf(5.991465, 2) == pytest.approx(0.05, abs=1e-5)
        assert chi2_sf(0.0, 1) == 1.0 and chi2_sf(math.inf, 2) == 0.0
        with pytest.raises(ValueError):
            chi2_sf(1.0, 3)

    def test_kupiec_accepts_the_expected_rate(self) -> None:
        violations = [i % 10 == 0 for i in range(100)]
        assert kupiec_pof(violations, 0.1).statistic == pytest.approx(0.0, abs=1e-9)

    def test_kupiec_known_value(self) -> None:
        # 10 violations in 100 against p = 0.05.
        violations = [i < 10 for i in range(100)]
        result = kupiec_pof(violations, 0.05)
        expected = -2 * (
            90 * math.log(0.95) + 10 * math.log(0.05) - 90 * math.log(0.9) - 10 * math.log(0.1)
        )
        assert result.statistic == pytest.approx(expected)
        assert result.rejects(0.05)

    def test_kupiec_with_a_zero_floor(self) -> None:
        assert kupiec_pof([False] * 10, 0.0).p_value == 1.0
        assert kupiec_pof([True] + [False] * 9, 0.0).p_value == 0.0
        assert kupiec_pof([], 0.1).p_value == 1.0

    def test_transition_counts(self) -> None:
        assert transition_counts([False, True, True, False]) == (0, 1, 1, 1)

    def test_independence_flags_clustering(self) -> None:
        clustered = [False] * 20 + [True] * 10 + [False] * 20
        assert christoffersen_independence(clustered).rejects(0.05)

    def test_independence_accepts_iid_violations(self) -> None:
        rng = random.Random(5)
        iid = [rng.random() < 0.2 for _ in range(200)]
        assert not christoffersen_independence(iid).rejects(0.05)

    def test_independence_also_rejects_strict_alternation(self) -> None:
        # Never two in a row is dependence too; the report's `clustered`
        # flag separates this from findings that follow findings.
        spread = [i % 5 == 0 for i in range(50)]
        assert christoffersen_independence(spread).rejects(0.05)

    def test_independence_degenerate_sequences(self) -> None:
        assert christoffersen_independence([]).p_value == 1.0
        assert christoffersen_independence([True] * 5).p_value == 1.0

    def test_conditional_coverage_is_the_sum(self) -> None:
        seq = [False] * 20 + [True] * 10 + [False] * 20
        joint = conditional_coverage(seq, 0.05)
        assert joint.df == 2
        assert joint.statistic == pytest.approx(
            kupiec_pof(seq, 0.05).statistic + christoffersen_independence(seq).statistic
        )


def sweep(tasks: int = 6, **pref5: object) -> list:  # type: ignore[type-arg]
    runs = []
    specs = {}
    for i in range(tasks):
        tid = f"t{i}"
        specs[tid] = task(tid)
        runs += arm_runs("baseline_direct", task_id=tid)
        runs += arm_runs("pref_1", task_id=tid)
        runs += arm_runs("pref_5", task_id=tid, served=CHEAP, **pref5)  # type: ignore[arg-type]
    return classify_sweep(specs, runs, config=config())


class TestAnalyze:
    def test_clean_sweep_reports_no_discovery(self) -> None:
        analysis = analyze(sweep(), config())
        assert analysis.discoveries == []
        assert analysis.family_size == 2 * 7
        assert {a.arm for a in analysis.arms} == {"pref_1", "pref_5"}
        assert analysis.arms[0].arm == "pref_1"
        assert all(f.flagged == 0 for f in analysis.floors)
        assert analysis.floors[0].runs == 30

    def test_silent_regression_is_discovered(self) -> None:
        analysis = analyze(sweep(calls=GOOD_CALLS[:3]), config())
        (hit,) = analysis.discoveries
        assert hit.arm == "pref_5" and hit.kind == FindingKind.MISSING_TOOL_CALL
        assert hit.flagged == 30 and hit.sprt_decision == "reject_null"
        assert analysis.silent_discoveries == [hit]
        pref5 = next(a for a in analysis.arms if a.arm == "pref_5")
        assert pref5.silent_rate == 1.0 and pref5.correct_rate == 1.0
        assert pref5.route_downgrade_rate == 1.0
        assert len(analysis.findings) == 30

    def test_backtests_cover_baseline_and_each_arm(self) -> None:
        analysis = analyze(sweep(), config())
        assert [b.arm for b in analysis.backtests] == ["baseline_direct", "pref_1", "pref_5"]
        assert analysis.backtest_for("pref_5") is not None
        assert analysis.backtest_for("nope") is None

    def test_clustered_findings_mark_the_arm_non_exchangeable(self) -> None:
        specs = {f"t{i}": task(f"t{i}") for i in range(6)}
        runs = []
        for tid in specs:
            runs += arm_runs("baseline_direct", task_id=tid)
            # Replicates 0 and 1 regress on every task; later ones do not.
            for r in range(5):
                calls = GOOD_CALLS[:3] if r < 2 else GOOD_CALLS
                runs.append(trajectory(task_id=tid, arm="pref_5", replicate=r, calls=calls))
        analysis = analyze(classify_sweep(specs, runs, config=config()), config())
        backtest = analysis.backtest_for("pref_5")
        assert backtest is not None and backtest.clustered
        assert backtest.rate_after_violation > backtest.rate_after_clean
        hit = next(h for h in analysis.hypotheses if h.kind == FindingKind.MISSING_TOOL_CALL)
        assert not hit.exchangeable

    def test_empty_and_mixed_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="no comparisons"):
            analyze([], config())
        comparisons = sweep(tasks=1)
        comparisons[0].mode = "both"
        with pytest.raises(ValueError, match="mix classify modes"):
            analyze(comparisons, config())

    def test_lookup_helpers(self) -> None:
        analysis = analyze(sweep(tasks=1), config())
        assert analysis.floor_for(FindingKind.WRONG_ANSWER) is not None
        assert analysis.floor_for(FindingKind.UNSUPPORTED_CLAIM) is None
