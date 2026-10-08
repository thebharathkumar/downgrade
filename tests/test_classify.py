"""Classifier tests: each detector, the priority ladder, the noise floor.

The property these defend: a finding is reported only when the baseline arm
reliably did something this run did not, and baseline compared with itself
produces nothing when it is stable.
"""

from __future__ import annotations

from typing import Any

import pytest
from _builders import (
    CHEAP,
    GOOD_CALLS,
    arm_runs,
    config,
    task,
    trajectory,
)

from downgrade.classify import (
    FingerprintMismatchError,
    build_profile,
    classify,
    classify_run,
    classify_sweep,
    compute_noise_floor,
    kinds_for_mode,
)
from downgrade.classify.answer import check_answer, has_expectation
from downgrade.classify.constraints import (
    PREDICATES,
    UnknownPredicateError,
    check_constraint,
    validate_constraints,
)
from downgrade.classify.judge import JudgeAssessment, JudgeIssue
from downgrade.classify.values import (
    canonical,
    collect_values,
    extract_numbers,
    is_grounded,
)
from downgrade.models import FindingKind, Trajectory
from downgrade.suite.spec import AnswerCheck, Constraint, TaskSpec


def profile_for(t: TaskSpec | None = None, runs: list[Trajectory] | None = None) -> Any:
    return build_profile(
        t or task(),
        runs if runs is not None else arm_runs("baseline_direct"),
        baseline_arm="baseline_direct",
        reliability_threshold=0.8,
    )


def run_kind(t: Trajectory, mode: str = "structural", judge: Any = None) -> FindingKind | None:
    finding = classify_run(task(), profile_for(), t, mode=mode, judge=judge)  # type: ignore[arg-type]
    return finding.kind if finding else None


class TestValues:
    def test_extracts_numbers_with_their_precision(self) -> None:
        tokens = extract_numbers("Revenue was 1,234.5 and margin 12%, up from -3.")
        assert [(t.value, t.decimals) for t in tokens] == [(1234.5, 1), (12.0, 0), (-3.0, 0)]

    def test_ignores_numbers_glued_to_words(self) -> None:
        assert extract_numbers("item1a and v2") == []

    def test_empty_text_has_no_numbers(self) -> None:
        assert extract_numbers(None) == []

    def test_collects_numbers_from_nested_tool_output(self) -> None:
        content = {"rows": [{"v": 3}, {"v": "4.5 million"}], "ok": True, "none": None}
        assert collect_values(content) == {3.0, 4.5}

    def test_rounded_answer_is_grounded_by_precise_value(self) -> None:
        (token,) = extract_numbers("1.2 billion")
        assert is_grounded(token, {1_234_000_000.0})

    def test_overstated_precision_is_not_grounded(self) -> None:
        (token,) = extract_numbers("1.25")
        assert not is_grounded(token, {1.234})

    def test_percent_matches_a_fraction(self) -> None:
        (token,) = extract_numbers("12.5%")
        assert is_grounded(token, {0.125})

    def test_small_integers_are_flagged_as_ordinals(self) -> None:
        (token,) = extract_numbers("3")
        assert token.is_small_integer

    def test_canonical_is_stable(self) -> None:
        assert canonical(1000.0) == canonical(1e3) == "1000"


class TestAnswerCheck:
    @pytest.mark.parametrize(
        ("check", "answer", "expected"),
        [
            (AnswerCheck(kind="numeric", value=1000.0), "It is 1,000.", True),
            (AnswerCheck(kind="numeric", value=1000.0), "It is 1,200.", False),
            (AnswerCheck(kind="exact", value="Yes"), " yes ", True),
            (AnswerCheck(kind="regex", pattern=r"acme"), "ACME leads", True),
            (AnswerCheck(kind="regex", value=r"^no$"), "yes", False),
            (AnswerCheck(kind="set", value=["ACME", "BOREAS"]), "acme and boreas", True),
            (AnswerCheck(kind="set", value="ACME"), "boreas", False),
            (AnswerCheck(kind="numeric", value=1.0), None, False),
        ],
    )
    def test_checks(self, check: AnswerCheck, answer: str | None, expected: bool) -> None:
        assert check_answer(check, answer) is expected

    def test_expectation(self) -> None:
        assert not has_expectation(AnswerCheck(kind="numeric", value=None))
        assert has_expectation(AnswerCheck(kind="regex", pattern="x"))


class TestConstraints:
    def check(self, predicate: str, args: dict[str, Any], t: Trajectory) -> bool:
        return check_constraint(Constraint(id="c", predicate=predicate, args=args), t).satisfied

    def test_every_predicate_on_a_good_run(self) -> None:
        t = trajectory()
        assert self.check("tool_called", {"name": "get_figure", "min": 4}, t)
        assert self.check("tool_not_called", {"name": "search"}, t)
        assert self.check(
            "tool_called_with", {"name": "get_figure", "arguments": {"company": "acme"}}, t
        )
        assert not self.check(
            "tool_called_with", {"name": "get_figure", "arguments": {"company": "ZED"}}, t
        )
        assert self.check("max_tool_calls", {"n": 4}, t)
        assert not self.check("max_tool_calls", {"n": 3}, t)
        assert self.check("answer_matches", {"pattern": "equity"}, t)
        assert self.check("answer_not_matches", {"pattern": "estimate"}, t)
        assert self.check("answer_max_words", {"n": 4}, t)
        assert self.check("answer_contains_all", {"values": ["equity", "1000"]}, t)
        assert not self.check("answer_contains_all", {"values": ["revenue"]}, t)

    def test_called_before(self) -> None:
        calls = [("list", {}, []), ("get_figure", {}, {"value": 1})]
        ordered = trajectory(calls=calls)
        reversed_ = trajectory(calls=list(reversed(calls)))
        skipped = trajectory(calls=[calls[1]])
        args = {"first": "list", "then": "get_figure"}
        assert self.check("called_before", args, ordered)
        assert not self.check("called_before", args, reversed_)
        assert not self.check("called_before", args, skipped)
        assert self.check("called_before", {"first": "list", "then": "nope"}, ordered)

    def test_unknown_predicate_fails_validation(self) -> None:
        bad = task(constraints=[Constraint(id="x", predicate="vibes_ok")])
        with pytest.raises(UnknownPredicateError, match="vibes_ok"):
            validate_constraints(bad)
        with pytest.raises(UnknownPredicateError):
            check_constraint(bad.constraints[0], trajectory())

    def test_missing_argument_is_an_authoring_error(self) -> None:
        with pytest.raises(UnknownPredicateError, match="missing argument"):
            check_constraint(Constraint(id="x", predicate="tool_called"), trajectory())

    def test_registry_is_closed(self) -> None:
        assert "judge" not in " ".join(PREDICATES)


class TestProfile:
    def test_rates_over_a_stable_baseline(self) -> None:
        p = profile_for()
        assert p.replicate_count == 5
        assert all(rate == 1.0 for rate in p.tool_signature_rates.values())
        assert p.tool_name_rates == {"get_figure": 1.0}
        assert p.correct_rate == 1.0
        assert p.constraint_satisfaction_rates == {"uses_lookup": 1.0, "short": 1.0}
        assert canonical(1000.0) in p.grounded_values
        assert p.distinct_answers == 1 and p.modal_answer_rate == 1.0
        assert p.tool_call_count_stdev == 0.0

    def test_empty_baseline(self) -> None:
        p = profile_for(runs=[])
        assert p.replicate_count == 0 and p.tool_signature_rates == {}

    def test_cited_documents_are_recorded(self) -> None:
        runs = [trajectory(calls=[("read", {"doc_id": "10k-2024"}, "text")])]
        assert profile_for(runs=runs).cited_documents == {"10k-2024"}

    def test_unreliable_answer_numbers_are_not_grounded(self) -> None:
        runs = arm_runs("baseline_direct", n=4) + [
            trajectory(replicate=9, answer="Combined equity is 1000, or 777.5 adjusted.")
        ]
        assert canonical(777.5) not in profile_for(runs=runs).grounded_values


class TestDetectors:
    def test_a_run_matching_baseline_is_clean(self) -> None:
        assert run_kind(trajectory(arm="pref_5", served=CHEAP)) is None

    def test_errored_run_is_loud(self) -> None:
        assert run_kind(trajectory(status="errored", error="HTTP 500")) == FindingKind.RUN_ERROR

    def test_no_termination_is_loud(self) -> None:
        assert run_kind(trajectory(status="no_termination")) == FindingKind.NO_TERMINATION

    def test_wrong_answer_is_loud_and_outranks_silent(self) -> None:
        t = trajectory(calls=GOOD_CALLS[:1], answer="Equity is 1200.")
        assert run_kind(t) == FindingKind.WRONG_ANSWER

    def test_missing_tool_call_with_the_right_answer_is_silent(self) -> None:
        t = trajectory(calls=GOOD_CALLS[:3])
        finding = classify_run(task(), profile_for(), t)
        assert finding is not None
        assert finding.kind == FindingKind.MISSING_TOOL_CALL
        assert not finding.is_loud
        assert "5/5 baseline runs" in (finding.evidence.baseline_reference or "")

    def test_missing_tool_name_when_signatures_vary(self) -> None:
        baseline = [
            trajectory(replicate=r, calls=[("lookup", {"q": f"v{r}"}, {"value": 1000.0})])
            for r in range(5)
        ]
        p = profile_for(runs=baseline)
        t = trajectory(calls=[("other", {}, {"value": 1000.0})])
        finding = classify_run(task(constraints=[]), p, t)
        assert finding is not None and finding.kind == FindingKind.MISSING_TOOL_CALL
        assert "lookup" in finding.evidence.detail

    def test_dropped_constraint(self) -> None:
        long = "Combined equity is 1000. " + "word " * 40
        finding = classify_run(task(), profile_for(), trajectory(answer=long))
        assert finding is not None and finding.kind == FindingKind.DROPPED_CONSTRAINT
        assert finding.evidence.constraint_id == "short"

    def test_constraint_unreliable_in_baseline_is_not_expected(self) -> None:
        long = "Combined equity is 1000. " + "word " * 40
        baseline = arm_runs("baseline_direct", n=3) + [
            trajectory(replicate=r, answer=long) for r in (3, 4)
        ]
        assert classify_run(task(), profile_for(runs=baseline), trajectory(answer=long)) is None

    def test_redundant_retry(self) -> None:
        calls = [*GOOD_CALLS, GOOD_CALLS[0], GOOD_CALLS[0]]
        finding = classify_run(task(), profile_for(), trajectory(calls=calls))
        assert finding is not None and finding.kind == FindingKind.REDUNDANT_RETRY
        assert finding.evidence.downgraded_value == "3"
        assert finding.evidence.step_index == 4

    def test_fabricated_value(self) -> None:
        t = trajectory(answer="Combined equity is 1000, about 4,321 per share.")
        finding = classify_run(task(), profile_for(), t)
        assert finding is not None and finding.kind == FindingKind.FABRICATED_VALUE
        assert finding.evidence.quoted_span == "4,321"

    def test_prompt_numbers_are_grounded(self) -> None:
        t = task(prompt="For fiscal 2024, report combined equity.")
        p = profile_for(t=t)
        run = trajectory(answer="In 2024 combined equity is 1000.")
        assert classify_run(t, p, run) is None

    def test_judge_mode_without_a_judge_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs a judge"):
            classify_run(task(), profile_for(), trajectory(), mode="judge")


class FakeJudge:
    def __init__(self, flagged: set[str], citations: set[str] | None = None) -> None:
        self.flagged = flagged
        self.citations = citations or set()
        self.calls: list[str] = []

    def assess(self, task: TaskSpec, trajectory: Trajectory) -> JudgeAssessment:
        self.calls.append(trajectory.run_id)
        claims = (
            [JudgeIssue(claim="x", reason="no source")] if trajectory.run_id in self.flagged else []
        )
        cites = (
            [JudgeIssue(claim="per 10-K", reason="wrong section")]
            if trajectory.run_id in self.citations
            else []
        )
        return JudgeAssessment(unsupported_claims=claims, unsupported_citations=cites)


class TestJudgeLadder:
    def test_unsupported_claim_outranks_structural_silent(self) -> None:
        t = trajectory(arm="pref_5", calls=GOOD_CALLS[:3])
        judge = FakeJudge({t.run_id})
        assert run_kind(t, mode="both", judge=judge) == FindingKind.UNSUPPORTED_CLAIM

    def test_judge_only_mode_skips_structural_silent(self) -> None:
        t = trajectory(arm="pref_5", calls=GOOD_CALLS[:3])
        assert run_kind(t, mode="judge", judge=FakeJudge(set())) is None

    def test_unsupported_citation(self) -> None:
        t = trajectory(arm="pref_5")
        judge = FakeJudge(set(), citations={t.run_id})
        assert run_kind(t, mode="judge", judge=judge) == FindingKind.UNSUPPORTED_CITATION

    def test_unavailable_assessment_yields_nothing(self) -> None:
        class Down:
            def assess(self, task: TaskSpec, trajectory: Trajectory) -> JudgeAssessment:
                return JudgeAssessment(
                    unsupported_claims=[JudgeIssue(claim="x", reason="y")], available=False
                )

        assert run_kind(trajectory(), mode="judge", judge=Down()) is None

    def test_loud_runs_never_reach_the_judge(self) -> None:
        judge = FakeJudge(set())
        run_kind(trajectory(status="errored"), mode="judge", judge=judge)
        assert judge.calls == []

    def test_modes_partition_the_silent_kinds(self) -> None:
        structural = set(kinds_for_mode("structural"))
        judged = set(kinds_for_mode("judge"))
        both = set(kinds_for_mode("both"))
        assert structural | judged == both
        assert structural & judged == {
            FindingKind.WRONG_ANSWER,
            FindingKind.RUN_ERROR,
            FindingKind.NO_TERMINATION,
        }


class TestNoiseFloor:
    def test_stable_baseline_has_a_zero_floor(self) -> None:
        floor = compute_noise_floor(task(), arm_runs("baseline_direct"), config=config())
        assert floor.replicate_count == 5
        assert floor.total_false_positive_rate == 0.0

    def test_variable_baseline_produces_false_positives(self) -> None:
        runs = arm_runs("baseline_direct", n=4) + [
            trajectory(replicate=4, calls=GOOD_CALLS[:2], answer="Equity is 1000.")
        ]
        floor = compute_noise_floor(task(), runs, config=config())
        assert floor.rate_for(FindingKind.MISSING_TOOL_CALL) == pytest.approx(0.2)

    def test_single_baseline_run_has_no_floor(self) -> None:
        floor = compute_noise_floor(task(), arm_runs("baseline_direct", n=1), config=config())
        assert floor.replicate_count == 0 and floor.finding_rates[0].rate == 0.0


class TestClassify:
    def test_clean_arm(self) -> None:
        comparison = classify(
            task(), arm_runs("baseline_direct"), arm_runs("pref_5", served=CHEAP), config=config()
        )
        assert comparison.verdict == "CLEAN"
        assert comparison.route_downgrade_rate == 1.0
        assert comparison.downgraded_correct_rate == 1.0

    def test_silent_arm(self) -> None:
        down = arm_runs("pref_5", calls=GOOD_CALLS[:3])
        comparison = classify(task(), arm_runs("baseline_direct"), down, config=config())
        assert comparison.verdict == "SILENT"
        assert comparison.excess_over_floor(FindingKind.MISSING_TOOL_CALL) == 1.0
        assert len(comparison.all_findings) == 5

    def test_fingerprint_drift_is_refused(self) -> None:
        drifted = arm_runs("pref_5", fingerprint=config(temperature=0.7).fingerprint)
        with pytest.raises(FingerprintMismatchError, match="drift"):
            classify(task(), arm_runs("baseline_direct"), drifted, config=config())

    def test_mixed_arms_are_refused(self) -> None:
        mixed = arm_runs("pref_4", n=2) + arm_runs("pref_5", n=2)
        with pytest.raises(ValueError, match="several arms"):
            classify(task(), arm_runs("baseline_direct"), mixed, config=config())

    def test_foreign_task_is_refused(self) -> None:
        with pytest.raises(ValueError, match="other tasks"):
            classify(
                task(),
                arm_runs("baseline_direct"),
                arm_runs("pref_5", task_id="other"),
                config=config(),
            )


class TestClassifySweep:
    def test_one_comparison_per_task_and_arm(self) -> None:
        runs = [
            *arm_runs("baseline_direct"),
            *arm_runs("pref_1"),
            *arm_runs("pref_5", calls=GOOD_CALLS[:3]),
        ]
        comparisons = classify_sweep({"equity": task()}, runs, config=config())
        assert [c.downgraded_arm for c in comparisons] == ["pref_1", "pref_5"]
        assert comparisons[0].noise_floor is comparisons[1].noise_floor

    def test_judge_is_called_once_per_run(self) -> None:
        runs = [*arm_runs("baseline_direct"), *arm_runs("pref_1"), *arm_runs("pref_5")]
        judge = FakeJudge(set())
        classify_sweep({"equity": task()}, runs, config=config(), mode="judge", judge=judge)
        assert len(judge.calls) == len(set(judge.calls)) == 15

    def test_unknown_task_and_missing_baseline(self) -> None:
        with pytest.raises(KeyError, match="no task spec"):
            classify_sweep({}, arm_runs("baseline_direct"), config=config())
        with pytest.raises(ValueError, match="no 'baseline_direct' runs"):
            classify_sweep({"equity": task()}, arm_runs("pref_5"), config=config())
