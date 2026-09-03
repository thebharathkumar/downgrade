"""Tests for the core data model.

The properties worth protecting here are the ones a later refactor could
quietly break: that usage never reports a plausible-looking zero, that the
router model string is built in the form the API expects, that a comparison
is set-shaped rather than pair-shaped, and that the config fingerprint
actually changes when a sampling setting drifts.
"""

from __future__ import annotations

import json

import pytest

from downgrade.models import (
    DETECTOR_FOR_KIND,
    LOUD_KINDS,
    ArmComparison,
    BaselineProfile,
    Evidence,
    Finding,
    FindingKind,
    FindingRate,
    RouteObservation,
    Step,
    SweepConfig,
    ToolCall,
    Trajectory,
    Usage,
    short_slug,
)


def make_route(turn: int = 0, served: str | None = "cheap-model", **kw: object) -> RouteObservation:
    defaults: dict[str, object] = {
        "turn_index": turn,
        "requested_model": "firerouter/primary/cheap",
        "served_model": served,
        "preference": 5,
        "temperature": 0.0,
        "usage": Usage(input_tokens=10, output_tokens=5, complete=True),
    }
    defaults.update(kw)
    return RouteObservation.model_validate(defaults)


def make_trajectory(*, served: list[str | None], **kw: object) -> Trajectory:
    steps = [Step(index=i, route=make_route(i, m)) for i, m in enumerate(served)]
    defaults: dict[str, object] = {
        "run_id": "r1",
        "task_id": "t1",
        "arm": "pref_5",
        "replicate": 0,
        "conversation_id": "conv-1",
        "config_fingerprint": "abc123",
        "steps": steps,
        "final_answer": "42",
    }
    defaults.update(kw)
    return Trajectory.model_validate(defaults)


class TestUsage:
    def test_addition_sums_tokens(self) -> None:
        total = Usage(input_tokens=10, output_tokens=5, complete=True) + Usage(
            input_tokens=3, output_tokens=2, complete=True
        )
        assert (total.input_tokens, total.output_tokens) == (13, 7)
        assert total.complete

    def test_incomplete_usage_poisons_the_total(self) -> None:
        """A truncated stream must not yield a confident-looking total.

        Fireworks sends usage only in the final chunk, so a partially consumed
        stream contributes zeros. Propagating `complete=False` is what stops
        the cost model from silently under-reporting.
        """
        total = Usage(input_tokens=10, output_tokens=5, complete=True) + Usage(complete=False)
        assert total.input_tokens == 10
        assert not total.complete

    def test_defaults_are_not_complete(self) -> None:
        assert not Usage().complete


class TestToolCall:
    def test_signature_is_argument_order_independent(self) -> None:
        a = ToolCall(call_id="1", name="query", arguments={"b": 2, "a": 1})
        b = ToolCall(call_id="2", name="query", arguments={"a": 1, "b": 2})
        assert a.signature() == b.signature()

    def test_signature_distinguishes_different_arguments(self) -> None:
        """Same tool, narrower filter, is not the same call."""
        a = ToolCall(call_id="1", name="query", arguments={"year": 2024})
        b = ToolCall(call_id="2", name="query", arguments={"year": 2023})
        assert a.signature() != b.signature()

    def test_signature_survives_non_json_arguments(self) -> None:
        call = ToolCall(call_id="1", name="q", arguments={"when": object()})
        assert call.signature().startswith("q(")


class TestTrajectory:
    def test_totals_sum_across_steps(self) -> None:
        traj = make_trajectory(served=["m", "m", "m"])
        assert traj.totals.input_tokens == 30
        assert traj.totals.output_tokens == 15

    def test_route_stability_true_for_single_model(self) -> None:
        assert make_trajectory(served=["m", "m", "m"]).route_stability

    def test_route_stability_false_when_route_drifts_mid_run(self) -> None:
        """Routing is cached per conversation, but the cache can still turn over."""
        traj = make_trajectory(served=["primary", "primary", "cheap"])
        assert not traj.route_stability
        assert traj.served_models == ["primary", "cheap"]

    def test_unattributed_steps_do_not_break_stability(self) -> None:
        assert make_trajectory(served=[None, None]).route_stability

    def test_tool_signatures_skip_stepswithout_calls(self) -> None:
        traj = make_trajectory(served=["m", "m"])
        traj.steps[0].tool_call = ToolCall(call_id="c", name="search", arguments={})
        assert traj.tool_names() == ["search"]
        assert len(traj.tool_signatures()) == 1

    def test_serialises_to_json(self) -> None:
        json.loads(make_trajectory(served=["m"]).model_dump_json())


class TestFinding:
    def test_loud_kinds_outrank_every_silent_kind(self) -> None:
        loud = [FindingKind.WRONG_ANSWER, FindingKind.RUN_ERROR, FindingKind.NO_TERMINATION]
        silent = [k for k in FindingKind if k not in LOUD_KINDS]
        from downgrade.models import SEVERITY_RANK

        assert max(SEVERITY_RANK[k] for k in loud) < min(SEVERITY_RANK[k] for k in silent)

    def test_unsupported_claim_is_the_highest_priority_silent_kind(self) -> None:
        """It evaded step-level judges most often, so it is checked first."""
        from downgrade.models import SEVERITY_RANK

        silent = [k for k in FindingKind if k not in LOUD_KINDS]
        assert min(silent, key=lambda k: SEVERITY_RANK[k]) is FindingKind.UNSUPPORTED_CLAIM

    def test_dropped_constraint_is_structural_not_judge(self) -> None:
        """Tasks declare constraints as predicates, so this is never an opinion."""
        assert DETECTOR_FOR_KIND[FindingKind.DROPPED_CONSTRAINT] == "structural"

    def test_only_two_kinds_need_a_judge(self) -> None:
        judged = {k for k, d in DETECTOR_FOR_KIND.items() if d == "judge"}
        assert judged == {FindingKind.UNSUPPORTED_CLAIM, FindingKind.UNSUPPORTED_CITATION}

    def test_every_kind_has_a_detector(self) -> None:
        assert set(DETECTOR_FOR_KIND) == set(FindingKind)


def make_finding(kind: FindingKind, replicate: int = 0) -> Finding:
    return Finding(
        kind=kind,
        detector=DETECTOR_FOR_KIND[kind],
        run_id=f"r{replicate}",
        task_id="t1",
        arm="pref_5",
        replicate=replicate,
        evidence=Evidence(detail="x", baseline_reference="5/5 baseline runs"),
    )


class TestFindingRate:
    def test_rate_is_occurrences_over_replicates(self) -> None:
        rate = FindingRate(
            kind=FindingKind.MISSING_TOOL_CALL,
            detector="structural",
            occurrences=2,
            replicate_count=5,
        )
        assert rate.rate == pytest.approx(0.4)

    def test_zero_replicates_does_not_divide_by_zero(self) -> None:
        rate = FindingRate(
            kind=FindingKind.MISSING_TOOL_CALL,
            detector="structural",
            occurrences=0,
            replicate_count=0,
        )
        assert rate.rate == 0.0


class TestArmComparison:
    def test_holds_replicate_lists_not_a_single_pair(self) -> None:
        """The whole point of the shape: one run per side cannot separate a
        downgrade from sampling noise."""
        cmp = ArmComparison(
            task_id="t1",
            baseline_arm="pref_1",
            downgraded_arm="pref_5",
            baseline_run_ids=["b0", "b1", "b2"],
            downgraded_run_ids=["d0", "d1", "d2"],
        )
        assert len(cmp.baseline_run_ids) == 3
        assert len(cmp.downgraded_run_ids) == 3

    def test_verdict_clean_with_no_findings(self) -> None:
        assert ArmComparison(task_id="t", baseline_arm="a", downgraded_arm="b").verdict == "CLEAN"

    def test_verdict_silent_when_only_silent_findings(self) -> None:
        cmp = ArmComparison(
            task_id="t",
            baseline_arm="a",
            downgraded_arm="b",
            finding_rates=[
                FindingRate(
                    kind=FindingKind.MISSING_TOOL_CALL,
                    detector="structural",
                    occurrences=1,
                    replicate_count=5,
                    findings=[make_finding(FindingKind.MISSING_TOOL_CALL)],
                )
            ],
        )
        assert cmp.verdict == "SILENT"

    def test_verdict_loud_dominates_silent(self) -> None:
        cmp = ArmComparison(
            task_id="t",
            baseline_arm="a",
            downgraded_arm="b",
            finding_rates=[
                FindingRate(
                    kind=FindingKind.MISSING_TOOL_CALL,
                    detector="structural",
                    occurrences=1,
                    replicate_count=5,
                    findings=[make_finding(FindingKind.MISSING_TOOL_CALL)],
                ),
                FindingRate(
                    kind=FindingKind.WRONG_ANSWER,
                    detector="structural",
                    occurrences=1,
                    replicate_count=5,
                    findings=[make_finding(FindingKind.WRONG_ANSWER)],
                ),
            ],
        )
        assert cmp.verdict == "LOUD"

    def test_a_rate_with_zero_occurrences_does_not_create_a_verdict(self) -> None:
        cmp = ArmComparison(
            task_id="t",
            baseline_arm="a",
            downgraded_arm="b",
            finding_rates=[
                FindingRate(
                    kind=FindingKind.MISSING_TOOL_CALL,
                    detector="structural",
                    occurrences=0,
                    replicate_count=5,
                )
            ],
        )
        assert cmp.verdict == "CLEAN"

    def test_all_findings_sorted_by_severity(self) -> None:
        cmp = ArmComparison(
            task_id="t",
            baseline_arm="a",
            downgraded_arm="b",
            finding_rates=[
                FindingRate(
                    kind=FindingKind.REDUNDANT_RETRY,
                    detector="structural",
                    occurrences=1,
                    replicate_count=5,
                    findings=[make_finding(FindingKind.REDUNDANT_RETRY)],
                ),
                FindingRate(
                    kind=FindingKind.WRONG_ANSWER,
                    detector="structural",
                    occurrences=1,
                    replicate_count=5,
                    findings=[make_finding(FindingKind.WRONG_ANSWER)],
                ),
            ],
        )
        assert [f.kind for f in cmp.all_findings] == [
            FindingKind.WRONG_ANSWER,
            FindingKind.REDUNDANT_RETRY,
        ]


class TestBaselineProfile:
    def test_reliability_threshold(self) -> None:
        profile = BaselineProfile(
            task_id="t1",
            baseline_arm="pref_1",
            replicate_count=5,
            tool_name_rates={"search": 1.0, "flaky": 0.4},
        )
        assert profile.is_reliable(profile.tool_name_rates, "search", 0.8)
        assert not profile.is_reliable(profile.tool_name_rates, "flaky", 0.8)
        assert not profile.is_reliable(profile.tool_name_rates, "absent", 0.8)

    def test_serialises_set_fields(self) -> None:
        profile = BaselineProfile(
            task_id="t1",
            baseline_arm="pref_1",
            replicate_count=5,
            grounded_values={"1.2", "3.4"},
        )
        assert sorted(json.loads(profile.model_dump_json())["grounded_values"]) == ["1.2", "3.4"]


class TestSweepConfig:
    def test_short_slug_strips_account_prefix(self) -> None:
        assert short_slug("accounts/fireworks/models/kimi-k3") == "kimi-k3"
        assert short_slug("kimi-k3") == "kimi-k3"

    def test_router_model_uses_short_slugs_by_default(self) -> None:
        cfg = SweepConfig(
            primary_model="accounts/fireworks/models/kimi-k3",
            secondary_model="accounts/fireworks/models/glm-5p2",
        )
        assert cfg.router_model == "firerouter/kimi-k3/glm-5p2"

    def test_router_model_can_use_fully_qualified_ids(self) -> None:
        cfg = SweepConfig(
            primary_model="accounts/fireworks/models/kimi-k3",
            secondary_model="accounts/fireworks/models/glm-5p2",
            short_slugs=False,
        )
        assert cfg.router_model.count("accounts/fireworks/models/") == 2

    def test_fingerprint_changes_when_temperature_drifts(self) -> None:
        """The guard that stops an arm comparison from measuring sampling drift."""
        a = SweepConfig(primary_model="p", secondary_model="s", temperature=0.0)
        b = SweepConfig(primary_model="p", secondary_model="s", temperature=0.7)
        assert a.fingerprint != b.fingerprint

    def test_fingerprint_changes_when_seed_drifts(self) -> None:
        a = SweepConfig(primary_model="p", secondary_model="s", seed=1)
        b = SweepConfig(primary_model="p", secondary_model="s", seed=2)
        assert a.fingerprint != b.fingerprint

    def test_fingerprint_ignores_the_arm_list(self) -> None:
        """Arms differ by construction; that must not look like config drift."""
        a = SweepConfig(primary_model="p", secondary_model="s", arms=["pref_1"])
        b = SweepConfig(primary_model="p", secondary_model="s", arms=["pref_5"])
        assert a.fingerprint == b.fingerprint

    def test_temperature_defaults_to_zero(self) -> None:
        assert SweepConfig(primary_model="p", secondary_model="s").temperature == 0.0
