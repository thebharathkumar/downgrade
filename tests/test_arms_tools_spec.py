"""Tests for the sweep matrix, the tool registry and the task schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from downgrade.arms import (
    ALL_ARMS,
    BASELINE_DIRECT,
    PREFERENCE_LABELS,
    ROUTER_ARMS,
    ROUTING_HEADER,
    arm_by_name,
    resolve_arms,
)
from downgrade.models import RouterFormatUnknownError, SweepConfig, ToolCall
from downgrade.suite.spec import AnswerCheck, Constraint, TaskSpec, load_suite
from downgrade.tools import FunctionTool, ToolDef, ToolError, ToolRegistry

PRIMARY = "accounts/fireworks/models/kimi-k3"


def config(**kw: Any) -> SweepConfig:
    defaults: dict[str, Any] = {
        "primary_model": PRIMARY,
        "secondary_model": "accounts/fireworks/models/glm-5p2",
        "short_slugs": True,
    }
    defaults.update(kw)
    return SweepConfig(**defaults)


class TestArms:
    def test_the_matrix_is_one_control_plus_five_preferences(self) -> None:
        assert len(ALL_ARMS) == 6
        assert [a.name for a in ROUTER_ARMS] == [f"pref_{p}" for p in range(1, 6)]

    def test_only_the_direct_arm_is_a_baseline(self) -> None:
        assert BASELINE_DIRECT.is_baseline
        assert not any(a.is_baseline for a in ROUTER_ARMS)

    def test_preference_one_is_a_router_arm_not_the_baseline(self) -> None:
        """Scoring pref_1 as an ordinary arm is what lets the sweep answer
        whether the router already downgrades at maximum intelligence."""
        assert arm_by_name("pref_1").use_router
        assert not arm_by_name("pref_1").is_baseline

    def test_labels_match_the_documented_names(self) -> None:
        assert PREFERENCE_LABELS[1] == "max-intelligence"
        assert PREFERENCE_LABELS[5] == "max-savings"
        assert arm_by_name("pref_3").label == "balanced"
        assert "no router" in BASELINE_DIRECT.label

    def test_direct_arm_uses_the_primary_model(self) -> None:
        assert BASELINE_DIRECT.model_for(config()) == PRIMARY

    def test_router_arms_use_the_pair_string(self) -> None:
        assert arm_by_name("pref_2").model_for(config()) == "firerouter/kimi-k3/glm-5p2"

    def test_a_router_arm_refuses_to_build_an_unprobed_model_string(self) -> None:
        with pytest.raises(RouterFormatUnknownError):
            arm_by_name("pref_2").model_for(config(short_slugs=None))

    def test_the_direct_arm_works_without_a_probe(self) -> None:
        """The control arm never touches the router, so it is not blocked on
        a wire-format question that does not apply to it."""
        assert BASELINE_DIRECT.model_for(config(short_slugs=None)) == PRIMARY

    def test_headers(self) -> None:
        assert arm_by_name("pref_4").request_headers() == {ROUTING_HEADER: "4"}
        assert BASELINE_DIRECT.request_headers() == {}

    def test_unknown_arm_names_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="pref_1"):
            arm_by_name("pref_9")

    def test_resolve_defaults_to_the_full_matrix(self) -> None:
        assert resolve_arms(None) == list(ALL_ARMS)
        assert resolve_arms([]) == list(ALL_ARMS)

    def test_resolve_always_includes_a_baseline(self) -> None:
        """Without it there is no profile to score against and no noise floor."""
        resolved = resolve_arms(["pref_5"])
        assert resolved[0] == BASELINE_DIRECT
        assert len(resolved) == 2

    def test_resolve_does_not_duplicate_an_explicit_baseline(self) -> None:
        resolved = resolve_arms(["baseline_direct", "pref_5"])
        assert [a.name for a in resolved] == ["baseline_direct", "pref_5"]


class TestToolRegistry:
    @staticmethod
    def registry() -> ToolRegistry:
        def lookup(concept: str, year: int = 2024) -> dict[str, Any]:
            return {"concept": concept, "year": year, "value": 1.0}

        def fails() -> None:
            raise ToolError("corpus unavailable")

        return ToolRegistry(
            [
                FunctionTool(
                    definition=ToolDef(
                        name="lookup",
                        description="Look up a value",
                        parameters={
                            "type": "object",
                            "properties": {"concept": {"type": "string"}},
                            "required": ["concept"],
                        },
                    ),
                    func=lookup,
                ),
                FunctionTool(
                    definition=ToolDef(name="fails", description="Fails", parameters={}),
                    func=fails,
                ),
            ]
        )

    def test_definitions_are_in_openai_shape(self) -> None:
        definition = self.registry().definitions()[1]
        assert definition["type"] == "function"
        assert definition["function"]["name"] == "lookup"
        assert definition["function"]["parameters"]["required"] == ["concept"]

    def test_names_and_membership(self) -> None:
        registry = self.registry()
        assert registry.names == ["fails", "lookup"]
        assert "lookup" in registry
        assert len(registry) == 2

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = self.registry()
        with pytest.raises(ValueError, match="already registered"):
            registry.register(
                FunctionTool(definition=ToolDef("lookup", "dup", {}), func=lambda: None)
            )

    def test_a_successful_call_returns_content(self) -> None:
        result = self.registry().invoke(
            ToolCall(call_id="c1", name="lookup", arguments={"concept": "Assets"})
        )
        assert result.ok
        assert result.content == {"concept": "Assets", "year": 2024, "value": 1.0}

    def test_a_tool_error_becomes_an_observation(self) -> None:
        """The model may recover; whether it did is what retry detection reads."""
        result = self.registry().invoke(ToolCall(call_id="c1", name="fails", arguments={}))
        assert not result.ok
        assert result.error == "corpus unavailable"

    def test_an_unknown_tool_lists_the_available_ones(self) -> None:
        result = self.registry().invoke(ToolCall(call_id="c1", name="ghost", arguments={}))
        assert not result.ok
        assert "No such tool 'ghost'" in (result.error or "")
        assert "lookup" in (result.error or "")

    def test_bad_arguments_come_back_as_a_correctable_error(self) -> None:
        result = self.registry().invoke(
            ToolCall(call_id="c1", name="lookup", arguments={"wrong": 1})
        )
        assert not result.ok
        assert "Invalid arguments" in (result.error or "")

    def test_invoke_never_raises(self) -> None:
        for call in [
            ToolCall(call_id="a", name="fails", arguments={}),
            ToolCall(call_id="b", name="ghost", arguments={}),
            ToolCall(call_id="c", name="lookup", arguments={"nope": True}),
        ]:
            assert self.registry().invoke(call).ok is False


class TestTaskSpec:
    def test_duplicate_constraint_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate constraint ids"):
            TaskSpec(
                task_id="t1",
                family="tabular_aggregation",
                prompt="p",
                answer_check=AnswerCheck(),
                constraints=[
                    Constraint(id="c1", predicate="tool_called"),
                    Constraint(id="c1", predicate="min_citations"),
                ],
            )

    def test_a_blank_constraint_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Constraint(id="  ", predicate="tool_called")

    def test_answer_check_defaults_to_numeric_with_tolerance(self) -> None:
        check = AnswerCheck()
        assert check.kind == "numeric"
        assert check.tolerance == 0.01


class TestLoadSuite:
    def _write(self, path: Path, name: str, task_id: str) -> None:
        path.joinpath(name).write_text(
            f"task_id: {task_id}\n"
            "family: tabular_aggregation\n"
            "prompt: Compute something\n"
            "answer_check:\n"
            "  kind: numeric\n"
            "  value: 42.0\n"
            "constraints:\n"
            "  - id: must_query\n"
            "    predicate: tool_called\n"
            "    args: {name: query_table}\n",
            encoding="utf-8",
        )

    def test_loads_every_yaml_in_a_directory(self, tmp_path: Path) -> None:
        self._write(tmp_path, "a.yaml", "t1")
        self._write(tmp_path, "b.yaml", "t2")
        suite = load_suite(tmp_path)
        assert len(suite) == 2
        assert sorted(suite.by_id()) == ["t1", "t2"]

    def test_loads_a_single_file(self, tmp_path: Path) -> None:
        self._write(tmp_path, "a.yaml", "t1")
        assert len(load_suite(tmp_path / "a.yaml")) == 1

    def test_constraints_survive_the_round_trip(self, tmp_path: Path) -> None:
        self._write(tmp_path, "a.yaml", "t1")
        constraint = load_suite(tmp_path).tasks[0].constraints[0]
        assert constraint.predicate == "tool_called"
        assert constraint.args == {"name": "query_table"}

    def test_duplicate_task_ids_across_files_are_rejected(self, tmp_path: Path) -> None:
        """A duplicate would merge two tasks' replicates into one comparison."""
        self._write(tmp_path, "a.yaml", "same")
        self._write(tmp_path, "b.yaml", "same")
        with pytest.raises(ValueError, match="duplicate task ids"):
            load_suite(tmp_path)

    def test_an_empty_file_is_skipped(self, tmp_path: Path) -> None:
        tmp_path.joinpath("empty.yaml").write_text("", encoding="utf-8")
        self._write(tmp_path, "a.yaml", "t1")
        assert len(load_suite(tmp_path)) == 1

    def test_a_list_of_tasks_in_one_file_is_accepted(self, tmp_path: Path) -> None:
        tmp_path.joinpath("many.yaml").write_text(
            "- task_id: t1\n"
            "  family: multi_hop_retrieval\n"
            "  prompt: p1\n"
            "  answer_check: {kind: exact, value: a}\n"
            "- task_id: t2\n"
            "  family: constrained_synthesis\n"
            "  prompt: p2\n"
            "  answer_check: {kind: exact, value: b}\n",
            encoding="utf-8",
        )
        assert len(load_suite(tmp_path)) == 2
