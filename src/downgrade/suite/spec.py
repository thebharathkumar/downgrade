"""Task specifications, loaded from YAML.

The important field is `constraints`. Each one names a predicate from a fixed
registry plus its arguments, which makes "was this constraint dropped?" a
computation over the trajectory rather than an LLM's reading of the prose. A
constraint that cannot be written as a predicate means the task is badly
authored and gets rewritten; there is no judge fallback for this sub-class.

`expected_tools` is documentation and a suite-authoring check, not a detector
input. The missing_tool_call detector reads what the BASELINE arm actually did
rather than what the task author expected it to do, because a task whose
baseline never calls a tool the author listed is a broken task, and a baseline
that reliably calls something the author did not anticipate is still a real
expectation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

TaskFamily = Literal[
    "multi_hop_retrieval",
    "tabular_aggregation",
    "cross_source_reconciliation",
    "constrained_synthesis",
]

AnswerCheckKind = Literal["numeric", "exact", "regex", "set"]


class Constraint(BaseModel):
    """One machine-checkable requirement stated in the task prompt."""

    id: str
    predicate: str
    args: dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    @field_validator("id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("constraint id must be non-empty")
        return value


class AnswerCheck(BaseModel):
    """How to decide whether a final answer is correct.

    `numeric` exists because most answers here are financial figures, where
    string equality would fail on formatting alone and a tolerance is the only
    honest comparison. `tolerance` is relative.
    """

    kind: AnswerCheckKind = "numeric"
    value: Any = None
    tolerance: float = 0.01
    pattern: str | None = None


class TaskSpec(BaseModel):
    task_id: str
    family: TaskFamily
    prompt: str
    answer_check: AnswerCheck
    constraints: list[Constraint] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    max_steps: int | None = None
    notes: str = ""

    @field_validator("constraints")
    @classmethod
    def _unique_constraint_ids(cls, value: list[Constraint]) -> list[Constraint]:
        seen = [c.id for c in value]
        if len(seen) != len(set(seen)):
            raise ValueError(f"duplicate constraint ids: {seen}")
        return value


class Suite(BaseModel):
    tasks: list[TaskSpec] = Field(default_factory=list)

    def by_id(self) -> dict[str, TaskSpec]:
        return {t.task_id: t for t in self.tasks}

    def __len__(self) -> int:
        return len(self.tasks)


def load_suite(path: Path) -> Suite:
    """Load every .yaml under `path` (or a single file) into a Suite.

    Task ids must be unique across the whole suite: a duplicate would silently
    merge two tasks' replicates into one comparison.
    """
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    tasks: list[TaskSpec] = []
    for file in files:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
        if raw is None:
            continue
        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            tasks.append(TaskSpec.model_validate(entry))

    ids = [t.task_id for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate task ids in suite: {', '.join(duplicates)}")
    return Suite(tasks=tasks)


__all__ = [
    "AnswerCheck",
    "AnswerCheckKind",
    "Constraint",
    "Suite",
    "TaskFamily",
    "TaskSpec",
    "load_suite",
]
