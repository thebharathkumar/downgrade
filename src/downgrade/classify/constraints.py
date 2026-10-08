"""The fixed registry of constraint predicates.

Every constraint a task declares names one of these. Each predicate is a pure
function of the trajectory, which is what makes dropped_constraint structural:
whether a constraint held is computed, not judged. A requirement that cannot
be written with these predicates means the task needs rewriting, and the
registry is extended deliberately rather than by adding a judge fallback.

An unknown predicate name raises. A typo in a task file must fail when the
suite is checked, not silently count as "constraint dropped" in every arm.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from downgrade.models import ConstraintResult, Trajectory
from downgrade.suite.spec import Constraint, TaskSpec

PredicateFn = Callable[[Trajectory, dict[str, Any]], tuple[bool, str]]


class UnknownPredicateError(ValueError):
    """A task named a predicate the registry does not have."""


def _calls(trajectory: Trajectory, name: str) -> list[dict[str, Any]]:
    return [
        call.arguments for step in trajectory.steps for call in step.tool_calls if call.name == name
    ]


def _tool_called(t: Trajectory, args: dict[str, Any]) -> tuple[bool, str]:
    name = str(args["name"])
    minimum = int(args.get("min", 1))
    count = len(_calls(t, name))
    return count >= minimum, f"{name} called {count} time(s), need >= {minimum}"


def _tool_not_called(t: Trajectory, args: dict[str, Any]) -> tuple[bool, str]:
    name = str(args["name"])
    count = len(_calls(t, name))
    return count == 0, f"{name} called {count} time(s), need 0"


def _tool_called_with(t: Trajectory, args: dict[str, Any]) -> tuple[bool, str]:
    name = str(args["name"])
    wanted: dict[str, Any] = dict(args.get("arguments") or {})
    for arguments in _calls(t, name):
        if all(_loose_equal(arguments.get(k), v) for k, v in wanted.items()):
            return True, f"{name} called with {wanted}"
    return False, f"no {name} call carried {wanted}"


def _max_tool_calls(t: Trajectory, args: dict[str, Any]) -> tuple[bool, str]:
    limit = int(args["n"])
    count = len(t.tool_names())
    return count <= limit, f"{count} tool call(s), limit {limit}"


def _called_before(t: Trajectory, args: dict[str, Any]) -> tuple[bool, str]:
    first, then = str(args["first"]), str(args["then"])
    names = t.tool_names()
    if then not in names:
        return True, f"{then} never called"
    if first not in names:
        return False, f"{then} called without {first} first"
    ok = names.index(first) < names.index(then)
    return ok, f"{first} at {names.index(first)}, {then} at {names.index(then)}"


def _answer_matches(t: Trajectory, args: dict[str, Any]) -> tuple[bool, str]:
    pattern = str(args["pattern"])
    ok = re.search(pattern, t.final_answer or "", flags=re.IGNORECASE) is not None
    return ok, f"answer {'matches' if ok else 'does not match'} /{pattern}/"


def _answer_not_matches(t: Trajectory, args: dict[str, Any]) -> tuple[bool, str]:
    ok, _ = _answer_matches(t, args)
    return not ok, f"answer {'contains' if ok else 'avoids'} /{args['pattern']}/"


def _answer_max_words(t: Trajectory, args: dict[str, Any]) -> tuple[bool, str]:
    limit = int(args["n"])
    words = len((t.final_answer or "").split())
    return words <= limit, f"{words} word(s), limit {limit}"


def _answer_contains_all(t: Trajectory, args: dict[str, Any]) -> tuple[bool, str]:
    answer = (t.final_answer or "").casefold()
    missing = [str(v) for v in args["values"] if str(v).casefold() not in answer]
    return not missing, f"missing {missing}" if missing else "all present"


def _loose_equal(actual: Any, expected: Any) -> bool:
    """Case-insensitive for strings, exact otherwise."""
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().casefold() == expected.strip().casefold()
    return bool(actual == expected)


PREDICATES: dict[str, PredicateFn] = {
    "tool_called": _tool_called,
    "tool_not_called": _tool_not_called,
    "tool_called_with": _tool_called_with,
    "max_tool_calls": _max_tool_calls,
    "called_before": _called_before,
    "answer_matches": _answer_matches,
    "answer_not_matches": _answer_not_matches,
    "answer_max_words": _answer_max_words,
    "answer_contains_all": _answer_contains_all,
}


def validate_constraints(task: TaskSpec) -> None:
    """Fail on any predicate the registry does not know."""
    unknown = sorted({c.predicate for c in task.constraints if c.predicate not in PREDICATES})
    if unknown:
        known = ", ".join(sorted(PREDICATES))
        raise UnknownPredicateError(
            f"Task '{task.task_id}' uses unknown predicate(s) {unknown}. Known: {known}"
        )


def check_constraint(constraint: Constraint, trajectory: Trajectory) -> ConstraintResult:
    fn = PREDICATES.get(constraint.predicate)
    if fn is None:
        raise UnknownPredicateError(f"Unknown predicate '{constraint.predicate}'")
    try:
        satisfied, detail = fn(trajectory, constraint.args)
    except KeyError as exc:
        raise UnknownPredicateError(
            f"Constraint '{constraint.id}' ({constraint.predicate}) is missing argument {exc}"
        ) from None
    return ConstraintResult(
        constraint_id=constraint.id,
        predicate=constraint.predicate,
        satisfied=satisfied,
        detail=detail,
    )


def check_constraints(task: TaskSpec, trajectory: Trajectory) -> list[ConstraintResult]:
    return [check_constraint(c, trajectory) for c in task.constraints]


__all__ = [
    "PREDICATES",
    "UnknownPredicateError",
    "check_constraint",
    "check_constraints",
    "validate_constraints",
]
