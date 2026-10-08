"""Deciding whether a final answer is correct.

This is the outcome check, the one an outcome-only judge also performs. It
produces the wrong_answer finding, which is loud. Everything else in the
classifier exists to find what this check misses.
"""

from __future__ import annotations

import re

from downgrade.classify.values import extract_numbers
from downgrade.suite.spec import AnswerCheck


def has_expectation(check: AnswerCheck) -> bool:
    """A check with nothing to compare against cannot mark anything wrong."""
    if check.kind == "regex":
        return bool(check.pattern)
    return check.value is not None


def check_answer(check: AnswerCheck, answer: str | None) -> bool:
    """True when `answer` satisfies `check`.

    `numeric` accepts the answer if any number in it is within the relative
    tolerance of the expected value. Answers here are prose around a figure,
    and requiring the figure to be the only number would mark a correct
    answer that also restates its inputs as wrong.
    """
    if answer is None or not answer.strip():
        return False

    if check.kind == "numeric":
        expected = float(check.value)
        allowed = abs(expected) * check.tolerance
        return any(abs(t.value - expected) <= allowed for t in extract_numbers(answer))

    if check.kind == "exact":
        return answer.strip().casefold() == str(check.value).strip().casefold()

    if check.kind == "regex":
        pattern = check.pattern or str(check.value)
        return re.search(pattern, answer, flags=re.IGNORECASE) is not None

    # "set": every expected item appears somewhere in the answer.
    items = check.value if isinstance(check.value, list) else [check.value]
    lowered = answer.casefold()
    return all(str(item).casefold() in lowered for item in items)


__all__ = ["check_answer", "has_expectation"]
