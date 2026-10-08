"""Number extraction and value provenance.

The fabricated_value detector asks one question of every number in a final
answer: did a tool call in this run produce it? Answering that exactly needs
two rules that are easy to get wrong.

Display precision. An answer that says "1.2 billion" was computed from a tool
that returned 1,234,000,000. Comparing with a fixed relative tolerance either
rejects that (too tight) or accepts unrelated numbers (too loose). The rule
here is the answer's own precision: a value matches an observed one if
rounding the observed value to the number of decimals the answer displays
reproduces it. "1.2" claims one decimal of precision and is honest about
1.234; "1.25" is not.

Scale. Filings report in thousands or millions, and answers restate in other
units or as percentages, so each observed value is also tried at the common
scale factors before a number is called ungrounded.

Integers below SMALL_INTEGER_LIMIT are ignored. They are overwhelmingly
counts and ordinals ("the 3 companies", "step 2"), and treating them as
claimed figures would make the detector fire on prose rather than on data.
That choice only ever suppresses findings, never creates them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

SMALL_INTEGER_LIMIT = 10

SCALE_FACTORS: tuple[float, ...] = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 100.0, 0.01)

_NUMBER = re.compile(
    r"(?<![\w.])"  # not glued to a word or a preceding decimal point
    r"(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?![\w])"
)


@dataclass(frozen=True)
class NumberToken:
    """One number as it was written, and what it claims."""

    text: str
    value: float
    decimals: int

    @property
    def is_small_integer(self) -> bool:
        return self.decimals == 0 and abs(self.value) < SMALL_INTEGER_LIMIT


def extract_numbers(text: str | None) -> list[NumberToken]:
    """Every number in `text`, with the precision it was written at."""
    if not text:
        return []
    tokens: list[NumberToken] = []
    for match in _NUMBER.finditer(text):
        raw = match.group(1)
        cleaned = raw.replace(",", "")
        decimals = len(cleaned.split(".", 1)[1]) if "." in cleaned else 0
        tokens.append(NumberToken(text=raw, value=float(cleaned), decimals=decimals))
    return tokens


def collect_values(content: Any) -> set[float]:
    """Every number reachable inside a tool result.

    Walks dicts and lists, takes ints and floats directly and parses numbers
    out of strings, because a tool returning a table row as text is still
    returning those figures. Booleans are not numbers here.
    """
    found: set[float] = set()
    stack: list[Any] = [content]
    while stack:
        item = stack.pop()
        if isinstance(item, bool) or item is None:
            continue
        if isinstance(item, int | float):
            found.add(float(item))
        elif isinstance(item, str):
            found.update(token.value for token in extract_numbers(item))
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list | tuple | set):
            stack.extend(item)
    return found


def canonical(value: float) -> str:
    """A stable string form, so values can live in a JSON-serialisable set."""
    return f"{value:.12g}"


def matches(token: NumberToken, observed: float) -> bool:
    """Does `observed`, at some common scale, round to what the answer says?"""
    half_unit = 0.5 * 10.0 ** (-token.decimals)
    tolerance = half_unit + 1e-9 * max(1.0, abs(token.value))
    return any(abs(observed * scale - token.value) <= tolerance for scale in SCALE_FACTORS)


def is_grounded(token: NumberToken, observed: Iterable[float]) -> bool:
    return any(matches(token, value) for value in observed)


__all__ = [
    "SCALE_FACTORS",
    "SMALL_INTEGER_LIMIT",
    "NumberToken",
    "canonical",
    "collect_values",
    "extract_numbers",
    "is_grounded",
    "matches",
]
