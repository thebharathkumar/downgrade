"""Multiple-comparison control across the whole hypothesis family.

A sweep tests every downgraded arm against every finding kind. At five arms
and nine kinds that is 45 tests, and at a 5% per-test level roughly two would
come up "significant" from noise alone. Benjamini-Hochberg bounds the
expected share of false discoveries among the reported ones instead.

The family is fixed before the data is seen: every (arm, kind) pair the
classify mode can produce, whether or not that kind occurred. Dropping kinds
that never fired would shrink m after looking, which is the same peeking the
anytime-valid p-values exist to avoid.

Benjamini-Yekutieli is offered for arbitrary dependence. Finding kinds on the
same runs are not independent (the ladder makes them mutually exclusive per
run), so BY is the conservative choice when that matters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Method = Literal["bh", "by"]


@dataclass(frozen=True)
class FDRResult:
    adjusted: list[float]
    rejected: list[bool]
    q: float
    method: Method

    @property
    def discoveries(self) -> int:
        return sum(self.rejected)


def benjamini_hochberg(
    pvalues: Sequence[float], q: float = 0.05, method: Method = "bh"
) -> FDRResult:
    """Step-up FDR control. Returns adjusted p-values in input order.

    The adjusted value for rank i is min over j >= i of p_(j) * m * c / j,
    clipped at 1, where c is 1 for BH and the harmonic number H_m for BY. A
    hypothesis is rejected when its adjusted value is at most q, which is
    equivalent to the textbook step-up rule.
    """
    m = len(pvalues)
    if m == 0:
        return FDRResult(adjusted=[], rejected=[], q=q, method=method)
    c = sum(1.0 / i for i in range(1, m + 1)) if method == "by" else 1.0
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        running = min(running, pvalues[index] * m * c / rank)
        adjusted[index] = min(1.0, running)
    return FDRResult(adjusted=adjusted, rejected=[a <= q for a in adjusted], q=q, method=method)


__all__ = ["FDRResult", "Method", "benjamini_hochberg"]
