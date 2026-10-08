"""Kupiec and Christoffersen backtests, borrowed from Value-at-Risk validation.

A VaR model says "losses exceed this level with probability p". A backtest
checks two things about the exceedances that actually happened: that they
occur at rate p (Kupiec's proportion-of-failures test) and that they do not
cluster in time (Christoffersen's independence test).

The noise floor makes the same kind of claim. If routing changed nothing,
an arm's runs draw findings at the baseline's own false-positive rate, and
they draw them independently. A "violation" here is a run with a finding.

Kupiec POF is two-sided: it rejects when the arm's finding rate is too high
OR too low against the floor. The sequential test already handles "too
high" with optional stopping; Kupiec is the classical fixed-sample check of
the same coverage, reported so the two can be read against each other.

Christoffersen independence is the one the rest of the analysis leans on.
The sequential test and the pooled rates assume runs are exchangeable. If
findings cluster in run order (a warm cache on early replicates, a rate
limit partway through, a router whose behaviour drifts during the sweep),
that assumption is false, the p-values are optimistic, and the report says
so for the affected arm rather than presenting its p-value as clean.

Run order is replicate-major across tasks: replicate 0 of every task, then
replicate 1 of every task, and so on. Ordering task-major instead would put
a hard task's replicates next to each other, and the independence test would
then flag ordinary task difficulty as clustering.

Chi-square tails are computed in closed form (df 1 via erfc, df 2 via exp),
so this needs no scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _xlogy(x: float, y: float) -> float:
    """x * log(y), with 0 * log(0) taken as 0."""
    if x == 0:
        return 0.0
    if y <= 0:
        return -math.inf
    return x * math.log(y)


def chi2_sf(statistic: float, df: int) -> float:
    """Upper tail of the chi-square distribution for df 1 or 2."""
    if statistic <= 0:
        return 1.0
    if math.isinf(statistic):
        return 0.0
    if df == 1:
        return math.erfc(math.sqrt(statistic / 2.0))
    if df == 2:
        return math.exp(-statistic / 2.0)
    raise ValueError(f"chi2_sf supports df 1 or 2, got {df}")


@dataclass(frozen=True)
class LRTestResult:
    statistic: float
    p_value: float
    df: int

    def rejects(self, alpha: float) -> bool:
        return self.p_value <= alpha


def kupiec_pof(violations: Sequence[bool], expected_rate: float) -> LRTestResult:
    """Kupiec's proportion-of-failures likelihood-ratio test."""
    n = len(violations)
    x = sum(1 for v in violations if v)
    if n == 0:
        return LRTestResult(0.0, 1.0, 1)
    p = min(max(expected_rate, 0.0), 1.0)
    observed = x / n
    null_ll = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    alt_ll = _xlogy(n - x, 1 - observed) + _xlogy(x, observed)
    statistic = -2.0 * (null_ll - alt_ll)
    statistic = math.inf if math.isnan(statistic) else max(0.0, statistic)
    return LRTestResult(statistic, chi2_sf(statistic, 1), 1)


def transition_counts(violations: Sequence[bool]) -> tuple[int, int, int, int]:
    """(n00, n01, n10, n11): counts of each consecutive pair of states."""
    n = [[0, 0], [0, 0]]
    for prev, curr in zip(violations, violations[1:], strict=False):
        n[int(prev)][int(curr)] += 1
    return n[0][0], n[0][1], n[1][0], n[1][1]


def christoffersen_independence(violations: Sequence[bool]) -> LRTestResult:
    """Does a violation make the next run more (or less) likely to violate?

    Compares a first-order Markov chain against independence. With no
    violations, or nothing but violations, there is no transition structure
    to test and the statistic is zero.
    """
    n00, n01, n10, n11 = transition_counts(violations)
    total = n00 + n01 + n10 + n11
    if total == 0 or (n01 + n11) in (0, total):
        return LRTestResult(0.0, 1.0, 1)
    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    null_ll = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    alt_ll = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    statistic = max(0.0, -2.0 * (null_ll - alt_ll))
    return LRTestResult(statistic, chi2_sf(statistic, 1), 1)


def conditional_coverage(violations: Sequence[bool], expected_rate: float) -> LRTestResult:
    """Christoffersen's joint test: right rate AND independent. df 2."""
    statistic = (
        kupiec_pof(violations, expected_rate).statistic
        + christoffersen_independence(violations).statistic
    )
    return LRTestResult(statistic, chi2_sf(statistic, 2), 2)


__all__ = [
    "LRTestResult",
    "chi2_sf",
    "christoffersen_independence",
    "conditional_coverage",
    "kupiec_pof",
    "transition_counts",
]
