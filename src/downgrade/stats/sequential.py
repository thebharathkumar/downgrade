"""Sequential tests on a stream of per-run finding indicators.

Each downgraded run either drew a finding of a given kind or it did not. The
question is whether that happens more often than the baseline noise floor
says it would by chance. Two tests answer it, for different purposes.

`anytime_pvalue` is a test martingale: a mixture likelihood ratio over a grid
of alternatives above the null rate. Its p-value stays valid however the
sample size was chosen, including stopping the sweep early because the
result already looked clear. That property is what lets these p-values feed
Benjamini-Hochberg without a correction for peeking.

Composite null. The null is "the rate is AT MOST p0", not "exactly p0". For
any alternative q >= p0 and any true rate p <= p0, the per-run factor
(q/p0)^x ((1-q)/(1-p0))^(1-x) has expectation p q/p0 + (1-p)(1-q)/(1-p0),
which is linear in p, equals 1 at p = p0 and is non-decreasing in p, so it is
at most 1. The product is a supermartingale under every rate in the null, and
Ville's inequality bounds P(sup M >= 1/alpha) by alpha. The grid is restricted
to q >= p0 for exactly this reason.

`sprt` is Wald's sequential probability ratio test between p0 and p0 + lift.
It answers a different question, "can the sweep stop adding replicates?",
and returns accept, reject or continue. It is reported alongside, not used to
gate the p-values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# Relative lifts: q = p0 + lift * (1 - p0). Uniform weights over the grid.
DEFAULT_LIFT_GRID: tuple[float, ...] = tuple(i / 20 for i in range(1, 20))

SPRTDecision = Literal["reject_null", "accept_null", "continue"]


def wilson_upper(successes: int, trials: int, z: float = 1.959964) -> float:
    """Upper end of the Wilson score interval for a binomial rate.

    Used as the null rate in place of the raw noise floor. The floor is
    estimated from few runs, and at k=0 the raw estimate is exactly zero,
    which would make a single downgraded finding infinitely significant. The
    upper bound charges the test for not knowing the floor precisely.
    """
    if trials <= 0:
        return 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2 * trials)
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return min(1.0, (centre + spread) / denom)


def _log_factor(flag: bool, p0: float, q: float) -> float:
    if flag:
        return math.log(q) - math.log(p0)
    return math.log1p(-q) - math.log1p(-p0)


def _logsumexp(values: Sequence[float]) -> float:
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(v - top) for v in values))


def anytime_pvalue(
    flags: Sequence[bool], null_rate: float, grid: Sequence[float] = DEFAULT_LIFT_GRID
) -> float:
    """Anytime-valid p-value for H0: rate <= null_rate against rate > null_rate.

    p = min(1, 1 / max_t M_t), where M_t is the mixture likelihood ratio after
    t observations. Taking the running maximum is what makes it valid at any
    stopping time rather than only at the planned one.
    """
    if not flags:
        return 1.0
    if null_rate >= 1.0:
        return 1.0
    if null_rate <= 0.0:
        # Under a zero null rate a single flag is impossible, so it refutes H0.
        return 0.0 if any(flags) else 1.0

    alternatives = [null_rate + lift * (1.0 - null_rate) for lift in grid]
    logs = [0.0] * len(alternatives)
    log_weight = -math.log(len(alternatives))
    best = 0.0
    for flag in flags:
        for i, q in enumerate(alternatives):
            logs[i] += _log_factor(flag, null_rate, q)
        best = max(best, log_weight + _logsumexp(logs))
    return min(1.0, math.exp(-best))


@dataclass(frozen=True)
class SPRTResult:
    decision: SPRTDecision
    log_likelihood_ratio: float
    lower: float
    upper: float
    observations_used: int


def sprt(
    flags: Sequence[bool],
    null_rate: float,
    alt_rate: float,
    *,
    alpha: float = 0.05,
    beta: float = 0.20,
) -> SPRTResult:
    """Wald's SPRT for H0: rate = null_rate against H1: rate = alt_rate.

    Stops at the first boundary crossing and reports how many observations
    that took. "continue" means the replicates so far do not decide it.
    """
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    p0 = min(max(null_rate, 1e-9), 1 - 1e-9)
    p1 = min(max(alt_rate, p0 + 1e-9), 1 - 1e-9)
    llr = 0.0
    for i, flag in enumerate(flags, start=1):
        llr += _log_factor(flag, p0, p1)
        if llr >= upper:
            return SPRTResult("reject_null", llr, lower, upper, i)
        if llr <= lower:
            return SPRTResult("accept_null", llr, lower, upper, i)
    return SPRTResult("continue", llr, lower, upper, len(flags))


__all__ = [
    "DEFAULT_LIFT_GRID",
    "SPRTDecision",
    "SPRTResult",
    "anytime_pvalue",
    "sprt",
    "wilson_upper",
]
