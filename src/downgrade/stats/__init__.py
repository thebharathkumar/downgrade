"""The statistics layer: sequential tests, FDR control and backtests."""

from downgrade.stats.analysis import (
    ArmSummary,
    Backtest,
    FloorEstimate,
    Hypothesis,
    SweepAnalysis,
    analyze,
)
from downgrade.stats.backtest import (
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
)
from downgrade.stats.multiple import benjamini_hochberg
from downgrade.stats.sequential import anytime_pvalue, sprt, wilson_upper

__all__ = [
    "ArmSummary",
    "Backtest",
    "FloorEstimate",
    "Hypothesis",
    "SweepAnalysis",
    "analyze",
    "anytime_pvalue",
    "benjamini_hochberg",
    "christoffersen_independence",
    "conditional_coverage",
    "kupiec_pof",
    "sprt",
    "wilson_upper",
]
