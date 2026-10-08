"""The published report, rendered as Markdown.

Section order is the argument, so it is fixed:

  1. What was run, and under which pre-registered settings.
  2. The noise floor: how often each detector fires on baseline vs itself.
     Every later number is read against this, so it comes first.
  3. Per-arm outcomes: routing, correctness, loud and silent rates.
  4. Tests: anytime-valid p-values, BH-adjusted across the whole family.
  5. Backtests: is each arm's finding rate and run-to-run independence what
     the floor predicts?
  6. Evidence for every discovery.
  7. What this report cannot tell you.

The headline is generated from the tests, never written by hand, and it says
"no regression detected" in exactly the cases where that is what the data
shows. A null result is a result.

This module quotes no external figures. Anything quoted from outside the repo
must be registered in docs/citations.yaml first, and unverified ones may not
appear here; tests/test_citations.py checks this file's source.
"""

from __future__ import annotations

from downgrade.models import Finding, FindingKind, RouterFormatUnknownError
from downgrade.stats.analysis import BACKTEST_ALPHA, Hypothesis, SweepAnalysis

MAX_EVIDENCE_PER_DISCOVERY = 3


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _p(value: float) -> str:
    if value < 1e-4:
        return "<0.0001"
    return f"{value:.4f}"


def _cell(text: str | None, limit: int = 80) -> str:
    if not text:
        return ""
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def headline(analysis: SweepAnalysis) -> str:
    q = analysis.config.fdr_q
    silent = analysis.silent_discoveries
    loud = [h for h in analysis.discoveries if h.is_loud]
    if not analysis.discoveries:
        return (
            f"No regression detected at any routing preference: none of the "
            f"{analysis.family_size} (arm, finding kind) hypotheses rejects at FDR q={q}."
        )
    parts = []
    if silent:
        arms = sorted({h.arm for h in silent})
        parts.append(
            f"SILENT regression detected in {len(silent)} hypothesis(es) across {', '.join(arms)}"
        )
    if loud:
        arms = sorted({h.arm for h in loud})
        parts.append(f"LOUD regression detected in {len(loud)} across {', '.join(arms)}")
    return "; ".join(parts) + f" (Benjamini-Hochberg, FDR q={q})."


def _setup(analysis: SweepAnalysis) -> list[str]:
    c = analysis.config
    try:
        router = c.router_model
    except RouterFormatUnknownError:
        router = f"{c.primary_model} / {c.secondary_model} (wire format not recorded)"
    rows = [
        ["primary model", f"`{c.primary_model}`"],
        ["secondary model", f"`{c.secondary_model}`"],
        ["router model string", f"`{router}`"],
        ["config fingerprint", f"`{c.fingerprint}`"],
        ["temperature / seed", f"{c.temperature} / {c.seed}"],
        ["replicates per arm per task", str(c.replicates)],
        ["tasks", str(len(analysis.task_ids))],
        ["classify mode", analysis.mode],
        ["reliability threshold", str(c.reliability_threshold)],
        ["FDR q", str(c.fdr_q)],
        ["SPRT alpha / beta / lift", f"{c.sprt_alpha} / {c.sprt_beta} / {c.min_detectable_lift}"],
    ]
    return ["## Setup", "", *_table(["setting", "value"], rows), ""]


def _floor(analysis: SweepAnalysis) -> list[str]:
    lines = [
        "## 1. Noise floor (read this first)",
        "",
        "Each baseline run scored leave-one-out against a profile of the other "
        "baseline runs. Every finding here is a false positive by construction. "
        "The test's null rate is the Wilson upper bound of the pooled floor, so "
        "imprecision in the floor counts against detecting a regression, not for it.",
        "",
    ]
    rows = [
        [f.kind.value, f.detector, f"{f.flagged}/{f.runs}", _pct(f.rate), _pct(f.null_rate)]
        for f in analysis.floors
    ]
    lines += _table(["kind", "detector", "flagged", "floor", "null rate used"], rows)
    total = sum(f.flagged for f in analysis.floors)
    runs = analysis.floors[0].runs if analysis.floors else 0
    if runs == 0:
        lines += ["", "**No floor could be computed: fewer than two baseline runs per task.**"]
    elif total:
        lines += [
            "",
            f"The classifier fires on {total} of {runs} baseline runs compared "
            "with themselves. That is a property of the detectors and the "
            "control arm's own variability, and it bounds what any arm below can show.",
        ]
    return [*lines, ""]


def _arms(analysis: SweepAnalysis) -> list[str]:
    rows = [
        [
            a.arm,
            f"{a.preference} {a.label}" if a.preference is not None else a.label,
            str(a.runs),
            _pct(a.route_downgrade_rate),
            f"{_pct(a.correct_rate)} vs {_pct(a.baseline_correct_rate)}",
            _pct(a.loud_rate),
            _pct(a.silent_rate),
        ]
        for a in analysis.arms
    ]
    headers = ["arm", "preference", "runs", "calls downgraded", "correct vs baseline"]
    return [
        "## 2. Per-arm outcomes",
        "",
        *_table([*headers, "loud", "silent"], rows),
        "",
        "Correctness is what an outcome-only judge sees. The silent column is what it does not.",
        "",
    ]


def _hypothesis_row(h: Hypothesis) -> list[str]:
    verdict = "**reject**" if h.rejected else "-"
    if h.rejected and not h.exchangeable:
        verdict += " (runs cluster, see backtests)"
    return [
        h.arm,
        h.kind.value,
        h.detector,
        f"{h.flagged}/{h.runs}",
        _pct(h.floor_rate),
        _pct(h.excess),
        _p(h.p_value),
        _p(h.p_adjusted),
        f"{h.sprt_decision} @{h.sprt_runs}",
        verdict,
    ]


def _tests(analysis: SweepAnalysis) -> list[str]:
    shown = [h for h in analysis.hypotheses if h.flagged or h.rejected]
    lines = [
        "## 3. Tests",
        "",
        f"Family: {analysis.family_size} pre-registered hypotheses, one per "
        "(downgraded arm, finding kind), each pooled across every task. "
        "p-values are anytime-valid (mixture likelihood ratio), so they hold "
        "under optional stopping. Adjusted with Benjamini-Hochberg at "
        f"q={analysis.config.fdr_q}. Rows with no findings are omitted from "
        "the table but counted in the family.",
        "",
    ]
    if not shown:
        return [*lines, "No downgraded arm drew any finding.", ""]
    headers = ["arm", "kind", "detector", "flagged", "floor", "excess", "p", "p (BH)"]
    return [*lines, *_table([*headers, "SPRT", "verdict"], [_hypothesis_row(h) for h in shown]), ""]


def _backtests(analysis: SweepAnalysis) -> list[str]:
    rows = [
        [
            b.arm,
            f"{b.violations}/{b.runs}",
            _pct(b.expected_rate),
            _p(b.kupiec_p),
            _p(b.independence_p),
            _p(b.coverage_p),
            "clustered" if b.clustered else "ok",
        ]
        for b in analysis.backtests
    ]
    headers = ["arm", "runs with a finding", "floor", "Kupiec POF p", "independence p"]
    lines = [
        "## 4. Backtests",
        "",
        "A violation is a run with any finding. Kupiec's proportion-of-failures "
        "test asks whether the violation rate matches the floor; Christoffersen's "
        "independence test asks whether violations cluster in run order "
        "(replicate-major across tasks). Clustering means runs are not "
        "exchangeable and that arm's p-values above are optimistic.",
        "",
        *_table([*headers, "joint p", f"at {BACKTEST_ALPHA}"], rows),
        "",
    ]
    clustered = [b.arm for b in analysis.backtests if b.clustered]
    if clustered:
        lines += [f"**Findings cluster in run order for: {', '.join(clustered)}.**", ""]
    return lines


def _evidence_line(f: Finding) -> str:
    e = f.evidence
    parts = [f"`{f.task_id}` replicate {f.replicate}"]
    if e.step_index is not None:
        parts.append(f"step {e.step_index}")
    text = f"- {', '.join(parts)}: {_cell(e.detail, 160)}"
    if e.baseline_reference:
        text += f" Baseline: {_cell(e.baseline_reference, 120)}."
    if e.quoted_span:
        text += f' Quoted: "{_cell(e.quoted_span, 120)}".'
    return text


def _evidence(analysis: SweepAnalysis) -> list[str]:
    if not analysis.discoveries:
        return []
    lines = ["## 5. Evidence", ""]
    for h in analysis.discoveries:
        matching = [f for f in analysis.findings if f.arm == h.arm and f.kind == h.kind]
        lines.append(f"### {h.arm}: {h.kind.value} ({h.flagged}/{h.runs} runs)")
        lines.append("")
        lines += [_evidence_line(f) for f in matching[:MAX_EVIDENCE_PER_DISCOVERY]]
        if len(matching) > MAX_EVIDENCE_PER_DISCOVERY:
            lines.append(f"- ... and {len(matching) - MAX_EVIDENCE_PER_DISCOVERY} more")
        lines.append("")
    return lines


def _limits(analysis: SweepAnalysis) -> list[str]:
    lines = ["## Limits of this report", ""]
    if analysis.mode == "structural":
        judged = ", ".join(
            k.value for k in (FindingKind.UNSUPPORTED_CLAIM, FindingKind.UNSUPPORTED_CITATION)
        )
        lines.append(f"- Judge detectors ({judged}) were not run. Those rows are absent, not zero.")
    lines += [
        "- One finding per run: the highest-priority sub-class. A run with a "
        "missing tool call AND a fabricated value counts only as the former.",
        f"- {analysis.config.replicates} replicates per arm per task. Per-task "
        "rates are coarse; the tests pool across tasks for that reason.",
        "- The null rate is a plug-in estimate from the same sweep, widened to "
        "its Wilson upper bound. It is conservative, not exact.",
    ]
    return [*lines, ""]


def render_markdown(analysis: SweepAnalysis) -> str:
    lines = [
        "# downgrade: sweep report",
        "",
        f"**{headline(analysis)}**",
        "",
        *_setup(analysis),
        *_floor(analysis),
        *_arms(analysis),
        *_tests(analysis),
        *_backtests(analysis),
        *_evidence(analysis),
        *_limits(analysis),
    ]
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["MAX_EVIDENCE_PER_DISCOVERY", "headline", "render_markdown"]
