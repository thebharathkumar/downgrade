"""Command line interface.

    downgrade doctor    probe the router wire format and check credentials
    downgrade smoke     run one task across all six arms and price the sweep
    downgrade verify    check the committed corpus against its manifest

`doctor` and `smoke` deliberately need no corpus. They answer the questions
that must be settled before a paid sweep (does the pair string work, does the
preference dial change anything, what does it cost) and blocking them on
corpus acquisition would put those answers after the money instead of before.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click

from downgrade.arms import ALL_ARMS, PREFERENCE_LABELS
from downgrade.client import FireworksClient
from downgrade.models import SweepConfig
from downgrade.probe import SmokeReport, probe_router_format, run_smoke
from downgrade.suite.corpus import CorpusManifest, verify_corpus
from downgrade.suite.spec import AnswerCheck, TaskSpec
from downgrade.tools import FunctionTool, ToolDef, ToolError, ToolRegistry
from downgrade.transport import HttpTransport

DEFAULT_CORPUS = Path("src/downgrade/suite/corpus")
API_KEY_ENV = "FIREWORKS_API_KEY"

# A small inline table so the probe task exercises real multi-step tool use
# without depending on the corpus. Values are arbitrary; only the arithmetic
# matters, and it is verifiable.
PROBE_TABLE: dict[str, dict[str, float]] = {
    "ACME": {"assets": 1200.0, "liabilities": 800.0},
    "BOREAS": {"assets": 900.0, "liabilities": 300.0},
}


def _probe_registry() -> ToolRegistry:
    def list_companies() -> list[str]:
        return sorted(PROBE_TABLE)

    def get_figure(company: str, concept: str) -> dict[str, Any]:
        row = PROBE_TABLE.get(company.upper())
        if row is None:
            raise ToolError(f"No such company '{company}'. Try: {', '.join(sorted(PROBE_TABLE))}")
        key = concept.lower()
        if key not in row:
            raise ToolError(f"No such concept '{concept}'. Try: assets, liabilities")
        return {"company": company.upper(), "concept": key, "value": row[key]}

    return ToolRegistry(
        [
            FunctionTool(
                definition=ToolDef(
                    name="list_companies",
                    description="List the companies available in the table.",
                    parameters={"type": "object", "properties": {}},
                ),
                func=list_companies,
            ),
            FunctionTool(
                definition=ToolDef(
                    name="get_figure",
                    description="Get one reported figure for one company.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "company": {"type": "string"},
                            "concept": {"type": "string", "enum": ["assets", "liabilities"]},
                        },
                        "required": ["company", "concept"],
                    },
                ),
                func=get_figure,
            ),
        ]
    )


PROBE_TASK = TaskSpec(
    task_id="probe_equity",
    family="tabular_aggregation",
    prompt=(
        "Using the tools, find total assets and total liabilities for every "
        "company in the table, then report the combined shareholders' equity "
        "(assets minus liabilities) across all of them. "
        "Answer with the number alone."
    ),
    answer_check=AnswerCheck(kind="numeric", value=1000.0),
    expected_tools=["list_companies", "get_figure"],
    max_steps=8,
)


def _resolve_key(explicit: str | None) -> str:
    key = explicit or os.environ.get(API_KEY_ENV, "")
    if not key:
        raise click.ClickException(
            f"No API key. Set {API_KEY_ENV} or pass --api-key. "
            "Both doctor and smoke make real, billable requests."
        )
    return key


def _config(primary: str, secondary: str, **kw: Any) -> SweepConfig:
    return SweepConfig(primary_model=primary, secondary_model=secondary, **kw)


@click.group()
@click.version_option(package_name="downgrade")
def main() -> None:
    """Measure silent quality regression when a model router downgrades you."""


@main.command()
@click.option("--primary", required=True, help="Primary (stronger) model of the router pair.")
@click.option("--secondary", required=True, help="Secondary (cheaper) model of the router pair.")
@click.option("--api-key", default=None, help=f"Defaults to ${API_KEY_ENV}.")
@click.option("--base-url", default="https://api.fireworks.ai/inference/v1")
def doctor(primary: str, secondary: str, api_key: str | None, base_url: str) -> None:
    """Probe the router wire format empirically. Run this before anything else.

    Sends one minimal request with the pair expressed as short slugs and one
    with fully-qualified IDs, and reports which the API accepted. Nothing is
    assumed: the sweep refuses to build a router model string until this has
    answered.
    """
    key = _resolve_key(api_key)
    config = _config(primary, secondary, base_url=base_url)
    client = FireworksClient(HttpTransport(), api_key=key, base_url=base_url)

    click.echo(f"Probing router wire format against {base_url}\n")
    result = probe_router_format(client, config)

    for attempt in result.attempts:
        mark = "ok  " if attempt.ok else "FAIL"
        click.echo(f"  [{mark}] {attempt.form:5s}  {attempt.model}")
        if attempt.detail:
            click.echo(f"           {attempt.detail[:160]}")

    click.echo("")
    click.echo(result.summary())
    if not result.resolved:
        click.echo(
            "\nBoth forms were rejected. Check the model slugs in your Fireworks "
            "dashboard and that the account has FireRouter enabled."
        )
        sys.exit(1)
    click.echo(
        f"\nPass this to the sweep:  --short-slugs={'true' if result.short_slugs else 'false'}"
    )


def _render_smoke(report: SmokeReport, tasks: int, replicates: int) -> None:
    click.echo("\nBaseline noise floor (read this first)")
    click.echo("-" * 64)
    baseline = report.baseline
    if baseline is None:
        click.echo("  no baseline arm ran")
    else:
        agreement = report.baseline_answer_agreement
        click.echo(
            f"  control arm agrees with itself : {agreement:.0%} over {baseline.replicates} runs"
        )
        click.echo(f"  distinct answers from control  : {baseline.distinct_answers}")
        click.echo(f"  runs with unstable routing     : {baseline.route_unstable_runs}")
        if agreement < 1.0:
            click.echo(
                "  NOTE: the control arm varies on its own. No downgraded arm's\n"
                "        finding rate can be believed below this level."
            )

    click.echo("\nPer-arm routing")
    click.echo("-" * 64)
    click.echo(
        f"  {'arm':16s} {'preference':20s} {'downgraded':>10s} {'steps':>6s} {'answers':>8s}"
    )
    for arm in report.arms:
        label = PREFERENCE_LABELS.get(arm.preference or 0, "no router")
        pref = f"{arm.preference} {label}" if arm.preference is not None else f"- {label}"
        click.echo(
            f"  {arm.arm:16s} {pref:20s} "
            f"{arm.downgrade_rate:>10.0%} {arm.mean_steps:>6.1f} {arm.distinct_answers:>8d}"
        )

    click.echo("\n  served models per arm")
    for arm in report.arms:
        counts = ", ".join(f"{m}={n}" for m, n in sorted(arm.served_model_counts.items())) or "none"
        click.echo(f"    {arm.arm:16s} {counts}")

    if not report.routing_responds_to_preference:
        click.echo(
            "\n  WARNING: the preference dial did not change which model served\n"
            "  these requests. On this task shape there may be no downgrade to\n"
            "  measure, which is a real result, not a bug to work around."
        )

    inputs, outputs = report.total_tokens()
    projection = report.extrapolate(tasks=tasks, replicates=replicates)
    click.echo(
        f"\nCost, measured on this task then scaled to {tasks} tasks x {replicates} replicates"
    )
    click.echo("-" * 64)
    click.echo(f"  measured   : {inputs:,} input + {outputs:,} output tokens")
    click.echo(
        f"  projected  : {projection['runs']:,.0f} runs, {projection['api_calls']:,.0f} API calls"
    )
    click.echo(
        f"               {projection['input_tokens']:,.0f} input + "
        f"{projection['output_tokens']:,.0f} output tokens"
    )
    click.echo(
        "  Multiply by your dashboard's per-million rates. Cost is linear in\n"
        "  replicates but quadratic in steps per run, since every turn resends\n"
        "  the conversation; the probe task is short, so treat this as a floor."
    )

    if not all(a.usage_complete for a in report.arms):
        click.echo(
            "\n  WARNING: at least one stream ended without a usage chunk, so these\n"
            "  token counts are under-reported."
        )


@main.command()
@click.option("--primary", required=True)
@click.option("--secondary", required=True)
@click.option(
    "--short-slugs/--full-ids",
    "short_slugs",
    default=None,
    required=True,
    help="From `downgrade doctor`. Not guessed.",
)
@click.option("--replicates", default=2, show_default=True, help="Runs per arm.")
@click.option("--temperature", default=0.0, show_default=True)
@click.option("--seed", default=20260903, show_default=True)
@click.option("--project-tasks", default=24, show_default=True, help="Suite size to price for.")
@click.option("--project-replicates", default=5, show_default=True)
@click.option("--api-key", default=None)
@click.option("--base-url", default="https://api.fireworks.ai/inference/v1")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Write raw JSON here.")
def smoke(
    primary: str,
    secondary: str,
    short_slugs: bool,
    replicates: int,
    temperature: float,
    seed: int,
    project_tasks: int,
    project_replicates: int,
    api_key: str | None,
    base_url: str,
    out: Path | None,
) -> None:
    """Run one task across all six arms, then price the full sweep.

    Answers the two things you cannot know from documentation: whether the
    preference dial actually changes the served model for tasks of this
    shape, and what the full sweep will cost. Reports the control arm's own
    variability first, because every other number is read against it.
    """
    key = _resolve_key(api_key)
    config = _config(
        primary,
        secondary,
        base_url=base_url,
        short_slugs=short_slugs,
        temperature=temperature,
        seed=seed,
        replicates=replicates,
        arms=[a.name for a in ALL_ARMS],
    )
    client = FireworksClient(HttpTransport(), api_key=key, base_url=base_url)

    click.echo(
        f"Smoke sweep: {len(ALL_ARMS)} arms x {replicates} replicates on '{PROBE_TASK.task_id}'"
    )
    click.echo(f"Router pair: {config.router_model}")
    click.echo(
        f"Sampling   : temperature={temperature}, seed={seed}  (fingerprint {config.fingerprint})"
    )

    report = run_smoke(client, _probe_registry(), config, PROBE_TASK, replicates=replicates)
    _render_smoke(report, tasks=project_tasks, replicates=project_replicates)

    if out is not None:
        payload = {
            "config": json.loads(config.model_dump_json()),
            "task_id": report.task_id,
            "arms": [
                {
                    "arm": arm.arm,
                    "preference": arm.preference,
                    "downgrade_rate": arm.downgrade_rate,
                    "served_model_counts": arm.served_model_counts,
                    "status_counts": arm.status_counts,
                    "answers": arm.answers,
                    "trajectories": [json.loads(t.model_dump_json()) for t in arm.trajectories],
                }
                for arm in report.arms
            ],
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        click.echo(f"\nWrote raw trajectories to {out}")


@main.command()
@click.option(
    "--corpus", type=click.Path(path_type=Path), default=DEFAULT_CORPUS, show_default=True
)
def verify(corpus: Path) -> None:
    """Check the committed corpus against its manifest.

    A corpus that changed between arms produces different tool results for
    reasons unrelated to routing, so this fails loudly rather than warning.
    """
    manifest_path = corpus / "manifest.json"
    if not manifest_path.is_file():
        raise click.ClickException(
            f"No manifest at {manifest_path}. Build the corpus first:\n"
            '  python scripts/fetch_corpus.py --user-agent "Your Name you@example.com"'
        )
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    report = verify_corpus(corpus, manifest)

    click.echo(f"Corpus: {corpus}")
    click.echo(f"  {report.summary()}")
    for path in report.missing:
        click.echo(f"  MISSING   {path}")
    for path in report.corrupted:
        click.echo(f"  CORRUPTED {path}")
    for path in report.untracked:
        click.echo(f"  untracked {path}")

    if not report.ok:
        raise click.ClickException("Corpus does not match its manifest.")
    click.echo("  corpus OK")


if __name__ == "__main__":
    main()
