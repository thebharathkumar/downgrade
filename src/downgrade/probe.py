"""Empirical probes that must run before a paid sweep.

Three questions have to be answered against the live API rather than guessed,
because getting any of them wrong wastes the whole sweep:

  1. Does `firerouter/<pair>` take short slugs or fully-qualified model IDs?
     Undocumented. Guessing wrong 404s on the first call.
  2. Does the routing preference actually change which model serves the
     request, for tasks of this shape? If the router never downgrades at
     preference 5, there is no regression to measure and that is the result.
  3. What does one task cost across all six arms? Multiply by tasks and
     replicates before committing to the full sweep, not after.

The smoke sweep answers 2 and 3 together and reports the baseline's own
variability first, because a downgraded arm's numbers cannot be read without
knowing how much the control arm moves on its own.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from downgrade.arms import ALL_ARMS, BASELINE_DIRECT, Arm
from downgrade.client import FireworksClient
from downgrade.models import SweepConfig, Trajectory, short_slug
from downgrade.runner import AgentLoop
from downgrade.suite.spec import TaskSpec
from downgrade.tools import ToolRegistry

PROBE_PROMPT = "Reply with exactly the word: ok"


@dataclass
class ProbeAttempt:
    form: str
    model: str
    ok: bool
    detail: str = ""


@dataclass
class FormatProbeResult:
    """Which wire format the API accepted, and the evidence for it."""

    short_slugs: bool | None
    attempts: list[ProbeAttempt] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.short_slugs is not None

    def summary(self) -> str:
        if self.short_slugs is None:
            return "Router format UNRESOLVED: neither form was accepted."
        form = "short slugs" if self.short_slugs else "fully-qualified IDs"
        return f"Router format resolved: {form}."


def probe_router_format(
    client: FireworksClient, config: SweepConfig, max_tokens: int = 8
) -> FormatProbeResult:
    """Send one minimal request in each form and keep whichever is accepted.

    Short form is tried first only because `firerouter/<a>/<b>` with up to
    eight models makes fully-qualified paths implausible, not because it is
    assumed correct. If both are accepted the short form wins and the result
    records that both worked.
    """
    candidates = [
        (
            "short",
            f"firerouter/{short_slug(config.primary_model)}/{short_slug(config.secondary_model)}",
        ),
        ("full", f"firerouter/{config.primary_model}/{config.secondary_model}"),
    ]
    attempts: list[ProbeAttempt] = []
    accepted: bool | None = None

    for form, model in candidates:
        result = client.chat(
            messages=[{"role": "user", "content": PROBE_PROMPT}],
            model=model,
            turn_index=0,
            temperature=config.temperature,
            max_tokens=max_tokens,
            seed=config.seed,
            preference=3,
            extra_headers={"x-routing-preference": "3"},
            primary_model=config.primary_model,
        )
        attempts.append(
            ProbeAttempt(
                form=form,
                model=model,
                ok=result.ok,
                detail=result.error or (result.route.served_model or "" if result.route else ""),
            )
        )
        if result.ok and accepted is None:
            accepted = form == "short"

    return FormatProbeResult(short_slugs=accepted, attempts=attempts)


@dataclass
class ArmSmoke:
    """What one arm did across its replicates in the smoke sweep."""

    arm: str
    preference: int | None
    trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def replicates(self) -> int:
        return len(self.trajectories)

    @property
    def served_model_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trajectory in self.trajectories:
            for step in trajectory.steps:
                model = step.route.served_model
                if model is not None:
                    counts[model] = counts.get(model, 0) + 1
        return counts

    @property
    def downgrade_rate(self) -> float:
        """Fraction of API calls served by something other than the primary."""
        calls = [
            s.route for t in self.trajectories for s in t.steps if s.route.downgraded is not None
        ]
        if not calls:
            return 0.0
        return sum(1 for r in calls if r.downgraded) / len(calls)

    @property
    def route_unstable_runs(self) -> int:
        return sum(1 for t in self.trajectories if not t.route_stability)

    @property
    def mean_input_tokens(self) -> float:
        return self._mean([float(t.totals.input_tokens) for t in self.trajectories])

    @property
    def mean_output_tokens(self) -> float:
        return self._mean([float(t.totals.output_tokens) for t in self.trajectories])

    @property
    def mean_steps(self) -> float:
        return self._mean([float(len(t.steps)) for t in self.trajectories])

    @property
    def answers(self) -> list[str]:
        return [t.final_answer or "" for t in self.trajectories]

    @property
    def distinct_answers(self) -> int:
        return len(set(self.answers))

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for trajectory in self.trajectories:
            counts[trajectory.status] = counts.get(trajectory.status, 0) + 1
        return counts

    @property
    def usage_complete(self) -> bool:
        return all(t.totals.complete for t in self.trajectories)

    @staticmethod
    def _mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0


@dataclass
class SmokeReport:
    """The numbers to look at before committing to a full sweep."""

    config: SweepConfig
    task_id: str
    arms: list[ArmSmoke] = field(default_factory=list)

    def arm(self, name: str) -> ArmSmoke | None:
        return next((a for a in self.arms if a.arm == name), None)

    @property
    def baseline(self) -> ArmSmoke | None:
        return self.arm(BASELINE_DIRECT.name)

    @property
    def baseline_answer_agreement(self) -> float:
        """How often the control arm agrees with itself.

        This is the noise floor in its crudest form, and it is the first thing
        to read. At temperature 0 it should be 1.0. Anything less means the
        control arm alone produces variation, and no downgraded arm's finding
        rate can be believed below that level.
        """
        baseline = self.baseline
        if baseline is None or not baseline.answers:
            return 0.0
        modal = max(set(baseline.answers), key=baseline.answers.count)
        return baseline.answers.count(modal) / len(baseline.answers)

    @property
    def routing_responds_to_preference(self) -> bool:
        """Did the preference dial change anything at all?

        If it did not, the full sweep has nothing to measure on this task
        shape, and that is a finding rather than a bug to work around.
        """
        rates = {a.arm: a.downgrade_rate for a in self.arms if a.preference is not None}
        return len(set(rates.values())) > 1

    def total_tokens(self) -> tuple[int, int]:
        inputs = sum(t.totals.input_tokens for a in self.arms for t in a.trajectories)
        outputs = sum(t.totals.output_tokens for a in self.arms for t in a.trajectories)
        return inputs, outputs

    def extrapolate(self, tasks: int, replicates: int) -> dict[str, float]:
        """Scale this one task's measured cost to a full sweep.

        Linear in tasks and replicates. Not linear in steps per run: every
        turn resends the conversation, so a task that takes twice as many
        steps costs roughly four times as much. The smoke task is short, so
        treat this as a floor rather than an estimate.
        """
        measured_replicates = max((a.replicates for a in self.arms), default=0)
        if measured_replicates == 0:
            return {"runs": 0.0, "input_tokens": 0.0, "output_tokens": 0.0, "api_calls": 0.0}

        scale = (tasks * replicates) / measured_replicates
        inputs, outputs = self.total_tokens()
        calls = sum(len(t.steps) for a in self.arms for t in a.trajectories)
        return {
            "runs": float(len(self.arms) * tasks * replicates),
            "api_calls": calls * scale,
            "input_tokens": inputs * scale,
            "output_tokens": outputs * scale,
        }


def run_smoke(
    client: FireworksClient,
    registry: ToolRegistry,
    config: SweepConfig,
    task: TaskSpec,
    arms: list[Arm] | None = None,
    replicates: int = 2,
) -> SmokeReport:
    """Run one task across every arm, a few times each.

    Deliberately runs the control arm first so that if the sweep is
    interrupted the baseline exists, which is the only arm the others are
    meaningless without.
    """
    selected = list(arms or ALL_ARMS)
    loop = AgentLoop(client, registry, config)
    report = SmokeReport(config=config, task_id=task.task_id)

    for arm in selected:
        smoke = ArmSmoke(arm=arm.name, preference=arm.preference)
        for replicate in range(replicates):
            smoke.trajectories.append(loop.run(task, arm, replicate=replicate))
        report.arms.append(smoke)
    return report


__all__ = [
    "PROBE_PROMPT",
    "ArmSmoke",
    "FormatProbeResult",
    "ProbeAttempt",
    "SmokeReport",
    "probe_router_format",
    "run_smoke",
]
