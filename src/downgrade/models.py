"""Core data model for downgrade.

Four objects carry the whole project:

  Trajectory      one task executed once, under one arm, in a fresh conversation
  BaselineProfile what the baseline arm does across its replicates, as a reference
  Finding         one detected degradation, with the evidence that produced it
  ArmComparison   one downgraded arm scored against a BaselineProfile

Two properties of this schema are load-bearing and deliberate.

1. Comparison is set-vs-set, never run-vs-run. At any temperature above
   zero a single downgraded run differing from a single baseline run is
   indistinguishable from sampling noise. Every arm therefore carries R
   replicate Trajectories, and a Finding records how often the behaviour
   occurred on each side rather than asserting it happened once. Weeks 2
   and 3 (sequential testing, Benjamini-Hochberg) consume those rates
   directly; there is no retrofit.

2. Every Finding names the detector that produced it. The structural
   detectors and the LLM-judge detectors are separately addressable so
   their precision can be measured independently later. `detector` is not
   decoration: `classify(mode=...)` gates which detectors run at all, and
   a judge-mode run and a structural-mode run over the same trajectories
   are directly comparable.

Route attribution note: the GenAI semantic conventions define no attribute
for "which model a router selected". `gen_ai.response.model` already
carries the served model, so RouteObservation records the served model
plus the things the spec has no home for (which arm, which preference,
how we learned the route). See downgrade.otel for the emitted names.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

SCHEMA_VERSION = "1"


# --------------------------------------------------------------------------
# Usage and routing
# --------------------------------------------------------------------------


class Usage(BaseModel):
    """Token counts for a single API call.

    Fireworks returns usage only in the FINAL streaming chunk, so a
    partially-consumed stream yields zeros here rather than a wrong number.
    `complete` records whether we actually saw the final chunk, because a
    silently-zero token count would corrupt the cost model.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    complete: bool = False

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            complete=self.complete and other.complete,
        )


RouteSource = Literal["response_model", "header", "unknown"]


class RouteObservation(BaseModel):
    """What the router did for ONE API call.

    Recorded per call, not per run, because FireRouter decides per request.
    Within a run the decision is cached, so `turn_index` lets us see whether
    a route drifted mid-conversation.

    `temperature` and `seed` are recorded here rather than only in SweepConfig
    so that a stored trajectory is self-describing: an arm whose sampling
    settings drifted is detectable from the artifact alone, without trusting
    the config that claims to have produced it.
    """

    turn_index: int
    requested_model: str
    served_model: str | None = None
    preference: int | None = None
    downgraded: bool | None = None
    source: RouteSource = "unknown"
    temperature: float
    seed: int | None = None
    response_id: str | None = None
    finish_reason: str | None = None
    response_headers: dict[str, str] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    latency_ms: int = 0


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------


class ToolCall(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def signature(self) -> str:
        """Stable identity for set-difference across runs.

        Name plus canonicalised arguments. Two calls with the same name but
        different arguments are different calls: a downgraded run that calls
        `query_table` with a narrower filter has not made the same call.
        """
        canonical = json.dumps(self.arguments, sort_keys=True, default=str)
        return f"{self.name}({canonical})"


class ToolResult(BaseModel):
    """What a tool returned for one call.

    A failed tool is not an exception: the model gets the error back as an
    observation and may recover from it. Whether it did is exactly what the
    redundant_retry detector reads.
    """

    call_id: str
    name: str
    content: Any = None
    ok: bool = True
    error: str | None = None
    latency_ms: int = 0


class Step(BaseModel):
    """One API call, and every tool call the resulting turn made.

    Deliberately one Step per API call rather than per tool call. Usage,
    latency and the route are properties of the request, so splitting a
    multi-call turn across several Steps would count the same tokens more
    than once in `Trajectory.totals` and record the same route twice.
    """

    index: int
    thought: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    route: RouteObservation
    error: str | None = None

    @property
    def failed_tool_calls(self) -> list[ToolResult]:
        return [r for r in self.tool_results if not r.ok]


RunStatus = Literal["completed", "errored", "no_termination"]


class Trajectory(BaseModel):
    """One task, one arm, one fresh conversation.

    `conversation_id` is minted by the runner and is unique per run. It is
    the artifact of the isolation guarantee: FireRouter caches its routing
    decision within a conversation, so a reused conversation would silently
    inherit a previous arm's route and make the whole sweep noise.
    """

    schema_version: str = SCHEMA_VERSION
    run_id: str
    task_id: str
    arm: str
    replicate: int
    conversation_id: str
    config_fingerprint: str
    steps: list[Step] = Field(default_factory=list)
    final_answer: str | None = None
    status: RunStatus = "completed"
    error: str | None = None
    wall_ms: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def totals(self) -> Usage:
        total = Usage(complete=True)
        for step in self.steps:
            total = total + step.route.usage
        return total

    @computed_field  # type: ignore[prop-decorator]
    @property
    def served_models(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            model = step.route.served_model
            if model is not None and model not in seen:
                seen.append(model)
        return seen

    @computed_field  # type: ignore[prop-decorator]
    @property
    def route_stability(self) -> bool:
        """True when every turn in this run was served by the same model."""
        return len(self.served_models) <= 1

    def tool_signatures(self) -> list[str]:
        return [call.signature() for step in self.steps for call in step.tool_calls]

    def tool_names(self) -> list[str]:
        return [call.name for step in self.steps for call in step.tool_calls]

    def tool_results(self) -> list[ToolResult]:
        return [result for step in self.steps for result in step.tool_results]


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


class FindingKind(StrEnum):
    # Loud: the outcome itself changed. An outcome-only judge catches these.
    WRONG_ANSWER = "wrong_answer"
    RUN_ERROR = "run_error"
    NO_TERMINATION = "no_termination"

    # Silent: the final answer survived, the path degraded. Ordered by the
    # priority in the brief. unsupported_claim is first because it evaded
    # step-level judges most often.
    UNSUPPORTED_CLAIM = "unsupported_claim"
    MISSING_TOOL_CALL = "missing_tool_call"
    DROPPED_CONSTRAINT = "dropped_constraint"
    REDUNDANT_RETRY = "redundant_retry"
    FABRICATED_VALUE = "fabricated_value"
    UNSUPPORTED_CITATION = "unsupported_citation"


LOUD_KINDS: frozenset[FindingKind] = frozenset(
    {FindingKind.WRONG_ANSWER, FindingKind.RUN_ERROR, FindingKind.NO_TERMINATION}
)

# Lower rank sorts first. Loud outranks every silent sub-class; silent
# sub-classes follow the brief's priority order.
SEVERITY_RANK: dict[FindingKind, int] = {
    FindingKind.WRONG_ANSWER: 0,
    FindingKind.RUN_ERROR: 1,
    FindingKind.NO_TERMINATION: 2,
    FindingKind.UNSUPPORTED_CLAIM: 3,
    FindingKind.MISSING_TOOL_CALL: 4,
    FindingKind.DROPPED_CONSTRAINT: 5,
    FindingKind.REDUNDANT_RETRY: 6,
    FindingKind.FABRICATED_VALUE: 7,
    FindingKind.UNSUPPORTED_CITATION: 8,
}

Detector = Literal["structural", "judge"]

# Which detector owns each kind. dropped_constraint is structural because
# every task declares its constraints as machine-checkable predicates; a
# constraint that cannot be written as a predicate means the task is badly
# authored and gets rewritten, not handed to a judge.
DETECTOR_FOR_KIND: dict[FindingKind, Detector] = {
    FindingKind.WRONG_ANSWER: "structural",
    FindingKind.RUN_ERROR: "structural",
    FindingKind.NO_TERMINATION: "structural",
    FindingKind.UNSUPPORTED_CLAIM: "judge",
    FindingKind.MISSING_TOOL_CALL: "structural",
    FindingKind.DROPPED_CONSTRAINT: "structural",
    FindingKind.REDUNDANT_RETRY: "structural",
    FindingKind.FABRICATED_VALUE: "structural",
    FindingKind.UNSUPPORTED_CITATION: "judge",
}


class Evidence(BaseModel):
    """Why the classifier believes what it believes.

    A label without this is not reportable. `baseline_reference` is a
    description of the baseline arm's behaviour (for example "called in 5/5
    baseline runs"), not a single run's value, because the baseline is a set.
    """

    step_index: int | None = None
    detail: str
    baseline_reference: str | None = None
    downgraded_value: str | None = None
    quoted_span: str | None = None
    constraint_id: str | None = None


class Finding(BaseModel):
    """One degradation detected in ONE downgraded run.

    Anchored to a specific run and, where meaningful, a specific step, so the
    evidence stays concrete. Aggregation to rates happens in ArmComparison.
    """

    kind: FindingKind
    detector: Detector
    run_id: str
    task_id: str
    arm: str
    replicate: int
    evidence: Evidence
    confidence: float = 1.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def severity(self) -> int:
        return SEVERITY_RANK[self.kind]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_loud(self) -> bool:
        return self.kind in LOUD_KINDS


# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------


class ConstraintResult(BaseModel):
    """Outcome of one machine-checkable constraint predicate on one run."""

    constraint_id: str
    predicate: str
    satisfied: bool
    detail: str = ""


# --------------------------------------------------------------------------
# Baseline profile and comparison
# --------------------------------------------------------------------------


class BaselineProfile(BaseModel):
    """What the baseline arm does for one task, across all its replicates.

    Built once per task from R baseline Trajectories, then every downgraded
    run is scored against it. This is deliberately not pairwise: pairing run
    i against run i is arbitrary, and all-pairs inflates dependence between
    observations. Scoring each downgraded run against a stable reference
    yields R independent observations per arm per task, which is the unit a
    sequential test consumes in week 2.

    Rates are fractions of baseline replicates, so a detector can require
    that a behaviour was RELIABLE in baseline before its absence counts as a
    regression. A tool called in 2 of 5 baseline runs is not an expectation.
    """

    task_id: str
    baseline_arm: str
    replicate_count: int
    reliability_threshold: float
    tool_signature_rates: dict[str, float] = Field(default_factory=dict)
    tool_name_rates: dict[str, float] = Field(default_factory=dict)
    median_tool_calls: float = 0.0
    call_count_by_signature: dict[str, float] = Field(default_factory=dict)
    constraint_satisfaction_rates: dict[str, float] = Field(default_factory=dict)
    grounded_values: set[str] = Field(default_factory=set)
    cited_documents: set[str] = Field(default_factory=set)
    correct_rate: float = 0.0
    answers: list[str] = Field(default_factory=list)

    # How variable the baseline arm is with itself. A profile whose own
    # behaviour scatters cannot support a claim that a downgraded arm
    # departed from it, so these travel with every rate rather than being
    # recomputed by whoever reads the report.
    tool_call_count_mean: float = 0.0
    tool_call_count_stdev: float = 0.0
    distinct_answers: int = 0
    modal_answer_rate: float = 0.0

    @property
    def sample_size(self) -> int:
        return self.replicate_count

    def rate_stderr(self, rate: float) -> float:
        """Binomial standard error of a rate at this profile's sample size.

        R is small by construction (5 by default), so every rate here is
        coarse: at R=5 a rate can only take six values and its standard error
        near 0.5 is about 0.22. Reporting the rate without this number invites
        reading 4/5 versus 5/5 as a signal when it is one run.
        """
        n = self.replicate_count
        if n <= 0:
            return 0.0
        return float((rate * (1.0 - rate) / n) ** 0.5)

    def is_reliable(self, rate_map: dict[str, float], key: str) -> bool:
        """Was this behaviour dependable enough in baseline to expect it?

        The threshold is `reliability_threshold`, taken from the pre-registered
        SweepConfig and copied onto the profile, not passed in by the caller.
        A caller-supplied threshold is a knob that can be turned after seeing
        results; this one cannot be moved without changing the config
        fingerprint and invalidating the comparison.
        """
        return rate_map.get(key, 0.0) >= self.reliability_threshold


class FindingRate(BaseModel):
    """One finding kind aggregated across an arm's replicates."""

    kind: FindingKind
    detector: Detector
    occurrences: int
    replicate_count: int
    findings: list[Finding] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rate(self) -> float:
        if self.replicate_count == 0:
            return 0.0
        return self.occurrences / self.replicate_count


ClassifyMode = Literal["structural", "judge", "both"]
Verdict = Literal["LOUD", "SILENT", "CLEAN"]


class NoiseFloor(BaseModel):
    """How often the classifier fires when nothing was downgraded at all.

    Computed leave-one-out over the baseline arm: for each baseline run, a
    profile is built from the other R-1 runs and that run is scored against
    it. Every finding this produces is a false positive by construction,
    because both sides are the same arm at the same settings.

    This is the first number the sweep reports, before any downgraded arm.
    A finding rate on a downgraded arm means nothing until it is read against
    the rate the same detector produces on baseline compared with itself. At
    temperature 0.0 the floor should be near zero, and if it is not, that is
    a result about the detector rather than about routing.
    """

    task_id: str
    baseline_arm: str
    replicate_count: int
    finding_rates: list[FindingRate] = Field(default_factory=list)

    def rate_for(self, kind: FindingKind) -> float:
        for entry in self.finding_rates:
            if entry.kind == kind:
                return entry.rate
        return 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_false_positive_rate(self) -> float:
        """Fraction of leave-one-out baseline runs that drew any finding."""
        if self.replicate_count == 0:
            return 0.0
        flagged = sum(entry.occurrences for entry in self.finding_rates)
        return min(1.0, flagged / self.replicate_count)


class ArmComparison(BaseModel):
    """One downgraded arm scored against the baseline arm, for one task.

    Holds the replicate run ids on both sides rather than a single pair.
    `verdict` is the arm-level roll-up: LOUD if any loud finding occurred,
    SILENT if only silent findings did, CLEAN if none did. The per-replicate
    detail lives in `finding_rates` so week 2 can test the rates rather than
    the roll-up.
    """

    schema_version: str = SCHEMA_VERSION
    task_id: str
    baseline_arm: str
    downgraded_arm: str
    baseline_run_ids: list[str] = Field(default_factory=list)
    downgraded_run_ids: list[str] = Field(default_factory=list)
    mode: ClassifyMode = "both"
    profile: BaselineProfile | None = None
    finding_rates: list[FindingRate] = Field(default_factory=list)
    noise_floor: NoiseFloor | None = None
    baseline_correct_rate: float = 0.0
    downgraded_correct_rate: float = 0.0
    route_downgrade_rate: float = 0.0

    def excess_over_floor(self, kind: FindingKind) -> float:
        """Finding rate minus the rate the same detector fires at on baseline.

        Never negative. This, not the raw rate, is what week 2 tests.
        """
        observed = next((e.rate for e in self.finding_rates if e.kind == kind), 0.0)
        floor = self.noise_floor.rate_for(kind) if self.noise_floor is not None else 0.0
        return max(0.0, observed - floor)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def verdict(self) -> Verdict:
        kinds = {fr.kind for fr in self.finding_rates if fr.occurrences > 0}
        if kinds & LOUD_KINDS:
            return "LOUD"
        if kinds:
            return "SILENT"
        return "CLEAN"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_findings(self) -> list[Finding]:
        out: list[Finding] = []
        for rate in self.finding_rates:
            out.extend(rate.findings)
        return sorted(out, key=lambda f: (f.severity, f.replicate))


# --------------------------------------------------------------------------
# Sweep configuration
# --------------------------------------------------------------------------


class RouterFormatUnknownError(RuntimeError):
    """Raised when the router model string is built before the format is known."""


def short_slug(model_id: str) -> str:
    """`accounts/fireworks/models/kimi-k3` -> `kimi-k3`; leave bare slugs alone."""
    return model_id.rsplit("/", 1)[-1]


class SweepConfig(BaseModel):
    """Everything that must be identical across arms for a sweep to be valid.

    `fingerprint` covers every field EXCEPT the arm list, and every Trajectory
    stores it. Comparing two arms whose fingerprints differ means sampling
    settings drifted between them, and the comparison is measuring the drift
    rather than the routing. classify() refuses that comparison rather than
    reporting a number nobody can interpret.
    """

    primary_model: str
    secondary_model: str
    base_url: str = "https://api.fireworks.ai/inference/v1"
    temperature: float = 0.0
    seed: int | None = 20260903
    max_tokens: int = 4096
    max_steps: int = 16
    replicates: int = 5
    # Pre-registered, not a knob. It sits inside the fingerprint, so a
    # threshold changed after seeing results produces a config that no
    # already-stored trajectory matches, and classify() refuses the run
    # rather than quietly reporting a tuned number.
    reliability_threshold: float = 0.8
    short_slugs: bool | None = None
    arms: list[str] = Field(default_factory=list)

    @property
    def router_model(self) -> str:
        """The model string for a custom router pair.

        Fireworks documents the custom-pair form as `firerouter/<modelA>/<modelB>`
        with up to 8 models, while full model IDs look like
        `accounts/fireworks/models/<slug>`. Eight full paths in one model string
        would be unwieldy, so the short slug is used here and the account prefix
        is stripped if a caller passes a fully-qualified ID.

        Which form the API accepts could not be confirmed from documentation,
        so it is not guessed: `short_slugs` starts as None and `downgrade
        doctor` determines it empirically by sending one probe request in each
        form and keeping whichever the API accepts. Reading `router_model`
        before that probe has run raises, rather than silently defaulting to a
        form that may produce a 404 on the first real call of a paid sweep.
        """
        if self.short_slugs is None:
            raise RouterFormatUnknownError(
                "Router wire format has not been probed. Run `downgrade doctor` "
                "to determine whether firerouter/<pair> takes short slugs or "
                "fully-qualified model IDs, then pass the result as short_slugs."
            )
        if not self.short_slugs:
            return f"firerouter/{self.primary_model}/{self.secondary_model}"
        return f"firerouter/{short_slug(self.primary_model)}/{short_slug(self.secondary_model)}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(exclude={"arms", "fingerprint"})
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "DETECTOR_FOR_KIND",
    "LOUD_KINDS",
    "SCHEMA_VERSION",
    "SEVERITY_RANK",
    "ArmComparison",
    "BaselineProfile",
    "ClassifyMode",
    "ConstraintResult",
    "Detector",
    "Evidence",
    "Finding",
    "FindingKind",
    "FindingRate",
    "NoiseFloor",
    "RouteObservation",
    "RouterFormatUnknownError",
    "RouteSource",
    "RunStatus",
    "short_slug",
    "Step",
    "SweepConfig",
    "ToolCall",
    "ToolResult",
    "Trajectory",
    "Usage",
    "Verdict",
]
