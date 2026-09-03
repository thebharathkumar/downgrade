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


class Step(BaseModel):
    """One assistant turn, plus the tool result it produced if any."""

    index: int
    thought: str = ""
    tool_call: ToolCall | None = None
    tool_result: Any = None
    tool_error: str | None = None
    route: RouteObservation
    error: str | None = None


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
        return [s.tool_call.signature() for s in self.steps if s.tool_call is not None]

    def tool_names(self) -> list[str]:
        return [s.tool_call.name for s in self.steps if s.tool_call is not None]


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
    tool_signature_rates: dict[str, float] = Field(default_factory=dict)
    tool_name_rates: dict[str, float] = Field(default_factory=dict)
    median_tool_calls: float = 0.0
    call_count_by_signature: dict[str, float] = Field(default_factory=dict)
    constraint_satisfaction_rates: dict[str, float] = Field(default_factory=dict)
    grounded_values: set[str] = Field(default_factory=set)
    cited_documents: set[str] = Field(default_factory=set)
    correct_rate: float = 0.0
    answers: list[str] = Field(default_factory=list)

    def is_reliable(self, rate_map: dict[str, float], key: str, threshold: float) -> bool:
        return rate_map.get(key, 0.0) >= threshold


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
    baseline_correct_rate: float = 0.0
    downgraded_correct_rate: float = 0.0
    route_downgrade_rate: float = 0.0

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
    short_slugs: bool = True
    arms: list[str] = Field(default_factory=list)

    @property
    def router_model(self) -> str:
        """The model string for a custom router pair.

        Fireworks documents the custom-pair form as `firerouter/<modelA>/<modelB>`
        with up to 8 models, while full model IDs look like
        `accounts/fireworks/models/<slug>`. Eight full paths in one model string
        would be unwieldy, so the short slug is used here and the account prefix
        is stripped if a caller passes a fully-qualified ID.

        This is the one wire-format detail that could not be confirmed against
        the Fireworks docs at authoring time (docs.fireworks.ai was unreachable).
        `downgrade doctor` issues a single probe request and reports which form
        the API accepts; if it is the fully-qualified form, set
        `short_slugs=False` and nothing else changes.
        """
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
    "RouteObservation",
    "RouteSource",
    "RunStatus",
    "short_slug",
    "Step",
    "SweepConfig",
    "ToolCall",
    "Trajectory",
    "Usage",
    "Verdict",
]
