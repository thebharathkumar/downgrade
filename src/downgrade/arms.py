"""Experimental arms: the six configurations a task is run under.

  baseline_direct   the primary model, called directly, no router in the path
  pref_1 .. pref_5  the router pair, at each routing preference

Two baselines exist on purpose. `baseline_direct` is the control the
BaselineProfile is built from, because it is the only arm with no router in
the path at all. `pref_1` is then scored as an ordinary downgraded arm, which
is what makes "does the router already downgrade at maximum intelligence?"
a question the sweep answers rather than assumes. A design that used pref_1
as the reference would measure every other arm against a possibly-already
contaminated baseline and understate the regression.

Preference semantics are Fireworks': 1 is most quality-protective, 5 is most
savings-focused, and a missing or out-of-range value falls back to 3.
"""

from __future__ import annotations

from dataclasses import dataclass

from downgrade.models import SweepConfig

ROUTING_HEADER = "x-routing-preference"

# Fireworks' own labels, kept verbatim so report output matches their docs.
PREFERENCE_LABELS: dict[int, str] = {
    1: "max-intelligence",
    2: "more-intelligence",
    3: "balanced",
    4: "more-savings",
    5: "max-savings",
}

MIN_PREFERENCE = 1
MAX_PREFERENCE = 5


@dataclass(frozen=True)
class Arm:
    """One row of the sweep matrix."""

    name: str
    preference: int | None
    use_router: bool

    @property
    def label(self) -> str:
        if self.preference is None:
            return "direct to primary, no router"
        return PREFERENCE_LABELS.get(self.preference, "unknown")

    @property
    def is_baseline(self) -> bool:
        return not self.use_router

    def model_for(self, config: SweepConfig) -> str:
        """Which model string to send.

        The direct arm names the primary model outright. Router arms use the
        pair string, which raises until the wire format has been probed.
        """
        return config.router_model if self.use_router else config.primary_model

    def request_headers(self) -> dict[str, str]:
        """Routing header, or nothing for the direct arm.

        The direct arm deliberately omits the header rather than sending a
        value: it is not routed at all, and sending a preference to a
        non-router request would misrepresent what the control arm is.
        """
        if self.preference is None:
            return {}
        return {ROUTING_HEADER: str(self.preference)}


BASELINE_DIRECT = Arm(name="baseline_direct", preference=None, use_router=False)

ROUTER_ARMS: tuple[Arm, ...] = tuple(
    Arm(name=f"pref_{p}", preference=p, use_router=True)
    for p in range(MIN_PREFERENCE, MAX_PREFERENCE + 1)
)

ALL_ARMS: tuple[Arm, ...] = (BASELINE_DIRECT, *ROUTER_ARMS)

ARMS_BY_NAME: dict[str, Arm] = {arm.name: arm for arm in ALL_ARMS}

DEFAULT_BASELINE_ARM = BASELINE_DIRECT.name


def arm_by_name(name: str) -> Arm:
    try:
        return ARMS_BY_NAME[name]
    except KeyError:
        known = ", ".join(sorted(ARMS_BY_NAME))
        raise KeyError(f"Unknown arm '{name}'. Known arms: {known}") from None


def resolve_arms(names: list[str] | None) -> list[Arm]:
    """Resolve arm names, defaulting to the full matrix.

    The baseline is always included even if a caller names only router arms,
    because without it there is no profile to score against and no noise floor
    to subtract.
    """
    if not names:
        return list(ALL_ARMS)
    resolved = [arm_by_name(n) for n in names]
    if all(arm.use_router for arm in resolved):
        resolved.insert(0, BASELINE_DIRECT)
    return resolved


__all__ = [
    "ALL_ARMS",
    "ARMS_BY_NAME",
    "BASELINE_DIRECT",
    "DEFAULT_BASELINE_ARM",
    "MAX_PREFERENCE",
    "MIN_PREFERENCE",
    "PREFERENCE_LABELS",
    "ROUTER_ARMS",
    "ROUTING_HEADER",
    "Arm",
    "arm_by_name",
    "resolve_arms",
]
