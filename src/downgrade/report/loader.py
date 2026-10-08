"""Loading stored runs for analysis.

Two on-disk shapes are accepted, both written by this package:

  smoke output   `downgrade smoke --out`: {"config", "arms": [{"trajectories"}]}
  run bundle     {"config", "trajectories": [...]}, what `save_bundle` writes

The config travels with the trajectories on purpose. Its fingerprint is
recomputed on load and every trajectory is checked against it, so a bundle
whose config was edited after the runs (a threshold moved, an FDR level
loosened) fails here rather than producing a quietly re-tuned report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from downgrade.classify.core import check_fingerprints
from downgrade.models import SweepConfig, Trajectory


@dataclass
class RunBundle:
    config: SweepConfig
    trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def task_ids(self) -> list[str]:
        return sorted({t.task_id for t in self.trajectories})

    @property
    def arms(self) -> list[str]:
        return sorted({t.arm for t in self.trajectories})


def _from_payload(payload: dict[str, Any]) -> RunBundle:
    if "config" not in payload:
        raise ValueError("run file has no 'config' block")
    config = SweepConfig.model_validate(payload["config"])
    raw: list[Any]
    if "trajectories" in payload:
        raw = list(payload["trajectories"])
    elif "arms" in payload:
        raw = [t for arm in payload["arms"] for t in arm.get("trajectories", [])]
    else:
        raise ValueError("run file has neither 'trajectories' nor 'arms'")
    trajectories = [Trajectory.model_validate(t) for t in raw]
    check_fingerprints(config, trajectories)
    return RunBundle(config=config, trajectories=trajectories)


def load_bundle(path: Path) -> RunBundle:
    """Load one run file, or merge every *.json in a directory.

    Merged files must share a fingerprint: runs produced under different
    settings do not belong in one analysis.
    """
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no run files under {path}")
    bundles = [_from_payload(json.loads(f.read_text(encoding="utf-8"))) for f in files]
    first = bundles[0]
    for other in bundles[1:]:
        if other.config.fingerprint != first.config.fingerprint:
            raise ValueError(
                f"run files disagree on config fingerprint "
                f"({first.config.fingerprint} vs {other.config.fingerprint})"
            )
        first.trajectories.extend(other.trajectories)
    ids = [t.run_id for t in first.trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run ids across run files")
    return first


def save_bundle(bundle: RunBundle, path: Path) -> None:
    payload = {
        "config": json.loads(bundle.config.model_dump_json()),
        "trajectories": [json.loads(t.model_dump_json()) for t in bundle.trajectories],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["RunBundle", "load_bundle", "save_bundle"]
