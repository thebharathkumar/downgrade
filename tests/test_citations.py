"""No unverified external figure may appear in anything this repo publishes.

docs/citations.yaml registers every figure quoted from outside the repo with
a `verified` flag. This test fails if the value of an unverified entry shows
up in the README or in the source of the report renderer and CLI, which is
where published text comes from.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = [
    ROOT / "README.md",
    ROOT / "src" / "downgrade" / "report" / "markdown.py",
    ROOT / "src" / "downgrade" / "cli.py",
]


def load_citations() -> list[dict[str, object]]:
    data = yaml.safe_load((ROOT / "docs" / "citations.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


def test_every_citation_is_well_formed() -> None:
    for entry in load_citations():
        assert {"id", "source", "claim", "value", "verified"} <= set(entry)


def test_unverified_figures_are_not_published() -> None:
    unverified = [str(e["value"]) for e in load_citations() if not e["verified"]]
    for path in PUBLISHED:
        text = path.read_text(encoding="utf-8")
        leaked = [value for value in unverified if value in text]
        assert not leaked, f"{path.name} quotes unverified figure(s) {leaked}"
