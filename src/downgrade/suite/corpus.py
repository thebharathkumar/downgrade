"""Corpus manifest, integrity checking, and 10-K section extraction.

The suite runs against a corpus of real SEC EDGAR filings committed into the
repo. Everything in this module is pure: no network, no clock, no filesystem
beyond reading files it is handed. The network shell that actually talks to
EDGAR lives in `scripts/fetch_corpus.py`, is run once on a machine that can
reach sec.gov, and produces exactly the manifest this module verifies.

That split is deliberate. Corpus acquisition happens once and cannot be
tested in CI; corpus integrity is checked on every run and must be. Keeping
the parsing and hashing here means the part that can silently corrupt an
experiment is the part that is covered by tests.

EDGAR filings are US government works placed in the public domain, so the
corpus is redistributable with attribution recorded per file in the manifest.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MANIFEST_SCHEMA_VERSION = "1"

FileKind = Literal["document", "table"]

# 10-K item headings we split on. Ordered as they appear in a filing so the
# span between one heading and the next is the section body.
ITEM_PATTERNS: tuple[tuple[str, str], ...] = (
    ("item_1", r"item\s*1\s*[.\-–—:]?\s*business"),
    ("item_1a", r"item\s*1a\s*[.\-–—:]?\s*risk\s*factors"),
    ("item_1b", r"item\s*1b\s*[.\-–—:]?\s*unresolved\s*staff\s*comments"),
    ("item_2", r"item\s*2\s*[.\-–—:]?\s*properties"),
    ("item_3", r"item\s*3\s*[.\-–—:]?\s*legal\s*proceedings"),
    ("item_5", r"item\s*5\s*[.\-–—:]?\s*market\s*for"),
    ("item_7", r"item\s*7\s*[.\-–—:]?\s*management['’]?s\s*discussion"),
    ("item_7a", r"item\s*7a\s*[.\-–—:]?\s*quantitative\s*and\s*qualitative"),
    ("item_8", r"item\s*8\s*[.\-–—:]?\s*financial\s*statements"),
    ("item_9a", r"item\s*9a\s*[.\-–—:]?\s*controls\s*and\s*procedures"),
)


class CorpusFile(BaseModel):
    """One committed corpus file, with everything needed to re-fetch it."""

    path: str
    kind: FileKind
    sha256: str
    bytes: int
    source_url: str
    retrieved_at: str
    title: str
    license: str = "public domain (17 U.S.C. 105, SEC EDGAR)"
    cik: str | None = None
    company: str | None = None
    accession: str | None = None
    form: str | None = None
    fiscal_year: int | None = None
    section: str | None = None
    columns: list[str] = Field(default_factory=list)
    row_count: int | None = None


class CorpusManifest(BaseModel):
    schema_version: str = MANIFEST_SCHEMA_VERSION
    source: str = "SEC EDGAR"
    generated_at: str
    files: list[CorpusFile] = Field(default_factory=list)

    def by_path(self) -> dict[str, CorpusFile]:
        return {f.path: f for f in self.files}

    @property
    def documents(self) -> list[CorpusFile]:
        return [f for f in self.files if f.kind == "document"]

    @property
    def tables(self) -> list[CorpusFile]:
        return [f for f in self.files if f.kind == "table"]


class VerifyReport(BaseModel):
    """Result of checking a corpus directory against its manifest.

    `ok` is the gate the runner uses. A corrupted or partially-committed
    corpus must fail loudly before a sweep starts, not produce quietly
    different tool results between the baseline arm and a downgraded arm run
    on a different day.
    """

    missing: list[str] = Field(default_factory=list)
    corrupted: list[str] = Field(default_factory=list)
    untracked: list[str] = Field(default_factory=list)
    verified: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.corrupted

    def summary(self) -> str:
        parts = [f"{len(self.verified)} verified"]
        if self.missing:
            parts.append(f"{len(self.missing)} missing")
        if self.corrupted:
            parts.append(f"{len(self.corrupted)} corrupted")
        if self.untracked:
            parts.append(f"{len(self.untracked)} untracked")
        return ", ".join(parts)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_corpus(root: Path, manifest: CorpusManifest) -> VerifyReport:
    """Check every manifest entry against the files actually on disk."""
    report = VerifyReport()
    tracked: set[Path] = set()

    for entry in manifest.files:
        target = root / entry.path
        tracked.add(target)
        if not target.is_file():
            report.missing.append(entry.path)
            continue
        if sha256_bytes(target.read_bytes()) != entry.sha256:
            report.corrupted.append(entry.path)
        else:
            report.verified.append(entry.path)

    for found in sorted(root.rglob("*")):
        if found.is_file() and found.name != "manifest.json" and found not in tracked:
            report.untracked.append(str(found.relative_to(root)))

    return report


def html_to_text(html: str) -> str:
    """Flatten filing HTML to plain text.

    Deliberately not a real HTML parser. EDGAR filings are enormous and
    inconsistently marked up; all the suite needs is readable prose with
    stable whitespace so that retrieval and citation offsets are reproducible.
    """
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|tr|h[1-6]|li)>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#8217;", "’")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
    )
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def split_10k_sections(text: str) -> dict[str, str]:
    """Split a flattened 10-K into item sections.

    The hard part is the table of contents: every item heading appears there
    too, usually within a few characters of the next one. Rather than guessing
    which occurrence is real, every occurrence of every heading is collected,
    the boundaries are sorted, and for each item the LONGEST span between that
    heading and the next heading of any kind is kept. Table-of-contents entries
    lose because they are adjacent to each other; the real section wins because
    it has a body.

    Returns only sections with a plausible amount of text, so a filing whose
    structure does not match returns fewer keys rather than garbage.
    """
    lowered = text.lower()
    hits: list[tuple[int, str]] = []
    for name, pattern in ITEM_PATTERNS:
        for match in re.finditer(pattern, lowered):
            hits.append((match.start(), name))

    if not hits:
        return {}

    hits.sort()
    boundaries = [pos for pos, _ in hits]

    best: dict[str, tuple[int, int]] = {}
    for index, (start, name) in enumerate(hits):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else len(text)
        span = end - start
        if name not in best or span > (best[name][1] - best[name][0]):
            best[name] = (start, end)

    sections: dict[str, str] = {}
    for name, (start, end) in best.items():
        body = text[start:end].strip()
        # A real section has a body; a surviving table-of-contents line does not.
        if len(body) >= 400:
            sections[name] = body
    return sections


__all__ = [
    "ITEM_PATTERNS",
    "MANIFEST_SCHEMA_VERSION",
    "CorpusFile",
    "CorpusManifest",
    "FileKind",
    "VerifyReport",
    "html_to_text",
    "sha256_bytes",
    "split_10k_sections",
    "verify_corpus",
]
