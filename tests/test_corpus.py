"""Tests for corpus integrity and 10-K section extraction.

The section splitter is the piece worth testing hardest. Every 10-K contains
each item heading at least twice, once in the table of contents and once as
the real section, and picking the wrong one silently yields a corpus of
one-line fragments that every task then fails on for reasons unrelated to
routing.
"""

from __future__ import annotations

import json
from pathlib import Path

from downgrade.suite.corpus import (
    CorpusFile,
    CorpusManifest,
    html_to_text,
    sha256_bytes,
    split_10k_sections,
    verify_corpus,
)


def make_entry(path: str, data: bytes, **kw: object) -> CorpusFile:
    defaults: dict[str, object] = {
        "path": path,
        "kind": "document",
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "source_url": f"https://www.sec.gov/Archives/{path}",
        "retrieved_at": "2026-09-03T00:00:00Z",
        "title": path,
    }
    defaults.update(kw)
    return CorpusFile.model_validate(defaults)


class TestVerifyCorpus:
    def test_clean_corpus_verifies(self, tmp_path: Path) -> None:
        data = b"risk factors body"
        (tmp_path / "a.txt").write_bytes(data)
        manifest = CorpusManifest(generated_at="t", files=[make_entry("a.txt", data)])
        report = verify_corpus(tmp_path, manifest)
        assert report.ok
        assert report.verified == ["a.txt"]

    def test_missing_file_is_reported_and_fails(self, tmp_path: Path) -> None:
        manifest = CorpusManifest(generated_at="t", files=[make_entry("gone.txt", b"x")])
        report = verify_corpus(tmp_path, manifest)
        assert not report.ok
        assert report.missing == ["gone.txt"]

    def test_modified_file_is_detected(self, tmp_path: Path) -> None:
        """A corpus that changed under the experiment invalidates every arm
        that ran against the old bytes."""
        entry = make_entry("a.txt", b"original")
        (tmp_path / "a.txt").write_bytes(b"tampered")
        report = verify_corpus(tmp_path, CorpusManifest(generated_at="t", files=[entry]))
        assert not report.ok
        assert report.corrupted == ["a.txt"]

    def test_untracked_file_is_noted_but_does_not_fail(self, tmp_path: Path) -> None:
        data = b"body"
        (tmp_path / "a.txt").write_bytes(data)
        (tmp_path / "stray.txt").write_bytes(b"not in manifest")
        report = verify_corpus(
            tmp_path, CorpusManifest(generated_at="t", files=[make_entry("a.txt", data)])
        )
        assert report.ok
        assert report.untracked == ["stray.txt"]

    def test_manifest_json_itself_is_not_untracked(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text("{}")
        report = verify_corpus(tmp_path, CorpusManifest(generated_at="t"))
        assert report.untracked == []

    def test_nested_files_are_checked(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        data = b"nested body"
        (tmp_path / "docs" / "a.txt").write_bytes(data)
        manifest = CorpusManifest(generated_at="t", files=[make_entry("docs/a.txt", data)])
        assert verify_corpus(tmp_path, manifest).ok

    def test_summary_mentions_each_category(self, tmp_path: Path) -> None:
        (tmp_path / "stray.txt").write_bytes(b"x")
        manifest = CorpusManifest(generated_at="t", files=[make_entry("gone.txt", b"y")])
        summary = verify_corpus(tmp_path, manifest).summary()
        assert "missing" in summary and "untracked" in summary


class TestManifest:
    def test_partitions_documents_and_tables(self) -> None:
        manifest = CorpusManifest(
            generated_at="t",
            files=[
                make_entry("a.txt", b"x"),
                make_entry("t.csv", b"y", kind="table", columns=["cik", "value"], row_count=3),
            ],
        )
        assert [f.path for f in manifest.documents] == ["a.txt"]
        assert [f.path for f in manifest.tables] == ["t.csv"]
        assert manifest.by_path()["t.csv"].row_count == 3

    def test_round_trips_through_json(self) -> None:
        manifest = CorpusManifest(generated_at="t", files=[make_entry("a.txt", b"x")])
        assert CorpusManifest.model_validate(json.loads(manifest.model_dump_json())) == manifest

    def test_license_defaults_to_public_domain(self) -> None:
        assert "public domain" in make_entry("a.txt", b"x").license


class TestHtmlToText:
    def test_strips_tags_and_entities(self) -> None:
        assert html_to_text("<p>Total&nbsp;assets &amp; debt</p>") == "Total assets & debt"

    def test_drops_script_and_style_bodies(self) -> None:
        out = html_to_text("<style>p{color:red}</style><script>var x=1</script><p>Body</p>")
        assert "color" not in out and "var x" not in out
        assert "Body" in out

    def test_block_tags_become_newlines(self) -> None:
        assert html_to_text("<div>One</div><div>Two</div>").splitlines() == ["One", "Two"]

    def test_collapses_runs_of_blank_lines(self) -> None:
        assert "\n\n\n" not in html_to_text("<p>A</p><p></p><p></p><p></p><p>B</p>")


def build_filing(*, toc: bool = True) -> str:
    """A miniature 10-K: a table of contents, then the real sections."""
    parts = []
    if toc:
        parts.append(
            "TABLE OF CONTENTS\n"
            "Item 1. Business 3\n"
            "Item 1A. Risk Factors 12\n"
            "Item 7. Management's Discussion and Analysis 40\n"
        )
    parts.append("Item 1. Business\n" + ("We operate in one segment. " * 40))
    parts.append("Item 1A. Risk Factors\n" + ("Our leverage may increase. " * 40))
    parts.append("Item 7. Management's Discussion and Analysis\n" + ("Revenue grew. " * 40))
    return "\n\n".join(parts)


class TestSplit10kSections:
    def test_extracts_each_item(self) -> None:
        sections = split_10k_sections(build_filing())
        assert set(sections) == {"item_1", "item_1a", "item_7"}

    def test_prefers_the_real_section_over_the_table_of_contents(self) -> None:
        """The TOC line and the real heading both match; the body must win."""
        sections = split_10k_sections(build_filing())
        assert "Our leverage may increase." in sections["item_1a"]
        assert "TABLE OF CONTENTS" not in sections["item_1a"]

    def test_works_without_a_table_of_contents(self) -> None:
        assert set(split_10k_sections(build_filing(toc=False))) == {"item_1", "item_1a", "item_7"}

    def test_sections_do_not_bleed_into_each_other(self) -> None:
        sections = split_10k_sections(build_filing())
        assert "Revenue grew" not in sections["item_1a"]

    def test_returns_empty_for_a_document_with_no_items(self) -> None:
        assert split_10k_sections("A press release with no item headings.") == {}

    def test_drops_headings_with_no_body(self) -> None:
        """A filing that is only a table of contents yields nothing, rather
        than a corpus of one-line fragments."""
        toc = "Item 1. Business 3\nItem 1A. Risk Factors 12\nItem 7. Management's Discussion 40\n"
        assert split_10k_sections(toc) == {}

    def test_matches_curly_apostrophe_in_managements_discussion(self) -> None:
        text = "Item 7. Management’s Discussion and Analysis\n" + ("Revenue grew. " * 40)
        assert "item_7" in split_10k_sections(text)

    def test_last_section_runs_to_end_of_document(self) -> None:
        sections = split_10k_sections(build_filing())
        assert sections["item_7"].rstrip().endswith("Revenue grew.")
