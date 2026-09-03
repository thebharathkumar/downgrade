#!/usr/bin/env python3
"""Fetch the SEC EDGAR corpus and write a hashed manifest.

Run this ONCE, on a machine that can reach sec.gov, then commit what it
produces. CI never runs it: the corpus is committed, and `downgrade suite
verify` checks it against the manifest on every sweep.

    python scripts/fetch_corpus.py --user-agent "Bharath kumar R you@example.com"

EDGAR requires a User-Agent identifying the requester and asks for no more
than 10 requests per second; both are honoured here. Filings are US
government works in the public domain, so the corpus is redistributable.

Everything this script does beyond HTTP lives in downgrade.suite.corpus,
which is pure and covered by tests. Keep it that way: if you need to change
how sections are cut or how the manifest is shaped, change it there.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from downgrade.suite.corpus import (  # noqa: E402
    CorpusFile,
    CorpusManifest,
    html_to_text,
    sha256_bytes,
    split_10k_sections,
)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"

# A spread of leverage profiles and sectors, so tasks that reconcile prose
# against reported figures have something to actually differ about.
DEFAULT_TICKERS = (
    "AAPL",
    "MSFT",
    "KO",
    "PG",  # low leverage, stable
    "T",
    "VZ",  # levered telecom
    "F",
    "GM",  # auto with captive finance arms
    "JPM",
    "GS",  # banks
    "DAL",
    "CCL",  # cyclical, high leverage
)

# Sections worth keeping. Risk factors and MD&A carry the prose claims that
# a cross-source task reconciles against the XBRL numbers.
KEEP_SECTIONS = ("item_1a", "item_7", "item_7a")

# Credit-relevant XBRL concepts, flattened into one table.
CONCEPTS = (
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "LongTermDebtNoncurrent",
    "LongTermDebtCurrent",
    "CashAndCashEquivalentsAtCarryingValue",
    "InterestExpense",
    "NetCashProvidedByUsedInOperatingActivities",
)

MIN_SECTION_BYTES = 2_000
MAX_SECTION_BYTES = 60_000


class Edgar:
    """Rate-limited EDGAR client. SEC asks for <= 10 requests/second."""

    def __init__(self, user_agent: str, delay: float = 0.15) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=60.0,
            follow_redirects=True,
        )
        self._delay = delay
        self._last = 0.0

    def get(self, url: str) -> httpx.Response:
        gap = time.monotonic() - self._last
        if gap < self._delay:
            time.sleep(self._delay - gap)
        response = self._client.get(url)
        self._last = time.monotonic()
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._client.close()


def resolve_ciks(edgar: Edgar, tickers: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """Map ticker -> (zero-padded CIK, company name), via SEC's own index.

    Resolved rather than hardcoded: a wrong CIK would silently build the
    corpus from the wrong issuer.
    """
    raw: dict[str, Any] = edgar.get(TICKER_MAP_URL).json()
    index = {row["ticker"].upper(): row for row in raw.values()}
    out: dict[str, tuple[str, str]] = {}
    for ticker in tickers:
        row = index.get(ticker.upper())
        if row is None:
            print(f"  ! {ticker}: not found in SEC ticker index, skipping")
            continue
        out[ticker.upper()] = (str(row["cik_str"]).zfill(10), str(row["title"]))
    return out


def latest_10k(edgar: Edgar, cik: str) -> tuple[str, str, str] | None:
    """Return (accession-no-dashes, primary document, filing date) or None."""
    recent = edgar.get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
    for form, accession, doc, date in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["filingDate"],
        strict=False,
    ):
        if form == "10-K":
            return accession.replace("-", ""), doc, date
    return None


def fetch_sections(
    edgar: Edgar, ticker: str, cik: str, company: str, out_dir: Path
) -> list[CorpusFile]:
    filing = latest_10k(edgar, cik)
    if filing is None:
        print(f"  ! {ticker}: no 10-K found")
        return []
    accession, doc, filed = filing
    url = ARCHIVE_URL.format(cik_int=int(cik), accession=accession, doc=doc)
    text = html_to_text(edgar.get(url).text)
    sections = split_10k_sections(text)

    entries: list[CorpusFile] = []
    for name in KEEP_SECTIONS:
        body = sections.get(name)
        if body is None or len(body) < MIN_SECTION_BYTES:
            print(f"  - {ticker}/{name}: absent or too short, skipping")
            continue
        body = body[:MAX_SECTION_BYTES]
        rel = f"documents/{ticker}_{filed[:4]}_10K_{name}.txt"
        payload = body.encode("utf-8")
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / rel).write_bytes(payload)
        entries.append(
            CorpusFile(
                path=rel,
                kind="document",
                sha256=sha256_bytes(payload),
                bytes=len(payload),
                source_url=url,
                retrieved_at=_now(),
                title=f"{company} FY{filed[:4]} 10-K, {name.replace('_', ' ').title()}",
                cik=cik,
                company=company,
                accession=accession,
                form="10-K",
                fiscal_year=int(filed[:4]),
                section=name,
            )
        )
        print(f"  + {rel} ({len(payload):,} bytes)")
    return entries


def fetch_facts(edgar: Edgar, resolved: dict[str, tuple[str, str]], out_dir: Path) -> CorpusFile:
    """Flatten selected XBRL concepts for every issuer into one CSV."""
    columns = [
        "ticker",
        "cik",
        "company",
        "concept",
        "unit",
        "fy",
        "fp",
        "period_start",
        "period_end",
        "value",
        "accession",
        "form",
    ]
    rows: list[dict[str, Any]] = []
    for ticker, (cik, company) in resolved.items():
        facts = edgar.get(COMPANYFACTS_URL.format(cik=cik)).json().get("facts", {})
        us_gaap = facts.get("us-gaap", {})
        for concept in CONCEPTS:
            entry = us_gaap.get(concept)
            if entry is None:
                continue
            for unit, observations in entry.get("units", {}).items():
                for obs in observations:
                    if obs.get("form") not in {"10-K", "10-Q"}:
                        continue
                    rows.append(
                        {
                            "ticker": ticker,
                            "cik": cik,
                            "company": company,
                            "concept": concept,
                            "unit": unit,
                            "fy": obs.get("fy"),
                            "fp": obs.get("fp"),
                            "period_start": obs.get("start", ""),
                            "period_end": obs.get("end", ""),
                            "value": obs.get("val"),
                            "accession": obs.get("accn", ""),
                            "form": obs.get("form", ""),
                        }
                    )
        print(f"  + {ticker}: {company}")

    rows.sort(key=lambda r: (r["ticker"], r["concept"], str(r["period_end"])))
    rel = "tables/xbrl_facts.csv"
    target = out_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    payload = target.read_bytes()
    print(f"  + {rel} ({len(rows):,} rows, {len(payload):,} bytes)")
    return CorpusFile(
        path=rel,
        kind="table",
        sha256=sha256_bytes(payload),
        bytes=len(payload),
        source_url="https://data.sec.gov/api/xbrl/companyfacts/",
        retrieved_at=_now(),
        title="Selected us-gaap concepts from XBRL company facts",
        columns=columns,
        row_count=len(rows),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-agent",
        required=True,
        help='Required by SEC, e.g. "Your Name you@example.com"',
    )
    parser.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    parser.add_argument("--out", default="src/downgrade/suite/corpus", type=Path)
    args = parser.parse_args()

    if "@" not in args.user_agent:
        parser.error("SEC requires a contact email in the User-Agent")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    edgar = Edgar(args.user_agent)
    try:
        print("Resolving CIKs...")
        resolved = resolve_ciks(edgar, tuple(args.tickers))
        print(f"\nFetching 10-K sections for {len(resolved)} issuers...")
        entries: list[CorpusFile] = []
        for ticker, (cik, company) in resolved.items():
            entries.extend(fetch_sections(edgar, ticker, cik, company, out_dir))
        print("\nFetching XBRL company facts...")
        entries.append(fetch_facts(edgar, resolved, out_dir))
    finally:
        edgar.close()

    manifest = CorpusManifest(generated_at=_now(), files=sorted(entries, key=lambda e: e.path))
    (out_dir / "manifest.json").write_text(
        json.dumps(json.loads(manifest.model_dump_json()), indent=2) + "\n", encoding="utf-8"
    )
    total = sum(e.bytes for e in entries)
    print(f"\nWrote {len(entries)} files ({total:,} bytes) and manifest.json to {out_dir}")
    print("Next: python -m downgrade.cli suite verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
