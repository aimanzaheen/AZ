#!/usr/bin/env python3
"""Ingest locally supplied paper files (PDF/TXT/MD) and cache their text for
reextract.py/verify.py, matching each file to its row in otto_output.csv.

Matching, in order of confidence:
  1. filename (minus extension) == Reference ID, e.g. "AIM-50.pdf" -> "AIM-50"
  2. a DOI found in the file's text matches a row's DOI
  3. fuzzy title match against the first ~4000 characters of the file's text

Writes the same data/cache/<paper_id>.json schema fetch_papers.py produces
(text_source="local_pdf" or "local_txt"), so reextract.py and verify.py work
unchanged regardless of whether text came from NCBI or a local folder.
Also writes data/scrape_manifest.csv and reports any otto_output.csv rows
that had no matching file, and any files that couldn't be matched at all.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

import common
import matching
import pdf_text


def run(args: argparse.Namespace) -> int:
    papers_dir = Path(args.papers_dir)
    if not papers_dir.is_dir():
        print(f"error: {papers_dir} is not a directory", file=sys.stderr)
        return 1

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    otto_rows = common.load_otto_rows(args.otto_csv)
    files = sorted(
        p
        for p in papers_dir.rglob("*")
        if p.suffix.lower() in pdf_text.SUPPORTED_SUFFIXES and p.stem.lower() != "readme"
    )
    if not files:
        print(f"error: no {sorted(pdf_text.SUPPORTED_SUFFIXES)} files found under {papers_dir}", file=sys.stderr)
        return 1

    claimed: dict[str, dict] = {}  # paper_id -> manifest row (best match so far)
    manifest_rows: list[dict] = []

    for path in files:
        text = pdf_text.load_text(path)
        if not text or not text.strip():
            manifest_rows.append(
                {
                    "file": str(path),
                    "paper_id": "",
                    "match_method": "extraction_failed",
                    "confidence": 0.0,
                    "status": "failed",
                }
            )
            print(f"{path.name}: FAILED (could not extract text)")
            continue

        result = matching.match_paper(path.stem, text, otto_rows)
        row = {
            "file": str(path),
            "paper_id": result.row.paper_id if result.row else "",
            "match_method": result.method,
            "confidence": result.confidence,
            "status": "matched" if result.row else "unmatched",
        }

        if result.row is None:
            manifest_rows.append(row)
            print(f"{path.name}: UNMATCHED (no filename/DOI/title match above threshold)")
            continue

        paper_id = result.row.paper_id
        existing = claimed.get(paper_id)
        if existing is not None and existing["confidence"] >= result.confidence:
            row["status"] = "duplicate_match"
            manifest_rows.append(row)
            print(
                f"{path.name}: DUPLICATE match for {paper_id} "
                f"(already claimed by {existing['file']} with >= confidence) - skipped"
            )
            continue

        if existing is not None:
            existing["status"] = "superseded"
            print(f"{existing['file']}: superseded by a higher-confidence match for {paper_id}")

        claimed[paper_id] = row
        manifest_rows.append(row)

        cache_entry = {
            "paper_id": paper_id,
            "doi": result.row.doi,
            "pmid": None,
            "pmcid": None,
            "text_source": "local_pdf" if path.suffix.lower() == ".pdf" else "local_txt",
            "status": "ok",
            "text": text,
            "error": None,
            "source_file": str(path),
            "match_method": result.method,
            "match_confidence": result.confidence,
            "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        common.write_json(cache_dir / f"{paper_id}.json", cache_entry)
        print(f"{path.name}: matched {paper_id} (via {result.method}, confidence={result.confidence})")

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        fields = ["file", "paper_id", "match_method", "confidence", "status"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    matched_ids = {r["paper_id"] for r in manifest_rows if r["status"] == "matched"}
    missing = [row for row in otto_rows if row.paper_id not in matched_ids]

    print(f"\nWrote manifest to {manifest_path}")
    print(f"Matched {len(matched_ids)}/{len(otto_rows)} otto_output.csv rows to a file.")
    if missing:
        print(f"Rows with no matching file ({len(missing)}):")
        for row in missing:
            print(f"  - {row.paper_id}: {row.title}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", default=str(common.PKG_DIR / "papers"))
    parser.add_argument("--otto-csv", default=str(common.DEFAULT_OTTO_CSV))
    parser.add_argument("--cache-dir", default=str(common.DEFAULT_CACHE_DIR))
    parser.add_argument("--manifest-out", default=str(common.PKG_DIR / "data" / "scrape_manifest.csv"))
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
