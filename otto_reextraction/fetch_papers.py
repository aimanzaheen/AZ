#!/usr/bin/env python3
"""Fetch full text (or abstract, as a fallback) for every paper in otto_output.csv.

Looks each paper's DOI up on PMC/PubMed via NCBI E-utilities, caches the raw
result to data/cache/<paper_id>.json so re-runs are free, and writes a
data/fetch_manifest.csv summarizing what was found for each paper.

NCBI usage note: without an NCBI_API_KEY this script self-limits to ~3
requests/sec (NCBI's unauthenticated cap); set NCBI_API_KEY to raise that to
~10/sec. See https://www.ncbi.nlm.nih.gov/books/NBK25497/.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
from pathlib import Path

import requests

import common
import ncbi

USER_AGENT = "otto-reextraction-pipeline (mailto:research-tools@local)"


def fetch_one(
    paper: common.OttoRow,
    session: requests.Session,
    limiter: common.RateLimiter,
    api_key: str | None,
) -> dict:
    result = {
        "paper_id": paper.paper_id,
        "doi": paper.doi,
        "pmid": None,
        "pmcid": None,
        "text_source": None,
        "status": None,
        "text": None,
        "error": None,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if not paper.doi:
        result["status"] = "no_doi"
        return result

    try:
        limiter.wait()
        ids = ncbi.doi_to_ids(paper.doi, session, api_key)
        pmid, pmcid = ids.get("pmid"), ids.get("pmcid")

        if not pmid:
            limiter.wait()
            pmid = ncbi.esearch_pmid_by_doi(paper.doi, session, api_key)

        result["pmid"], result["pmcid"] = pmid, pmcid

        text = None
        if pmcid:
            limiter.wait()
            text = ncbi.fetch_pmc_fulltext(pmcid, session, api_key)
            if text and len(text.split()) >= 200:
                result["text_source"] = "pmc_fulltext"
            else:
                text = None  # too short to trust / not actually OA full text

        if text is None and pmid:
            limiter.wait()
            text = ncbi.fetch_pubmed_abstract(pmid, session, api_key)
            if text:
                result["text_source"] = "pubmed_abstract"

        if text is None:
            result["status"] = "unavailable"
        else:
            result["status"] = "ok"
            result["text"] = text
    except requests.RequestException as exc:
        result["status"] = "failed"
        result["error"] = str(exc)

    return result


def run(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    papers = common.load_otto_rows(args.otto_csv)
    if args.paper_id:
        papers = [p for p in papers if p.paper_id == args.paper_id]
    if args.limit:
        papers = papers[: args.limit]

    api_key = os.environ.get("NCBI_API_KEY") or None
    interval = 1.0 / 10 if api_key else 1.0 / 3
    limiter = common.RateLimiter(interval)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    manifest_rows = []
    for i, paper in enumerate(papers, 1):
        cache_path = cache_dir / f"{paper.paper_id}.json"
        cached = common.read_json(cache_path)
        if cached and not args.force and cached.get("status") != "failed":
            print(f"[{i}/{len(papers)}] {paper.paper_id}: cached ({cached.get('status')})")
            manifest_rows.append(cached)
            continue

        result = fetch_one(paper, session, limiter, api_key)
        common.write_json(cache_path, result)
        print(
            f"[{i}/{len(papers)}] {paper.paper_id}: {result['status']}"
            + (f" ({result['text_source']})" if result["text_source"] else "")
            + (f" - {result['error']}" if result["error"] else "")
        )
        manifest_rows.append(result)

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        fields = ["paper_id", "doi", "pmid", "pmcid", "text_source", "status", "fetched_at", "error"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({k: row.get(k) for k in fields})

    counts: dict[str, int] = {}
    for row in manifest_rows:
        key = row.get("text_source") or row.get("status")
        counts[key] = counts.get(key, 0) + 1
    print(f"\nWrote manifest to {manifest_path}")
    print("Summary:", counts)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--otto-csv", default=str(common.DEFAULT_OTTO_CSV))
    parser.add_argument("--cache-dir", default=str(common.DEFAULT_CACHE_DIR))
    parser.add_argument("--manifest-out", default=str(common.DEFAULT_FETCH_MANIFEST_CSV))
    parser.add_argument("--force", action="store_true", help="Refetch even if a cache entry exists.")
    parser.add_argument("--limit", type=int, help="Only process the first N papers (for testing).")
    parser.add_argument("--paper-id", help="Only process this single Reference ID (for testing).")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
