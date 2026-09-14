#!/usr/bin/env python3
"""Fetch full text (or abstract, as a fallback) for every paper in search_results.csv.

Looks each PMID up on PMC via the ID Converter, fetches PMC full text if the article is
open-access there, otherwise falls back to the PubMed abstract. Caches the raw result to
data/cache/<pmid>.json so re-runs are free, and writes data/fetch_manifest.csv summarizing
what was found for each paper.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

import requests

import common
import ncbi

USER_AGENT = "lh-injection-harvester (mailto:research-tools@local)"


def fetch_one(
    paper: common.SearchResult,
    session: requests.Session,
    limiter: common.RateLimiter,
    api_key: str | None,
) -> dict:
    result = {
        "pmid": paper.pmid,
        "pmcid": None,
        "text_source": None,
        "status": None,
        "text": None,
        "error": None,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    try:
        limiter.wait()
        pmcid = ncbi.pmid_to_pmcid(paper.pmid, session, api_key)
        result["pmcid"] = pmcid

        text = None
        if pmcid:
            limiter.wait()
            text = ncbi.fetch_pmc_fulltext(pmcid, session, api_key)
            if text and len(text.split()) >= 200:
                result["text_source"] = "pmc_fulltext"
            else:
                text = None  # too short to trust / not actually OA full text

        if text is None:
            limiter.wait()
            text = ncbi.fetch_pubmed_abstract(paper.pmid, session, api_key)
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

    papers = common.load_search_results(args.search_results_csv)
    if args.pmid:
        papers = [p for p in papers if p.pmid == args.pmid]
    if args.limit:
        papers = papers[: args.limit]

    if not papers:
        print(f"No papers found in {args.search_results_csv} - run search_papers.py first.", file=sys.stderr)
        return 1

    api_key = os.environ.get("NCBI_API_KEY") or None
    interval = 1.0 / 10 if api_key else 1.0 / 3
    limiter = common.RateLimiter(interval)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    manifest_rows = []
    for i, paper in enumerate(papers, 1):
        cache_path = cache_dir / f"{paper.pmid}.json"
        cached = common.read_json(cache_path)
        if cached and not args.force and cached.get("status") != "failed":
            print(f"[{i}/{len(papers)}] {paper.pmid}: cached ({cached.get('status')})")
            manifest_rows.append(cached)
            continue

        result = fetch_one(paper, session, limiter, api_key)
        common.write_json(cache_path, result)
        print(
            f"[{i}/{len(papers)}] {paper.pmid}: {result['status']}"
            + (f" ({result['text_source']})" if result["text_source"] else "")
            + (f" - {result['error']}" if result["error"] else "")
        )
        manifest_rows.append(result)

    common.write_csv(args.manifest_out, common.FETCH_MANIFEST_FIELDS, manifest_rows)

    counts: dict[str, int] = {}
    for row in manifest_rows:
        key = row.get("text_source") or row.get("status")
        counts[key] = counts.get(key, 0) + 1
    print(f"\nWrote manifest to {args.manifest_out}")
    print("Summary:", counts)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-results-csv", default=str(common.DEFAULT_SEARCH_RESULTS_CSV))
    parser.add_argument("--cache-dir", default=str(common.DEFAULT_CACHE_DIR))
    parser.add_argument("--manifest-out", default=str(common.DEFAULT_FETCH_MANIFEST_CSV))
    parser.add_argument("--force", action="store_true", help="Refetch even if a cache entry exists.")
    parser.add_argument("--limit", type=int, help="Only process the first N papers (for testing).")
    parser.add_argument("--pmid", help="Only process this single PMID (for testing).")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
