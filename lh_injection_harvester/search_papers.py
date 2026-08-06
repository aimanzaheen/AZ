#!/usr/bin/env python3
"""Search PubMed for papers matching the injection query and write data/search_results.csv.

Default query targets retrograde tracer injections into the mouse lateral hypothalamus (LH).
Override with --query to target a different injection type/tracer/region.
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

import common
import ncbi

USER_AGENT = "lh-injection-harvester (mailto:research-tools@local)"


def run(args: argparse.Namespace) -> int:
    api_key = os.environ.get("NCBI_API_KEY") or None
    interval = 1.0 / 10 if api_key else 1.0 / 3
    limiter = common.RateLimiter(interval)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    limiter.wait()
    search = ncbi.esearch(args.query, session, api_key, retmax=args.max_results)
    pmids = search["idlist"]
    print(f"Query matched {search['count']} PubMed records; fetching metadata for {len(pmids)}.")

    rows: list[dict] = []
    batch_size = 50
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        limiter.wait()
        summaries = ncbi.esummary(batch, session, api_key)
        for pmid in batch:
            summary = summaries.get(pmid)
            if not summary:
                continue
            rows.append(summary)

    common.write_csv(args.out, common.SEARCH_RESULTS_FIELDS, rows)
    print(f"Wrote {len(rows)} candidate papers to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=common.DEFAULT_QUERY, help="PubMed search query.")
    parser.add_argument("--max-results", type=int, default=100, help="Max PubMed records to pull.")
    parser.add_argument("--out", default=str(common.DEFAULT_SEARCH_RESULTS_CSV))
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
