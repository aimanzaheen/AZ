#!/usr/bin/env python3
"""Run the full pipeline end to end: search -> fetch -> extract -> render.

Convenience wrapper around search_papers.py, fetch_papers.py, extract.py, and render_table.py -
see README.md for what each stage does and for running them individually.
"""

from __future__ import annotations

import argparse
import sys

import extract
import fetch_papers
import render_table
import search_papers
import common


def run(args: argparse.Namespace) -> int:
    search_args = search_papers.build_parser().parse_args(
        ["--query", args.query, "--max-results", str(args.max_results), "--out", str(common.DEFAULT_SEARCH_RESULTS_CSV)]
    )
    rc = search_papers.run(search_args)
    if rc != 0:
        return rc

    fetch_args_list = ["--search-results-csv", str(common.DEFAULT_SEARCH_RESULTS_CSV)]
    if args.limit:
        fetch_args_list += ["--limit", str(args.limit)]
    if args.force:
        fetch_args_list.append("--force")
    rc = fetch_papers.run(fetch_papers.build_parser().parse_args(fetch_args_list))
    if rc != 0:
        return rc

    extract_args_list = ["--search-results-csv", str(common.DEFAULT_SEARCH_RESULTS_CSV), "--model", args.model]
    if args.limit:
        extract_args_list += ["--limit", str(args.limit)]
    if args.force:
        extract_args_list.append("--force")
    rc = extract.run(extract.build_parser().parse_args(extract_args_list))
    if rc != 0:
        return rc

    return render_table.run(render_table.build_parser().parse_args(["--query", args.query]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=common.DEFAULT_QUERY, help="PubMed search query.")
    parser.add_argument("--max-results", type=int, default=100, help="Max PubMed records to pull.")
    parser.add_argument("--model", default="claude-sonnet-5", help="Anthropic model for extraction.")
    parser.add_argument("--limit", type=int, help="Only process the first N papers end to end (for testing).")
    parser.add_argument("--force", action="store_true", help="Redo fetch/extraction even if cached.")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
