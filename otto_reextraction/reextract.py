#!/usr/bin/env python3
"""Re-run the 4 extraction prompt templates against freshly fetched paper text.

Requires ANTHROPIC_API_KEY in the environment. Requires fetch_papers.py to
have already populated data/cache/<paper_id>.json for the papers being
processed (skips papers with no usable cached text).

Each (paper, modality) API call is cached to data/reextracted_raw/ so
re-runs are free/resumable; pass --force to redo them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anthropic

import common
import flatten
import llm_client


def run(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    modalities = args.modalities.split(",") if args.modalities else list(common.MODALITIES)
    for m in modalities:
        if m not in common.MODALITIES:
            print(f"error: unknown modality '{m}'. Choices: {list(common.MODALITIES)}", file=sys.stderr)
            return 1

    papers = common.load_otto_rows(args.otto_csv)
    if args.paper_id:
        papers = [p for p in papers if p.paper_id == args.paper_id]
    if args.limit:
        papers = papers[: args.limit]

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    model = args.model or llm_client.DEFAULT_MODEL

    all_csv_rows: list[dict] = []
    skipped_no_text = 0
    n_calls = 0

    for paper in papers:
        cached_fetch = common.read_json(cache_dir / f"{paper.paper_id}.json")
        text = (cached_fetch or {}).get("text")
        text_source = (cached_fetch or {}).get("text_source")
        if not text:
            skipped_no_text += 1
            print(f"{paper.paper_id}: skipped (no cached paper text - run fetch_papers.py first)")
            continue

        for modality in modalities:
            raw_path = raw_dir / f"{paper.paper_id}__{modality}.json"
            raw = common.read_json(raw_path)
            if raw is not None and not args.force:
                print(f"{paper.paper_id}/{modality}: using cached response")
            else:
                instructions = common.load_prompt(modality)
                print(f"{paper.paper_id}/{modality}: calling {model}...")
                try:
                    raw = llm_client.call_extraction(client, model, text, text_source, instructions)
                except Exception as exc:  # noqa: BLE001 - record and keep going
                    print(f"{paper.paper_id}/{modality}: ERROR - {exc}", file=sys.stderr)
                    continue
                common.write_json(raw_path, raw)
                n_calls += 1

            all_csv_rows.extend(flatten.flatten_raw(paper.paper_id, modality, raw))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flatten.REEXTRACTED_FIELDS)
        writer.writeheader()
        writer.writerows(all_csv_rows)

    print(f"\nMade {n_calls} new API call(s); skipped {skipped_no_text} paper(s) with no cached text.")
    print(f"Wrote {len(all_csv_rows)} row(s) to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--otto-csv", default=str(common.DEFAULT_OTTO_CSV))
    parser.add_argument("--cache-dir", default=str(common.DEFAULT_CACHE_DIR))
    parser.add_argument("--raw-dir", default=str(common.DEFAULT_RAW_DIR))
    parser.add_argument("--out", default=str(common.DEFAULT_REEXTRACTED_CSV))
    parser.add_argument("--model", default=None, help=f"Default: {llm_client.DEFAULT_MODEL}")
    parser.add_argument(
        "--modalities", default=None, help="Comma-separated subset of: " + ",".join(common.MODALITIES)
    )
    parser.add_argument("--force", action="store_true", help="Redo API calls even if a raw response is cached.")
    parser.add_argument("--limit", type=int, help="Only process the first N papers (for testing).")
    parser.add_argument("--paper-id", help="Only process this single Reference ID (for testing).")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
