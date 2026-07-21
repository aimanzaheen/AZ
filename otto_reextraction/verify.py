#!/usr/bin/env python3
"""Audit Otto's original extraction against the paper text + fresh re-extraction.

For each paper and modality, asks the model to judge whether each of Otto's
recorded field values is accurate and reasonably complete given the source
paper text (using the independent re-extraction from reextract.py as a
cross-check, not as ground truth on its own). Writes data/verification.csv
with a match/partial/mismatch/unverifiable verdict, confidence, a short
explanation, and a supporting quote for every (paper, Otto field) pair - so
you can sort straight to the likely mistakes instead of eyeballing all 400+
rows equally.

Requires ANTHROPIC_API_KEY. Requires fetch_papers.py and reextract.py to
have already run for the papers being audited.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import anthropic

import common
import flatten
import llm_client


def load_reextracted_grouped(path: Path) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    if not path.exists():
        return grouped
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            grouped.setdefault((row["paper_id"], row["modality"]), []).append(row)
    return grouped


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

    grouped_reextraction = load_reextracted_grouped(Path(args.reextracted_csv))

    client = anthropic.Anthropic()
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
            otto_fields = common.MODALITIES[modality]["otto_fields"]
            otto_values = {f: paper.raw.get(f, "") for f in otto_fields}

            raw_path = raw_dir / f"{paper.paper_id}__{modality}.json"
            raw = common.read_json(raw_path)
            if raw is not None and not args.force:
                print(f"{paper.paper_id}/{modality}: using cached verdict")
            else:
                reext_rows = grouped_reextraction.get((paper.paper_id, modality), [])
                reextraction_summary = flatten.render_modality_summary(reext_rows)
                print(f"{paper.paper_id}/{modality}: calling {model}...")
                try:
                    raw = llm_client.call_verification(
                        client, model, text, text_source, reextraction_summary, otto_values
                    )
                except Exception as exc:  # noqa: BLE001 - record and keep going
                    print(f"{paper.paper_id}/{modality}: ERROR - {exc}", file=sys.stderr)
                    continue
                common.write_json(raw_path, raw)
                n_calls += 1

            all_csv_rows.extend(flatten.flatten_verification(paper.paper_id, modality, raw))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flatten.VERIFICATION_FIELDS)
        writer.writeheader()
        writer.writerows(
            sorted(all_csv_rows, key=lambda r: flatten.VERDICT_SEVERITY.get(r["verdict"], 99))
        )

    counts: dict[str, int] = {}
    for row in all_csv_rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    print(f"\nMade {n_calls} new API call(s); skipped {skipped_no_text} paper(s) with no cached text.")
    print(f"Wrote {len(all_csv_rows)} row(s) to {out_path}")
    print("Verdict counts:", counts)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--otto-csv", default=str(common.DEFAULT_OTTO_CSV))
    parser.add_argument("--cache-dir", default=str(common.DEFAULT_CACHE_DIR))
    parser.add_argument("--reextracted-csv", default=str(common.DEFAULT_REEXTRACTED_CSV))
    parser.add_argument("--raw-dir", default=str(common.DEFAULT_VERIFICATION_RAW_DIR))
    parser.add_argument("--out", default=str(common.DEFAULT_VERIFICATION_CSV))
    parser.add_argument("--model", default=None, help=f"Default: {llm_client.DEFAULT_MODEL}")
    parser.add_argument(
        "--modalities", default=None, help="Comma-separated subset of: " + ",".join(common.MODALITIES)
    )
    parser.add_argument("--force", action="store_true", help="Redo API calls even if a verdict is cached.")
    parser.add_argument("--limit", type=int, help="Only process the first N papers (for testing).")
    parser.add_argument("--paper-id", help="Only process this single Reference ID (for testing).")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
