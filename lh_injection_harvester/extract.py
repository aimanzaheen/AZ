#!/usr/bin/env python3
"""Run LLM extraction of the 6 target fields for every cached paper.

Calls the Anthropic API once per paper with the paper's cached text (full text or abstract) and
the LH retrograde-injection extraction prompt, writes the raw structured result to
data/extracted_raw/<pmid>.json, then rebuilds the flattened data/extracted.csv (one row per
distinct LH injection experiment) from every raw file on disk.

Requires ANTHROPIC_API_KEY. Resumable: skips papers already in extracted_raw/ unless --force.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anthropic

import common
import llm_client


def rebuild_extracted_csv(raw_dir: Path, search_results_csv: Path, out_csv: Path) -> int:
    metadata = {r.pmid: r for r in common.load_search_results(search_results_csv)}
    rows: list[dict] = []
    for raw_path in sorted(raw_dir.glob("*.json")):
        data = common.read_json(raw_path)
        if not data or not data.get("lh_retrograde_injection_present"):
            continue
        pmid = data.get("pmid", raw_path.stem)
        meta = metadata.get(pmid)
        for idx, exp in enumerate(data.get("experiments", []), 1):
            rows.append(
                {
                    "pmid": pmid,
                    "paper_name": meta.title if meta else "",
                    "year": meta.year if meta else "",
                    "journal": meta.journal if meta else "",
                    "experiment_index": idx,
                    "volume_injected_lh": exp.get("volume_injected_lh", ""),
                    "stereotaxic_coordinates": exp.get("stereotaxic_coordinates", ""),
                    "tracer": exp.get("tracer", ""),
                    "projections_found": exp.get("projections_found", ""),
                    "survival_time": exp.get("survival_time", ""),
                    "figure_ref": exp.get("figure_ref", ""),
                    "source_quote": exp.get("source_quote", ""),
                    "text_source": data.get("text_source", ""),
                }
            )
    common.write_csv(out_csv, common.EXTRACTED_FIELDS, rows)
    return len(rows)


def run(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    papers = common.load_search_results(args.search_results_csv)
    if args.pmid:
        papers = [p for p in papers if p.pmid == args.pmid]
    if args.limit:
        papers = papers[: args.limit]

    if not papers:
        print(f"No papers found in {args.search_results_csv} - run search_papers.py first.", file=sys.stderr)
        return 1

    instructions = common.load_prompt(Path(args.prompt_file))
    client = anthropic.Anthropic()

    processed = skipped = failed = 0
    for i, paper in enumerate(papers, 1):
        raw_path = raw_dir / f"{paper.pmid}.json"
        if raw_path.exists() and not args.force:
            print(f"[{i}/{len(papers)}] {paper.pmid}: already extracted")
            skipped += 1
            continue

        cached = common.read_json(cache_dir / f"{paper.pmid}.json")
        if not cached or cached.get("status") != "ok" or not cached.get("text"):
            print(f"[{i}/{len(papers)}] {paper.pmid}: no cached text, skipping (run fetch_papers.py first)")
            skipped += 1
            continue

        try:
            result = llm_client.call_extraction(
                client, args.model, cached["text"], cached["text_source"], instructions
            )
        except Exception as exc:  # noqa: BLE001 - log and continue with the rest of the batch
            print(f"[{i}/{len(papers)}] {paper.pmid}: extraction failed - {exc}", file=sys.stderr)
            failed += 1
            continue

        result["pmid"] = paper.pmid
        result["text_source"] = cached["text_source"]
        common.write_json(raw_path, result)
        match = "match" if result.get("lh_retrograde_injection_present") else "no match"
        n_exp = len(result.get("experiments", []))
        print(f"[{i}/{len(papers)}] {paper.pmid}: {match} ({n_exp} experiment(s))")
        processed += 1

    n_rows = rebuild_extracted_csv(raw_dir, Path(args.search_results_csv), Path(args.out))
    print(f"\nExtracted {processed} new, skipped {skipped}, failed {failed}.")
    print(f"Wrote {n_rows} experiment rows to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-results-csv", default=str(common.DEFAULT_SEARCH_RESULTS_CSV))
    parser.add_argument("--cache-dir", default=str(common.DEFAULT_CACHE_DIR))
    parser.add_argument("--raw-dir", default=str(common.DEFAULT_EXTRACTED_RAW_DIR))
    parser.add_argument("--out", default=str(common.DEFAULT_EXTRACTED_CSV))
    parser.add_argument("--prompt-file", default=str(common.PROMPT_FILE))
    parser.add_argument("--model", default=llm_client.DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Re-run extraction even if cached.")
    parser.add_argument("--limit", type=int, help="Only process the first N papers (for testing).")
    parser.add_argument("--pmid", help="Only process this single PMID (for testing).")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
