#!/usr/bin/env python3
"""Render a single-page, searchable/sortable HTML dashboard combining paper-match
status, Otto's original field text, and the verification audit (if it's been run).

Unlike render_review.py/render_verification.py (two separate flat tables), this
produces one page: a paper-level table (match status + Otto's anatomical/
photometry/ephys text, click to expand) plus a field-level verification table
(sortable by verdict, mismatches first) - built from data/otto_output.csv,
data/scrape_manifest.csv, and data/verification.csv. The verification section is
simply empty if verify.py/verification.csv hasn't been run yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import common

TEMPLATE_PATH = common.PKG_DIR / "dashboard_template.html"


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["paper_id"]: row for row in csv.DictReader(fh) if row.get("paper_id")}


def load_verification(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_paper_rows(otto_rows: list[common.OttoRow], manifest: dict[str, dict]) -> list[dict]:
    rows = []
    for paper in otto_rows:
        r = paper.raw
        m = manifest.get(paper.paper_id)
        year = (r.get("Authors and Publication Year") or "").split(";")[-1].strip()
        rows.append(
            {
                "id": paper.paper_id,
                "title": r.get("Title", ""),
                "authors": r.get("Authors", ""),
                "year": year,
                "doi": r.get("DOI", ""),
                "animal_model": r.get("Animal Model", ""),
                "anatomical_text": (r.get("Anatomical connections") or "").strip(),
                "photometry_text": (r.get("Photometry Data") or "").strip(),
                "ephys_text": (r.get("Electrophysiology Data") or "").strip(),
                "has_anatomical": bool((r.get("Anatomical connections") or "").strip()),
                "has_photometry": bool((r.get("Photometry Data") or "").strip()),
                "has_ephys": bool((r.get("Electrophysiology Data") or "").strip()),
                "has_functional": bool((r.get("Purpose of this circuit") or "").strip()),
                "matched": bool(m),
                "confidence": float(m["confidence"]) if m else None,
                "match_method": m["match_method"] if m else None,
            }
        )
    return rows


def run(args: argparse.Namespace) -> int:
    otto_rows = common.load_otto_rows(args.otto_csv)
    manifest = load_manifest(Path(args.manifest_csv))
    verification_rows = load_verification(Path(args.verification_csv))

    data = build_paper_rows(otto_rows, manifest)

    if not TEMPLATE_PATH.exists():
        print(f"error: template not found at {TEMPLATE_PATH}", file=sys.stderr)
        return 1
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    html_out = template.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    html_out = html_out.replace("__VERIFICATION_JSON__", json.dumps(verification_rows, ensure_ascii=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_out, encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"  {len(data)} papers ({sum(1 for d in data if d['matched'])} matched)")
    print(f"  {len(verification_rows)} verification verdicts")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--otto-csv", default=str(common.DEFAULT_OTTO_CSV))
    parser.add_argument("--manifest-csv", default=str(common.PKG_DIR / "data" / "scrape_manifest.csv"))
    parser.add_argument("--verification-csv", default=str(common.DEFAULT_VERIFICATION_CSV))
    parser.add_argument("--out", default=str(common.PKG_DIR / "data" / "dashboard.html"))
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
