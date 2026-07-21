#!/usr/bin/env python3
"""Join otto_output.csv and reextracted.csv into a side-by-side comparison CSV.

No auto-resolution: every (paper, field) just lands next to Otto's original
value so a human can eyeball it. One row per re-extracted item (pathway /
population-epoch / cell-type group / summary) - a paper with 5 traced
pathways produces 5 rows under "Anatomical connections", plus one
"(overall summary)" row.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import common
import flatten

SIDE_BY_SIDE_FIELDS = [
    "paper_id",
    "title",
    "modality",
    "field_name",
    "row_label",
    "otto_value",
    "reextracted_value",
    "source_quote",
    "figure_ref",
]


def load_reextracted(path: Path) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if not path.exists():
        return grouped
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            grouped[(row["paper_id"], row["modality"])].append(row)
    return grouped


def build_side_by_side(otto_rows: list[common.OttoRow], grouped: dict[tuple[str, str], list[dict]]) -> list[dict]:
    out: list[dict] = []
    for paper in otto_rows:
        for modality, spec in common.MODALITIES.items():
            reext_rows = grouped.get((paper.paper_id, modality), [])
            summary_text = reext_rows[0]["summary_paragraph"].strip() if reext_rows else ""
            item_rows = [r for r in reext_rows if json.loads(r["fields_json"] or "{}")]

            for otto_field in spec["otto_fields"]:
                otto_value = paper.raw.get(otto_field, "")
                base = {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "modality": modality,
                    "field_name": otto_field,
                    "otto_value": otto_value,
                }

                if not reext_rows:
                    out.append(
                        {
                            **base,
                            "row_label": "",
                            "reextracted_value": "[not re-extracted yet - run reextract.py]",
                            "source_quote": "",
                            "figure_ref": "",
                        }
                    )
                    continue

                if summary_text:
                    out.append(
                        {
                            **base,
                            "row_label": "(overall summary)",
                            "reextracted_value": summary_text,
                            "source_quote": "",
                            "figure_ref": "",
                        }
                    )

                for row in item_rows:
                    out.append(
                        {
                            **base,
                            "row_label": row["row_label"],
                            "reextracted_value": flatten.render_row_value(row),
                            "source_quote": row["source_quote"],
                            "figure_ref": row["figure_ref"],
                        }
                    )

                if not summary_text and not item_rows:
                    out.append(
                        {
                            **base,
                            "row_label": "",
                            "reextracted_value": "[model returned no data]",
                            "source_quote": "",
                            "figure_ref": "",
                        }
                    )
    return out


def run(args: argparse.Namespace) -> int:
    otto_rows = common.load_otto_rows(args.otto_csv)
    grouped = load_reextracted(Path(args.reextracted_csv))
    side_by_side = build_side_by_side(otto_rows, grouped)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SIDE_BY_SIDE_FIELDS)
        writer.writeheader()
        writer.writerows(side_by_side)

    print(f"Wrote {len(side_by_side)} row(s) to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--otto-csv", default=str(common.DEFAULT_OTTO_CSV))
    parser.add_argument("--reextracted-csv", default=str(common.DEFAULT_REEXTRACTED_CSV))
    parser.add_argument("--out", default=str(common.DEFAULT_SIDE_BY_SIDE_CSV))
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
