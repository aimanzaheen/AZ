#!/usr/bin/env python3
"""Render side_by_side.csv as a self-contained, collapsible HTML review page."""

from __future__ import annotations

import argparse
import csv
import html
import sys
from collections import defaultdict
from pathlib import Path

import common

CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; padding: 1.5rem;
       max-width: 1100px; margin-inline: auto; color: #1a1a1a; }
h1 { font-size: 1.3rem; }
.paper { border: 1px solid #ddd; border-radius: 8px; margin-bottom: 0.75rem; }
.paper > summary { padding: 0.6rem 0.9rem; cursor: pointer; font-weight: 600; }
.field { border-top: 1px solid #eee; padding: 0.6rem 0.9rem; }
.field h3 { margin: 0 0 0.4rem 0; font-size: 0.95rem; color: #444; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th, td { text-align: left; padding: 0.3rem 0.5rem; vertical-align: top; border-bottom: 1px solid #f0f0f0; }
th { color: #666; font-weight: 600; width: 12%; }
.otto-row td { background: #fff8e1; }
.item-row td { background: #f4f8ff; }
.summary-row td { background: #f1f1f1; font-style: italic; }
.placeholder { color: #999; font-style: italic; }
input#filter { width: 100%; padding: 0.5rem; margin-bottom: 1rem; font-size: 1rem;
               box-sizing: border-box; }
"""

JS = """
document.getElementById('filter').addEventListener('input', (e) => {
  const q = e.target.value.toLowerCase();
  document.querySelectorAll('.paper').forEach((p) => {
    p.style.display = p.dataset.search.includes(q) ? '' : 'none';
  });
});
"""


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def render(rows: list[dict]) -> str:
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_paper[row["paper_id"]].append(row)

    papers_html = []
    for paper_id, paper_rows in sorted(by_paper.items()):
        title = paper_rows[0]["title"]
        by_field: dict[str, list[dict]] = defaultdict(list)
        for row in paper_rows:
            by_field[row["field_name"]].append(row)

        fields_html = []
        for field_name, field_rows in by_field.items():
            otto_value = field_rows[0]["otto_value"]
            item_html = [
                f"<tr class='otto-row'><th>Otto</th>"
                f"<td colspan='3'>{esc(otto_value) or '<span class=placeholder>(empty)</span>'}</td></tr>"
            ]
            for r in field_rows:
                css_class = "summary-row" if r["row_label"] == "(overall summary)" else "item-row"
                label = esc(r["row_label"]) or "&mdash;"
                value = esc(r["reextracted_value"]) or "<span class=placeholder>(empty)</span>"
                quote = esc(r["source_quote"])
                fig = esc(r["figure_ref"])
                item_html.append(
                    f"<tr class='{css_class}'><th>{label}</th><td>{value}</td>"
                    f"<td>{quote}</td><td>{fig}</td></tr>"
                )
            fields_html.append(
                f"<div class='field'><h3>{esc(field_name)}</h3><table>"
                f"<tr><th></th><th>Value</th><th>Source quote</th><th>Figure ref</th></tr>"
                f"{''.join(item_html)}</table></div>"
            )

        search_blob = esc(f"{paper_id} {title}").lower()
        papers_html.append(
            f"<details class='paper' data-search='{search_blob}'>"
            f"<summary>{esc(paper_id)} &mdash; {esc(title)}</summary>"
            f"{''.join(fields_html)}</details>"
        )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Otto re-extraction review</title><style>{CSS}</style></head>
<body>
<h1>Otto re-extraction review</h1>
<input id="filter" placeholder="Filter by paper ID or title...">
{''.join(papers_html)}
<script>{JS}</script>
</body></html>"""


def run(args: argparse.Namespace) -> int:
    in_path = Path(args.side_by_side_csv)
    if not in_path.exists():
        print(f"error: {in_path} not found - run compare.py first", file=sys.stderr)
        return 1

    with in_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side-by-side-csv", default=str(common.DEFAULT_SIDE_BY_SIDE_CSV))
    parser.add_argument("--out", default=str(common.PKG_DIR / "data" / "review.html"))
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
