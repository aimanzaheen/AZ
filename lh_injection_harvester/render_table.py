#!/usr/bin/env python3
"""Render extracted.csv as a sortable, filterable HTML table - the final harvested-papers webpage."""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path

import common

CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; padding: 1.5rem;
       max-width: 1500px; margin-inline: auto; color: #1a1a1a; }
h1 { font-size: 1.3rem; }
.summary { font-size: 0.85rem; color: #555; margin-bottom: 0.75rem; }
.controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }
input#filter { flex: 1; min-width: 240px; padding: 0.5rem; font-size: 1rem; box-sizing: border-box; }
table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
th, td { text-align: left; padding: 0.45rem 0.6rem; vertical-align: top; border-bottom: 1px solid #eee; }
th { position: sticky; top: 0; background: #fff; cursor: pointer; user-select: none; }
th.sorted::after { content: " \\25BE"; }
tr:hover td { background: #fafafa; }
td.na { color: #999; font-style: italic; }
a { color: #2454b8; }
.quote { max-width: 260px; color: #666; font-size: 0.78rem; }
"""

JS = """
const rows = Array.from(document.querySelectorAll('#tbl tbody tr'));
const filterInput = document.getElementById('filter');

function applyFilter() {
  const q = filterInput.value.toLowerCase();
  rows.forEach(r => { r.style.display = r.dataset.search.includes(q) ? '' : 'none'; });
}
filterInput.addEventListener('input', applyFilter);

document.querySelectorAll('th[data-col]').forEach((th, colIdx) => {
  th.addEventListener('click', () => {
    const tbody = document.querySelector('#tbl tbody');
    const asc = th.dataset.asc !== 'true';
    document.querySelectorAll('th[data-col]').forEach(h => { h.classList.remove('sorted'); h.dataset.asc = ''; });
    th.classList.add('sorted');
    th.dataset.asc = asc;
    const sorted = rows.slice().sort((a, b) => {
      const av = a.children[colIdx].innerText.toLowerCase();
      const bv = b.children[colIdx].innerText.toLowerCase();
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    });
    sorted.forEach(r => tbody.appendChild(r));
  });
});
"""


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def cell(value: str) -> str:
    value = (value or "").strip()
    if not value or value.lower() == "not explicitly stated":
        return "<td class='na'>not explicitly stated</td>"
    return f"<td>{esc(value)}</td>"


def render(rows: list[dict], query: str) -> str:
    n_papers = len({r["pmid"] for r in rows})
    body_rows = []
    for r in rows:
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{esc(r['pmid'])}/"
        search_blob = esc(
            " ".join(
                str(r.get(k, ""))
                for k in ["paper_name", "year", "journal", "tracer", "stereotaxic_coordinates", "projections_found"]
            )
        ).lower()
        source_label = {"pmc_fulltext": "full text", "pubmed_abstract": "abstract only"}.get(
            r.get("text_source", ""), r.get("text_source", "")
        )
        body_rows.append(
            f"<tr data-search='{search_blob}'>"
            f"<td><a href='{pubmed_url}' target='_blank' rel='noopener'>{esc(r['paper_name']) or r['pmid']}</a></td>"
            f"<td>{esc(r.get('year', ''))}</td>"
            f"<td>{esc(r.get('journal', ''))}</td>"
            + cell(r.get("volume_injected_lh", ""))
            + cell(r.get("stereotaxic_coordinates", ""))
            + cell(r.get("tracer", ""))
            + cell(r.get("projections_found", ""))
            + cell(r.get("survival_time", ""))
            + f"<td>{esc(source_label)}</td>"
            f"<td class='quote'>{esc(r.get('source_quote', ''))}</td>"
            f"</tr>"
        )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>LH retrograde injection literature table</title><style>{CSS}</style></head>
<body>
<h1>Mouse LH retrograde tracer injection papers</h1>
<div class="summary">{len(rows)} injection experiment(s) across {n_papers} paper(s).
Search query: <code>{esc(query)}</code></div>
<div class="controls">
  <input id="filter" placeholder="Filter by paper, tracer, coordinates, projections...">
</div>
<table id="tbl">
<thead><tr>
<th data-col>Paper name</th><th data-col>Year</th><th data-col>Journal</th>
<th data-col>Volume injected into LH</th><th data-col>Stereotaxic coordinates</th>
<th data-col>Tracer</th><th data-col>Projections found</th>
<th data-col>Survival time before perfusion</th><th data-col>Source</th><th data-col>Quote</th>
</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
<script>{JS}</script>
</body></html>"""


def run(args: argparse.Namespace) -> int:
    in_path = Path(args.extracted_csv)
    if not in_path.exists():
        print(f"error: {in_path} not found - run extract.py first", file=sys.stderr)
        return 1

    with in_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(rows, args.query), encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} rows)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-csv", default=str(common.DEFAULT_EXTRACTED_CSV))
    parser.add_argument("--out", default=str(common.DEFAULT_TABLE_HTML))
    parser.add_argument("--query", default=common.DEFAULT_QUERY, help="Search query to show in the page header.")
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
