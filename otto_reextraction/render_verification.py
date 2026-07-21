#!/usr/bin/env python3
"""Render verification.csv as a sortable, filterable HTML triage table.

Sorted mismatches-first by default so you can go straight to the fields
where Otto's original extraction is most likely wrong, instead of
eyeballing everything with equal weight.
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path

import common

CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; padding: 1.5rem;
       max-width: 1400px; margin-inline: auto; color: #1a1a1a; }
h1 { font-size: 1.3rem; }
.controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }
input#filter { flex: 1; min-width: 240px; padding: 0.5rem; font-size: 1rem; box-sizing: border-box; }
.verdict-toggle label { margin-right: 0.75rem; font-size: 0.85rem; cursor: pointer; }
table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
th, td { text-align: left; padding: 0.4rem 0.55rem; vertical-align: top; border-bottom: 1px solid #eee; }
th { position: sticky; top: 0; background: #fff; cursor: pointer; user-select: none; }
th.sorted::after { content: " \\25BE"; }
tr:hover td { background: #fafafa; }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem;
         font-weight: 600; color: #fff; white-space: nowrap; }
.badge-mismatch { background: #d64545; }
.badge-partial { background: #d99a1f; }
.badge-unverifiable { background: #888; }
.badge-match { background: #3a9d5c; }
.summary { font-size: 0.85rem; color: #555; margin-bottom: 0.75rem; }
.otto-val, .quote { max-width: 260px; }
"""

JS = """
const rows = Array.from(document.querySelectorAll('#tbl tbody tr'));
const filterInput = document.getElementById('filter');
const verdictBoxes = Array.from(document.querySelectorAll('.verdict-toggle input'));

function applyFilters() {
  const q = filterInput.value.toLowerCase();
  const activeVerdicts = new Set(verdictBoxes.filter(b => b.checked).map(b => b.value));
  rows.forEach(r => {
    const matchesText = r.dataset.search.includes(q);
    const matchesVerdict = activeVerdicts.has(r.dataset.verdict);
    r.style.display = (matchesText && matchesVerdict) ? '' : 'none';
  });
}
filterInput.addEventListener('input', applyFilters);
verdictBoxes.forEach(b => b.addEventListener('change', applyFilters));

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


def render(rows: list[dict], otto_by_paper: dict[str, common.OttoRow]) -> str:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    total = len(rows)
    summary = " | ".join(f"{v}: {counts.get(v, 0)}" for v in ["mismatch", "partial", "unverifiable", "match"])

    body_rows = []
    for r in rows:
        otto = otto_by_paper.get(r["paper_id"])
        title = otto.title if otto else ""
        otto_value = otto.raw.get(r["otto_field"], "") if otto else ""
        search_blob = esc(f"{r['paper_id']} {title} {r['otto_field']} {r['modality']}").lower()
        body_rows.append(
            f"<tr data-verdict='{esc(r['verdict'])}' data-search='{search_blob}'>"
            f"<td>{esc(r['paper_id'])}</td>"
            f"<td>{esc(title)}</td>"
            f"<td>{esc(r['modality'])}</td>"
            f"<td>{esc(r['otto_field'])}</td>"
            f"<td><span class='badge badge-{esc(r['verdict'])}'>{esc(r['verdict'])}</span></td>"
            f"<td>{esc(r['confidence'])}</td>"
            f"<td class='otto-val'>{esc(otto_value)}</td>"
            f"<td>{esc(r['explanation'])}</td>"
            f"<td class='quote'>{esc(r['evidence_quote'])}</td>"
            f"</tr>"
        )

    verdict_toggles = "".join(
        f"<label><input type='checkbox' value='{v}' checked> {v}</label>"
        for v in ["mismatch", "partial", "unverifiable", "match"]
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Otto verification triage</title><style>{CSS}</style></head>
<body>
<h1>Otto verification triage</h1>
<div class="summary">{total} field verdicts across {len(otto_by_paper)} papers &mdash; {summary}</div>
<div class="controls">
  <input id="filter" placeholder="Filter by paper ID, title, field, or modality...">
  <div class="verdict-toggle">{verdict_toggles}</div>
</div>
<table id="tbl">
<thead><tr>
<th data-col>Paper</th><th data-col>Title</th><th data-col>Modality</th><th data-col>Field</th>
<th data-col>Verdict</th><th data-col>Confidence</th><th data-col>Otto value</th>
<th data-col>Explanation</th><th data-col>Evidence quote</th>
</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table>
<script>{JS}</script>
</body></html>"""


def run(args: argparse.Namespace) -> int:
    in_path = Path(args.verification_csv)
    if not in_path.exists():
        print(f"error: {in_path} not found - run verify.py first", file=sys.stderr)
        return 1

    with in_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    otto_by_paper = {p.paper_id: p for p in common.load_otto_rows(args.otto_csv)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(rows, otto_by_paper), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-csv", default=str(common.DEFAULT_VERIFICATION_CSV))
    parser.add_argument("--otto-csv", default=str(common.DEFAULT_OTTO_CSV))
    parser.add_argument("--out", default=str(common.PKG_DIR / "data" / "verification.html"))
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
