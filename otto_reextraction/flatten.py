"""Flatten a raw record_extraction JSON blob into long-format CSV rows."""

from __future__ import annotations

import json

REEXTRACTED_FIELDS = [
    "paper_id",
    "modality",
    "row_index",
    "row_label",
    "figure_ref",
    "source_quote",
    "fields_json",
    "summary_paragraph",
]


def flatten_raw(paper_id: str, modality: str, raw: dict) -> list[dict]:
    """Turn one model response ({"rows": [...], "summary_paragraph": ...}) into CSV rows."""
    summary = (raw.get("summary_paragraph") or "").strip()
    rows = raw.get("rows") or []

    if not rows:
        return [
            {
                "paper_id": paper_id,
                "modality": modality,
                "row_index": 0,
                "row_label": "(summary only - no itemized rows returned)",
                "figure_ref": "",
                "source_quote": "",
                "fields_json": "{}",
                "summary_paragraph": summary,
            }
        ]

    out = []
    for i, row in enumerate(rows):
        out.append(
            {
                "paper_id": paper_id,
                "modality": modality,
                "row_index": i,
                "row_label": row.get("row_label", ""),
                "figure_ref": row.get("figure_ref", ""),
                "source_quote": row.get("source_quote", ""),
                "fields_json": json.dumps(row.get("fields") or {}, ensure_ascii=False),
                "summary_paragraph": summary,
            }
        )
    return out


def render_row_value(row: dict) -> str:
    """Human-readable one-line-per-field rendering of a flattened row, for side-by-side viewing."""
    fields = json.loads(row.get("fields_json") or "{}")
    parts = [f"{k}: {v}" for k, v in fields.items()]
    label = row.get("row_label") or ""
    body = "; ".join(parts)
    return f"{label} — {body}" if label and body else (label or body)
