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

VERIFICATION_FIELDS = [
    "paper_id",
    "modality",
    "otto_field",
    "verdict",
    "confidence",
    "explanation",
    "evidence_quote",
]

VERDICT_SEVERITY = {"mismatch": 0, "partial": 1, "unverifiable": 2, "match": 3}


def flatten_verification(paper_id: str, modality: str, raw: dict) -> list[dict]:
    """Turn one model response ({"verdicts": [...]}) into CSV rows."""
    verdicts = raw.get("verdicts") or []
    return [
        {
            "paper_id": paper_id,
            "modality": modality,
            "otto_field": v.get("otto_field", ""),
            "verdict": v.get("verdict", "unverifiable"),
            "confidence": v.get("confidence", ""),
            "explanation": v.get("explanation", ""),
            "evidence_quote": v.get("evidence_quote", ""),
        }
        for v in verdicts
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


def render_modality_summary(rows: list[dict]) -> str:
    """Reconstruct a readable text block for a (paper, modality) group of flattened rows.

    Used to give the verification model the fresh re-extraction as context.
    """
    if not rows:
        return "(no re-extraction available for this modality)"

    lines = []
    summary = (rows[0].get("summary_paragraph") or "").strip()
    if summary:
        lines.append(f"Overall summary: {summary}")

    for row in rows:
        fields = json.loads(row.get("fields_json") or "{}")
        if not fields:
            continue
        detail = "; ".join(f"{k}: {v}" for k, v in fields.items())
        fig = row.get("figure_ref") or ""
        line = f"- {row.get('row_label', '')}: {detail}"
        if fig:
            line += f" [{fig}]"
        lines.append(line)

    return "\n".join(lines) if lines else "(model returned no itemized data)"
