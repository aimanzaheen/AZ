#!/usr/bin/env python3
"""Otto re-extraction pipeline, as a single self-contained script with a local web UI.

This is a one-file rewrite of the otto_reextraction/ pipeline (scrape_papers.py +
reextract.py + compare.py + verify.py + the two render_*.py scripts) plus a
Flask app so you can drive the whole thing from a browser instead of the CLI:
scan papers/, re-extract, compare against Otto, verify Otto's accuracy, and
view the results - all from one page.

Run it:

    pip install flask anthropic requests
    export ANTHROPIC_API_KEY=sk-ant-...
    python webapp.py

Then open the URL it prints - http://127.0.0.1:5000 by default. It only
binds to localhost: nothing outside this machine can reach it.

Inputs (unchanged from the modular pipeline):
    data/otto_output.csv   Otto's original extraction (one row per paper)
    prompts/*.txt          your 4 extraction prompt templates, used verbatim
    papers/                drop your PDF/TXT/MD files here

Outputs (unchanged):
    data/cache/<paper_id>.json                 extracted paper text
    data/reextracted_raw/<paper_id>__<modality>.json
    data/reextracted.csv
    data/verification_raw/<paper_id>__<modality>.json
    data/verification.csv
    data/scrape_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, Response, jsonify, request

# --------------------------------------------------------------------------
# Config / paths
# --------------------------------------------------------------------------

PKG_DIR = Path(__file__).parent
OTTO_CSV = PKG_DIR / "data" / "otto_output.csv"
PAPERS_DIR = PKG_DIR / "papers"
CACHE_DIR = PKG_DIR / "data" / "cache"
RAW_DIR = PKG_DIR / "data" / "reextracted_raw"
VERIFICATION_RAW_DIR = PKG_DIR / "data" / "verification_raw"
REEXTRACTED_CSV = PKG_DIR / "data" / "reextracted.csv"
VERIFICATION_CSV = PKG_DIR / "data" / "verification.csv"
SCRAPE_MANIFEST_CSV = PKG_DIR / "data" / "scrape_manifest.csv"
PROMPTS_DIR = PKG_DIR / "prompts"

PAPER_ID_COLUMN = "Reference ID"
DOI_COLUMN = "DOI"

MODALITIES: dict[str, dict] = {
    "anatomical": {
        "prompt_file": PROMPTS_DIR / "anatomical.txt",
        "otto_fields": ["Anatomical connections", "Verification of Zona Incerta Targeting"],
    },
    "photometry": {
        "prompt_file": PROMPTS_DIR / "photometry.txt",
        "otto_fields": ["Photometry Data"],
    },
    "ephys": {
        "prompt_file": PROMPTS_DIR / "ephys.txt",
        "otto_fields": ["Electrophysiology Data"],
    },
    "functional": {
        "prompt_file": PROMPTS_DIR / "functional.txt",
        "otto_fields": ["Purpose of this circuit", "Necessity vs. sufficiency or Both"],
    },
}

DEFAULT_MODEL = "claude-sonnet-5"
MAX_RETRIES = 5
PDF_TEXT_SUFFIXES = {".pdf"}
PLAIN_TEXT_SUFFIXES = {".txt", ".md"}
TITLE_MATCH_THRESHOLD = 0.5

# --------------------------------------------------------------------------
# Shared state for the web UI (progress log + "is a stage running" guard)
# --------------------------------------------------------------------------

_state_lock = threading.Lock()
_log: deque[str] = deque(maxlen=300)
_running_stage: str | None = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _state_lock:
        _log.append(line)
    print(line)


def try_start_stage(name: str) -> bool:
    global _running_stage
    with _state_lock:
        if _running_stage is not None:
            return False
        _running_stage = name
        return True


def finish_stage() -> None:
    global _running_stage
    with _state_lock:
        _running_stage = None


def current_stage() -> str | None:
    with _state_lock:
        return _running_stage


def recent_log(n: int = 80) -> list[str]:
    with _state_lock:
        return list(_log)[-n:]


# --------------------------------------------------------------------------
# otto_output.csv loading
# --------------------------------------------------------------------------


@dataclass
class OttoRow:
    paper_id: str
    doi: str | None
    title: str
    raw: dict[str, str]


def normalize_doi(doi: str) -> str | None:
    doi = (doi or "").strip()
    if not doi:
        return None
    doi = re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", doi)
    return doi.strip("/ ") or None


def load_otto_rows(csv_path: Path | None = None) -> list[OttoRow]:
    # csv_path defaults to the *current* value of the OTTO_CSV global (not a value bound at
    # def-time) so tests can monkeypatch webapp.OTTO_CSV and have every caller see it.
    csv_path = csv_path if csv_path is not None else OTTO_CSV
    rows: list[OttoRow] = []
    if not csv_path.exists():
        return rows
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        for raw in csv.DictReader(fh):
            paper_id = raw.get(PAPER_ID_COLUMN, "").strip()
            if not paper_id:
                continue
            rows.append(
                OttoRow(
                    paper_id=paper_id,
                    doi=normalize_doi(raw.get(DOI_COLUMN, "")),
                    title=raw.get("Title", "").strip(),
                    raw=raw,
                )
            )
    return rows


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_prompt(modality: str) -> str:
    return MODALITIES[modality]["prompt_file"].read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------
# Local PDF/TXT text loading
# --------------------------------------------------------------------------


def load_paper_text(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in PLAIN_TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in PDF_TEXT_SUFFIXES:
        return _extract_pdf_text(path)
    return None


def _extract_pdf_text(path: Path) -> str | None:
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        text = "\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass  # pdfplumber missing/unusable - fall back to pypdf

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text or None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Paper <-> otto_output.csv matching
# --------------------------------------------------------------------------

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>,;]+", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")
_SEPARATOR_RE = re.compile(r"[-_\s]+")

# A paper's own DOI is essentially always on the first page (title block, running header,
# or abstract). Bibliography citations are not - so restricting the DOI search to a header
# window avoids matching on a DOI the paper merely *cites*, which matters a lot in a corpus
# where every paper is about the same circuit and heavily cites the other papers in it.
DOI_SEARCH_WINDOW = 3000


def _normalize_key(s: str) -> str:
    # Collapse dashes/underscores/whitespace only (the separators real Reference IDs use,
    # e.g. "AIM-50") - do NOT strip other punctuation. Stripping everything would merge
    # unrelated digit groups together, e.g. a browser's duplicate-download suffix
    # "AIM-1(1).pdf" would otherwise collapse to "aim11" and collide with "AIM-11".
    return _SEPARATOR_RE.sub("", s.lower())


def _words(s: str) -> set[str]:
    return set(_WORD_RE.findall(s.lower()))


def filename_match(filename_stem: str, otto_rows: list[OttoRow]) -> OttoRow | None:
    target = _normalize_key(filename_stem)
    if not target:
        return None
    for row in otto_rows:
        if _normalize_key(row.paper_id) == target:
            return row
    return None


def doi_match(pdf_text: str, otto_rows: list[OttoRow]) -> OttoRow | None:
    by_doi = {row.doi: row for row in otto_rows if row.doi}
    if not by_doi:
        return None

    # Ordered, de-duplicated (not a set - iteration order of a set of strings is not
    # stable across runs, which previously made this function pick a different, wrong
    # paper depending on Python's per-process hash seed).
    seen: list[str] = []
    for m in _DOI_RE.finditer(pdf_text[:DOI_SEARCH_WINDOW]):
        doi = normalize_doi(m.group(0).rstrip(").,;"))
        if doi and doi not in seen:
            seen.append(doi)

    for doi in seen:
        if doi in by_doi:
            return by_doi[doi]
    return None


def title_match(
    pdf_text: str, otto_rows: list[OttoRow], threshold: float = TITLE_MATCH_THRESHOLD
) -> tuple[OttoRow | None, float]:
    haystack = _words(pdf_text[:4000])
    if not haystack:
        return None, 0.0
    best_row, best_score = None, 0.0
    for row in otto_rows:
        title_words = _words(row.title)
        if not title_words:
            continue
        score = len(title_words & haystack) / len(title_words)
        if score > best_score:
            best_row, best_score = row, score
    if best_row is not None and best_score >= threshold:
        return best_row, best_score
    return None, best_score


@dataclass
class MatchResult:
    row: OttoRow | None
    method: str
    confidence: float


def match_paper(filename_stem: str, pdf_text: str, otto_rows: list[OttoRow]) -> MatchResult:
    row = filename_match(filename_stem, otto_rows)
    if row is not None:
        return MatchResult(row, "filename", 1.0)
    row = doi_match(pdf_text, otto_rows)
    if row is not None:
        return MatchResult(row, "doi", 0.95)
    row, score = title_match(pdf_text, otto_rows)
    if row is not None:
        return MatchResult(row, "title", round(score, 3))
    return MatchResult(None, "unmatched", 0.0)


# --------------------------------------------------------------------------
# Anthropic API wrapper (tool-use forces structured JSON output)
# --------------------------------------------------------------------------

EXTRACTION_TOOL = {
    "name": "record_extraction",
    "description": "Record the structured result of the extraction instructions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "description": "One entry per distinct item the instructions ask you to "
                "enumerate. Leave empty only if the instructions ask a single question with "
                "no table.",
                "items": {
                    "type": "object",
                    "properties": {
                        "row_label": {"type": "string"},
                        "figure_ref": {"type": "string"},
                        "source_quote": {"type": "string"},
                        "fields": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                    "required": ["row_label", "fields"],
                },
            },
            "summary_paragraph": {"type": "string"},
        },
        "required": ["rows", "summary_paragraph"],
    },
}

VERIFICATION_TOOL = {
    "name": "record_verification",
    "description": "Record a verdict on whether each of Otto's original extracted values "
    "is accurate and reasonably complete given the paper text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "otto_field": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["match", "partial", "mismatch", "unverifiable"],
                        },
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "explanation": {"type": "string"},
                        "evidence_quote": {"type": "string"},
                    },
                    "required": ["otto_field", "verdict", "confidence", "explanation"],
                },
            }
        },
        "required": ["verdicts"],
    },
}

EXTRACTION_SYSTEM_PROMPT = (
    "You are a meticulous scientific data extraction assistant reviewing a neuroscience "
    "paper. You will be given the paper's text followed by extraction instructions written "
    "by the research team. Follow the instructions exactly, including their row/table "
    "structure. If a value is not explicitly stated in the provided text, write 'not "
    "explicitly stated' - never estimate or infer a number that isn't given. Call the "
    "record_extraction tool exactly once with your complete answer."
)

VERIFICATION_SYSTEM_PROMPT = (
    "You are auditing a research-extraction team's work on a neuroscience paper. You will "
    "be given the paper's text, a fresh independent re-extraction performed separately, and "
    "the original team's ('Otto's') recorded value for one or more fields. For EACH field "
    "listed in <otto_original_values>, decide whether Otto's value is accurate and "
    "reasonably complete given the actual paper text - use the fresh re-extraction as a "
    "helpful cross-check, but base your verdict on the paper text itself. An empty Otto "
    "value is 'match' only if the paper genuinely reports nothing for that field. Call "
    "record_verification exactly once, with one verdict entry per field."
)


def _source_note(text_source: str) -> str:
    return {
        "local_pdf": "FULL TEXT (extracted from a locally supplied PDF)",
        "local_txt": "FULL TEXT (from a locally supplied text file)",
    }.get(text_source, text_source or "UNKNOWN SOURCE")


def _call_tool(client, model: str, system: str, tool: dict, user_message: str, max_tokens: int) -> dict:
    import anthropic

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
                messages=[{"role": "user", "content": user_message}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == tool["name"]:
                    return block.input
            raise RuntimeError(f"Model response did not include a {tool['name']} tool call.")
        except (
            anthropic.RateLimitError,
            anthropic.OverloadedError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"{tool['name']} call failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc


def call_extraction(client, model: str, paper_text: str, text_source: str, instructions: str) -> dict:
    user_message = (
        f'<paper_text source="{_source_note(text_source)}">\n{paper_text}\n</paper_text>\n\n'
        f"<extraction_instructions>\n{instructions}\n</extraction_instructions>"
    )
    return _call_tool(client, model, EXTRACTION_SYSTEM_PROMPT, EXTRACTION_TOOL, user_message, 8000)


def call_verification(
    client, model: str, paper_text: str, text_source: str, reextraction_summary: str, otto_values: dict
) -> dict:
    otto_block = "\n".join(f'{name}: "{value}"' for name, value in otto_values.items())
    user_message = (
        f'<paper_text source="{_source_note(text_source)}">\n{paper_text}\n</paper_text>\n\n'
        f"<fresh_independent_reextraction>\n{reextraction_summary}\n</fresh_independent_reextraction>\n\n"
        f"<otto_original_values>\n{otto_block}\n</otto_original_values>"
    )
    return _call_tool(client, model, VERIFICATION_SYSTEM_PROMPT, VERIFICATION_TOOL, user_message, 4000)


def get_anthropic_client():
    import os

    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")
    return anthropic.Anthropic()


# --------------------------------------------------------------------------
# Flatten / render helpers
# --------------------------------------------------------------------------

REEXTRACTED_FIELDS = [
    "paper_id", "modality", "row_index", "row_label", "figure_ref",
    "source_quote", "fields_json", "summary_paragraph",
]
VERIFICATION_FIELDS = [
    "paper_id", "modality", "otto_field", "verdict", "confidence", "explanation", "evidence_quote",
]
VERDICT_SEVERITY = {"mismatch": 0, "partial": 1, "unverifiable": 2, "match": 3}


def flatten_raw(paper_id: str, modality: str, raw: dict) -> list[dict]:
    summary = (raw.get("summary_paragraph") or "").strip()
    rows = raw.get("rows") or []
    if not rows:
        return [
            {
                "paper_id": paper_id, "modality": modality, "row_index": 0,
                "row_label": "(summary only - no itemized rows returned)",
                "figure_ref": "", "source_quote": "", "fields_json": "{}",
                "summary_paragraph": summary,
            }
        ]
    out = []
    for i, row in enumerate(rows):
        out.append(
            {
                "paper_id": paper_id, "modality": modality, "row_index": i,
                "row_label": row.get("row_label", ""), "figure_ref": row.get("figure_ref", ""),
                "source_quote": row.get("source_quote", ""),
                "fields_json": json.dumps(row.get("fields") or {}, ensure_ascii=False),
                "summary_paragraph": summary,
            }
        )
    return out


def render_row_value(row: dict) -> str:
    fields = json.loads(row.get("fields_json") or "{}")
    parts = [f"{k}: {v}" for k, v in fields.items()]
    label = row.get("row_label") or ""
    body = "; ".join(parts)
    return f"{label} — {body}" if label and body else (label or body)


def render_modality_summary(rows: list[dict]) -> str:
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


def flatten_verification(paper_id: str, modality: str, raw: dict) -> list[dict]:
    verdicts = raw.get("verdicts") or []
    return [
        {
            "paper_id": paper_id, "modality": modality,
            "otto_field": v.get("otto_field", ""), "verdict": v.get("verdict", "unverifiable"),
            "confidence": v.get("confidence", ""), "explanation": v.get("explanation", ""),
            "evidence_quote": v.get("evidence_quote", ""),
        }
        for v in verdicts
    ]


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------


def _paper_files() -> list[Path]:
    """Every candidate paper file in papers/, excluding the placeholder README."""
    if not PAPERS_DIR.is_dir():
        return []
    return sorted(
        p
        for p in PAPERS_DIR.rglob("*")
        if p.suffix.lower() in (PDF_TEXT_SUFFIXES | PLAIN_TEXT_SUFFIXES) and p.stem.lower() != "readme"
    )


def stage_scrape() -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    otto_rows = load_otto_rows()
    if not PAPERS_DIR.is_dir():
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    files = _paper_files()

    if not files:
        log(f"No PDF/TXT/MD files found in {PAPERS_DIR} - nothing to scan.")
        return {"matched": 0, "unmatched": 0, "missing": len(otto_rows)}

    claimed: dict[str, dict] = {}
    manifest_rows: list[dict] = []
    log(f"Scanning {len(files)} file(s) in {PAPERS_DIR}...")

    for path in files:
        text = load_paper_text(path)
        if not text or not text.strip():
            manifest_rows.append({"file": str(path), "paper_id": "", "match_method": "extraction_failed", "confidence": 0.0, "status": "failed"})
            log(f"{path.name}: FAILED (could not extract text)")
            continue

        result = match_paper(path.stem, text, otto_rows)
        row = {"file": str(path), "paper_id": result.row.paper_id if result.row else "", "match_method": result.method, "confidence": result.confidence, "status": "matched" if result.row else "unmatched"}

        if result.row is None:
            manifest_rows.append(row)
            log(f"{path.name}: unmatched")
            continue

        paper_id = result.row.paper_id
        existing = claimed.get(paper_id)
        if existing is not None and existing["confidence"] >= result.confidence:
            row["status"] = "duplicate_match"
            manifest_rows.append(row)
            log(f"{path.name}: duplicate match for {paper_id} - skipped")
            continue
        if existing is not None:
            existing["status"] = "superseded"

        claimed[paper_id] = row
        manifest_rows.append(row)
        write_json(
            CACHE_DIR / f"{paper_id}.json",
            {
                "paper_id": paper_id, "doi": result.row.doi, "text_source": "local_pdf" if path.suffix.lower() == ".pdf" else "local_txt",
                "status": "ok", "text": text, "source_file": str(path),
                "match_method": result.method, "match_confidence": result.confidence,
            },
        )
        log(f"{path.name}: matched {paper_id} (via {result.method}, confidence={result.confidence})")

    SCRAPE_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SCRAPE_MANIFEST_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "paper_id", "match_method", "confidence", "status"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    matched = len(claimed)
    missing = len(otto_rows) - matched
    log(f"Scrape done: {matched}/{len(otto_rows)} otto rows matched, {missing} missing.")
    return {"matched": matched, "unmatched": sum(1 for r in manifest_rows if r["status"] == "unmatched"), "missing": missing}


def stage_reextract(limit: int | None, modalities: list[str]) -> dict:
    client = get_anthropic_client()
    otto_rows = load_otto_rows()
    if limit:
        otto_rows = otto_rows[:limit]

    n_calls = 0
    for paper in otto_rows:
        cached = read_json(CACHE_DIR / f"{paper.paper_id}.json")
        text = (cached or {}).get("text")
        text_source = (cached or {}).get("text_source")
        if not text:
            log(f"{paper.paper_id}: skipped (no cached text - run Scrape Papers first)")
            continue
        for modality in modalities:
            raw_path = RAW_DIR / f"{paper.paper_id}__{modality}.json"
            raw = read_json(raw_path)
            if raw is None:
                instructions = load_prompt(modality)
                log(f"{paper.paper_id}/{modality}: calling {DEFAULT_MODEL}...")
                try:
                    raw = call_extraction(client, DEFAULT_MODEL, text, text_source, instructions)
                except Exception as exc:  # noqa: BLE001
                    log(f"{paper.paper_id}/{modality}: ERROR - {exc}")
                    continue
                write_json(raw_path, raw)
                n_calls += 1

    # Rebuild the full CSV from every cached raw response (not just this run's rows).
    all_flat: list[dict] = []
    for raw_path in sorted(RAW_DIR.glob("*__*.json")):
        paper_id, modality = raw_path.stem.split("__", 1)
        raw = read_json(raw_path)
        if raw:
            all_flat.extend(flatten_raw(paper_id, modality, raw))

    REEXTRACTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REEXTRACTED_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REEXTRACTED_FIELDS)
        writer.writeheader()
        writer.writerows(all_flat)

    log(f"Re-extract done: {n_calls} new API call(s). data/reextracted.csv has {len(all_flat)} row(s).")
    return {"new_calls": n_calls, "rows": len(all_flat)}


def stage_verify(limit: int | None, modalities: list[str]) -> dict:
    client = get_anthropic_client()
    otto_rows = load_otto_rows()
    if limit:
        otto_rows = otto_rows[:limit]

    grouped_reextraction: dict[tuple[str, str], list[dict]] = {}
    if REEXTRACTED_CSV.exists():
        with REEXTRACTED_CSV.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                grouped_reextraction.setdefault((row["paper_id"], row["modality"]), []).append(row)

    n_calls = 0
    for paper in otto_rows:
        cached = read_json(CACHE_DIR / f"{paper.paper_id}.json")
        text = (cached or {}).get("text")
        text_source = (cached or {}).get("text_source")
        if not text:
            log(f"{paper.paper_id}: skipped (no cached text)")
            continue
        for modality in modalities:
            otto_fields = MODALITIES[modality]["otto_fields"]
            otto_values = {f: paper.raw.get(f, "") for f in otto_fields}
            raw_path = VERIFICATION_RAW_DIR / f"{paper.paper_id}__{modality}.json"
            if read_json(raw_path) is not None:
                continue
            reext_rows = grouped_reextraction.get((paper.paper_id, modality), [])
            reextraction_summary = render_modality_summary(reext_rows)
            log(f"{paper.paper_id}/{modality}: verifying against {DEFAULT_MODEL}...")
            try:
                raw = call_verification(client, DEFAULT_MODEL, text, text_source, reextraction_summary, otto_values)
            except Exception as exc:  # noqa: BLE001
                log(f"{paper.paper_id}/{modality}: ERROR - {exc}")
                continue
            write_json(raw_path, raw)
            n_calls += 1

    all_flat: list[dict] = []
    for raw_path in sorted(VERIFICATION_RAW_DIR.glob("*__*.json")):
        paper_id, modality = raw_path.stem.split("__", 1)
        raw = read_json(raw_path)
        if raw:
            all_flat.extend(flatten_verification(paper_id, modality, raw))
    all_flat.sort(key=lambda r: VERDICT_SEVERITY.get(r["verdict"], 99))

    VERIFICATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    with VERIFICATION_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=VERIFICATION_FIELDS)
        writer.writeheader()
        writer.writerows(all_flat)

    log(f"Verify done: {n_calls} new API call(s). data/verification.csv has {len(all_flat)} row(s).")
    return {"new_calls": n_calls, "rows": len(all_flat)}


def build_side_by_side() -> list[dict]:
    otto_rows = load_otto_rows()
    grouped: dict[tuple[str, str], list[dict]] = {}
    if REEXTRACTED_CSV.exists():
        with REEXTRACTED_CSV.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                grouped.setdefault((row["paper_id"], row["modality"]), []).append(row)

    out: list[dict] = []
    for paper in otto_rows:
        for modality, spec in MODALITIES.items():
            reext_rows = grouped.get((paper.paper_id, modality), [])
            summary_text = reext_rows[0]["summary_paragraph"].strip() if reext_rows else ""
            item_rows = [r for r in reext_rows if json.loads(r["fields_json"] or "{}")]

            for otto_field in spec["otto_fields"]:
                base = {"paper_id": paper.paper_id, "title": paper.title, "modality": modality, "field_name": otto_field, "otto_value": paper.raw.get(otto_field, "")}
                if not reext_rows:
                    out.append({**base, "row_label": "", "reextracted_value": "[not re-extracted yet]", "source_quote": "", "figure_ref": ""})
                    continue
                if summary_text:
                    out.append({**base, "row_label": "(overall summary)", "reextracted_value": summary_text, "source_quote": "", "figure_ref": ""})
                for row in item_rows:
                    out.append({**base, "row_label": row["row_label"], "reextracted_value": render_row_value(row), "source_quote": row["source_quote"], "figure_ref": row["figure_ref"]})
                if not summary_text and not item_rows:
                    out.append({**base, "row_label": "", "reextracted_value": "[model returned no data]", "source_quote": "", "figure_ref": ""})
    return out


# --------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------

app = Flask(__name__)

PAGE_CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; padding: 1.5rem;
       max-width: 1300px; margin-inline: auto; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.4rem; margin-bottom: 0.2rem; }
.subtitle { color: #666; font-size: 0.85rem; margin-bottom: 1.25rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem; }
.card { background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; padding: 0.9rem; }
.card .n { font-size: 1.5rem; font-weight: 700; }
.card .l { font-size: 0.78rem; color: #666; }
.panel { background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; padding: 1rem; margin-bottom: 1.25rem; }
.panel h2 { font-size: 1rem; margin-top: 0; }
.actions { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: end; }
.action label { display: block; font-size: 0.75rem; color: #555; margin-bottom: 0.2rem; }
.action input[type=number] { width: 70px; padding: 0.3rem; }
button { background: #2a5bd7; color: #fff; border: none; border-radius: 6px; padding: 0.5rem 0.9rem;
         font-size: 0.85rem; cursor: pointer; }
button:disabled { background: #aaa; cursor: not-allowed; }
button.secondary { background: #555; }
a.btn { display: inline-block; background: #2a5bd7; color: #fff; text-decoration: none; border-radius: 6px;
        padding: 0.5rem 0.9rem; font-size: 0.85rem; }
#log { background: #111; color: #ddd; font-family: ui-monospace, monospace; font-size: 0.78rem;
       padding: 0.75rem; border-radius: 6px; height: 220px; overflow-y: auto; white-space: pre-wrap; }
.status-badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.75rem;
                font-weight: 600; }
.status-idle { background: #e2f3e6; color: #1c7a34; }
.status-running { background: #fff3d6; color: #9a6c00; }
.warn { color: #a33; font-size: 0.82rem; }
"""

DASHBOARD_JS = """
async function refresh() {
  const r = await fetch('/api/status');
  const s = await r.json();
  document.getElementById('stat-papers').innerText = s.papers_in_folder;
  document.getElementById('stat-matched').innerText = s.matched;
  document.getElementById('stat-reextracted').innerText = s.reextracted_pairs;
  document.getElementById('stat-verified').innerText = s.verified_pairs;
  const badge = document.getElementById('running-badge');
  if (s.running) {
    badge.textContent = 'running: ' + s.running;
    badge.className = 'status-badge status-running';
  } else {
    badge.textContent = 'idle';
    badge.className = 'status-badge status-idle';
  }
  document.querySelectorAll('button.stage-btn').forEach(b => b.disabled = !!s.running);
  document.getElementById('log').textContent = s.log.join('\\n');
  document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;
}
async function run(stage) {
  const form = document.getElementById('form-' + stage);
  const body = form ? new URLSearchParams(new FormData(form)) : new URLSearchParams();
  await fetch('/api/run/' + stage, {method: 'POST', body});
  refresh();
}
setInterval(refresh, 1500);
refresh();
"""


def _dashboard_html() -> str:
    api_key_set = bool(__import__("os").environ.get("ANTHROPIC_API_KEY"))
    key_warning = "" if api_key_set else "<p class='warn'>ANTHROPIC_API_KEY is not set - Re-extract and Verify will fail until you export it.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Otto Pipeline (running locally)</title><style>{PAGE_CSS}</style></head>
<body>
<h1>Otto Re-extraction Pipeline <span id="running-badge" class="status-badge status-idle">idle</span></h1>
<div class="subtitle">Running locally on this machine only, at http://127.0.0.1 - not exposed to the internet.</div>
{key_warning}

<div class="grid">
  <div class="card"><div class="n" id="stat-papers">-</div><div class="l">files in papers/</div></div>
  <div class="card"><div class="n" id="stat-matched">-</div><div class="l">papers matched &amp; cached</div></div>
  <div class="card"><div class="n" id="stat-reextracted">-</div><div class="l">(paper, modality) re-extracted</div></div>
  <div class="card"><div class="n" id="stat-verified">-</div><div class="l">(paper, modality) verified</div></div>
</div>

<div class="panel">
  <h2>1. Scan papers/ folder</h2>
  <p>Matches every file in <code>papers/</code> to a Reference ID in <code>data/otto_output.csv</code>
  (by filename, then DOI, then fuzzy title) and caches its text.</p>
  <button class="stage-btn" onclick="run('scrape')">Scan Papers</button>
</div>

<div class="panel">
  <h2>2. Re-extract</h2>
  <p>Re-runs the 4 prompt templates against every cached paper's text via the Anthropic API.</p>
  <form id="form-reextract" class="actions" onsubmit="event.preventDefault(); run('reextract');">
    <div class="action"><label>Limit (papers)</label><input type="number" name="limit" value="3" min="1"></div>
    <button class="stage-btn" type="submit">Re-extract</button>
  </form>
</div>

<div class="panel">
  <h2>3. Compare &amp; Verify</h2>
  <p>Compare builds a side-by-side view (no API calls). Verify asks the model to judge whether
  each of Otto's original values is correct against the paper text.</p>
  <div class="actions">
    <button class="stage-btn" onclick="run('compare')">Build Comparison</button>
    <form id="form-verify" style="display:contents" onsubmit="event.preventDefault(); run('verify');">
      <div class="action"><label>Limit (papers)</label><input type="number" name="limit" value="3" min="1"></div>
      <button class="stage-btn" type="submit">Verify Accuracy</button>
    </form>
  </div>
</div>

<div class="panel">
  <h2>Results</h2>
  <div class="actions">
    <a class="btn" href="/review">Side-by-side review &rarr;</a>
    <a class="btn" href="/verification">Verification triage &rarr;</a>
  </div>
</div>

<div class="panel">
  <h2>Log</h2>
  <div id="log"></div>
</div>

<script>{DASHBOARD_JS}</script>
</body></html>"""


@app.get("/")
def dashboard() -> Response:
    return Response(_dashboard_html(), mimetype="text/html")


@app.get("/api/status")
def api_status():
    papers_in_folder = len(_paper_files())
    matched = len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.is_dir() else 0
    reextracted_pairs = len(list(RAW_DIR.glob("*__*.json"))) if RAW_DIR.is_dir() else 0
    verified_pairs = len(list(VERIFICATION_RAW_DIR.glob("*__*.json"))) if VERIFICATION_RAW_DIR.is_dir() else 0
    return jsonify(
        {
            "papers_in_folder": papers_in_folder,
            "matched": matched,
            "reextracted_pairs": reextracted_pairs,
            "verified_pairs": verified_pairs,
            "running": current_stage(),
            "log": recent_log(),
        }
    )


def _run_in_background(name: str, fn, *args) -> bool:
    if not try_start_stage(name):
        return False

    def target():
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            log(f"{name}: FATAL ERROR - {exc}")
        finally:
            finish_stage()

    threading.Thread(target=target, daemon=True).start()
    return True


@app.post("/api/run/scrape")
def api_run_scrape():
    ok = _run_in_background("scrape", stage_scrape)
    return jsonify({"started": ok})


@app.post("/api/run/reextract")
def api_run_reextract():
    limit = request.form.get("limit", type=int)
    ok = _run_in_background("reextract", stage_reextract, limit, list(MODALITIES))
    return jsonify({"started": ok})


@app.post("/api/run/verify")
def api_run_verify():
    limit = request.form.get("limit", type=int)
    ok = _run_in_background("verify", stage_verify, limit, list(MODALITIES))
    return jsonify({"started": ok})


@app.post("/api/run/compare")
def api_run_compare():
    def _compare():
        rows = build_side_by_side()
        log(f"Comparison built: {len(rows)} side-by-side row(s). Open /review to view it.")

    ok = _run_in_background("compare", _compare)
    return jsonify({"started": ok})


def _escape(s: str) -> str:
    import html

    return html.escape(s or "", quote=True)


@app.get("/review")
def review() -> Response:
    from collections import defaultdict

    rows = build_side_by_side()
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
            item_html = [f"<tr class='otto-row'><th>Otto</th><td colspan='3'>{_escape(otto_value) or '(empty)'}</td></tr>"]
            for r in field_rows:
                item_html.append(
                    f"<tr><th>{_escape(r['row_label']) or '&mdash;'}</th>"
                    f"<td>{_escape(r['reextracted_value']) or '(empty)'}</td>"
                    f"<td>{_escape(r['source_quote'])}</td><td>{_escape(r['figure_ref'])}</td></tr>"
                )
            fields_html.append(
                f"<div class='panel'><h3>{_escape(field_name)}</h3><table style='width:100%;font-size:0.85rem'>"
                f"<tr><th></th><th>Value</th><th>Quote</th><th>Fig</th></tr>{''.join(item_html)}</table></div>"
            )
        papers_html.append(f"<details><summary><b>{_escape(paper_id)}</b> &mdash; {_escape(title)}</summary>{''.join(fields_html)}</details>")

    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>Review (local)</title>
<style>{PAGE_CSS} table{{border-collapse:collapse}} td,th{{border-bottom:1px solid #eee;padding:4px;text-align:left;vertical-align:top}}</style></head>
<body><h1>Side-by-side review</h1><p><a href="/">&larr; back to dashboard</a></p>{''.join(papers_html) or '<p>No data yet - run Scan Papers, Re-extract, then Build Comparison.</p>'}</body></html>"""
    return Response(body, mimetype="text/html")


@app.get("/verification")
def verification() -> Response:
    if not VERIFICATION_CSV.exists():
        body = "<p>No verification data yet - run Verify Accuracy first.</p><p><a href='/'>&larr; back</a></p>"
        return Response(f"<!doctype html><html><body>{body}</body></html>", mimetype="text/html")

    with VERIFICATION_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    otto_by_paper = {p.paper_id: p for p in load_otto_rows()}

    body_rows = []
    for r in rows:
        otto = otto_by_paper.get(r["paper_id"])
        title = otto.title if otto else ""
        otto_value = otto.raw.get(r["otto_field"], "") if otto else ""
        body_rows.append(
            f"<tr><td>{_escape(r['paper_id'])}</td><td>{_escape(title)}</td><td>{_escape(r['modality'])}</td>"
            f"<td>{_escape(r['otto_field'])}</td><td><b>{_escape(r['verdict'])}</b></td>"
            f"<td>{_escape(r['confidence'])}</td><td>{_escape(otto_value)}</td>"
            f"<td>{_escape(r['explanation'])}</td><td>{_escape(r['evidence_quote'])}</td></tr>"
        )

    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>Verification (local)</title>
<style>{PAGE_CSS} table{{border-collapse:collapse;width:100%;font-size:0.83rem}} td,th{{border-bottom:1px solid #eee;padding:5px;text-align:left;vertical-align:top}}</style></head>
<body><h1>Verification triage</h1><p><a href="/">&larr; back to dashboard</a></p>
<table><tr><th>Paper</th><th>Title</th><th>Modality</th><th>Field</th><th>Verdict</th><th>Conf.</th><th>Otto value</th><th>Explanation</th><th>Quote</th></tr>
{''.join(body_rows)}</table></body></html>"""
    return Response(body, mimetype="text/html")


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, localhost only)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    url = f"http://{args.host}:{args.port}"
    print(f"Starting Otto pipeline web app LOCALLY at {url} (bound to {args.host} - not reachable from outside this machine)")
    print("Open that URL in your browser. Press Ctrl+C to stop.")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
