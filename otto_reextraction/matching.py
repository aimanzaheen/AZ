"""Match a locally supplied paper file to a row in otto_output.csv.

Tries, in order of confidence: exact filename == Reference ID, a DOI found
in the paper text matching a row's DOI, then a fuzzy title match against
the first page of text. Pure functions over plain strings/OttoRow objects
so this is fully unit-testable without real PDFs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import common

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>,;]+", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")

TITLE_MATCH_THRESHOLD = 0.5


@dataclass
class MatchResult:
    row: common.OttoRow | None
    method: str  # "filename" | "doi" | "title" | "unmatched"
    confidence: float  # 0-1


def _normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _words(s: str) -> set[str]:
    return set(_WORD_RE.findall(s.lower()))


def filename_match(filename_stem: str, otto_rows: list[common.OttoRow]) -> common.OttoRow | None:
    target = _normalize_key(filename_stem)
    if not target:
        return None
    for row in otto_rows:
        if _normalize_key(row.paper_id) == target:
            return row
    return None


def doi_match(pdf_text: str, otto_rows: list[common.OttoRow]) -> common.OttoRow | None:
    found_dois = {common.normalize_doi(m.group(0).rstrip(").,;")) for m in _DOI_RE.finditer(pdf_text)}
    found_dois.discard(None)
    if not found_dois:
        return None
    by_doi = {row.doi: row for row in otto_rows if row.doi}
    for doi in found_dois:
        if doi in by_doi:
            return by_doi[doi]
    return None


def title_match(
    pdf_text: str, otto_rows: list[common.OttoRow], threshold: float = TITLE_MATCH_THRESHOLD
) -> tuple[common.OttoRow | None, float]:
    haystack_words = _words(pdf_text[:4000])
    if not haystack_words:
        return None, 0.0

    best_row, best_score = None, 0.0
    for row in otto_rows:
        title_words = _words(row.title)
        if not title_words:
            continue
        overlap = len(title_words & haystack_words)
        score = overlap / len(title_words)
        if score > best_score:
            best_row, best_score = row, score

    if best_row is not None and best_score >= threshold:
        return best_row, best_score
    return None, best_score


def match_paper(filename_stem: str, pdf_text: str, otto_rows: list[common.OttoRow]) -> MatchResult:
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
