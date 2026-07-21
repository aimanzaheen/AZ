"""Heuristic extraction of structured fields from research article text."""

from __future__ import annotations

import re
from pathlib import Path

from research_extractor.models import Article
from research_extractor.text_extraction import load_text

# Canonical section name -> aliases seen in the wild (compared lower-case).
SECTION_ALIASES: dict[str, str] = {
    "abstract": "abstract",
    "summary": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "related work": "related_work",
    "literature review": "related_work",
    "materials and methods": "methods",
    "material and methods": "methods",
    "methods": "methods",
    "methodology": "methods",
    "experimental setup": "methods",
    "experiments": "methods",
    "results": "results",
    "results and discussion": "results_and_discussion",
    "findings": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "conclusion and future work": "conclusion",
    "conclusions and future work": "conclusion",
    "limitations": "limitations",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "references": "references",
    "bibliography": "references",
}

_HEADER_NUMBERING_RE = re.compile(
    r"^\s*(?:[0-9]+(?:\.[0-9]+)*\.?|[IVXLC]+\.?)\s+", re.IGNORECASE
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>,;]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_KEYWORDS_RE = re.compile(
    r"(?im)^\s*(?:key\s*words|keywords|index\s+terms)\s*[:\-]\s*(.+)$"
)
_JOURNAL_HINT_RE = re.compile(r"(?i)\bjournal\b|\bproceedings\b|\btransactions\b")
_CITATION_MARKER_RE = re.compile(r"(?m)^\s*(?:\[\d+\]|\d{1,3}\.)\s+")


def _clean_header(line: str) -> str:
    stripped = _HEADER_NUMBERING_RE.sub("", line.strip())
    return stripped.strip(" :\t").lower()


def _find_sections(lines: list[str]) -> list[tuple[int, str, str]]:
    """Return (line_index, canonical_name, raw_header_text) for header lines."""
    found = []
    for i, line in enumerate(lines):
        text = line.strip()
        if not text or len(text) > 60:
            continue
        canonical = SECTION_ALIASES.get(_clean_header(text))
        if canonical:
            found.append((i, canonical, text))
    return found


def _extract_sections(text: str) -> tuple[dict[str, str], int | None]:
    """Split text into named sections. Returns (sections, index of first header line)."""
    lines = text.splitlines()
    headers = _find_sections(lines)
    if not headers:
        return {}, None

    sections: dict[str, str] = {}
    for idx, (line_no, name, _raw) in enumerate(headers):
        start = line_no + 1
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if name in sections:
            sections[name] = f"{sections[name]}\n{body}".strip()
        else:
            sections[name] = body
    return sections, headers[0][0]


def _extract_title_and_authors(
    lines: list[str], header_stop: int | None
) -> tuple[str | None, list[str]]:
    stop = header_stop if header_stop is not None else min(len(lines), 40)
    candidates = [ln.strip() for ln in lines[:stop] if ln.strip()]

    def looks_like_metadata(ln: str) -> bool:
        low = ln.lower()
        if _EMAIL_RE.search(ln) or _DOI_RE.search(ln):
            return True
        if low.startswith(("issn", "doi", "http", "www.", "vol.", "volume", "©")):
            return True
        if _JOURNAL_HINT_RE.search(low) and len(ln) < 120:
            return True
        return False

    title = None
    title_idx = None
    for i, ln in enumerate(candidates):
        if looks_like_metadata(ln):
            continue
        if len(ln) < 15 or len(ln) > 300:
            continue
        if ln.isupper() and len(ln.split()) <= 3:
            continue
        title = ln
        title_idx = i
        break

    authors: list[str] = []
    if title_idx is not None:
        for ln in candidates[title_idx + 1 : title_idx + 4]:
            if looks_like_metadata(ln) and not _EMAIL_RE.search(ln):
                break
            low = ln.lower()
            if low.startswith(("university", "department", "institute", "school of")):
                break
            if "," in ln or " and " in low or "&" in ln:
                parts = re.split(r",|\band\b|&", ln)
                names = [p.strip(" .*0123456789") for p in parts if p.strip(" .*0123456789")]
                authors.extend(n for n in names if 1 <= len(n.split()) <= 5)
                break

    return title, authors


def _strip_keywords_line(abstract: str) -> str:
    return _KEYWORDS_RE.sub("", abstract).strip()


def _extract_abstract(text: str, sections: dict[str, str]) -> str | None:
    if "abstract" in sections and sections["abstract"]:
        return _strip_keywords_line(sections["abstract"])
    match = re.search(
        r"(?is)\babstract\b\s*[:\-]?\s*(.+?)(?=\n\s*(?:keywords|key\s*words|1\.?\s*introduction|introduction)\b|\Z)",
        text,
    )
    if match:
        candidate = _strip_keywords_line(match.group(1).strip())
        if candidate:
            return candidate
    return None


def _extract_keywords(text: str) -> list[str]:
    match = _KEYWORDS_RE.search(text)
    if not match:
        return []
    raw = match.group(1).strip()
    parts = re.split(r"[;,]", raw)
    return [p.strip(" .") for p in parts if p.strip(" .")]


def _extract_journal(lines: list[str], header_stop: int | None) -> str | None:
    stop = header_stop if header_stop is not None else min(len(lines), 15)
    for ln in lines[:stop]:
        text = ln.strip()
        if text and _JOURNAL_HINT_RE.search(text) and len(text) < 150:
            return text
    return None


def _extract_year(text: str) -> str | None:
    head = text[:2000]
    for line in head.splitlines():
        if re.search(r"(?i)received|accepted|published|copyright|©", line):
            m = _YEAR_RE.search(line)
            if m:
                return m.group(0)
    m = _YEAR_RE.search(head)
    return m.group(0) if m else None


def _extract_references(sections: dict[str, str]) -> list[str]:
    raw = sections.get("references")
    if not raw:
        return []
    if _CITATION_MARKER_RE.search(raw):
        parts = _CITATION_MARKER_RE.split(raw)
        return [p.strip().replace("\n", " ") for p in parts if p.strip()]
    entries = [ln.strip() for ln in raw.splitlines() if len(ln.strip()) > 5]
    return entries


def parse_text(text: str, source_file: str = "<text>") -> Article:
    """Parse raw article text into a structured Article."""
    lines = text.splitlines()
    sections, header_stop = _extract_sections(text)

    title, authors = _extract_title_and_authors(lines, header_stop)
    abstract = _extract_abstract(text, sections)
    keywords = _extract_keywords(text)
    doi_match = _DOI_RE.search(text)
    doi = doi_match.group(0).rstrip(").,;") if doi_match else None
    emails = list(dict.fromkeys(_EMAIL_RE.findall(text)))
    journal = _extract_journal(lines, header_stop)
    year = _extract_year(text)
    references = _extract_references(sections)

    body_sections = {k: v for k, v in sections.items() if k != "references"}
    if abstract and "abstract" in body_sections:
        body_sections["abstract"] = abstract

    return Article(
        source_file=source_file,
        title=title,
        authors=authors,
        emails=emails,
        abstract=abstract,
        keywords=keywords,
        doi=doi,
        journal=journal,
        publication_year=year,
        sections=body_sections,
        references=references,
        word_count=len(text.split()),
    )


def parse_article(path: str | Path) -> Article:
    """Load a PDF/TXT/MD article file and extract structured information."""
    path = Path(path)
    text = load_text(path)
    return parse_text(text, source_file=str(path))
