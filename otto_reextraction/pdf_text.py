"""Minimal local PDF/TXT text loader, self-contained so otto_reextraction doesn't
depend on the sibling research_extractor package."""

from __future__ import annotations

from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES


def load_text(path: str | Path) -> str | None:
    """Return the plain text contents of a local paper file, or None if extraction fails."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in PDF_SUFFIXES:
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
        pass  # pdfplumber missing or unusable here - fall back to pypdf

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text or None
    except Exception:
        return None
