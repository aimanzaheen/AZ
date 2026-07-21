"""Load raw text out of a research article file (PDF, TXT, or Markdown)."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


class UnsupportedFileType(ValueError):
    pass


def load_text(path: str | Path) -> str:
    """Return the plain text contents of a supported article file."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_text(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    raise UnsupportedFileType(
        f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def _extract_pdf_text(path: Path) -> str:
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
        # pdfplumber missing, or unusable in this environment - fall back to pypdf.
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError as exc:
        raise ImportError(
            "PDF extraction requires 'pdfplumber' or 'pypdf'. "
            "Install with: pip install pdfplumber pypdf"
        ) from exc
