from pathlib import Path

import pytest

import pdf_text

reportlab = pytest.importorskip("reportlab")


def _build_pdf(tmp_path: Path, text: str) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    t = c.beginText(40, 740)
    t.setFont("Helvetica", 10)
    for line in text.splitlines():
        t.textLine(line)
    c.drawText(t)
    c.save()
    return pdf_path


def test_load_text_extracts_pdf(tmp_path):
    pdf_path = _build_pdf(tmp_path, "Full paper text about the zona incerta circuit.\nDOI: 10.1000/test.001")
    text = pdf_text.load_text(pdf_path)
    assert "zona incerta circuit" in text
    assert "10.1000/test.001" in text


def test_load_text_reads_plain_txt(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("hello world")
    assert pdf_text.load_text(txt_path) == "hello world"


def test_load_text_returns_none_for_unsupported_suffix(tmp_path):
    docx_path = tmp_path / "sample.docx"
    docx_path.write_bytes(b"not really a docx")
    assert pdf_text.load_text(docx_path) is None
