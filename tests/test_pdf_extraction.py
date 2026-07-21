from pathlib import Path

import pytest

from research_extractor.parser import parse_article

reportlab = pytest.importorskip("reportlab")

FIXTURE_TXT = Path(__file__).parent / "fixtures" / "sample_article.txt"


def _build_pdf(tmp_path: Path) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "sample_article.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    text_obj = c.beginText(40, 740)
    text_obj.setFont("Helvetica", 10)
    for line in FIXTURE_TXT.read_text().splitlines():
        if text_obj.getY() < 40:
            c.drawText(text_obj)
            c.showPage()
            text_obj = c.beginText(40, 740)
            text_obj.setFont("Helvetica", 10)
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.save()
    return pdf_path


def test_parse_article_reads_pdf_file(tmp_path):
    pdf_path = _build_pdf(tmp_path)
    article = parse_article(pdf_path)

    assert article.doi == "10.1234/jamlr.2024.00567"
    assert article.title is not None
    assert "methods" in article.sections
