import common
import render_verification


def test_render_produces_valid_html_with_expected_rows():
    rows = [
        {
            "paper_id": "AIM-1",
            "modality": "anatomical",
            "otto_field": "Anatomical connections",
            "verdict": "mismatch",
            "confidence": "high",
            "explanation": "No such pathway reported.",
            "evidence_quote": "not found in text",
        }
    ]
    otto_by_paper = {
        "AIM-1": common.OttoRow(
            paper_id="AIM-1",
            doi="10.1/x",
            title="A Test Paper",
            raw={"Anatomical connections": "ZI -> PVT"},
        )
    }
    html_out = render_verification.render(rows, otto_by_paper)
    assert "<table" in html_out
    assert "AIM-1" in html_out
    assert "A Test Paper" in html_out
    assert "badge-mismatch" in html_out
    assert "ZI -&gt; PVT" in html_out  # otto value is HTML-escaped
