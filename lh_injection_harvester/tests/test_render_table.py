import argparse

import common
import render_table


SAMPLE_ROWS = [
    {
        "pmid": "1",
        "paper_name": "A paper about <LH> tracing",
        "year": "2021",
        "journal": "J Neurosci",
        "experiment_index": "1",
        "transmission_type": "retrograde",
        "volume_injected_lh": "50 nL",
        "stereotaxic_coordinates": "AP -1.4, ML -1.0, DV -5.2",
        "lha_subdivided": "single/undivided site",
        "tracer": "CTB-Alexa555",
        "projections_found": "PVT, ARC",
        "survival_time": "7 days",
        "figure_ref": "Fig 1",
        "source_quote": "we injected CTB",
        "text_source": "pmc_fulltext",
    },
    {
        "pmid": "2",
        "paper_name": "Another paper",
        "year": "2019",
        "journal": "Neuron",
        "experiment_index": "1",
        "transmission_type": "anterograde",
        "volume_injected_lh": "not explicitly stated",
        "stereotaxic_coordinates": "not explicitly stated",
        "lha_subdivided": "rostral vs caudal LH",
        "tracer": "BDA",
        "projections_found": "not explicitly stated",
        "survival_time": "not explicitly stated",
        "figure_ref": "",
        "source_quote": "",
        "text_source": "pubmed_abstract",
    },
]


def test_render_escapes_html_and_includes_all_columns():
    out = render_table.render(SAMPLE_ROWS, common.DEFAULT_QUERY)
    assert "&lt;LH&gt;" in out  # escaped, not raw injected HTML
    assert "CTB-Alexa555" in out
    assert "AP -1.4, ML -1.0, DV -5.2" in out
    assert "PVT, ARC" in out
    assert "pubmed.ncbi.nlm.nih.gov/1/" in out
    assert "pubmed.ncbi.nlm.nih.gov/2/" in out
    assert "Volume injected into LH" in out
    assert "Stereotaxic coordinates" in out
    assert "Survival time before perfusion" in out
    assert "Transmission type" in out
    assert "LHA subdivided?" in out
    assert "rostral vs caudal LH" in out


def test_render_shows_transmission_type_badges():
    out = render_table.render(SAMPLE_ROWS, common.DEFAULT_QUERY)
    assert "badge-retro" in out
    assert "badge-antero" in out
    assert "retrograde" in out
    assert "anterograde" in out


def test_render_marks_not_explicitly_stated_cells():
    out = render_table.render(SAMPLE_ROWS, common.DEFAULT_QUERY)
    assert out.count("class='na'") >= 3


def test_run_writes_html_file(tmp_path):
    csv_path = tmp_path / "extracted.csv"
    common.write_csv(csv_path, common.EXTRACTED_FIELDS, SAMPLE_ROWS)
    out_path = tmp_path / "table.html"

    args = argparse.Namespace(extracted_csv=str(csv_path), out=str(out_path), query="test query")
    rc = render_table.run(args)

    assert rc == 0
    assert out_path.exists()
    assert "test query" in out_path.read_text()


def test_run_errors_when_extracted_csv_missing(tmp_path, capsys):
    args = argparse.Namespace(
        extracted_csv=str(tmp_path / "missing.csv"), out=str(tmp_path / "table.html"), query="q"
    )
    rc = render_table.run(args)
    assert rc == 1
