import csv
import json
import re
from pathlib import Path

import common
import render_dashboard


def test_build_paper_rows_matched_and_unmatched():
    otto_rows = [
        common.OttoRow(
            paper_id="AIM-1",
            doi="10.1/x",
            title="A Test Paper",
            raw={
                "Title": "A Test Paper",
                "Authors": "Doe J",
                "Authors and Publication Year": "Jane Doe; 2020",
                "DOI": "10.1/x",
                "Animal Model": "Mouse",
                "Anatomical connections": "ZI -> PVT",
                "Photometry Data": "",
                "Electrophysiology Data": "",
                "Purpose of this circuit": "Feeding",
            },
        ),
        common.OttoRow(
            paper_id="AIM-2",
            doi=None,
            title="Unmatched Paper",
            raw={
                "Title": "Unmatched Paper",
                "Authors": "",
                "Authors and Publication Year": "",
                "DOI": "",
                "Animal Model": "",
                "Anatomical connections": "",
                "Photometry Data": "",
                "Electrophysiology Data": "",
                "Purpose of this circuit": "",
            },
        ),
    ]
    manifest = {"AIM-1": {"paper_id": "AIM-1", "confidence": "1.0", "match_method": "filename"}}

    rows = render_dashboard.build_paper_rows(otto_rows, manifest)

    assert rows[0]["id"] == "AIM-1"
    assert rows[0]["matched"] is True
    assert rows[0]["confidence"] == 1.0
    assert rows[0]["year"] == "2020"
    assert rows[0]["has_anatomical"] is True
    assert rows[0]["has_photometry"] is False

    assert rows[1]["id"] == "AIM-2"
    assert rows[1]["matched"] is False
    assert rows[1]["confidence"] is None


def test_load_manifest_and_verification_missing_files_return_empty(tmp_path):
    assert render_dashboard.load_manifest(tmp_path / "nope.csv") == {}
    assert render_dashboard.load_verification(tmp_path / "nope.csv") == []


def test_run_end_to_end_produces_valid_html_with_embedded_json(tmp_path):
    otto_csv = tmp_path / "otto_output.csv"
    with otto_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "Reference ID",
                "Title",
                "Authors",
                "DOI",
                "Authors and Publication Year",
                "Animal Model",
                "Anatomical connections",
                "Verification of Zona Incerta Targeting",
                "Photometry Data",
                "Electrophysiology Data",
                "Purpose of this circuit",
                "Necessity vs. sufficiency or Both",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Reference ID": "AIM-1",
                "Title": "A Test Paper",
                "Authors": "Doe J",
                "DOI": "10.1/x",
                "Authors and Publication Year": "Jane Doe; 2020",
                "Animal Model": "Mouse",
                "Anatomical connections": "ZI -> PVT",
                "Verification of Zona Incerta Targeting": "",
                "Photometry Data": "",
                "Electrophysiology Data": "",
                "Purpose of this circuit": "Feeding",
                "Necessity vs. sufficiency or Both": "",
            }
        )

    manifest_csv = tmp_path / "scrape_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["paper_id", "confidence", "match_method"])
        writer.writeheader()
        writer.writerow({"paper_id": "AIM-1", "confidence": "1.0", "match_method": "filename"})

    verification_csv = tmp_path / "verification.csv"
    with verification_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["paper_id", "modality", "otto_field", "verdict", "confidence", "explanation", "evidence_quote"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "paper_id": "AIM-1",
                "modality": "anatomical",
                "otto_field": "Anatomical connections",
                "verdict": "match",
                "confidence": "high",
                "explanation": "Matches text.",
                "evidence_quote": "ZI projects to PVT",
            }
        )

    out_path = tmp_path / "dashboard.html"
    args = render_dashboard.build_parser().parse_args(
        [
            "--otto-csv",
            str(otto_csv),
            "--manifest-csv",
            str(manifest_csv),
            "--verification-csv",
            str(verification_csv),
            "--out",
            str(out_path),
        ]
    )
    exit_code = render_dashboard.run(args)
    assert exit_code == 0
    assert out_path.exists()

    html_out = out_path.read_text(encoding="utf-8")
    assert "__DATA_JSON__" not in html_out
    assert "__VERIFICATION_JSON__" not in html_out

    data_match = re.search(r"const DATA = (\[.*?\]);\nconst VDATA", html_out, re.S)
    vdata_match = re.search(r"const VDATA = (\[.*?\]);\n\nconst generatedAt", html_out, re.S)
    assert data_match is not None
    assert vdata_match is not None

    data = json.loads(data_match.group(1))
    vdata = json.loads(vdata_match.group(1))
    assert data == [
        {
            "id": "AIM-1",
            "title": "A Test Paper",
            "authors": "Doe J",
            "year": "2020",
            "doi": "10.1/x",
            "animal_model": "Mouse",
            "anatomical_text": "ZI -> PVT",
            "photometry_text": "",
            "ephys_text": "",
            "has_anatomical": True,
            "has_photometry": False,
            "has_ephys": False,
            "has_functional": True,
            "matched": True,
            "confidence": 1.0,
            "match_method": "filename",
        }
    ]
    assert len(vdata) == 1
    assert vdata[0]["verdict"] == "match"


def test_run_without_verification_file_still_succeeds(tmp_path):
    otto_csv = tmp_path / "otto_output.csv"
    with otto_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Reference ID", "Title"])
        writer.writeheader()
        writer.writerow({"Reference ID": "AIM-1", "Title": "A Test Paper"})

    out_path = tmp_path / "dashboard.html"
    args = render_dashboard.build_parser().parse_args(
        [
            "--otto-csv",
            str(otto_csv),
            "--manifest-csv",
            str(tmp_path / "does_not_exist.csv"),
            "--verification-csv",
            str(tmp_path / "also_missing.csv"),
            "--out",
            str(out_path),
        ]
    )
    assert render_dashboard.run(args) == 0
    html_out = out_path.read_text(encoding="utf-8")
    assert "const VDATA = [];" in html_out
