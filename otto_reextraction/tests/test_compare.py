import common
import compare


def make_otto_row(**overrides):
    raw = {
        "Anatomical connections": "ZI -> PVT (anterograde AAV)",
        "Verification of Zona Incerta Targeting": "Histological verification confined to ZI",
        "Photometry Data": "",
        "Electrophysiology Data": "",
        "Purpose of this circuit": "Mediates feeding",
        "Necessity vs. sufficiency or Both": "Both",
    }
    raw.update(overrides)
    return common.OttoRow(paper_id="AIM-1", doi="10.1/x", title="A Test Paper", raw=raw)


def test_build_side_by_side_no_reextraction_yet():
    rows = compare.build_side_by_side([make_otto_row()], grouped={})
    anatomical_rows = [r for r in rows if r["field_name"] == "Anatomical connections"]
    assert len(anatomical_rows) == 1
    assert anatomical_rows[0]["reextracted_value"] == "[not re-extracted yet - run reextract.py]"
    assert anatomical_rows[0]["otto_value"] == "ZI -> PVT (anterograde AAV)"


def test_build_side_by_side_with_item_rows_and_summary():
    grouped = {
        ("AIM-1", "anatomical"): [
            {
                "row_label": "ZI -> PVT",
                "figure_ref": "Fig 1",
                "source_quote": "we traced ZI to PVT",
                "fields_json": '{"Source region": "ZI", "Target region": "PVT"}',
                "summary_paragraph": "ZI -> PVT (Vgat+, necessity).",
            }
        ]
    }
    rows = compare.build_side_by_side([make_otto_row()], grouped)
    anatomical_rows = [r for r in rows if r["field_name"] == "Anatomical connections"]
    # one summary row + one item row
    assert len(anatomical_rows) == 2
    labels = {r["row_label"] for r in anatomical_rows}
    assert labels == {"(overall summary)", "ZI -> PVT"}
    item_row = next(r for r in anatomical_rows if r["row_label"] == "ZI -> PVT")
    assert item_row["figure_ref"] == "Fig 1"
    assert item_row["source_quote"] == "we traced ZI to PVT"
    assert "Source region: ZI" in item_row["reextracted_value"]

    # "Verification of Zona Incerta Targeting" shares the anatomical modality's re-extraction
    verification_rows = [r for r in rows if r["field_name"] == "Verification of Zona Incerta Targeting"]
    assert len(verification_rows) == 2
    assert all(r["otto_value"] == "Histological verification confined to ZI" for r in verification_rows)


def test_build_side_by_side_covers_all_six_otto_fields_per_paper():
    rows = compare.build_side_by_side([make_otto_row()], grouped={})
    field_names = {r["field_name"] for r in rows}
    assert field_names == {
        "Anatomical connections",
        "Verification of Zona Incerta Targeting",
        "Photometry Data",
        "Electrophysiology Data",
        "Purpose of this circuit",
        "Necessity vs. sufficiency or Both",
    }
