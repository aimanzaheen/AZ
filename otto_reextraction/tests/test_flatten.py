import flatten


def test_flatten_raw_with_rows():
    raw = {
        "summary_paragraph": "ZI -> PVT (Vgat+, necessity).",
        "rows": [
            {
                "row_label": "ZI (Vgat+) -> PVT",
                "figure_ref": "Fig 1",
                "source_quote": "we traced ZI to PVT",
                "fields": {"Source region": "ZI", "Target region": "PVT"},
            }
        ],
    }
    rows = flatten.flatten_raw("AIM-1", "anatomical", raw)
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "AIM-1"
    assert rows[0]["modality"] == "anatomical"
    assert rows[0]["row_label"] == "ZI (Vgat+) -> PVT"
    assert rows[0]["figure_ref"] == "Fig 1"
    assert rows[0]["summary_paragraph"] == "ZI -> PVT (Vgat+, necessity)."
    import json

    assert json.loads(rows[0]["fields_json"]) == {"Source region": "ZI", "Target region": "PVT"}


def test_flatten_raw_with_no_rows_falls_back_to_placeholder():
    raw = {"summary_paragraph": "This circuit mediates feeding.", "rows": []}
    rows = flatten.flatten_raw("AIM-2", "functional", raw)
    assert len(rows) == 1
    assert rows[0]["fields_json"] == "{}"
    assert rows[0]["summary_paragraph"] == "This circuit mediates feeding."


def test_render_row_value_combines_label_and_fields():
    row = {"row_label": "ZI -> PVT", "fields_json": '{"Source region": "ZI", "Target region": "PVT"}'}
    value = flatten.render_row_value(row)
    assert value.startswith("ZI -> PVT")
    assert "Source region: ZI" in value
    assert "Target region: PVT" in value
