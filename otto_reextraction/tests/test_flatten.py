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


def test_render_modality_summary_with_no_rows():
    assert "no re-extraction available" in flatten.render_modality_summary([])


def test_render_modality_summary_includes_overall_summary_and_items():
    rows = [
        {
            "row_label": "ZI -> PVT",
            "figure_ref": "Fig 1",
            "fields_json": '{"Source region": "ZI"}',
            "summary_paragraph": "ZI drives PVT inhibition.",
        },
        {
            "row_label": "PL -> ZI",
            "figure_ref": "",
            "fields_json": '{"Source region": "PL"}',
            "summary_paragraph": "ZI drives PVT inhibition.",
        },
    ]
    text = flatten.render_modality_summary(rows)
    assert "Overall summary: ZI drives PVT inhibition." in text
    assert "ZI -> PVT: Source region: ZI [Fig 1]" in text
    assert "PL -> ZI: Source region: PL" in text


def test_flatten_verification():
    raw = {
        "verdicts": [
            {
                "otto_field": "Anatomical connections",
                "verdict": "partial",
                "confidence": "medium",
                "explanation": "Otto missed the PL->ZI pathway.",
                "evidence_quote": "we also traced PL inputs to ZI",
            }
        ]
    }
    rows = flatten.flatten_verification("AIM-1", "anatomical", raw)
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "AIM-1"
    assert rows[0]["modality"] == "anatomical"
    assert rows[0]["verdict"] == "partial"
    assert rows[0]["otto_field"] == "Anatomical connections"


def test_verdict_severity_orders_mismatch_first():
    order = sorted(["match", "mismatch", "unverifiable", "partial"], key=lambda v: flatten.VERDICT_SEVERITY[v])
    assert order == ["mismatch", "partial", "unverifiable", "match"]
