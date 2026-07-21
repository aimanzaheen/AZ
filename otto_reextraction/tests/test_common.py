from pathlib import Path

import common

FIXTURE = Path(__file__).parent / "fixtures" / "mini_otto.csv"


def test_normalize_doi_strips_url_prefix():
    assert common.normalize_doi("https://doi.org/10.1000/test.001") == "10.1000/test.001"
    assert common.normalize_doi("10.1000/test.001") == "10.1000/test.001"
    assert common.normalize_doi("") is None
    assert common.normalize_doi("  ") is None


def test_load_otto_rows():
    rows = common.load_otto_rows(FIXTURE)
    assert len(rows) == 1
    row = rows[0]
    assert row.paper_id == "AIM-TEST"
    assert row.doi == "10.1000/test.001"
    assert row.title == "A Test Paper About the Zona Incerta"
    assert row.raw["Anatomical connections"] == "ZI (VGAT+) -> PVT (anterograde AAV)"


def test_modalities_cover_all_extraction_columns():
    covered = {field for spec in common.MODALITIES.values() for field in spec["otto_fields"]}
    assert covered == {
        "Anatomical connections",
        "Verification of Zona Incerta Targeting",
        "Photometry Data",
        "Electrophysiology Data",
        "Purpose of this circuit",
        "Necessity vs. sufficiency or Both",
    }


def test_prompts_load_for_every_modality():
    for modality in common.MODALITIES:
        text = common.load_prompt(modality)
        assert text.strip()
