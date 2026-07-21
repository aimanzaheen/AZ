import common
import matching


def make_row(paper_id, doi, title):
    return common.OttoRow(paper_id=paper_id, doi=doi, title=title, raw={})


ROWS = [
    make_row("AIM-50", "10.1126/science.aam7100", "Rapid binge-like eating and body weight gain"),
    make_row("AIM-41", "10.1101/2025.06.05.658170", "Bidirectional modulation of accumbens dopamine"),
]


def test_filename_match_is_case_and_punctuation_insensitive():
    assert matching.filename_match("aim_50", ROWS).paper_id == "AIM-50"
    assert matching.filename_match("AIM-50", ROWS).paper_id == "AIM-50"
    assert matching.filename_match("AIM50", ROWS).paper_id == "AIM-50"
    assert matching.filename_match("nope", ROWS) is None


def test_doi_match_finds_doi_anywhere_in_text():
    text = "Some preamble.\nDOI: 10.1101/2025.06.05.658170\nMore text."
    result = matching.doi_match(text, ROWS)
    assert result.paper_id == "AIM-41"


def test_doi_match_returns_none_when_no_doi_present():
    assert matching.doi_match("no doi here at all", ROWS) is None


def test_title_match_above_threshold():
    text = "Rapid binge-like eating and body weight gain driven by zona incerta neurons. Introduction..."
    row, score = matching.title_match(text, ROWS)
    assert row.paper_id == "AIM-50"
    assert score >= matching.TITLE_MATCH_THRESHOLD


def test_title_match_below_threshold_returns_none():
    row, score = matching.title_match("completely unrelated content about crustaceans", ROWS)
    assert row is None


def test_match_paper_prefers_filename_over_doi_and_title():
    text = f"DOI: 10.1101/2025.06.05.658170 Rapid binge-like eating and body weight gain"
    result = matching.match_paper("AIM-50", text, ROWS)
    assert result.method == "filename"
    assert result.row.paper_id == "AIM-50"


def test_match_paper_prefers_doi_over_title():
    text = "DOI: 10.1101/2025.06.05.658170 Rapid binge-like eating and body weight gain"
    result = matching.match_paper("unnamed_file", text, ROWS)
    assert result.method == "doi"
    assert result.row.paper_id == "AIM-41"


def test_match_paper_unmatched():
    result = matching.match_paper("random_file", "totally unrelated text about crustaceans", ROWS)
    assert result.method == "unmatched"
    assert result.row is None
