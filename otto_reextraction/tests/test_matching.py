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


# --- Regression tests for two real bugs found via adversarial/synthetic ground truth ---
# (see git history: over-aggressive filename normalization + non-deterministic DOI matching
# were both silently producing wrong matches at high reported confidence.)

COLLISION_ROWS = [
    make_row("AIM-1", "10.1000/aim1", "The real paper being scanned"),
    make_row("AIM-11", "10.1000/aim11", "A different, unrelated corpus paper"),
]


def test_filename_match_does_not_collide_on_duplicate_download_suffix():
    # A browser-downloaded duplicate is commonly suffixed like "AIM-1(1).pdf". Naive
    # normalization that strips ALL punctuation collapses this to "aim11", which
    # collided with a completely different paper, "AIM-11".
    assert matching.filename_match("AIM-1(1)", COLLISION_ROWS) is None
    assert matching.filename_match("AIM-1 (2)", COLLISION_ROWS) is None
    assert matching.filename_match("AIM-1_copy2", COLLISION_ROWS) is None


def test_filename_match_still_ignores_missing_separator():
    # The fix must not regress the legitimate case of a filename that just omits the
    # dash, e.g. "AIM50.pdf" for "AIM-50".
    rows = [make_row("AIM-50", None, "")]
    assert matching.filename_match("AIM50", rows).paper_id == "AIM-50"
    assert matching.filename_match("aim_50", rows).paper_id == "AIM-50"


def test_doi_match_prefers_paper_own_doi_over_a_cited_doi():
    # This corpus is 71 interrelated papers about the same brain circuit that constantly
    # cite each other. A paper's own DOI is near the top; cited papers' DOIs show up in
    # the References section. doi_match must not pick up a citation instead of the
    # paper's own identity, no matter what Python's string-hash seed happens to be.
    paper_text = (
        "The Real Paper Being Scanned\nSome Author\nDOI: 10.1000/aim1\n\n"
        + ("filler introduction and methods text " * 400)
        + "\nReferences\n1. Some Other Author. DOI: 10.1000/aim11\n"
    )
    for _ in range(20):  # repeat: a set-based implementation would occasionally pick wrong
        result = matching.doi_match(paper_text, COLLISION_ROWS)
        assert result is not None
        assert result.paper_id == "AIM-1"


def test_doi_match_ignores_a_doi_that_only_appears_in_the_references_section():
    # If the paper's OWN doi isn't found near the top at all, don't fall back to
    # whatever DOI happens to appear later - that's how a References-section citation
    # got mistaken for the paper's own identity in the first place.
    paper_text = (
        "A paper with no DOI printed near the top.\n"
        + ("filler introduction and methods text " * 400)
        + "\nReferences\n1. Some Other Author. DOI: 10.1000/aim11\n"
    )
    assert matching.doi_match(paper_text, COLLISION_ROWS) is None


def test_match_paper_end_to_end_does_not_silently_mismatch_on_collision_filename():
    # Full match_paper cascade: a colliding filename with no usable DOI/title signal
    # must fall through to "unmatched" rather than silently claiming AIM-11 at 1.0.
    result = matching.match_paper("AIM-1(1)", "some unrelated filler text", COLLISION_ROWS)
    assert result.method == "unmatched"
    assert result.row is None
