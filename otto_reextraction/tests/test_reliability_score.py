import reliability_score as rs


def make_row(paper_id="AIM-1", modality="anatomical", field="Anatomical connections", verdict="match", confidence="high"):
    return {"paper_id": paper_id, "modality": modality, "otto_field": field, "verdict": verdict, "confidence": confidence}


def test_all_high_confidence_matches_scores_near_ten_after_shrinkage():
    rows = [make_row(paper_id=f"AIM-{i}") for i in range(50)]
    result = rs.compute_reliability(rows)
    assert result["raw_score_10"] == 10.0
    # with 50 high-confidence matches, shrinkage toward 5.0 should barely move the score
    assert result["shrunk_score_10"] >= 9.0


def test_all_mismatches_scores_near_zero_after_shrinkage():
    rows = [make_row(paper_id=f"AIM-{i}", verdict="mismatch") for i in range(50)]
    result = rs.compute_reliability(rows)
    assert result["raw_score_10"] == 0.0
    assert result["shrunk_score_10"] <= 1.0


def test_small_sample_shrinks_toward_five():
    # A single high-confidence match with a small n should NOT score a full 10/10 -
    # shrinkage should pull it well below the raw score.
    rows = [make_row()]
    result = rs.compute_reliability(rows)
    assert result["raw_score_10"] == 10.0
    assert result["shrunk_score_10"] < 8.0
    assert result["shrunk_score_10"] > 5.0


def test_unverifiable_rows_are_excluded_from_scoring_but_counted():
    rows = [make_row(), make_row(verdict="unverifiable", confidence="")]
    result = rs.compute_reliability(rows)
    assert result["n_field_verdicts"] == 2
    assert result["n_scored_verdicts"] == 1
    assert result["verdict_counts"]["unverifiable"] == 1


def test_partial_scores_half_a_match():
    high_match = rs.compute_reliability([make_row(verdict="match")])
    high_partial = rs.compute_reliability([make_row(verdict="partial")])
    high_mismatch = rs.compute_reliability([make_row(verdict="mismatch")])
    assert high_mismatch["raw_score_10"] < high_partial["raw_score_10"] < high_match["raw_score_10"]
    assert high_partial["raw_score_10"] == 5.0


def test_confidence_weighting_pulls_low_confidence_verdicts_toward_less_influence():
    # A high-confidence mismatch should drag the score down more than a
    # low-confidence mismatch mixed with a high-confidence match.
    mostly_bad = rs.compute_reliability([make_row(verdict="mismatch", confidence="high"), make_row(verdict="match", confidence="low")])
    mostly_good = rs.compute_reliability([make_row(verdict="mismatch", confidence="low"), make_row(verdict="match", confidence="high")])
    assert mostly_good["raw_score_10"] > mostly_bad["raw_score_10"]


def test_empty_input_has_no_raw_score():
    result = rs.compute_reliability([])
    assert result["raw_score_10"] is None
    assert result["shrunk_score_10"] == 5.0  # pure prior, no evidence either way
    assert result["n_papers_audited"] == 0


def test_field_and_modality_breakdowns_are_populated():
    rows = [
        make_row(field="Anatomical connections", modality="anatomical", verdict="partial"),
        make_row(field="Photometry Data", modality="photometry", verdict="match"),
    ]
    result = rs.compute_reliability(rows)
    assert result["field_counts"]["Anatomical connections"]["partial"] == 1
    assert result["modality_counts"]["photometry"]["match"] == 1


def test_more_evidence_shrinks_less_than_little_evidence():
    small = rs.compute_reliability([make_row() for _ in range(2)])
    large = rs.compute_reliability([make_row() for _ in range(200)])
    # both are all-matches (raw=10.0), but the larger sample should sit closer to 10
    assert large["shrunk_score_10"] > small["shrunk_score_10"]
