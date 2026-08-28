import json
import time

import pytest

import webapp


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Point every webapp.* path constant at a scratch directory so tests never
    touch the real otto_reextraction/data or papers/ folders."""
    otto_csv = tmp_path / "otto_output.csv"
    otto_csv.write_text(
        '"Reference ID","Title","DOI","Anatomical connections","Verification of Zona Incerta Targeting",'
        '"Photometry Data","Electrophysiology Data","Purpose of this circuit","Necessity vs. sufficiency or Both"\n'
        '"AIM-TEST","A Test Paper About the Zona Incerta","https://doi.org/10.1000/test.001",'
        '"ZI -> PVT","Verified","","","Mediates feeding","Both"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(webapp, "OTTO_CSV", otto_csv)
    monkeypatch.setattr(webapp, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(webapp, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(webapp, "RAW_DIR", tmp_path / "reextracted_raw")
    monkeypatch.setattr(webapp, "VERIFICATION_RAW_DIR", tmp_path / "verification_raw")
    monkeypatch.setattr(webapp, "REEXTRACTED_CSV", tmp_path / "reextracted.csv")
    monkeypatch.setattr(webapp, "VERIFICATION_CSV", tmp_path / "verification.csv")
    monkeypatch.setattr(webapp, "SCRAPE_MANIFEST_CSV", tmp_path / "scrape_manifest.csv")
    yield tmp_path


def test_load_otto_rows(isolated_paths):
    rows = webapp.load_otto_rows(webapp.OTTO_CSV)
    assert len(rows) == 1
    assert rows[0].paper_id == "AIM-TEST"
    assert rows[0].doi == "10.1000/test.001"


def test_matching_cascade():
    rows = [webapp.OttoRow(paper_id="AIM-TEST", doi="10.1000/test.001", title="A Test Paper About the Zona Incerta", raw={})]
    assert webapp.match_paper("AIM-TEST", "irrelevant text", rows).method == "filename"
    assert webapp.match_paper("unnamed", "DOI: 10.1000/test.001", rows).method == "doi"
    assert webapp.match_paper("unnamed", "A Test Paper About the Zona Incerta, intro...", rows).method == "title"
    assert webapp.match_paper("unnamed", "totally unrelated crustacean text", rows).method == "unmatched"


def test_filename_match_does_not_collide_on_duplicate_download_suffix():
    # Regression test: normalization used to strip ALL punctuation, so a browser's
    # duplicate-download suffix "AIM-1(1).pdf" collapsed to "aim11" and collided with
    # a completely different paper, "AIM-11" - a silent wrong match at confidence 1.0.
    rows = [
        webapp.OttoRow(paper_id="AIM-1", doi="10.1000/aim1", title="", raw={}),
        webapp.OttoRow(paper_id="AIM-11", doi="10.1000/aim11", title="", raw={}),
    ]
    assert webapp.filename_match("AIM-1(1)", rows) is None
    assert webapp.filename_match("AIM50", [webapp.OttoRow(paper_id="AIM-50", doi=None, title="", raw={})]).paper_id == "AIM-50"


def test_doi_match_prefers_own_doi_over_a_cited_doi_deterministically():
    # Regression test: doi_match used to collect candidates in a `set`, whose iteration
    # order (and thus which paper got matched) depended on Python's per-process string
    # hash seed - and it could pick up a DOI cited in the References section instead of
    # the paper's own DOI near the top.
    rows = [
        webapp.OttoRow(paper_id="AIM-1", doi="10.1000/aim1", title="", raw={}),
        webapp.OttoRow(paper_id="AIM-11", doi="10.1000/aim11", title="", raw={}),
    ]
    paper_text = (
        "The Real Paper\nDOI: 10.1000/aim1\n\n"
        + ("filler text " * 400)
        + "\nReferences\n1. Other Author. DOI: 10.1000/aim11\n"
    )
    for _ in range(20):
        result = webapp.doi_match(paper_text, rows)
        assert result is not None and result.paper_id == "AIM-1"


def test_flatten_raw_with_and_without_rows():
    with_rows = webapp.flatten_raw("AIM-TEST", "anatomical", {"summary_paragraph": "s", "rows": [{"row_label": "L", "fields": {"a": "b"}}]})
    assert len(with_rows) == 1 and with_rows[0]["row_label"] == "L"

    without_rows = webapp.flatten_raw("AIM-TEST", "functional", {"summary_paragraph": "the purpose", "rows": []})
    assert len(without_rows) == 1 and without_rows[0]["summary_paragraph"] == "the purpose"


def test_stage_scrape_matches_and_caches(isolated_paths):
    papers_dir = isolated_paths / "papers"
    papers_dir.mkdir()
    (papers_dir / "AIM-TEST.txt").write_text("Full paper text about the zona incerta circuit.")
    # a README placeholder must not be treated as a paper
    (papers_dir / "README.md").write_text("Drop paper files here.")

    result = webapp.stage_scrape()
    assert result["matched"] == 1

    cached = json.loads((isolated_paths / "cache" / "AIM-TEST.json").read_text())
    assert cached["text_source"] == "local_txt"
    assert "zona incerta circuit" in cached["text"]
    assert not (isolated_paths / "cache" / "README.json").exists()


def test_build_side_by_side_without_reextraction_shows_placeholder(isolated_paths):
    rows = webapp.build_side_by_side()
    anatomical = [r for r in rows if r["field_name"] == "Anatomical connections"]
    assert len(anatomical) == 1
    assert anatomical[0]["reextracted_value"] == "[not re-extracted yet]"
    assert anatomical[0]["otto_value"] == "ZI -> PVT"


@pytest.fixture
def client():
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


def test_dashboard_loads(client, isolated_paths):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Otto Re-extraction Pipeline" in resp.data
    assert b"127.0.0.1" in resp.data


def test_api_status_reports_counts(client, isolated_paths):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["papers_in_folder"] == 0
    assert data["running"] is None


def test_review_page_renders_without_crashing(client, isolated_paths):
    resp = client.get("/review")
    assert resp.status_code == 200
    assert b"AIM-TEST" in resp.data


def test_verification_page_handles_missing_data(client, isolated_paths):
    resp = client.get("/verification")
    assert resp.status_code == 200
    assert b"No verification data yet" in resp.data


def test_run_scrape_endpoint_starts_and_blocks_concurrent_runs(client, isolated_paths, monkeypatch):
    real_stage_scrape = webapp.stage_scrape

    def slow_stage_scrape():
        time.sleep(0.2)
        return real_stage_scrape()

    monkeypatch.setattr(webapp, "stage_scrape", slow_stage_scrape)

    resp1 = client.post("/api/run/scrape")
    assert resp1.get_json()["started"] is True
    # A concurrent request while the background thread is still "running" must be refused.
    resp2 = client.post("/api/run/scrape")
    assert resp2.get_json()["started"] is False

    # Wait for the background thread to finish *before* the isolated_paths fixture tears down
    # its monkeypatches - otherwise the thread can run after teardown and touch real repo paths.
    for _ in range(50):
        if webapp.current_stage() is None:
            break
        time.sleep(0.05)
    assert webapp.current_stage() is None
