import json
from pathlib import Path

import scrape_papers

FIXTURE_OTTO = Path(__file__).parent / "fixtures" / "mini_otto.csv"


def run_scrape(papers_dir, cache_dir, manifest_out, otto_csv=FIXTURE_OTTO):
    args = scrape_papers.build_parser().parse_args(
        [
            "--papers-dir",
            str(papers_dir),
            "--otto-csv",
            str(otto_csv),
            "--cache-dir",
            str(cache_dir),
            "--manifest-out",
            str(manifest_out),
        ]
    )
    return scrape_papers.run(args)


def test_scrape_matches_by_filename_and_caches_text(tmp_path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "AIM-TEST.txt").write_text("Full paper text about the zona incerta circuit.")

    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "scrape_manifest.csv"
    exit_code = run_scrape(papers_dir, cache_dir, manifest_path)
    assert exit_code == 0

    cached = json.loads((cache_dir / "AIM-TEST.json").read_text())
    assert cached["text_source"] == "local_txt"
    assert "zona incerta circuit" in cached["text"]
    assert cached["match_method"] == "filename"

    assert manifest_path.exists()
    manifest_text = manifest_path.read_text()
    assert "AIM-TEST" in manifest_text
    assert "matched" in manifest_text


def test_scrape_reports_unmatched_file_and_missing_row(tmp_path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "unrelated_paper.txt").write_text("Something about crustacean locomotion, nothing to do with ZI.")

    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "scrape_manifest.csv"
    run_scrape(papers_dir, cache_dir, manifest_path)

    assert not (cache_dir / "AIM-TEST.json").exists()
    manifest_text = manifest_path.read_text()
    assert "unmatched" in manifest_text


def test_scrape_flags_duplicate_match_and_keeps_higher_confidence(tmp_path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    # Both files match AIM-TEST: one via filename (confidence 1.0), one via DOI (confidence 0.95).
    (papers_dir / "AIM-TEST.txt").write_text("Full paper text about the zona incerta circuit.")
    (papers_dir / "duplicate_copy.txt").write_text("See DOI: 10.1000/test.001 for details on this circuit.")

    cache_dir = tmp_path / "cache"
    manifest_path = tmp_path / "scrape_manifest.csv"
    run_scrape(papers_dir, cache_dir, manifest_path)

    cached = json.loads((cache_dir / "AIM-TEST.json").read_text())
    assert cached["match_method"] == "filename"  # the higher-confidence match won
    manifest_text = manifest_path.read_text()
    assert "duplicate_match" in manifest_text
