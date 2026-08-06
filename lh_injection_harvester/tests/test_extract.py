import argparse
import json

import common
import extract


def test_rebuild_extracted_csv_includes_only_matches_and_flattens_experiments(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    search_csv = tmp_path / "search_results.csv"
    common.write_csv(
        search_csv,
        common.SEARCH_RESULTS_FIELDS,
        [
            {"pmid": "1", "title": "Match paper", "authors": "", "year": "2020", "journal": "J1", "doi": ""},
            {"pmid": "2", "title": "No match paper", "authors": "", "year": "2019", "journal": "J2", "doi": ""},
        ],
    )

    (raw_dir / "1.json").write_text(
        json.dumps(
            {
                "pmid": "1",
                "text_source": "pmc_fulltext",
                "lh_injection_present": True,
                "experiments": [
                    {
                        "transmission_type": "retrograde",
                        "volume_injected_lh": "50 nL",
                        "stereotaxic_coordinates": "AP -1.4",
                        "lha_subdivided": "single/undivided site",
                        "tracer": "CTB",
                        "projections_found": "PVT",
                        "survival_time": "7 days",
                        "figure_ref": "Fig 1",
                        "source_quote": "we injected CTB",
                    },
                    {
                        "transmission_type": "anterograde",
                        "volume_injected_lh": "100 nL",
                        "stereotaxic_coordinates": "AP -1.6",
                        "lha_subdivided": "rostral LH",
                        "tracer": "BDA",
                        "projections_found": "ARC",
                        "survival_time": "10 days",
                        "figure_ref": "Fig 2",
                        "source_quote": "second experiment",
                    },
                ],
                "notes": "two experiments",
            }
        )
    )
    (raw_dir / "2.json").write_text(
        json.dumps(
            {
                "pmid": "2",
                "text_source": "pubmed_abstract",
                "lh_injection_present": False,
                "experiments": [],
                "notes": "not actually about LH injections",
            }
        )
    )

    out_csv = tmp_path / "extracted.csv"
    n = extract.rebuild_extracted_csv(raw_dir, search_csv, out_csv)

    assert n == 2
    with out_csv.open() as fh:
        import csv as csvmod

        rows = list(csvmod.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["paper_name"] == "Match paper"
    assert rows[0]["tracer"] == "CTB"
    assert rows[0]["transmission_type"] == "retrograde"
    assert rows[0]["lha_subdivided"] == "single/undivided site"
    assert rows[0]["experiment_index"] == "1"
    assert rows[1]["tracer"] == "BDA"
    assert rows[1]["transmission_type"] == "anterograde"
    assert rows[1]["experiment_index"] == "2"
    assert all(r["pmid"] != "2" for r in rows)


class FakeAnthropic:
    def __init__(self, *args, **kwargs):
        pass


def test_run_skips_papers_without_cached_text(tmp_path, monkeypatch, capsys):
    search_csv = tmp_path / "search_results.csv"
    common.write_csv(
        search_csv,
        common.SEARCH_RESULTS_FIELDS,
        [{"pmid": "1", "title": "t", "authors": "", "year": "2020", "journal": "j", "doi": ""}],
    )
    monkeypatch.setattr(extract.anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(
        extract.llm_client,
        "call_extraction",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called without cached text")),
    )

    args = argparse.Namespace(
        search_results_csv=str(search_csv),
        cache_dir=str(tmp_path / "cache"),
        raw_dir=str(tmp_path / "raw"),
        out=str(tmp_path / "extracted.csv"),
        prompt_file=str(common.PROMPT_FILE),
        model="claude-sonnet-5",
        force=False,
        limit=None,
        pmid=None,
    )
    rc = extract.run(args)
    assert rc == 0
    assert not (tmp_path / "raw" / "1.json").exists()


def test_run_calls_llm_and_writes_raw_json(tmp_path, monkeypatch):
    search_csv = tmp_path / "search_results.csv"
    common.write_csv(
        search_csv,
        common.SEARCH_RESULTS_FIELDS,
        [{"pmid": "1", "title": "t", "authors": "", "year": "2020", "journal": "j", "doi": ""}],
    )
    cache_dir = tmp_path / "cache"
    common.write_json(cache_dir / "1.json", {"status": "ok", "text_source": "pmc_fulltext", "text": "paper text"})

    monkeypatch.setattr(extract.anthropic, "Anthropic", FakeAnthropic)
    payload = {
        "lh_injection_present": True,
        "experiments": [
            {
                "transmission_type": "retrograde",
                "volume_injected_lh": "50 nL",
                "stereotaxic_coordinates": "AP -1.4",
                "lha_subdivided": "single/undivided site",
                "tracer": "CTB",
                "projections_found": "PVT",
                "survival_time": "7 days",
                "figure_ref": "Fig 1",
                "source_quote": "quote",
            }
        ],
        "notes": "ok",
    }
    monkeypatch.setattr(extract.llm_client, "call_extraction", lambda *a, **k: dict(payload))

    args = argparse.Namespace(
        search_results_csv=str(search_csv),
        cache_dir=str(cache_dir),
        raw_dir=str(tmp_path / "raw"),
        out=str(tmp_path / "extracted.csv"),
        prompt_file=str(common.PROMPT_FILE),
        model="claude-sonnet-5",
        force=False,
        limit=None,
        pmid=None,
    )
    rc = extract.run(args)
    assert rc == 0

    raw = common.read_json(tmp_path / "raw" / "1.json")
    assert raw["pmid"] == "1"
    assert raw["lh_injection_present"] is True

    with open(args.out) as fh:
        import csv as csvmod

        rows = list(csvmod.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["tracer"] == "CTB"
    assert rows[0]["transmission_type"] == "retrograde"
