from pathlib import Path

import common


def test_write_csv_and_load_search_results_round_trip(tmp_path):
    csv_path = tmp_path / "search_results.csv"
    rows = [
        {
            "pmid": "111",
            "title": "A paper",
            "authors": "Smith J",
            "year": "2021",
            "journal": "J Neurosci",
            "doi": "10.1/abc",
        },
        {"pmid": "", "title": "skip me - no pmid", "authors": "", "year": "", "journal": "", "doi": ""},
    ]
    common.write_csv(csv_path, common.SEARCH_RESULTS_FIELDS, rows)

    results = common.load_search_results(csv_path)
    assert len(results) == 1
    assert results[0].pmid == "111"
    assert results[0].title == "A paper"
    assert results[0].doi == "10.1/abc"


def test_load_search_results_missing_file_returns_empty_list(tmp_path):
    assert common.load_search_results(tmp_path / "nope.csv") == []


def test_read_json_missing_file_returns_none(tmp_path):
    assert common.read_json(tmp_path / "nope.json") is None


def test_write_json_then_read_json_round_trip(tmp_path):
    path = tmp_path / "sub" / "data.json"
    common.write_json(path, {"a": 1})
    assert common.read_json(path) == {"a": 1}


def test_rate_limiter_waits_at_least_min_interval(monkeypatch):
    sleeps = []
    monkeypatch.setattr(common.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(common.time, "monotonic", lambda: 100.0)

    limiter = common.RateLimiter(min_interval_seconds=0.5)
    limiter.wait()  # first call: no prior call recorded (last_call=0.0), long elapsed -> no sleep
    assert sleeps == []

    limiter.wait()  # second call: elapsed since last call is 0 -> should sleep ~0.5s
    assert sleeps == [0.5]


def test_default_query_targets_lh_and_retrograde():
    assert "lateral hypothalamus" in common.DEFAULT_QUERY
    assert "retrograde" in common.DEFAULT_QUERY


def test_load_prompt_reads_prompt_file():
    text = common.load_prompt()
    assert "lateral hypothalamus" in text.lower()
    assert "retrograde" in text.lower()
