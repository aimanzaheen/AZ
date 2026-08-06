import argparse
import json

import common
import fetch_papers


class FakeResponse:
    def __init__(self, json_data=None, text_data=""):
        self._json = json_data
        self.text = text_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class FakeSession:
    def __init__(self, responder):
        self.headers = {}
        self.responder = responder

    def get(self, url, params=None, timeout=None):
        return self.responder(url, params)


PMC_XML = """<pmc-articleset><article><body>
<sec><title>Methods</title>
<p>%s</p></sec></body></article></pmc-articleset>""" % (
    "We injected CTB into the LH. " * 60
)


def _limiter():
    limiter = common.RateLimiter(0.0)
    limiter.wait = lambda: None
    return limiter


def test_fetch_one_prefers_pmc_fulltext_when_available():
    def responder(url, params):
        if "idconv" in url:
            return FakeResponse(json_data={"records": [{"pmcid": "PMC999"}]})
        if "efetch" in url:
            return FakeResponse(text_data=PMC_XML)
        raise AssertionError(url)

    paper = common.SearchResult(pmid="111", title="t", authors="a", year="2020", journal="j", doi=None)
    result = fetch_papers.fetch_one(paper, FakeSession(responder), _limiter(), None)

    assert result["status"] == "ok"
    assert result["text_source"] == "pmc_fulltext"
    assert result["pmcid"] == "PMC999"
    assert "CTB" in result["text"]


def test_fetch_one_falls_back_to_abstract_when_no_pmcid():
    def responder(url, params):
        if "idconv" in url:
            return FakeResponse(json_data={"records": [{"status": "error"}]})
        if "efetch" in url:
            return FakeResponse(text_data="Title.\n\nAbstract about LH retrograde tracing.")
        raise AssertionError(url)

    paper = common.SearchResult(pmid="222", title="t", authors="a", year="2020", journal="j", doi=None)
    result = fetch_papers.fetch_one(paper, FakeSession(responder), _limiter(), None)

    assert result["status"] == "ok"
    assert result["text_source"] == "pubmed_abstract"
    assert result["pmcid"] is None


def test_fetch_one_falls_back_to_abstract_when_pmc_fulltext_too_short():
    def responder(url, params):
        if "idconv" in url:
            return FakeResponse(json_data={"records": [{"pmcid": "PMC1"}]})
        if "efetch" in url and params["db"] == "pmc":
            return FakeResponse(text_data="<pmc-articleset><article><body><p>too short</p></body></article></pmc-articleset>")
        if "efetch" in url and params["db"] == "pubmed":
            return FakeResponse(text_data="Abstract text.")
        raise AssertionError(url)

    paper = common.SearchResult(pmid="333", title="t", authors="a", year="2020", journal="j", doi=None)
    result = fetch_papers.fetch_one(paper, FakeSession(responder), _limiter(), None)

    assert result["text_source"] == "pubmed_abstract"


def test_fetch_one_marks_unavailable_when_nothing_found():
    def responder(url, params):
        if "idconv" in url:
            return FakeResponse(json_data={"records": [{"status": "error"}]})
        if "efetch" in url:
            return FakeResponse(text_data="")
        raise AssertionError(url)

    paper = common.SearchResult(pmid="444", title="t", authors="a", year="2020", journal="j", doi=None)
    result = fetch_papers.fetch_one(paper, FakeSession(responder), _limiter(), None)

    assert result["status"] == "unavailable"
    assert result["text"] is None


def test_run_skips_already_cached_papers(tmp_path, monkeypatch):
    search_csv = tmp_path / "search_results.csv"
    common.write_csv(
        search_csv,
        common.SEARCH_RESULTS_FIELDS,
        [{"pmid": "555", "title": "t", "authors": "a", "year": "2020", "journal": "j", "doi": ""}],
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "555.json").write_text(json.dumps({"pmid": "555", "status": "ok", "text": "x"}))

    def responder(url, params):
        raise AssertionError("should not hit the network for a cached paper")

    monkeypatch.setattr(fetch_papers.requests, "Session", lambda: FakeSession(responder))

    args = argparse.Namespace(
        search_results_csv=str(search_csv),
        cache_dir=str(cache_dir),
        manifest_out=str(tmp_path / "manifest.csv"),
        force=False,
        limit=None,
        pmid=None,
    )
    rc = fetch_papers.run(args)
    assert rc == 0
