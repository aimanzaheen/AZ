import argparse

import search_papers


class FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class FakeSession:
    def __init__(self, responder):
        self.headers = {}
        self.responder = responder
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responder(url, params)


def test_run_writes_search_results_csv(tmp_path, monkeypatch):
    def responder(url, params):
        if "esearch" in url:
            return FakeResponse({"esearchresult": {"idlist": ["111", "222"], "count": "2"}})
        if "esummary" in url:
            return FakeResponse(
                {
                    "result": {
                        "uids": ["111", "222"],
                        "111": {
                            "title": "LH retrograde tracing paper.",
                            "authors": [{"name": "Smith J"}],
                            "pubdate": "2020",
                            "fulljournalname": "J Neurosci",
                            "articleids": [{"idtype": "doi", "value": "10.1/x"}],
                        },
                        "222": {
                            "title": "Another paper",
                            "authors": [],
                            "pubdate": "2018",
                            "fulljournalname": "Neuron",
                            "articleids": [],
                        },
                    }
                }
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(search_papers.requests, "Session", lambda: FakeSession(responder))
    monkeypatch.setattr(search_papers.common.RateLimiter, "wait", lambda self: None)

    out_csv = tmp_path / "search_results.csv"
    args = argparse.Namespace(query="lateral hypothalamus retrograde", max_results=100, out=str(out_csv))
    rc = search_papers.run(args)

    assert rc == 0
    results = search_papers.common.load_search_results(out_csv)
    assert {r.pmid for r in results} == {"111", "222"}
    assert results[0].title == "LH retrograde tracing paper"


def test_build_parser_defaults():
    args = search_papers.build_parser().parse_args([])
    assert args.query == search_papers.common.DEFAULT_QUERY
    assert args.max_results == 100
