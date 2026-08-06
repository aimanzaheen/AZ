import ncbi

SAMPLE_JATS = """<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <body>
      <sec>
        <title>Introduction</title>
        <p>Lateral hypothalamus neurons receive input from the <italic>PVT</italic>.</p>
        <p>This is a second paragraph with more detail.</p>
      </sec>
      <sec>
        <title>Methods</title>
        <p>We injected CTB-Alexa555 into the LH.</p>
        <fig>
          <caption><p>Figure 1. Injection site verification.</p></caption>
        </fig>
      </sec>
    </body>
  </article>
</pmc-articleset>
"""


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
        self.responder = responder
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responder(url, params)


def test_esearch_returns_idlist_and_count():
    def responder(url, params):
        assert "esearch" in url
        assert params["term"] == "lateral hypothalamus retrograde"
        return FakeResponse(json_data={"esearchresult": {"idlist": ["1", "2"], "count": "2"}})

    session = FakeSession(responder)
    result = ncbi.esearch("lateral hypothalamus retrograde", session)
    assert result == {"idlist": ["1", "2"], "count": 2}


def test_esummary_parses_title_authors_year_journal_doi():
    def responder(url, params):
        assert "esummary" in url
        assert params["id"] == "111,222"
        return FakeResponse(
            json_data={
                "result": {
                    "uids": ["111", "222"],
                    "111": {
                        "title": "A paper about LH tracing.",
                        "authors": [{"name": "Smith J"}, {"name": "Doe A"}],
                        "pubdate": "2021 Jun",
                        "fulljournalname": "J Neurosci",
                        "articleids": [{"idtype": "doi", "value": "10.1/abc"}],
                    },
                    "222": {
                        "title": "Another paper",
                        "authors": [],
                        "sortpubdate": "2019/01/01 00:00",
                        "source": "Neuron",
                        "articleids": [],
                    },
                }
            }
        )

    session = FakeSession(responder)
    result = ncbi.esummary(["111", "222"], session)
    assert result["111"]["title"] == "A paper about LH tracing"
    assert result["111"]["authors"] == "Smith J, Doe A"
    assert result["111"]["year"] == "2021"
    assert result["111"]["journal"] == "J Neurosci"
    assert result["111"]["doi"] == "10.1/abc"
    assert result["222"]["year"] == "2019"
    assert result["222"]["journal"] == "Neuron"


def test_esummary_returns_empty_dict_for_no_pmids():
    session = FakeSession(lambda url, params: FakeResponse())
    assert ncbi.esummary([], session) == {}
    assert session.calls == []


def test_pmid_to_pmcid_parses_idconv_response():
    def responder(url, params):
        assert "idconv" in url
        return FakeResponse(json_data={"records": [{"pmid": "123", "pmcid": "PMC456"}]})

    session = FakeSession(responder)
    assert ncbi.pmid_to_pmcid("123", session) == "PMC456"


def test_pmid_to_pmcid_returns_none_on_error_status():
    def responder(url, params):
        return FakeResponse(json_data={"records": [{"status": "error"}]})

    session = FakeSession(responder)
    assert ncbi.pmid_to_pmcid("999", session) is None


def test_fetch_pmc_fulltext_strips_pmc_prefix_for_efetch_id():
    def responder(url, params):
        assert params["id"] == "456"
        return FakeResponse(text_data=SAMPLE_JATS)

    session = FakeSession(responder)
    text = ncbi.fetch_pmc_fulltext("PMC456", session)
    assert "Lateral hypothalamus neurons" in text


def test_fetch_pubmed_abstract_returns_text():
    def responder(url, params):
        return FakeResponse(text_data="Title.\n\nAbstract text here.")

    session = FakeSession(responder)
    text = ncbi.fetch_pubmed_abstract("123", session)
    assert text == "Title.\n\nAbstract text here."


def test_parse_pmc_fulltext_xml_extracts_paragraphs_and_titles_without_duplicating_captions():
    text = ncbi.parse_pmc_fulltext_xml(SAMPLE_JATS)
    assert "## Introduction" in text
    assert "## Methods" in text
    assert "We injected CTB-Alexa555 into the LH." in text
    assert text.count("Figure 1. Injection site verification.") == 1


def test_parse_pmc_fulltext_xml_returns_none_on_malformed_xml():
    assert ncbi.parse_pmc_fulltext_xml("<not><valid") is None
