"""Minimal client for the NCBI E-utilities / PMC ID Converter APIs used to search for and
fetch papers.

Kept as thin, mockable functions (each takes a `requests.Session`) so search_papers.py and
fetch_papers.py can be unit-tested without live network access.

NCBI usage note: without an NCBI_API_KEY requests self-limit to ~3/sec (NCBI's unauthenticated
cap); set NCBI_API_KEY to raise that to ~10/sec. See https://www.ncbi.nlm.nih.gov/books/NBK25497/.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import requests

IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TIMEOUT = 30


def _common_params(api_key: str | None) -> dict:
    return {"api_key": api_key} if api_key else {}


def esearch(
    query: str,
    session: requests.Session,
    api_key: str | None = None,
    retmax: int = 100,
    retstart: int = 0,
) -> dict:
    """Search PubMed and return {"idlist": [pmid, ...], "count": int}."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "retstart": retstart,
        **_common_params(api_key),
    }
    resp = session.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    result = resp.json().get("esearchresult", {})
    return {"idlist": result.get("idlist", []), "count": int(result.get("count", 0) or 0)}


def esummary(pmids: list[str], session: requests.Session, api_key: str | None = None) -> dict[str, dict]:
    """Fetch DocSum metadata (title, authors, year, journal, DOI) for a batch of PMIDs."""
    if not pmids:
        return {}
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        **_common_params(api_key),
    }
    resp = session.get(f"{EUTILS_BASE}/esummary.fcgi", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    result = resp.json().get("result", {})
    summaries: dict[str, dict] = {}
    for uid in result.get("uids", []):
        doc = result.get(uid, {})
        authors = ", ".join(a.get("name", "") for a in doc.get("authors", []) if a.get("name"))
        doi = ""
        for article_id in doc.get("articleids", []):
            if article_id.get("idtype") == "doi":
                doi = article_id.get("value", "")
                break
        pubdate = doc.get("pubdate", "") or doc.get("sortpubdate", "")
        year_match = re.match(r"\d{4}", pubdate)
        year = year_match.group(0) if year_match else ""
        summaries[uid] = {
            "pmid": uid,
            "title": doc.get("title", "").strip().rstrip("."),
            "authors": authors,
            "year": year,
            "journal": doc.get("fulljournalname", "") or doc.get("source", ""),
            "doi": doi,
        }
    return summaries


def pmid_to_pmcid(pmid: str, session: requests.Session, api_key: str | None = None) -> str | None:
    """Look up the PMCID for a PMID via the PMC ID Converter, if the article is in PMC."""
    params = {"ids": pmid, "format": "json", **_common_params(api_key)}
    resp = session.get(IDCONV_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    records = resp.json().get("records") or []
    if not records or records[0].get("status") == "error":
        return None
    return records[0].get("pmcid")


def fetch_pmc_fulltext(pmcid: str, session: requests.Session, api_key: str | None = None) -> str | None:
    """Fetch and flatten full text for a PMC record. Returns None if not available (non-OA)."""
    numeric_id = pmcid[3:] if pmcid.upper().startswith("PMC") else pmcid
    params = {
        "db": "pmc",
        "id": numeric_id,
        "rettype": "full",
        "retmode": "xml",
        **_common_params(api_key),
    }
    resp = session.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return parse_pmc_fulltext_xml(resp.text)


def fetch_pubmed_abstract(pmid: str, session: requests.Session, api_key: str | None = None) -> str | None:
    """Fetch the title + abstract text for a PMID."""
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "abstract",
        "retmode": "text",
        **_common_params(api_key),
    }
    resp = session.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    text = resp.text.strip()
    return text or None


def parse_pmc_fulltext_xml(xml_text: str) -> str | None:
    """Extract a flattened plain-text version of the <body> of a PMC JATS XML record."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    body = root.find(".//body")
    if body is None:
        return None

    lines: list[str] = []
    for elem in body.iter():
        tag = elem.tag.split("}")[-1]  # strip any namespace
        if tag == "title":
            text = " ".join("".join(elem.itertext()).split())
            if text:
                lines.append(f"\n## {text}")
        elif tag == "p":
            text = "".join(elem.itertext()).strip()
            text = " ".join(text.split())
            if text:
                lines.append(text)

    flattened = "\n".join(lines).strip()
    return flattened or None
