"""Minimal client for the NCBI E-utilities / ID Converter APIs used to fetch paper text.

Kept as thin, mockable functions (each takes a `requests.Session`) so the
fetch logic in fetch_papers.py can be unit-tested without live network access.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import requests

IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TIMEOUT = 30


def _common_params(api_key: str | None) -> dict:
    return {"api_key": api_key} if api_key else {}


def doi_to_ids(doi: str, session: requests.Session, api_key: str | None = None) -> dict:
    """Return {"pmid": str|None, "pmcid": str|None} for a DOI via the PMC ID Converter."""
    params = {"ids": doi, "format": "json", **_common_params(api_key)}
    resp = session.get(IDCONV_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    records = data.get("records") or []
    if not records or records[0].get("status") == "error":
        return {"pmid": None, "pmcid": None}
    record = records[0]
    return {"pmid": record.get("pmid"), "pmcid": record.get("pmcid")}


def esearch_pmid_by_doi(doi: str, session: requests.Session, api_key: str | None = None) -> str | None:
    """Fallback: look up a PMID directly on PubMed when the ID Converter has no record."""
    params = {
        "db": "pubmed",
        "term": f"{doi}[DOI]",
        "retmode": "json",
        **_common_params(api_key),
    }
    resp = session.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


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
