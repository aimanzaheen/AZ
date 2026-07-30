"""Shared config, IO helpers, and small utilities for the LH injection harvester."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).parent
DATA_DIR = PKG_DIR / "data"
DEFAULT_SEARCH_RESULTS_CSV = DATA_DIR / "search_results.csv"
DEFAULT_CACHE_DIR = DATA_DIR / "cache"
DEFAULT_FETCH_MANIFEST_CSV = DATA_DIR / "fetch_manifest.csv"
DEFAULT_EXTRACTED_RAW_DIR = DATA_DIR / "extracted_raw"
DEFAULT_EXTRACTED_CSV = DATA_DIR / "extracted.csv"
DEFAULT_TABLE_HTML = DATA_DIR / "table.html"
PROMPT_FILE = PKG_DIR / "prompts" / "lh_retrograde_injection.txt"

# Default PubMed search: retrograde tracer injections into the lateral hypothalamus (LH) in mice.
# Tune this per your specific injection type/tracer of interest with --query.
DEFAULT_QUERY = (
    '("lateral hypothalamus"[tiab] OR "lateral hypothalamic area"[tiab] OR "LHA"[tiab]) '
    "AND (retrograde[tiab]) "
    'AND (tracer[tiab] OR tracing[tiab] OR "cholera toxin"[tiab] OR CTB[tiab] OR fluorogold[tiab] '
    'OR retrobead*[tiab] OR "retrograde AAV"[tiab] OR AAVretro[tiab] OR rabies[tiab]) '
    'AND (mouse[tiab] OR mice[tiab] OR "Mus musculus"[tiab])'
)

SEARCH_RESULTS_FIELDS = ["pmid", "title", "authors", "year", "journal", "doi"]
FETCH_MANIFEST_FIELDS = ["pmid", "pmcid", "text_source", "status", "fetched_at", "error"]
EXTRACTED_FIELDS = [
    "pmid",
    "paper_name",
    "year",
    "journal",
    "experiment_index",
    "volume_injected_lh",
    "stereotaxic_coordinates",
    "tracer",
    "projections_found",
    "survival_time",
    "figure_ref",
    "source_quote",
    "text_source",
]


def load_prompt(path: Path = PROMPT_FILE) -> str:
    return path.read_text(encoding="utf-8").strip()


@dataclass
class SearchResult:
    pmid: str
    title: str
    authors: str
    year: str
    journal: str
    doi: str | None


def load_search_results(csv_path: str | Path = DEFAULT_SEARCH_RESULTS_CSV) -> list[SearchResult]:
    csv_path = Path(csv_path)
    results: list[SearchResult] = []
    if not csv_path.exists():
        return results
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        for raw in csv.DictReader(fh):
            pmid = (raw.get("pmid") or "").strip()
            if not pmid:
                continue
            results.append(
                SearchResult(
                    pmid=pmid,
                    title=(raw.get("title") or "").strip(),
                    authors=(raw.get("authors") or "").strip(),
                    year=(raw.get("year") or "").strip(),
                    journal=(raw.get("journal") or "").strip(),
                    doi=(raw.get("doi") or "").strip() or None,
                )
            )
    return results


def write_csv(path: str | Path, fieldnames: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class RateLimiter:
    """Simple sleep-based limiter to stay under a requests-per-second cap."""

    min_interval_seconds: float
    _last_call: float = field(default=0.0, init=False)

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()
