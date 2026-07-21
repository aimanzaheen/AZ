"""Shared config, IO helpers, and small utilities for the Otto re-extraction pipeline."""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

PKG_DIR = Path(__file__).parent
DEFAULT_OTTO_CSV = PKG_DIR / "data" / "otto_output.csv"
DEFAULT_CACHE_DIR = PKG_DIR / "data" / "cache"
DEFAULT_RAW_DIR = PKG_DIR / "data" / "reextracted_raw"
DEFAULT_REEXTRACTED_CSV = PKG_DIR / "data" / "reextracted.csv"
DEFAULT_SIDE_BY_SIDE_CSV = PKG_DIR / "data" / "side_by_side.csv"
DEFAULT_FETCH_MANIFEST_CSV = PKG_DIR / "data" / "fetch_manifest.csv"
PROMPTS_DIR = PKG_DIR / "prompts"

PAPER_ID_COLUMN = "Reference ID"
DOI_COLUMN = "DOI"

# Modality name -> (prompt file stem, Otto columns compared against that modality's re-extraction).
# See README.md "Modality mapping" for why some Otto columns (e.g. "Verification of Zona
# Incerta Targeting") share a modality with no dedicated prompt of their own.
MODALITIES: dict[str, dict] = {
    "anatomical": {
        "prompt_file": PROMPTS_DIR / "anatomical.txt",
        "otto_fields": ["Anatomical connections", "Verification of Zona Incerta Targeting"],
    },
    "photometry": {
        "prompt_file": PROMPTS_DIR / "photometry.txt",
        "otto_fields": ["Photometry Data"],
    },
    "ephys": {
        "prompt_file": PROMPTS_DIR / "ephys.txt",
        "otto_fields": ["Electrophysiology Data"],
    },
    "functional": {
        "prompt_file": PROMPTS_DIR / "functional.txt",
        "otto_fields": ["Purpose of this circuit", "Necessity vs. sufficiency or Both"],
    },
}


def load_prompt(modality: str) -> str:
    return MODALITIES[modality]["prompt_file"].read_text(encoding="utf-8").strip()


@dataclass
class OttoRow:
    paper_id: str
    doi: str | None
    title: str
    raw: dict[str, str]


def load_otto_rows(csv_path: str | Path = DEFAULT_OTTO_CSV) -> list[OttoRow]:
    csv_path = Path(csv_path)
    rows: list[OttoRow] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        for raw in csv.DictReader(fh):
            paper_id = raw.get(PAPER_ID_COLUMN, "").strip()
            if not paper_id:
                continue
            rows.append(
                OttoRow(
                    paper_id=paper_id,
                    doi=normalize_doi(raw.get(DOI_COLUMN, "")),
                    title=raw.get("Title", "").strip(),
                    raw=raw,
                )
            )
    return rows


def normalize_doi(doi: str) -> str | None:
    doi = (doi or "").strip()
    if not doi:
        return None
    doi = re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", doi)
    return doi.strip("/ ") or None


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


class RateLimiter:
    """Simple sleep-based limiter to stay under a requests-per-second cap."""

    def __init__(self, min_interval_seconds: float):
        self.min_interval = min_interval_seconds
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()
