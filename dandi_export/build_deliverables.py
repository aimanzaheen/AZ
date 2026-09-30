"""Assemble the "Dandi Data sets" deliverable folder.

For every Dandiset it collects, under "Dandi Data sets/DANDI_<id>/":
    DANDI_<id>_report.pdf       the neutral PDF report (if built)
    metadata/dandiset.yaml      the Dandiset's published metadata (DANDI API), as YAML
    metadata/dandiset.json      the same, as JSON
    metadata/assets_manifest.csv  every file in the Dandiset: path, size, asset id, download URL
    metadata/nwb_file_metadata.json, nwb_descriptions.csv
                                file-level fields and column descriptions from an NWB file
    exported_tables/            the CSV tables produced by the export scripts
                                (Parquet tables are converted to CSV)
    original_files/             non-NWB files the authors published alongside the
                                data (left as downloaded; see README.md)

Re-run any time; it overwrites what it generates and leaves original_files/ alone.

Usage:
    python dandi_export/build_deliverables.py
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "dandi_export"
OUT = ROOT / "Dandi Data sets"
API = "https://api.dandiarchive.org/api"
MAX_PLAIN_CSV = 90e6  # larger CSVs are written gzip-compressed (.csv.gz)

# dandiset -> (export output folder, report pdf, pinned version or None for latest)
DATASETS = {
    "000017": ("output", "DANDI_000017_report.pdf", "0.240329.1926"),
    "000409": ("output_000409", "DANDI_000409_report.pdf", "0.260309.1324"),
    "001333": ("output_001333", "DANDI_001333_report.pdf", "0.250327.2220"),
    "000021": ("output_000021", "DANDI_000021_report.pdf", None),
    "000022": ("output_000022", "DANDI_000022_report.pdf", None),
    "000458": ("output_000458", "DANDI_000458_report.pdf", None),
    "001051": ("output_001051", "DANDI_001051_report.pdf", None),
    "001326": ("output_001326", "DANDI_001326_report.pdf", None),
    "001416": ("output_001416", "DANDI_001416_report.pdf", None),
    "001417": ("output_001417", "DANDI_001417_report.pdf", None),
    "001637": ("output_001637", "DANDI_001637_report.pdf", None),
}


def latest(ds: str) -> str:
    d = requests.get(f"{API}/dandisets/{ds}/", timeout=60).json()
    return (d.get("most_recent_published_version") or d["draft_version"])["version"]


def assets(ds: str, ver: str) -> pd.DataFrame:
    url, rows = f"{API}/dandisets/{ds}/versions/{ver}/assets/?page_size=1000", []
    while url:
        j = requests.get(url, timeout=60).json()
        rows += [{"path": a["path"], "size_bytes": a["size"], "asset_id": a["asset_id"],
                  "created": a.get("created"), "modified": a.get("modified"),
                  "download_url": f"{API}/assets/{a['asset_id']}/download/"} for a in j["results"]]
        url = j["next"]
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    index = []
    for ds, (folder, pdf, ver) in DATASETS.items():
        ver = ver or latest(ds)
        dst = OUT / f"DANDI_{ds}"
        meta, tabs = dst / "metadata", dst / "exported_tables"
        for d in (meta, tabs):
            d.mkdir(parents=True, exist_ok=True)
        m = requests.get(f"{API}/dandisets/{ds}/versions/{ver}/", timeout=60).json()
        (meta / "dandiset.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
        (meta / "dandiset.yaml").write_text(yaml.safe_dump(m, sort_keys=False, allow_unicode=True))
        man = assets(ds, ver)
        man.to_csv(meta / "assets_manifest.csv", index=False)

        src = EXP / folder / "tables"
        n_tables = 0
        if src.exists():
            for f in ("nwb_file_metadata.json", "nwb_descriptions.csv"):
                if (src / f).exists():
                    shutil.copy2(src / f, meta / f)
            for f in sorted(src.iterdir()):
                if f.name in ("nwb_file_metadata.json", "nwb_descriptions.csv", "dataset_metadata.json"):
                    continue
                if f.suffix == ".csv":
                    if f.stat().st_size > MAX_PLAIN_CSV:  # GitHub rejects files > 100 MB
                        (tabs / f.name).unlink(missing_ok=True)
                        with open(f, "rb") as src_f, gzip.open(tabs / f"{f.name}.gz", "wb") as dst_f:
                            shutil.copyfileobj(src_f, dst_f)
                    else:
                        shutil.copy2(f, tabs / f.name)
                    n_tables += 1
                elif f.suffix == ".parquet":
                    df = pd.read_parquet(f)
                    target = tabs / f"{f.stem}.csv"
                    df.to_csv(target, index=False)
                    if target.stat().st_size > MAX_PLAIN_CSV:  # GitHub rejects files > 100 MB
                        target.unlink()
                        df.to_csv(tabs / f"{f.stem}.csv.gz", index=False, compression="gzip")
                    else:
                        (tabs / f"{f.stem}.csv.gz").unlink(missing_ok=True)
                    n_tables += 1
        if (EXP / folder / "tables" / "psd.csv").exists():
            pass  # copied above
        report = EXP / pdf
        has_pdf = report.exists()
        if has_pdf:
            shutil.copy2(report, dst / pdf)
        orig = dst / "original_files"
        index.append({"dandiset": ds, "version": ver, "name": m.get("name"),
                      "files_in_dandiset": len(man), "size_gb": round(man.size_bytes.sum() / 1e9, 1),
                      "report": pdf if has_pdf else "(export still running)",
                      "exported_tables": n_tables,
                      "original_files": ", ".join(p.name for p in orig.iterdir()) if orig.exists() else ""})
        print(f"DANDI:{ds} {ver}: {len(man)} assets, {n_tables} tables, pdf={'yes' if has_pdf else 'no'}")
    pd.DataFrame(index).to_csv(OUT / "index.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
