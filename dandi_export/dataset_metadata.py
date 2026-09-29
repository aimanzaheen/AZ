"""Capture a Dandiset's published metadata and one NWB file's own
descriptions, verbatim, so reports can quote what the authors reported.

Usage:
    python dandi_export/dataset_metadata.py 000017 0.240329.1926 some_file.nwb dandi_export/output

Writes to <out_dir>/tables/:
    dataset_metadata.json   DANDI version metadata (name, description, citation,
                            contributors, license, keywords, assets summary, ...)
    nwb_file_metadata.json  file-level NWB fields (session/experiment
                            description, institution, lab, subject, devices,
                            electrode groups, related publications)
    nwb_descriptions.csv    every table column / data interface in the file
                            with its description as written by the authors
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")
API = "https://api.dandiarchive.org/api"


def dandi_metadata(dandiset: str, version: str) -> dict:
    r = requests.get(f"{API}/dandisets/{dandiset}/versions/{version}/", timeout=60)
    r.raise_for_status()
    m = r.json()
    keep = ["name", "description", "citation", "doi", "url", "version", "license", "keywords",
            "studyTarget", "protocol", "ethicsApproval", "acknowledgement", "relatedResource",
            "about", "wasGeneratedBy", "assetsSummary", "datePublished", "dateCreated"]
    out = {k: m[k] for k in keep if k in m}
    out["contributor"] = [
        {k: c.get(k) for k in ("name", "roleName", "affiliation", "awardNumber") if c.get(k)}
        for c in m.get("contributor", [])
    ]
    out["identifier"] = dandiset
    return out


def _s(x):
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool, dict)):
        return x
    if isinstance(x, (tuple, list)):
        return [_s(v) for v in x]
    try:  # numpy arrays and h5py (string) datasets
        return [v.decode() if isinstance(v, bytes) else _s(v) for v in list(x[:])]
    except Exception:
        return str(x)


def nwb_metadata(path: Path) -> tuple[dict, pd.DataFrame]:
    from pynwb import NWBHDF5IO

    rows = []

    def table_cols(where, t):
        if t is None:
            return
        rows.append({"location": where, "name": "(table)", "description": t.description})
        for c in t.columns:
            if c.name.endswith("_index"):
                continue
            rows.append({"location": where, "name": c.name, "description": c.description})

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        subj = nwb.subject
        meta = {
            "session_description": nwb.session_description,
            "experiment_description": nwb.experiment_description,
            "institution": nwb.institution,
            "lab": nwb.lab,
            "experimenter": _s(nwb.experimenter),
            "keywords": _s(nwb.keywords),
            "related_publications": _s(nwb.related_publications),
            "protocol": nwb.protocol,
            "surgery": nwb.surgery,
            "virus": nwb.virus,
            "pharmacology": nwb.pharmacology,
            "stimulus_notes": nwb.stimulus_notes,
            "data_collection": nwb.data_collection,
            "notes": nwb.notes,
            "subject": {f: _s(getattr(subj, f, None)) for f in
                        ("subject_id", "species", "strain", "genotype", "sex", "age", "description")}
            if subj else None,
            "devices": {k: _s(d.description) for k, d in nwb.devices.items()},
            "electrode_groups": {k: {"description": g.description, "location": g.location}
                                 for k, g in nwb.electrode_groups.items()},
        }
        meta = {k: v for k, v in meta.items() if v not in (None, "", [], {})}

        table_cols("electrodes", nwb.electrodes)
        table_cols("units", nwb.units)
        for name, t in nwb.intervals.items():
            table_cols(f"intervals/{name}", t)
        for name, ts in nwb.acquisition.items():
            rows.append({"location": "acquisition", "name": name,
                         "description": getattr(ts, "description", "")})
        for mod_name, mod in nwb.processing.items():
            rows.append({"location": f"processing/{mod_name}", "name": "(module)",
                         "description": mod.description})
            for iname, iface in mod.data_interfaces.items():
                children = {}
                for attr in ("time_series", "interval_series", "electrical_series", "spatial_series"):
                    children.update(getattr(iface, attr, None) or {})
                if hasattr(iface, "columns"):
                    table_cols(f"processing/{mod_name}/{iname}", iface)
                elif children:
                    for cname, ts in children.items():
                        rows.append({"location": f"processing/{mod_name}/{iname}", "name": cname,
                                     "description": getattr(ts, "description", "")})
                else:
                    rows.append({"location": f"processing/{mod_name}", "name": iname,
                                 "description": getattr(iface, "description", "")})
    return meta, pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dandiset")
    ap.add_argument("version")
    ap.add_argument("nwb", type=Path, help="one representative NWB file")
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args(argv)
    t = a.out_dir / "tables"
    t.mkdir(parents=True, exist_ok=True)
    (t / "dataset_metadata.json").write_text(json.dumps(dandi_metadata(a.dandiset, a.version), indent=1, default=str))
    meta, desc = nwb_metadata(a.nwb)
    meta["source_file"] = a.nwb.name
    (t / "nwb_file_metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    desc.to_csv(t / "nwb_descriptions.csv", index=False)
    print(f"wrote metadata for {a.dandiset} ({len(desc)} described items) -> {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
