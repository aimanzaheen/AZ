"""Export DANDI Neuropixels Dandisets to tables by streaming only the tables
needed from each NWB file (HTTP range requests), without downloading the raw
voltage, LFP or video data.

Used for 000458, 001051, 001326, 001416, 001417 and 001637. The export is
dataset-neutral: every units column the authors provide is kept as is, and
region labels are the electrode `location` strings stored in the files.

Usage:
    python dandi_export/export_streamed.py 001417 dandi_export/output_001417 [--version V] [--workers 4]

Outputs (per Dandiset):
    tables/sessions.csv          one row per NWB file: subject, session, counts
    tables/units.parquet         one row per unit: every scalar units column,
                                 plus session_id, region (electrode location),
                                 n_spikes (from spike_times_index)
    tables/region_summary.csv    units per region (all units, and any provided
                                 firing_rate column summarised)
    tables/intervals_summary.csv every intervals table: rows and columns per session
    intervals/<name>.parquet     each intervals table, all sessions concatenated
    tables/dataset_metadata.json, nwb_file_metadata.json, nwb_descriptions.csv
                                 published metadata, quoted verbatim
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

API = "https://api.dandiarchive.org/api"
# Columns that hold per-spike or per-sample arrays; everything else is kept.
SKIP_UNIT_COLS = {"spike_times", "spike_amplitudes", "spike_amplitudes_uV", "waveform_mean", "waveform_sd",
                  "waveforms", "electrodes", "electrode_group", "obs_intervals", "spike_depths",
                  "spike_distances_from_probe_tip_um"}


def version_of(dandiset: str) -> str:
    d = requests.get(f"{API}/dandisets/{dandiset}/", timeout=60).json()
    v = d.get("most_recent_published_version") or d["draft_version"]
    return v["version"]


def list_assets(dandiset: str, version: str) -> list[dict]:
    url = f"{API}/dandisets/{dandiset}/versions/{version}/assets/?page_size=1000"
    out = []
    while url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        j = r.json()
        out += [a for a in j["results"] if a["path"].endswith(".nwb")]
        url = j["next"]
    if dandiset == "001051":  # per-probe *_ecephys.nwb files hold LFP only; units are in the session files
        out = [a for a in out if not a["path"].endswith("_ecephys.nwb")]
    return sorted(out, key=lambda a: a["path"])


def session_id(path: str) -> str:
    return Path(path).stem


def open_remote(asset_id: str):
    import h5py
    import remfile
    from pynwb import NWBHDF5IO

    url = requests.head(f"{API}/assets/{asset_id}/download/", allow_redirects=True, timeout=60).url
    h = h5py.File(remfile.File(url), "r")
    io = NWBHDF5IO(file=h, load_namespaces=True)
    return io, io.read()


def is_scalar_col(col) -> bool:
    name = col.name
    if name in SKIP_UNIT_COLS or name.endswith("_index"):
        return False
    shape = getattr(col.data, "shape", None)
    return shape is not None and len(shape) == 1


def decode(v):
    return v.decode() if isinstance(v, bytes) else v


def unit_regions(nwb, u) -> tuple[list, str]:
    """Electrode location per unit, and a note on how it was found."""
    for c in ("brain_region", "location", "structure_acronym", "ecephys_structure_acronym", "brain_area"):
        if c in u.colnames:
            return [decode(v) for v in u[c].data[:]], f"units.{c}"
    if nwb.electrodes is None or "location" not in nwb.electrodes.colnames:
        return [""] * len(u), "no electrodes.location"
    loc = np.array([decode(v) for v in nwb.electrodes["location"].data[:]], dtype=object)
    ids = np.asarray(nwb.electrodes.id[:])
    for c in ("peak_channel_id", "peak_channel"):
        if c in u.colnames:
            v = np.asarray(u[c].data[:]).ravel().astype(int)
            pos = {i: k for k, i in enumerate(ids)}
            if all(x in pos for x in v):
                return [loc[pos[x]] for x in v], f"electrodes.location at units.{c} (electrode id)"
    cols = {c.name: c for c in u.columns}
    if "electrodes" in cols:
        reg = cols["electrodes"]  # DynamicTableRegion (u["electrodes"] would return its index)
        data = np.asarray(reg.data[:]).astype(int)
        vi = next((c for c in u.columns if type(c).__name__ == "VectorIndex" and c.target is reg), None)
        if vi is None:
            return list(loc[data]), "electrodes.location at units.electrodes (one electrode per unit)"
        ends = np.asarray(vi.data[:]).astype(int)
        starts = np.concatenate([[0], ends[:-1]])
        n = ends - starts
        if (n == 1).all():
            return list(loc[data[starts]]), "electrodes.location at units.electrodes (one electrode per unit)"
        if "extremum_channel_index" in u.colnames:
            k = np.asarray(u["extremum_channel_index"].data[:]).astype(int)
            if (k < n).all():
                return (list(loc[data[starts + k]]),
                        "electrodes.location at units.electrodes[extremum_channel_index] (each unit lists all "
                        "channels of its probe; position checked against units.depth)")
        e = nwb.electrodes
        if "depth" in u.colnames and "rel_y" in e.colnames:
            ry = np.asarray(e["rel_y"].data[:], float)
            rx = np.asarray(e["rel_x"].data[:], float) if "rel_x" in e.colnames else np.zeros_like(ry)
            dep = np.asarray(u["depth"].data[:], float)
            ex = np.asarray(u["estimated_x"].data[:], float) if "estimated_x" in u.colnames else None
            out = []
            for i, (s0, e0) in enumerate(zip(starts, ends)):
                cand = data[s0:e0]
                dist = (ry[cand] - dep[i]) ** 2 + (0 if ex is None else (rx[cand] - ex[i]) ** 2)
                out.append(loc[cand[np.argmin(dist)]])
            return out, ("electrodes.location at the electrode (among the unit's units.electrodes) nearest to "
                         "units.depth" + ("/estimated_x" if ex is not None else "") + " (file has no peak-channel column)")
        return list(loc[data[starts]]), "electrodes.location at the first of several electrodes in units.electrodes"
    return [""] * len(u), "units have no electrode reference"


def extract(asset: dict) -> dict:
    from dataset_metadata import nwb_metadata_from

    sid = session_id(asset["path"])
    for attempt in range(4):
        try:
            io, nwb = open_remote(asset["asset_id"])
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    try:
        subj = nwb.subject
        out = {"session_id": sid}
        units = None
        region_note = ""
        u = nwb.units
        if u is not None and len(u):
            cols = {}
            for c in u.columns:
                if is_scalar_col(c):
                    try:
                        cols[c.name] = [decode(v) for v in c.data[:]]
                    except Exception:
                        pass
            units = pd.DataFrame(cols)
            units.insert(0, "unit_row", np.arange(len(u)))
            units.insert(0, "unit_id", np.asarray(u.id[:]))
            units.insert(0, "session_id", sid)
            regions, region_note = unit_regions(nwb, u)
            units.insert(3, "region", regions)
            if "spike_times_index" in u.colnames:
                ends = np.asarray(u["spike_times_index"].data[:])
                units["n_spikes"] = np.diff(np.concatenate([[0], ends]))
        intervals = {}
        for name, t in nwb.intervals.items():
            try:
                df = t.to_dataframe(index=True)
                df = df[[c for c in df.columns if df[c].map(lambda v: np.isscalar(v) or v is None).all()]]
                df.insert(0, "session_id", sid)
                intervals[name] = df.reset_index()
            except Exception:
                pass
        if nwb.trials is not None and "trials" not in intervals:
            intervals["trials"] = nwb.trials.to_dataframe().reset_index().assign(session_id=sid)
        n_elec = len(nwb.electrodes) if nwb.electrodes is not None else 0
        session = {
            "session_id": sid,
            "asset_path": asset["path"],
            "size_gb": round(asset["size"] / 1e9, 2),
            "subject_id": getattr(subj, "subject_id", None),
            "species": getattr(subj, "species", None),
            "sex": getattr(subj, "sex", None),
            "age": getattr(subj, "age", None),
            "genotype": getattr(subj, "genotype", None),
            "strain": getattr(subj, "strain", None),
            "session_start_time": nwb.session_start_time.isoformat(),
            "session_description": nwb.session_description,
            "institution": nwb.institution,
            "lab": nwb.lab,
            "n_units": 0 if units is None else len(units),
            "n_electrodes": n_elec,
            "electrode_groups": ";".join(nwb.electrode_groups),
            "region_source": region_note,
            "regions": "" if units is None else ";".join(sorted({str(r) for r in units.region if r})),
            "intervals": ";".join(f"{k}({len(v)})" for k, v in intervals.items()),
            "acquisition": ";".join(nwb.acquisition),
            "processing": ";".join(f"{m}:{'/'.join(mod.data_interfaces)}" for m, mod in nwb.processing.items()),
        }
        meta = nwb_metadata_from(nwb)
        meta = (meta[0] | {"source_file": Path(asset["path"]).name}, meta[1])
    finally:
        io.close()
    return {"session": session, "units": units, "intervals": intervals, "meta": meta}


def process(asset: dict, out: Path) -> str:
    sid = session_id(asset["path"])
    sdir = out / "_sessions" / sid
    if (sdir / "done").exists():
        return f"skip {sid}"
    sdir.mkdir(parents=True, exist_ok=True)
    r = extract(asset)
    if r["units"] is not None:
        r["units"].to_parquet(sdir / "units.parquet", index=False)
    for name, df in r["intervals"].items():
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        df.astype({c: str for c in df.columns if df[c].dtype == object}).to_parquet(
            sdir / f"intervals__{safe}.parquet", index=False)
    (sdir / "session.json").write_text(json.dumps(r["session"], default=str))
    (sdir / "meta.json").write_text(json.dumps(r["meta"][0], default=str))
    r["meta"][1].to_csv(sdir / "descriptions.csv", index=False)
    (sdir / "done").touch()
    return f"ok   {sid} ({r['session']['n_units']} units, {len(r['intervals'])} interval tables)"


def summarise(out: Path, dandiset: str, version: str) -> None:
    from dataset_metadata import dandi_metadata

    dirs = [d for d in sorted((out / "_sessions").iterdir()) if (d / "done").exists()]
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    sessions = pd.DataFrame([json.loads((d / "session.json").read_text()) for d in dirs])
    sessions.to_csv(tables / "sessions.csv", index=False)

    ups = [pd.read_parquet(d / "units.parquet") for d in dirs if (d / "units.parquet").exists()]
    units = pd.concat(ups, ignore_index=True) if ups else pd.DataFrame(columns=["session_id", "region"])
    units = units.merge(sessions[["session_id", "subject_id"]], on="session_id", how="left")
    units.to_parquet(tables / "units.parquet", index=False)

    g = units.assign(region=units.region.replace("", "(none)")).groupby("region")
    rs = g.agg(n_units=("session_id", "size"), n_sessions=("session_id", "nunique"),
               n_subjects=("subject_id", "nunique"))
    if "firing_rate" in units:
        fr = pd.to_numeric(units.firing_rate, errors="coerce")
        rs["firing_rate_mean_hz"] = fr.groupby(units.region.replace("", "(none)")).mean()
        rs["firing_rate_median_hz"] = fr.groupby(units.region.replace("", "(none)")).median()
    rs.sort_values("n_units", ascending=False).reset_index().to_csv(tables / "region_summary.csv", index=False)

    (out / "intervals").mkdir(exist_ok=True)
    rows = []
    names = sorted({p.name.removeprefix("intervals__").removesuffix(".parquet")
                    for d in dirs for p in d.glob("intervals__*.parquet")})
    for name in names:
        parts = [pd.read_parquet(p) for d in dirs for p in [d / f"intervals__{name}.parquet"] if p.exists()]
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(out / "intervals" / f"{name}.parquet", index=False)
        rows.append({"table": name, "n_sessions": df.session_id.nunique(), "n_rows": len(df),
                     "columns": ", ".join(c for c in df.columns if c not in ("session_id", "id"))})
    pd.DataFrame(rows).to_csv(tables / "intervals_summary.csv", index=False)

    (tables / "dataset_metadata.json").write_text(json.dumps(dandi_metadata(dandiset, version), indent=1, default=str))
    (tables / "nwb_file_metadata.json").write_text((dirs[0] / "meta.json").read_text())
    pd.read_csv(dirs[0] / "descriptions.csv", keep_default_na=False).to_csv(tables / "nwb_descriptions.csv", index=False)
    print(f"summarised {len(sessions)} sessions, {len(units):,} units -> {tables}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dandiset")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--version")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    version = a.version or version_of(a.dandiset)
    assets = list_assets(a.dandiset, version)[: a.limit]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "VERSION").write_text(version)
    print(f"DANDI:{a.dandiset} {version}: {len(assets)} NWB files", flush=True)
    failed = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, x, a.out_dir): x for x in assets}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                print(f"[{i}/{len(assets)}] {fu.result()}", flush=True)
            except Exception as e:
                failed.append(futs[fu]["path"])
                print(f"[{i}/{len(assets)}] FAIL {futs[fu]['path']}: {e!r}", flush=True)
    if failed:
        print(f"{len(failed)} failed (re-run to retry):", *failed, sep="\n  ")
    summarise(a.out_dir, a.dandiset, version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
