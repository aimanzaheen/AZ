"""Export DANDI:000409 (IBL Brain Wide Map) processed NWB files to tables.

The Dandiset is ~50 TB, mostly raw electrophysiology and video. This script
uses only the 459 `desc-processed_behavior+ecephys.nwb` files (~0.5 TB): each
is downloaded, reduced to tables, and deleted, so disk use stays at a few GB.
It is resumable: finished sessions are skipped on re-run.

Usage:
    python dandi_export/export_000409.py dandi_export/output_000409 \
        --tmp /path/to/scratch --workers 4

Outputs (see dandi_export/README_000409.md):
    tables/sessions.csv        one row per session
    tables/units.csv           one row per unit: Allen region (acronym + name),
                               CCF coords, IBL quality metrics, firing rates
    tables/trials.csv          one row per trial
    tables/region_summary.csv  per-region firing-rate summary (good units)
    trial_unit_counts.parquet  spike counts per (trial, unit) in fixed windows
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

DANDISET, VERSION = "000409", "0.260309.1324"
API = "https://api.dandiarchive.org/api"
HERE = Path(__file__).parent

# Windows (seconds) relative to each trial event, for trial_unit_counts.
WINDOWS = {
    "prestim": ("gabor_stimulus_onset_time", -0.4, -0.1),
    "stim": ("gabor_stimulus_onset_time", 0.0, 0.25),
    "premove": ("wheel_movement_onset_time", -0.25, 0.0),
    "feedback": ("feedback_time", 0.0, 0.5),
}


def list_processed_assets() -> list[dict]:
    url = f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/?page_size=1000"
    out = []
    while url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        j = r.json()
        out += [a for a in j["results"] if "desc-processed" in a["path"]]
        url = j["next"]
    return sorted(out, key=lambda a: a["path"])


def session_id(path: str) -> str:
    # sub-NYU-11/sub-NYU-11_ses-6713a4a7-..._desc-processed_... -> NYU-11_6713a4a7
    subj, ses = Path(path).name.split("_")[:2]
    return f"{subj.removeprefix('sub-')}_{ses.removeprefix('ses-')[:8]}"


def download(asset_id: str, dest: Path) -> None:
    url = f"{API}/assets/{asset_id}/download/"
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(8 << 20):
                        f.write(chunk)
            return
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))


def norm_name(name: str) -> str:
    # IBL drops punctuation from Allen names ("Entorhinal area lateral part layer 5").
    return " ".join(re.sub(r"[^a-z0-9/ ]", " ", str(name).lower()).split())


def acronym_lookup() -> dict[str, str]:
    s = pd.read_csv(HERE / "allen_structures.csv")
    return dict(zip(s.name.map(norm_name), s.acronym))


def ragged(index) -> list[np.ndarray]:
    # `units["spike_times"]` returns the VectorIndex itself; its target holds the data.
    data = np.asarray(index.target.data[:])
    ends = np.asarray(index.data[:])
    starts = np.concatenate([[0], ends[:-1]])
    return [data[a:b] for a, b in zip(starts, ends)]


def extract(nwb_path: Path, sid: str, asset_path: str) -> dict:
    from pynwb import NWBHDF5IO

    acr = acronym_lookup()
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        subj = nwb.subject

        trials = nwb.trials.to_dataframe().reset_index(names="trial_id")
        trials.insert(0, "session_id", sid)

        u = nwb.units
        n = len(u)
        elec = nwb.electrodes.to_dataframe().reset_index(drop=True)
        me = np.asarray(u["max_electrode"].data[:]).astype(int)
        loc = elec.iloc[me].reset_index(drop=True)

        scalar = [
            "unit_name", "cluster_uuid", "probe_name", "kilosort2_label", "ibl_quality_score",
            "firing_rate", "peak_to_trough_duration_ms", "median_spike_amplitude_uV",
            "presence_ratio", "isi_violations_ratio", "sliding_rp_violation", "noise_cutoff",
            "missed_spikes_estimate", "distance_from_probe_tip_um", "spike_count",
        ]
        cols = {c: np.asarray(u[c].data[:]) for c in scalar if c in u.colnames}
        for c in ("unit_name", "cluster_uuid", "probe_name", "kilosort2_label"):
            if c in cols:
                cols[c] = [v.decode() if isinstance(v, bytes) else v for v in cols[c]]

        spikes = ragged(u["spike_times_index"])
        t0, t1 = float(trials.start_time.min()), float(trials.stop_time.max())
        counts = {w: np.zeros((len(trials), n), np.int16) for w in WINDOWS}
        task_rate = np.zeros(n)
        for i, st in enumerate(spikes):
            task_rate[i] = (np.searchsorted(st, t1) - np.searchsorted(st, t0)) / (t1 - t0)
            for w, (ev, lo, hi) in WINDOWS.items():
                e = trials[ev].to_numpy(float)
                ok = ~np.isnan(e)
                c = np.zeros(len(e), np.int32)
                c[ok] = np.searchsorted(st, e[ok] + hi) - np.searchsorted(st, e[ok] + lo)
                counts[w][:, i] = np.minimum(c, np.iinfo(np.int16).max)

        lo, hi = WINDOWS["prestim"][1:]
        region_name = loc["location"].astype(str)
        units = pd.DataFrame(
            {
                "session_id": sid,
                "unit_id": np.arange(n),
                "unit_name": cols.get("unit_name"),
                "cluster_uuid": cols.get("cluster_uuid"),
                "probe": cols.get("probe_name"),
                "region": [acr.get(norm_name(r), "") for r in region_name],
                "region_name": region_name,
                "ccf_x": loc["x"], "ccf_y": loc["y"], "ccf_z": loc["z"],
                "distance_from_probe_tip_um": cols.get("distance_from_probe_tip_um"),
                "kilosort2_label": cols.get("kilosort2_label"),
                "ibl_quality_score": cols.get("ibl_quality_score"),
                "good": np.isclose(cols["ibl_quality_score"], 1.0),
                "peak_to_trough_ms": np.where(cols["peak_to_trough_duration_ms"] < 0, np.nan, cols["peak_to_trough_duration_ms"]),
                "median_amplitude_uV": cols.get("median_spike_amplitude_uV"),
                "presence_ratio": cols.get("presence_ratio"),
                "isi_violations_ratio": cols.get("isi_violations_ratio"),
                "spike_count": cols.get("spike_count"),
                "firing_rate_hz": cols.get("firing_rate"),
                "task_rate_hz": task_rate,
                "prestim_rate_hz": counts["prestim"].mean(axis=0) / (hi - lo),
                "stim_rate_hz": counts["stim"].mean(axis=0) / 0.25,
            }
        )

        tuc = pd.DataFrame(
            {
                "session_id": sid,
                "trial_id": np.repeat(trials.trial_id.to_numpy(np.int32), n),
                "unit_id": np.tile(np.arange(n, dtype=np.int32), len(trials)),
                **{f"count_{w}": counts[w].ravel() for w in WINDOWS},
            }
        )

        dob = getattr(subj, "date_of_birth", None)
        start = nwb.session_start_time
        session = {
            "session_id": sid,
            "session_uuid": nwb.session_id,
            "asset_path": asset_path,
            "subject_id": subj.subject_id,
            "lab": nwb.lab,
            "institution": nwb.institution,
            "sex": subj.sex,
            "age_days": (start - dob).days if dob is not None else None,
            "weight_kg": getattr(subj, "weight", None),
            "session_start_time": start.isoformat(),
            "protocol": nwb.protocol,
            "n_probes": len(nwb.electrode_groups),
            "n_trials": len(trials),
            "n_units": n,
            "n_good_units": int(units.good.sum()),
            "task_duration_s": t1 - t0,
            "regions": ";".join(sorted({r for r in units.region if r})),
        }
    return {"session": session, "units": units, "trials": trials, "counts": tuc}


def process(asset: dict, out: Path, tmp: Path) -> str:
    sid = session_id(asset["path"])
    sdir = out / "_sessions" / sid
    if (sdir / "done").exists():
        return f"skip {sid}"
    sdir.mkdir(parents=True, exist_ok=True)
    f = tmp / f"{sid}.nwb"
    try:
        download(asset["asset_id"], f)
        r = extract(f, sid, asset["path"])
    finally:
        f.unlink(missing_ok=True)
    r["units"].to_csv(sdir / "units.csv", index=False)
    r["trials"].to_csv(sdir / "trials.csv", index=False)
    r["counts"].to_parquet(sdir / "counts.parquet", index=False)
    (sdir / "session.json").write_text(json.dumps(r["session"], default=str))
    (sdir / "done").touch()
    return f"ok   {sid} ({asset['size'] / 1e9:.2f} GB, {r['session']['n_units']} units)"


def region_summary(units: pd.DataFrame) -> pd.DataFrame:
    g = units[units.good & ~units.region.isin(["", "root", "void"]) & units.region.notna()]
    s = g.groupby(["region", "region_name"]).agg(
        n_units=("unit_id", "size"),
        n_sessions=("session_id", "nunique"),
        n_mice=("subject_id", "nunique"),
        firing_rate_mean_hz=("firing_rate_hz", "mean"),
        firing_rate_sd_hz=("firing_rate_hz", "std"),
        firing_rate_median_hz=("firing_rate_hz", "median"),
        prestim_rate_mean_hz=("prestim_rate_hz", "mean"),
        prestim_rate_median_hz=("prestim_rate_hz", "median"),
        stim_rate_mean_hz=("stim_rate_hz", "mean"),
        peak_to_trough_ms_median=("peak_to_trough_ms", "median"),
    )
    return s.sort_values("n_units", ascending=False).reset_index()


def merge(out: Path) -> None:
    sessions, units, trials, counts = [], [], [], []
    for d in sorted((out / "_sessions").iterdir()):
        if not (d / "done").exists():
            continue
        sessions.append(json.loads((d / "session.json").read_text()))
        units.append(pd.read_csv(d / "units.csv", keep_default_na=False, na_values=[""]))
        trials.append(pd.read_csv(d / "trials.csv"))
        counts.append(d / "counts.parquet")
    sessions = pd.DataFrame(sessions)
    units = pd.concat(units, ignore_index=True)
    units["region"] = units.region.fillna("")
    units = units.merge(sessions[["session_id", "subject_id"]], on="session_id")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(tables / "sessions.csv", index=False)
    units.to_csv(tables / "units.csv", index=False)
    pd.concat(trials, ignore_index=True).to_csv(tables / "trials.csv", index=False)
    region_summary(units).to_csv(tables / "region_summary.csv", index=False)

    import pyarrow.parquet as pq

    writer = None
    for p in counts:
        t = pq.read_table(p)
        writer = writer or pq.ParquetWriter(out / "trial_unit_counts.parquet", t.schema)
        writer.write_table(t)
    if writer:
        writer.close()
    print(f"merged {len(sessions)} sessions, {len(units):,} units -> {tables}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tmp", type=Path, required=True, help="scratch dir for downloads")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="only the first N sessions (testing)")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.tmp.mkdir(parents=True, exist_ok=True)

    if not a.merge_only:
        assets = list_processed_assets()[: a.limit]
        print(f"{len(assets)} processed sessions, {sum(x['size'] for x in assets) / 1e9:.0f} GB", flush=True)
        failed = []
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(process, x, a.out_dir, a.tmp): x for x in assets}
            for i, fu in enumerate(as_completed(futs), 1):
                try:
                    print(f"[{i}/{len(assets)}] {fu.result()}", flush=True)
                except Exception as e:  # keep going; re-run retries failures
                    failed.append(futs[fu]["path"])
                    print(f"[{i}/{len(assets)}] FAIL {futs[fu]['path']}: {e!r}", flush=True)
        if failed:
            print(f"{len(failed)} failed (re-run to retry):", *failed, sep="\n  ")
    merge(a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
