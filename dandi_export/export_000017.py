"""Convert DANDI:000017 (Steinmetz et al. 2019, Neuropixels, mouse visual
discrimination task) NWB files into plain tables for modelling.

Usage:
    dandi download DANDI:000017/0.240329.1926          # -> ./000017/
    python dandi_export/export_000017.py 000017 dandi_export/output

Outputs (see dandi_export/README.md for column descriptions):
    tables/sessions.csv          one row per session (subject metadata, counts)
    tables/units.csv             one row per unit: brain region, CCF coords,
                                 waveform width, mean + spontaneous firing rate
    tables/trials.csv            one row per trial: stimulus, choice, feedback
    tables/region_summary.csv    per brain region firing-rate summary
                                 (good units only) - directly comparable to
                                 literature values like "spontaneous firing rate"
    trial_unit_counts.parquet    spike counts per (trial, unit) in fixed
                                 windows around stimulus / response / feedback
    spikes/<session>.parquet     every spike: unit_id, time (s), amp, depth
    behavior/<session>__<name>.parquet
                                 licks, wheel movements, wheel position (100 Hz),
                                 pupil, face motion
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

warnings.filterwarnings("ignore")

# Windows (seconds) relative to each trial event, for trial_unit_counts.
WINDOWS = {
    "prestim": ("visual_stimulus_time", -0.5, 0.0),
    "stim": ("visual_stimulus_time", 0.0, 0.25),
    "premove": ("response_time", -0.25, 0.0),
    "feedback": ("feedback_time", 0.0, 0.5),
}
WHEEL_DOWNSAMPLE_HZ = 100.0


def session_id(path: Path) -> str:
    # sub-Cori_ses-20161214T120000.nwb -> Cori_2016-12-14
    subj, ses = path.stem.split("_")
    d = ses.removeprefix("ses-")[:8]
    return f"{subj.removeprefix('sub-')}_{d[:4]}-{d[4:6]}-{d[6:]}"


def true_rate(ts) -> float:
    # Several series in this dataset store the sampling *period* in `rate`
    # (e.g. wheel_position rate=0.0004 means 2500 Hz).
    return 1.0 / ts.rate if ts.rate < 1 else ts.rate


def series_frame(ts, downsample_hz: float | None = None) -> pd.DataFrame:
    data = np.asarray(ts.data[:])
    if ts.timestamps is not None:
        t = np.asarray(ts.timestamps[:])
        if len(t) == 2 and len(data) > 2:
            # Some sessions store only the first/last frame time (evenly sampled video).
            t = np.linspace(t[0], t[1], len(data))
    else:
        t = ts.starting_time + np.arange(len(data)) / true_rate(ts)
    if downsample_hz and ts.timestamps is None and true_rate(ts) > downsample_hz:
        step = int(round(true_rate(ts) / downsample_hz))
        t, data = t[::step], data[::step]
    if data.ndim == 1:
        return pd.DataFrame({"time": t, ts.name: data})
    cols = {f"{ts.name}_{i}": data[:, i] for i in range(data.shape[1])}
    return pd.DataFrame({"time": t, **cols})


def intervals_frame(ts) -> pd.DataFrame:
    data = np.asarray(ts.data[:])
    t = np.asarray(ts.timestamps[:])
    if len(t) == 2 * len(data):
        # This dataset stores (start, stop) timestamp pairs with one label per
        # interval (wheel_moves: 0 = flinch/unclassified, 1 = left, 2 = right).
        return pd.DataFrame({"start_time": t[0::2], "stop_time": t[1::2], "type": data})
    # Standard IntervalSeries: data > 0 marks a start, < 0 a stop.
    return pd.DataFrame({"start_time": t[data > 0], "stop_time": t[data < 0]})


def export_behavior(nwb, sid: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for name, ts in nwb.acquisition.items():
        if name == "lickPiezo":
            continue  # raw piezo voltage; lick_times below is the extracted signal
        frames[name] = series_frame(ts, WHEEL_DOWNSAMPLE_HZ)
    beh = nwb.processing.get("behavior")
    if beh is not None:
        for iface in beh.data_interfaces.values():
            kind = type(iface).__name__
            if kind == "BehavioralEpochs":
                for name, ts in iface.interval_series.items():
                    frames[name] = intervals_frame(ts)
            elif kind == "BehavioralEvents":
                for name, ts in iface.time_series.items():
                    t = ts.timestamps[:] if ts.timestamps is not None else ts.data[:]
                    frames[name] = pd.DataFrame({"time": np.asarray(t)})
            else:
                for name, ts in iface.time_series.items():
                    frames[name] = series_frame(ts)
    for name, df in frames.items():
        df.to_parquet(out / f"{sid}__{name}.parquet", index=False)


def waveform_duration_samples(u) -> tuple[np.ndarray, str]:
    """Trough-to-peak waveform duration in samples, plus where it came from.

    In 11 of the 39 sessions the published `waveform_duration` column is
    actually a copy of `cluster_depths` (r > 0.9999, values up to ~3800).
    There it is recomputed from `waveform_mean` (82 samples x 50 nearest
    channels) on the highest-amplitude channel, which reproduces the stored
    value closely (r ~ 0.96) in the sessions where the stored value is valid.
    """
    stored = np.asarray(u["waveform_duration"].data[:], float).ravel()
    depth = np.asarray(u["cluster_depths"].data[:], float).ravel()
    if stored.max() < 100 and not np.allclose(stored, depth):
        return stored, "nwb"
    wm = np.asarray(u["waveform_mean"].data[:], float)  # (units, samples, channels)
    amp = wm.max(axis=1) - wm.min(axis=1)
    best = wm[np.arange(len(wm)), :, amp.argmax(axis=1)]  # (units, samples)
    trough = best.argmin(axis=1)
    after = np.where(np.arange(best.shape[1]) >= trough[:, None], best, -np.inf)
    return (after.argmax(axis=1) - trough).astype(float), "recomputed_from_waveform_mean"


def export_session(path: Path, out: Path):
    sid = session_id(path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        subj = nwb.subject

        # --- trials ---------------------------------------------------------
        trials = nwb.trials.to_dataframe().reset_index(names="trial_id")
        trials.insert(0, "session_id", sid)

        # --- electrodes -> region per unit (via peak channel) ---------------
        elec = nwb.electrodes.to_dataframe()
        elec_cols = ["location", "ccf_ap", "ccf_dv", "ccf_lr", "group_name"]
        elec_rows = elec[elec_cols].reset_index(drop=True)

        u = nwb.units
        unit_ids = np.asarray(u.id[:])
        peak = np.asarray(u["peak_channel"].data[:]).ravel().astype(int)
        # peak_channel is 1-based (the first electrode listed for each unit,
        # its closest channel, is always peak_channel - 1).
        reg = elec_rows.iloc[peak - 1].reset_index(drop=True)

        spont = nwb.intervals["spontaneous"].to_dataframe()
        spont_dur = float((spont.stop_time - spont.start_time).sum())
        t_start, t_end = float(trials.start_time.min()), float(trials.stop_time.max())

        st_col, amp_col, dep_col = u["spike_times"], u["spike_amps"], u["spike_depths"]
        n = len(unit_ids)
        n_spikes = np.zeros(n, int)
        spont_counts = np.zeros(n, int)
        task_counts = np.zeros(n, int)
        last_spike = np.zeros(n)
        spike_frames = []
        counts = {w: np.zeros((len(trials), n), np.int32) for w in WINDOWS}
        for i in range(n):
            st = np.asarray(st_col[i]).ravel()
            n_spikes[i] = len(st)
            last_spike[i] = st[-1] if len(st) else 0.0
            for a, b in zip(spont.start_time, spont.stop_time):
                spont_counts[i] += np.searchsorted(st, b) - np.searchsorted(st, a)
            task_counts[i] = np.searchsorted(st, t_end) - np.searchsorted(st, t_start)
            for w, (ev, lo, hi) in WINDOWS.items():
                e = trials[ev].to_numpy()
                ok = ~np.isnan(e)
                c = np.zeros(len(e), np.int32)
                c[ok] = np.searchsorted(st, e[ok] + hi) - np.searchsorted(st, e[ok] + lo)
                counts[w][:, i] = c
            spike_frames.append(
                pd.DataFrame(
                    {
                        "unit_id": np.full(len(st), unit_ids[i], np.int32),
                        "time": st,
                        "amp": np.asarray(amp_col[i], np.float32).ravel(),
                        "depth": np.asarray(dep_col[i], np.float32).ravel(),
                    }
                )
            )

        wheel = nwb.acquisition["wheel_position"]
        wheel_end = (wheel.starting_time or 0.0) + wheel.data.shape[0] / true_rate(wheel)
        rec_dur = max(wheel_end, t_end, last_spike.max(initial=0.0))
        fs = np.asarray(u["sampling_rate"].data[:]).ravel()
        quality = np.asarray(u["phy_annotations"].data[:]).ravel().astype(int)
        wave_samples, wave_source = waveform_duration_samples(u)
        units = pd.DataFrame(
            {
                "session_id": sid,
                "unit_id": unit_ids,
                "region": reg["location"],
                "probe": reg["group_name"],
                "ccf_ap": reg["ccf_ap"],
                "ccf_dv": reg["ccf_dv"],
                "ccf_lr": reg["ccf_lr"],
                "peak_channel": peak,
                "depth_um": np.asarray(u["cluster_depths"].data[:]).ravel(),
                "phy_annotation": quality,
                "good": quality >= 2,
                "waveform_duration_ms": wave_samples / fs * 1000,
                "waveform_duration_source": wave_source,
                "n_spikes": n_spikes,
                "mean_rate_hz": n_spikes / rec_dur,
                "task_rate_hz": task_counts / (t_end - t_start),
                "spontaneous_rate_hz": spont_counts / spont_dur if spont_dur else np.nan,
            }
        )

        spikes = pd.concat(spike_frames, ignore_index=True).sort_values("time")
        (out / "spikes").mkdir(parents=True, exist_ok=True)
        spikes.to_parquet(out / "spikes" / f"{sid}.parquet", index=False)

        tuc = pd.DataFrame(
            {
                "session_id": sid,
                "trial_id": np.repeat(trials.trial_id.to_numpy(), n),
                "unit_id": np.tile(unit_ids, len(trials)),
                **{f"count_{w}": counts[w].ravel() for w in WINDOWS},
            }
        )

        export_behavior(nwb, sid, out / "behavior")

        session = {
            "session_id": sid,
            "file": str(path),
            "subject_id": subj.subject_id,
            "sex": subj.sex,
            "age": subj.age,
            "genotype": subj.genotype,
            "strain": subj.description,
            "session_start_time": nwb.session_start_time.isoformat(),
            "duration_s": rec_dur,
            "n_trials": len(trials),
            "n_units": n,
            "n_good_units": int((quality >= 2).sum()),
            "regions": ";".join(sorted(set(units.region))),
            "spontaneous_duration_s": spont_dur,
        }
    return session, units, trials, tuc


def region_summary(units: pd.DataFrame) -> pd.DataFrame:
    g = units[units.good & (units.region != "root")].groupby("region")
    s = g.agg(
        n_units=("unit_id", "size"),
        n_sessions=("session_id", "nunique"),
        spont_rate_mean_hz=("spontaneous_rate_hz", "mean"),
        spont_rate_sd_hz=("spontaneous_rate_hz", "std"),
        spont_rate_median_hz=("spontaneous_rate_hz", "median"),
        mean_rate_mean_hz=("mean_rate_hz", "mean"),
        waveform_duration_ms_median=("waveform_duration_ms", "median"),
    )
    return s.sort_values("n_units", ascending=False).reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dandiset_dir", type=Path, help="folder from `dandi download`")
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args(argv)

    files = sorted(args.dandiset_dir.glob("sub-*/*.nwb"))
    if not files:
        print(f"no NWB files under {args.dandiset_dir}", file=sys.stderr)
        return 1
    tables = args.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    sessions, units, trials, tucs = [], [], [], []
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}", flush=True)
        s, u, t, c = export_session(f, args.out_dir)
        sessions.append(s)
        units.append(u)
        trials.append(t)
        tucs.append(c)

    units_df = pd.concat(units, ignore_index=True)
    pd.DataFrame(sessions).to_csv(tables / "sessions.csv", index=False)
    units_df.to_csv(tables / "units.csv", index=False)
    pd.concat(trials, ignore_index=True).to_csv(tables / "trials.csv", index=False)
    region_summary(units_df).to_csv(tables / "region_summary.csv", index=False)
    pd.concat(tucs, ignore_index=True).to_parquet(
        args.out_dir / "trial_unit_counts.parquet", index=False
    )
    print(f"done -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
