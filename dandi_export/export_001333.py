"""Export DANDI:001333 (Parkinson's Electrophysiological Signal Dataset, PESD)
to plain tables.

All 52 files are *simulated* signals from the Fleming et al. (2020)
cortico-basal-ganglia model (NEURON), not recordings. Each file holds one
signal: either an STN local field potential (LFP, 2 kHz) or the beta-band
average rectified value (ARV, 50 Hz) derived from it, for a "healthy" or a
"parkinsonian" network.

Usage:
    dandi download DANDI:001333/0.250327.2220        # ~29 MB -> ./001333/
    python dandi_export/export_001333.py 001333 dandi_export/output_001333

Outputs (see dandi_export/README_001333.md):
    tables/sessions.csv   one row per file: group, signal type, duration,
                          duplicate detection, amplitude and beta-band metrics
    tables/psd.csv        Welch power spectrum (0-100 Hz) of every LFP file
    signals/<group>_<signal>_ses-<n>.csv   time (s), voltage (V)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.signal import welch

warnings.filterwarnings("ignore")

BETA = (13.0, 30.0)  # Hz, as in the dataset description
TOTAL = (1.0, 100.0)  # Hz, denominator for relative beta power
PEAK_SEARCH = (5.0, 45.0)
WELCH_SEG_S = 2.0  # 0.5 Hz resolution


def band_power(f, p, lo, hi) -> float:
    m = (f >= lo) & (f <= hi)
    return float(np.trapezoid(p[m], f[m]))


def read(path: Path) -> dict:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        es = next(iter(nwb.processing["ecephys"]["LFP"].electrical_series.values()))
        data = np.asarray(es.data[:], float) * (es.conversion or 1.0)
        if es.timestamps is not None:
            t = np.asarray(es.timestamps[:], float)
        else:
            t = (es.starting_time or 0.0) + np.arange(len(data)) / es.rate
        return {"series": es.name, "t": t, "v": data, "subject_id": nwb.subject.subject_id}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dandiset_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args(argv)

    files = sorted(a.dandiset_dir.glob("sub-*/*.nwb"))
    if not files:
        print(f"no NWB files under {a.dandiset_dir}", file=sys.stderr)
        return 1
    (a.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (a.out_dir / "signals").mkdir(parents=True, exist_ok=True)

    rows, psd_rows, first_seen = [], [], {}
    for f in files:
        subj = f.parent.name.removeprefix("sub-")  # e.g. parkinson-simulated-lfp
        ses = f.stem.split("_ses-")[1].split("_")[0]
        group = "parkinsonian" if subj.startswith("parkinson") else "healthy"
        signal = "beta_arv" if subj.endswith("beta") else "lfp"
        r = read(f)
        t, v = r["t"], r["v"]
        digest = hashlib.md5(v.tobytes()).hexdigest()
        key = (signal, digest)
        sid = f"{group}_{signal}_ses-{ses}" + ("_data" if subj.endswith("data") else "")
        dup_of = first_seen.setdefault(key, sid)
        fs = 1.0 / np.median(np.diff(t))

        row = {
            "sample_id": sid,
            "file": f"{f.parent.name}/{f.name}",
            "group": group,
            "signal": signal,
            "session": ses,
            "nwb_series": r["series"],
            "n_samples": len(v),
            "fs_hz": round(fs, 3),
            "t_start_s": t[0],
            "t_end_s": t[-1],
            "duration_s": t[-1] - t[0] + 1 / fs,
            "md5": digest,
            "duplicate_of": "" if dup_of == sid else dup_of,
            "unique": dup_of == sid,
            "mean_v": v.mean(),
            "sd_v": v.std(),
            "rms_v": float(np.sqrt(np.mean(v ** 2))),
            "min_v": v.min(),
            "max_v": v.max(),
        }
        if signal == "lfp":
            fr, p = welch(v - v.mean(), fs=fs, nperseg=int(WELCH_SEG_S * fs))
            beta = band_power(fr, p, *BETA)
            m = (fr >= PEAK_SEARCH[0]) & (fr <= PEAK_SEARCH[1])
            row.update(
                beta_power_v2=beta,
                total_power_1_100_v2=band_power(fr, p, *TOTAL),
                relative_beta_power=beta / band_power(fr, p, *TOTAL),
                peak_freq_hz=float(fr[m][np.argmax(p[m])]),
                peak_psd_v2_per_hz=float(p[m].max()),
            )
            keep = fr <= 100
            psd_rows.append(pd.DataFrame({"sample_id": sid, "group": group, "unique": row["unique"],
                                          "freq_hz": fr[keep], "psd_v2_per_hz": p[keep]}))
        rows.append(row)
        pd.DataFrame({"time_s": t, "voltage_v": v}).to_csv(a.out_dir / "signals" / f"{sid}.csv", index=False)

    s = pd.DataFrame(rows).sort_values(["signal", "group", "sample_id"])
    s.to_csv(a.out_dir / "tables" / "sessions.csv", index=False)
    pd.concat(psd_rows).to_csv(a.out_dir / "tables" / "psd.csv", index=False)
    u = s[s.unique]
    print(f"{len(s)} files, {len(u)} unique signals "
          f"({u.groupby(['group', 'signal']).size().to_dict()}) -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
