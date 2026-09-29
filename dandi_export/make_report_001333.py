"""Build a PDF report of the DANDI:001333 (PESD) export.

The report quotes the dataset's own published metadata and NWB descriptions,
then summarises every file. Numbers computed by the export are labelled as
such.

Usage:
    python dandi_export/make_report_001333.py dandi_export/output_001333 \
        dandi_export/DANDI_001333_report.pdf
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, Spacer, Table

sys.path.insert(0, str(Path(__file__).parent))
from export_001333 import BETA, PEAK_SEARCH, TOTAL, WELCH_SEG_S  # noqa: E402
from report_common import (  # noqa: E402
    BODY, H1, H2, NOTE, SMALL, SRC_DANDI, SRC_EXPORT, SRC_NWB, W, descriptions_section, doc, esc,
    fig_to_image, fmt, legend_para, load_metadata, published_section, table,
)

VERSION = "0.250327.2220"
COLORS = {"healthy": "#2f5d8a", "parkinsonian": "#c0504d"}


def sig(out: Path, sid: str) -> pd.DataFrame:
    return pd.read_csv(out / "signals" / f"{sid}.csv")


def figures(out, s, psd, tmp):
    imgs = {}
    u = s[s.unique]
    first = {(g, sg): u[(u.group == g) & (u.signal == sg)].sample_id.iloc[0]
             for g in COLORS for sg in ("lfp", "beta_arv")}

    fig, axs = plt.subplots(2, 1, figsize=(11, 3.4), sharex=True)
    for ax, g in zip(axs, COLORS):
        sid = first[(g, "lfp")]
        d = sig(out, sid)
        w = d[(d.time_s >= 20) & (d.time_s < 21)]
        ax.plot(w.time_s, w.voltage_v * 1e3, color=COLORS[g], lw=0.7)
        ax.set_ylabel("mV", fontsize=8)
        ax.set_title(f"{sid}: 1 s excerpt (20–21 s)", fontsize=8, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axs[1].set_xlabel("Time (s)")
    imgs["lfp"] = fig_to_image(fig, tmp, "lfp", 24)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for g, c in COLORS.items():
        q = psd[(psd.group == g) & psd.unique]
        m = q.groupby("freq_hz").psd_v2_per_hz
        mean, sd = m.mean(), m.std().fillna(0)
        n = q.sample_id.nunique()
        ax.semilogy(mean.index, mean.values, color=c, label=f"{g} (n = {n} unique LFP files)")
        if n > 1:
            ax.fill_between(mean.index, (mean - sd).clip(lower=1e-15), mean + sd, color=c, alpha=0.2, lw=0)
    for f in BETA:
        ax.axvline(f, color="k", lw=0.6, ls="--")
    ax.set_xlim(0, 60)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("LFP Welch PSD, mean ± SD (dashed: 13 and 30 Hz)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["psd"] = fig_to_image(fig, tmp, "psd", 12)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for g, c in COLORS.items():
        sid = first[(g, "beta_arv")]
        d = sig(out, sid)
        ax.plot(d.time_s, d.voltage_v * 1e6, color=c, lw=0.8, label=sid)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Beta_Band_Voltage (µV)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("Beta ARV series (one file per group)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["arv"] = fig_to_image(fig, tmp, "arv", 12)

    fig, axs = plt.subplots(1, 3, figsize=(11, 2.8))
    specs = [("lfp", "beta_power_v2", 1e9, "LFP 13–30 Hz power (×10⁻⁹ V²)"),
             ("lfp", "relative_beta_power", 1, "LFP 13–30 Hz / 1–100 Hz power"),
             ("beta_arv", "mean_v", 1e6, "Mean Beta_Band_Voltage (µV)")]
    rng = np.random.default_rng(0)
    for ax, (sg, col, k, lab) in zip(axs, specs):
        for i, (g, c) in enumerate(COLORS.items()):
            y = u[(u.group == g) & (u.signal == sg)][col].values * k
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=14, color=c, alpha=0.8)
            ax.hlines(np.mean(y), i - 0.25, i + 0.25, color="k", lw=1)
        ax.set_xticks([0, 1], list(COLORS))
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(lab, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Per unique file (identical files counted once; line = mean)", fontsize=8)
    imgs["strip"] = fig_to_image(fig, tmp, "strip", 24)
    return imgs


def build(out: Path, pdf: Path) -> None:
    s = pd.read_csv(out / "tables" / "sessions.csv", keep_default_na=False, na_values=["", "NaN", "nan"])
    s["unique"] = s.unique.astype(str).str.lower() == "true"
    psd = pd.read_csv(out / "tables" / "psd.csv")
    psd["unique"] = psd.unique.astype(str).str.lower() == "true"
    dm, fm, desc = load_metadata(out)
    u = s[s.unique]
    tmp = Path(tempfile.mkdtemp())
    imgs = figures(out, s, psd, tmp)

    d, footer = doc(pdf, f"DANDI:001333 {dm['name']}", f"DANDI:001333 v{VERSION} · {dm['name']}")
    story = [Paragraph(f"DANDI:001333: {esc(dm['name'])}", H1), legend_para(), Spacer(1, 4)]
    story += published_section(dm, fm, "1")

    story += [Spacer(1, 6), Paragraph("2. Contents of the files", H2)]
    lfp, arv = s[s.signal == "lfp"], s[s.signal == "beta_arv"]
    rows = [["Quantity", "Value", "Source"],
            ["Files per NWB subject_id", ", ".join(f"{k} ({v})" for k, v in
                                                   s.file.str.split("/").str[0].str.removeprefix("sub-").value_counts().sort_index().items()), SRC_NWB],
            ["subject.description (all files)", fm.get("subject", {}).get("description", ""), SRC_NWB],
            ["subject species / sex / age (all files)", f"{fm['subject'].get('species')} / {fm['subject'].get('sex')} / {fm['subject'].get('age')}", SRC_NWB],
            ["Device", "; ".join(f"{k}: {v}" for k, v in fm.get("devices", {}).items()), SRC_NWB],
            ["Electrode groups", f"{len(fm.get('electrode_groups', {}))} groups; location '{next(iter(fm['electrode_groups'].values()))['location']}'", SRC_NWB],
            ["LFP series ('LFP')", f"{len(lfp)} files; {lfp.fs_hz.iloc[0]:g} Hz; durations " +
             ", ".join(f"{d_:g} s ({n})" for d_, n in lfp.duration_s.round(2).value_counts().items()), SRC_NWB],
            ["Beta ARV series ('Beta_Band_Voltage')", f"{len(arv)} files; timestamps every {1 / arv.fs_hz.iloc[0]:g} s "
             f"from {arv.t_start_s.min():.2f} to {arv.t_end_s.max():.2f} s ({arv.n_samples.iloc[0]} samples)", SRC_NWB],
            ["Beta ARV definition", "Given in the DANDI description (section 1): 'fully rectifying the filtered LFP signal using a "
             "fourth-order Chebyshev band-pass filter with an 8 Hz bandwidth, centered around the peak of the LFP power spectrum'", SRC_DANDI],
            ["Unique signals (by MD5)", ", ".join(f"{g} {sg}: {int(q.unique.sum())} of {len(q)}"
                                                  for (g, sg), q in s.groupby(["group", "signal"])), SRC_EXPORT]]
    story.append(table(rows, [5.5 * cm, W - 9.5 * cm, 4 * cm]))

    story += [Paragraph("3. Export parameters", H2)]
    rows = [["Parameter", "Definition", "Source"],
            ["group, signal", "from the NWB subject_id: 'healthy'/'parkinson' prefix; '-beta' → beta_arv, '-lfp' or '-data' → lfp", SRC_EXPORT],
            ["Power spectrum", f"Welch, {WELCH_SEG_S:g} s Hann segments (0.5 Hz resolution), mean removed; 0–100 Hz kept", SRC_EXPORT],
            ["beta_power_v2", f"PSD integrated over {BETA[0]:g}–{BETA[1]:g} Hz (the band named in the DANDI description)", SRC_EXPORT],
            ["relative_beta_power", f"beta_power / power over {TOTAL[0]:g}–{TOTAL[1]:g} Hz", SRC_EXPORT],
            ["peak_freq_hz", f"frequency of the PSD maximum within {PEAK_SEARCH[0]:g}–{PEAK_SEARCH[1]:g} Hz", SRC_EXPORT],
            ["unique, duplicate_of", "MD5 of each file's signal values; duplicate_of names the first file with identical values", SRC_EXPORT]]
    story.append(table(rows, [4 * cm, W - 8 * cm, 4 * cm]))

    story += [PageBreak(), Paragraph(f"4. Figures ({SRC_EXPORT})", H2), imgs["lfp"], Spacer(1, 4),
              Table([[imgs["psd"], imgs["arv"]]], colWidths=[W / 2, W / 2])]
    story += [PageBreak(), imgs["strip"], Spacer(1, 6)]

    story.append(Paragraph(f"5. Summary per group and signal, unique files ({SRC_EXPORT})", H2))
    rows = [["Group", "Signal", "Unique files", "13–30 Hz power mean (V²)", "SD", "Relative power mean", "SD",
             "Peak frequencies (Hz)", "Signal SD mean (V)", "Signal mean (V)"]]
    for (g, sg), q in u.groupby(["group", "signal"]):
        is_lfp = sg == "lfp"
        rows.append([g, sg, len(q),
                     f"{q.beta_power_v2.mean():.3e}" if is_lfp else "",
                     f"{q.beta_power_v2.std():.2e}" if is_lfp and len(q) > 1 else "",
                     fmt(q.relative_beta_power.mean(), 3) if is_lfp else "",
                     fmt(q.relative_beta_power.std(), 3) if is_lfp and len(q) > 1 else "",
                     ", ".join(f"{x:g}" for x in sorted(q.peak_freq_hz.unique())) if is_lfp else "",
                     f"{q.sd_v.mean():.3e}", f"{q.mean_v.mean():.3e}"])
    story.append(table(rows, [2.6 * cm, 2 * cm, 1.8 * cm] + [(W - 6.4 * cm) / 7] * 7))

    story += [PageBreak(), Paragraph("6. All files", H2)]
    rows = [["Sample", "Group", "Signal", "Samples", "fs (Hz)", "Duration (s)", "Unique", "Identical to",
             "SD (V)", "13–30 Hz power (V²)", "Relative", "Peak (Hz)"]]
    for r in s.itertuples(index=False):
        is_lfp = r.signal == "lfp"
        rows.append([r.sample_id, r.group, r.signal, r.n_samples, fmt(r.fs_hz, 0), fmt(r.duration_s),
                     "yes" if r.unique else "no", r.duplicate_of if isinstance(r.duplicate_of, str) else "",
                     f"{r.sd_v:.3e}", f"{r.beta_power_v2:.3e}" if is_lfp else "",
                     fmt(r.relative_beta_power, 3) if is_lfp else "", fmt(r.peak_freq_hz, 1) if is_lfp else ""])
    story.append(table(rows, [4.6 * cm, 2.3 * cm, 1.7 * cm, 1.6 * cm, 1.4 * cm, 1.8 * cm, 1.3 * cm, 4.6 * cm]
                       + [(W - 19.3 * cm) / 4] * 4))
    story.append(Paragraph("Signal values: NWB file. Durations, uniqueness and spectral values: computed by this export.", SMALL))

    dup = s[~s.unique]
    story += [PageBreak(), Paragraph("7. Data checks performed by this export", H2),
              Paragraph(f"<b>Identical files.</b> Comparing the MD5 hash of each file's signal values: {len(u)} of {len(s)} files "
                        f"are unique. {len(dup)} files are identical to another file: "
                        + "; ".join(f"{g} {sg}: {len(q)} files identical to {q.duplicate_of.iloc[0]}"
                                    for (g, sg), q in dup.groupby(["group", "signal"]))
                        + ". Every parkinsonian file is unique.", NOTE),
              Paragraph("<b>Beta ARV timing.</b> The Beta_Band_Voltage series has no <i>rate</i> field; timing comes from its "
                        "timestamps (every 0.02 s).", NOTE)]
    story += descriptions_section(desc, "8", fm.get("source_file"))

    story += [Spacer(1, 8), Paragraph("9. Exported files", H2)]
    dd = [["File", "Rows", "Columns"],
          ["tables/sessions.csv", str(len(s)), ", ".join(s.columns)],
          ["tables/psd.csv", f"{len(psd):,}", "sample_id, group, unique, freq_hz, psd_v2_per_hz (LFP files, 0–100 Hz)"],
          ["tables/dataset_metadata.json, nwb_file_metadata.json, nwb_descriptions.csv", "", "the published metadata quoted in sections 1 and 8"],
          ["signals/<sample_id>.csv", "one per NWB file", "time_s, voltage_v"]]
    story.append(table(dd, [6 * cm, 3 * cm, W - 9 * cm]))
    d.build(story, onFirstPage=footer, onLaterPages=footer)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("pdf", type=Path)
    a = ap.parse_args(argv)
    build(a.out_dir, a.pdf)
    print(f"wrote {a.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
