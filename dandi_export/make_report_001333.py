"""Build a PDF report of the DANDI:001333 (PESD) export.

Usage:
    python dandi_export/make_report_001333.py dandi_export/output_001333 \
        dandi_export/DANDI_001333_report.pdf
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).parent))
from export_001333 import BETA, TOTAL, WELCH_SEG_S  # noqa: E402
from make_report import BODY, H1, H2, SMALL, fig_to_image, fmt, header_white, table  # noqa: E402

VERSION = "0.250327.2220"
HEALTHY, PD = "#2f5d8a", "#c0504d"
WARN = ParagraphStyle("warn", parent=BODY, backColor=colors.HexColor("#fdecea"),
                      borderColor=colors.HexColor("#c0504d"), borderWidth=0.8, borderPadding=6,
                      spaceBefore=4, spaceAfter=14)


def sig(out: Path, sid: str) -> pd.DataFrame:
    return pd.read_csv(out / "signals" / f"{sid}.csv")


def figures(out, s, psd, tmp):
    imgs = {}
    u = s[s.unique]
    h_lfp = u[(u.group == "healthy") & (u.signal == "lfp")].sample_id.iloc[0]
    p_lfp = u[(u.group == "parkinsonian") & (u.signal == "lfp")].sample_id.iloc[0]
    h_b = u[(u.group == "healthy") & (u.signal == "beta_arv")].sample_id.iloc[0]
    p_b = u[(u.group == "parkinsonian") & (u.signal == "beta_arv")].sample_id.iloc[0]

    fig, axs = plt.subplots(2, 1, figsize=(11, 3.6), sharex=True)
    for ax, sid, c, lab in [(axs[0], h_lfp, HEALTHY, "healthy"), (axs[1], p_lfp, PD, "parkinsonian")]:
        d = sig(out, sid)
        w = d[(d.time_s >= 20) & (d.time_s < 21)]
        ax.plot(w.time_s, w.voltage_v * 1e3, color=c, lw=0.7)
        ax.set_ylabel("mV", fontsize=8)
        ax.set_title(f"STN LFP, {lab} ({sid}), 1 s excerpt", fontsize=8, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axs[1].set_xlabel("Time (s)")
    imgs["lfp"] = fig_to_image(fig, tmp, "lfp", 24)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for g, c in [("healthy", HEALTHY), ("parkinsonian", PD)]:
        q = psd[(psd.group == g) & psd.unique]
        m = q.groupby("freq_hz").psd_v2_per_hz
        mean, sd = m.mean(), m.std().fillna(0)
        n = q.sample_id.nunique()
        ax.semilogy(mean.index, mean.values, color=c, label=f"{g} (n = {n} unique)")
        if n > 1:
            ax.fill_between(mean.index, (mean - sd).clip(lower=1e-15), mean + sd, color=c, alpha=0.2, lw=0)
    ax.axvspan(*BETA, color="#e8a33d", alpha=0.15, lw=0, label="beta 13–30 Hz")
    ax.set_xlim(0, 60)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("LFP power spectrum (Welch, mean ± SD)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["psd"] = fig_to_image(fig, tmp, "psd", 12)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for sid, c, lab in [(h_b, HEALTHY, "healthy"), (p_b, PD, "parkinsonian")]:
        d = sig(out, sid)
        ax.plot(d.time_s, d.voltage_v * 1e6, color=c, lw=0.8, label=f"{lab} ({sid.split('_ses-')[1]})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Beta ARV (µV)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Beta-band ARV signal (50 Hz)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["arv"] = fig_to_image(fig, tmp, "arv", 12)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
    specs = [("lfp", "beta_power_v2", 1e9, "LFP beta power (×10⁻⁹ V²)"),
             ("lfp", "relative_beta_power", 1, "LFP relative beta power (13–30 / 1–100 Hz)"),
             ("beta_arv", "mean_v", 1e6, "Mean beta ARV (µV)")]
    rng = np.random.default_rng(0)
    for ax, (sg, col, k, lab) in zip(axs, specs):
        for i, (g, c) in enumerate([("healthy", HEALTHY), ("parkinsonian", PD)]):
            y = u[(u.group == g) & (u.signal == sg)][col].values * k
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=14, color=c, alpha=0.8)
            ax.hlines(np.mean(y), i - 0.25, i + 0.25, color="k", lw=1)
        ax.set_xticks([0, 1], ["healthy", "parkinsonian"])
        ax.set_title(lab, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Unique signals only (duplicate files counted once)", fontsize=8)
    imgs["strip"] = fig_to_image(fig, tmp, "strip", 24)
    return imgs


def build(out: Path, pdf: Path) -> None:
    s = pd.read_csv(out / "tables" / "sessions.csv", keep_default_na=False, na_values=["", "NaN", "nan"])
    s["unique"] = s.unique.astype(str).str.lower() == "true"
    psd = pd.read_csv(out / "tables" / "psd.csv")
    psd["unique"] = psd.unique.astype(str).str.lower() == "true"
    u = s[s.unique]

    doc = SimpleDocTemplate(
        str(pdf), pagesize=landscape(A4),
        leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm,
        title="DANDI:001333 PESD: parameters and data",
    )
    W = landscape(A4)[0] - 3 * cm
    tmp = Path(tempfile.mkdtemp())
    imgs = figures(out, s, psd, tmp)
    story = [
        Paragraph("DANDI:001333: Parkinson's Electrophysiological Signal Dataset (PESD)", H1),
        Paragraph(
            "Biswas, A. (2025). <i>Parkinson's Electrophysiological Signal Dataset (PESD)</i> "
            f"(Version {VERSION}) [Data set]. DANDI Archive. doi:10.48324/dandi.001333/{VERSION}. "
            "BrainX Lab, Michigan Technological University. Companion paper: arXiv:2407.17756. "
            "Converted with <font face='Courier'>dandi_export/export_001333.py</font>.", BODY),
        Spacer(1, 10),
        Paragraph(
            "<b>These are simulated signals, not recordings.</b> All 52 files come from the computational "
            "cortico-basal-ganglia network model of Fleming et al. (2020, <i>Front. Neurosci.</i> 14:166, "
            "doi:10.3389/fnins.2020.00166), run in NEURON. The 'subjects' are simulation conditions; species "
            "'Homo sapiens', age 0 and sex 'U' in the files are placeholders. "
            f"<b>The {int((~s.unique).sum())} duplicate files:</b> the 10 healthy beta-ARV files are byte-identical, and "
            "the 10 healthy LFP files plus 'healthy-simulated-data' are byte-identical, so the healthy condition is a "
            f"<b>single simulation run</b>. Only {len(u)} of the 52 files are unique. Healthy-vs-parkinsonian "
            "statistics that count the healthy files as independent samples are pseudo-replicated.", WARN),
    ]
    counts = s.groupby(["group", "signal"]).agg(files=("sample_id", "size"), unique=("unique", "sum")).reset_index()
    rows = [["Group", "Signal", "Files", "Unique signals"]] + [[r.group, r.signal, r.files, int(r.unique)] for r in counts.itertuples(index=False)]
    story.append(table(header_white(rows), [4 * cm, 4 * cm, 3 * cm, 3 * cm]))

    story.append(Paragraph("1. Parameters", H2))
    lfp, arv = s[s.signal == "lfp"], s[s.signal == "beta_arv"]
    params = [
        ["Group", "Parameter", "Value"],
        ["Source", "Model", "Cortico-basal-ganglia network (cortex, STN, GPe, GPi, thalamus), Fleming et al. 2020; simulated in NEURON"],
        ["Source", "Conditions", "healthy vs parkinsonian network parameters (parkinsonian shows exaggerated 13–30 Hz beta oscillations in STN)"],
        ["Source", "'Electrodes'", "12 virtual contacts (4 shanks × 3) on a 'NEURON_Simulator' virtual probe; location 'Simulated Cortico-basal-ganglia network of brain'"],
        ["LFP", "Signal", "STN local field potential, 1 channel per file (V)"],
        ["LFP", "Sampling rate", f"{lfp.fs_hz.iloc[0]:g} Hz"],
        ["LFP", "Duration", ", ".join(f"{d:g} s ({n} files)" for d, n in lfp.duration_s.round(2).value_counts().items())],
        ["Beta ARV", "Definition", "average rectified value of the beta band: LFP band-pass filtered with a 4th-order Chebyshev filter, "
                                   "8 Hz bandwidth centred on the LFP spectral peak, then fully rectified and averaged (from the dataset description)"],
        ["Beta ARV", "Sampling / span", f"{arv.fs_hz.iloc[0]:g} Hz, t = {arv.t_start_s.min():.2f}–{arv.t_end_s.max():.2f} s ({arv.n_samples.iloc[0]} samples)"],
        ["Export", "Power spectrum", f"Welch, {WELCH_SEG_S:g} s Hann segments (0.5 Hz resolution), mean removed, 0–100 Hz kept"],
        ["Export", "Beta power", f"integral of PSD over {BETA[0]:g}–{BETA[1]:g} Hz; relative = beta / {TOTAL[0]:g}–{TOTAL[1]:g} Hz power; peak searched 5–45 Hz"],
        ["Export", "Duplicates", "MD5 of each file's signal; duplicate_of names the first file with identical data"],
    ]
    story.append(table(header_white(params), [2.3 * cm, 3.5 * cm, W - 5.8 * cm]))

    story += [PageBreak(), Paragraph("2. Figures", H2), imgs["lfp"], Spacer(1, 4),
              Table([[imgs["psd"], imgs["arv"]]], colWidths=[W / 2, W / 2])]
    story += [PageBreak(), imgs["strip"], Spacer(1, 6)]

    story.append(Paragraph("3. Group summary (unique signals only)", H2))
    rows = [["Group", "Signal", "n", "Beta power mean (V²)", "Beta power SD", "Relative beta mean", "Relative beta SD",
             "Peak freq (Hz)", "Signal SD mean (V)", "Mean value (V)"]]
    for (g, sg), q in u.groupby(["group", "signal"]):
        rows.append([g, sg, len(q),
                     f"{q.beta_power_v2.mean():.3e}" if sg == "lfp" else "", f"{q.beta_power_v2.std():.2e}" if sg == "lfp" and len(q) > 1 else "",
                     fmt(q.relative_beta_power.mean(), 3) if sg == "lfp" else "", fmt(q.relative_beta_power.std(), 3) if sg == "lfp" and len(q) > 1 else "",
                     ", ".join(f"{x:g}" for x in sorted(q.peak_freq_hz.unique())) if sg == "lfp" else "",
                     f"{q.sd_v.mean():.3e}", f"{q.mean_v.mean():.3e}"])
    story.append(table(header_white(rows), [2.6 * cm, 2 * cm, 1 * cm] + [(W - 5.6 * cm) / 7] * 7))
    ratio = u[(u.signal == "lfp") & (u.group == "parkinsonian")].beta_power_v2.mean() / \
        u[(u.signal == "lfp") & (u.group == "healthy")].beta_power_v2.mean()
    story.append(Paragraph(f"Parkinsonian LFP beta power is {ratio:.1f}× the single healthy run's.", SMALL))

    story += [PageBreak(), Paragraph("4. All files", H2)]
    rows = [["Sample", "Group", "Signal", "Samples", "fs (Hz)", "Duration (s)", "Unique", "Duplicate of",
             "SD (V)", "Beta power (V²)", "Rel. beta", "Peak (Hz)"]]
    for r in s.itertuples(index=False):
        lfp_row = r.signal == "lfp"
        rows.append([r.sample_id, r.group, r.signal, r.n_samples, fmt(r.fs_hz, 0), fmt(r.duration_s),
                     "yes" if r.unique else "no", r.duplicate_of if isinstance(r.duplicate_of, str) else "",
                     f"{r.sd_v:.3e}", f"{r.beta_power_v2:.3e}" if lfp_row else "",
                     fmt(r.relative_beta_power, 3) if lfp_row else "", fmt(r.peak_freq_hz, 1) if lfp_row else ""])
    hl = [i for i, r in enumerate(s.itertuples(index=False), start=1) if not r.unique]
    t = table(header_white(rows), [4.6 * cm, 2.3 * cm, 1.7 * cm, 1.6 * cm, 1.4 * cm, 1.8 * cm, 1.3 * cm, 4.6 * cm,
                                   (W - 19.3 * cm) / 4] + [(W - 19.3 * cm) / 4] * 3, highlight=hl)
    story += [Paragraph("Duplicate files are highlighted.", SMALL), t]

    story += [PageBreak(), Paragraph("5. Data files and columns", H2)]
    dd = [
        ["File", "Rows", "Columns"],
        ["tables/sessions.csv", str(len(s)), ", ".join(s.columns)],
        ["tables/psd.csv", f"{len(psd):,}", "sample_id, group, unique, freq_hz, psd_v2_per_hz (LFP files only, 0–100 Hz)"],
        ["signals/&lt;sample_id&gt;.csv", "one file per NWB file", "time_s, voltage_v (the raw signal, SI units)"],
    ]
    story.append(table(header_white(dd), [5 * cm, 3.5 * cm, W - 8.5 * cm]))
    story += [Spacer(1, 6), Paragraph(
        "For modelling, filter <font face='Courier'>sessions.csv</font> to <font face='Courier'>unique == True</font>. "
        "Beta-ARV and LFP files sharing a session number within a group appear to be paired (the ARV derived from that "
        "LFP), but the files do not state this explicitly. Column descriptions: dandi_export/README_001333.md.", BODY)]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.5 * cm, 0.7 * cm, f"DANDI:001333 v{VERSION} · PESD (simulated)")
        canvas.drawRightString(landscape(A4)[0] - 1.5 * cm, 0.7 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


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
