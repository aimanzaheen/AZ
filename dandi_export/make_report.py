"""Build a PDF report of the DANDI:000017 export: recording/task/export
parameters, per-session and per-region data tables, figures and a data
dictionary.

Usage:
    python dandi_export/make_report.py dandi_export/output dandi_export/DANDI_000017_report.pdf
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
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).parent))
from export_000017 import WHEEL_DOWNSAMPLE_HZ, WINDOWS  # noqa: E402

LIT_REGIONS = {"ZI", "LH"}  # regions studied in otto_reextraction/
ACCENT = colors.HexColor("#2f5d8a")
HILITE = colors.HexColor("#fff2cc")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=ACCENT, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=ACCENT, spaceBefore=8)
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=9.5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7, leading=8.5)


def table(rows, col_widths=None, header_rows=1, highlight=()):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    style = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]
    for r in highlight:
        style.append(("BACKGROUND", (0, r), (-1, r), HILITE))
    t.setStyle(TableStyle(style))
    return t


def header_white(rows):
    # Header cells need white text; wrap them in a font tag.
    rows[0] = [f'<font color="white"><b>{c}</b></font>' for c in rows[0]]
    return rows


def fmt(x, nd=2):
    if isinstance(x, (float, np.floating)):
        return "" if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def fig_to_image(fig, tmp: Path, name: str, width_cm: float) -> Image:
    p = tmp / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    w, h = fig.get_size_inches()
    return Image(str(p), width=width_cm * cm, height=width_cm * cm * h / w)


def figures(units, trials, regsum, tmp):
    imgs = {}
    good = units[units.good & (units.region != "root")]

    # 1. Spontaneous rate by region (regions with >= 20 good units).
    top = regsum[regsum.n_units >= 20].sort_values("spont_rate_median_hz")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    data = [good.loc[good.region == r, "spontaneous_rate_hz"].values for r in top.region]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, r in zip(bp["boxes"], top.region):
        patch.set_facecolor("#e8a33d" if r in LIT_REGIONS else "#9db8d3")
    ax.set_xticks(range(1, len(top) + 1), top.region, rotation=90, fontsize=7)
    ax.set_ylabel("Spontaneous rate (Hz)")
    ax.set_title("Spontaneous firing rate by region (good units; regions with ≥20 units; ZI/LH in orange)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["spont"] = fig_to_image(fig, tmp, "spont", 25)

    # 2. Waveform duration distribution.
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(good.waveform_duration_ms, bins=np.arange(0, 1.6, 0.0333), color="#5b85aa")
    ax.axvline(0.4, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Trough-to-peak waveform duration (ms)")
    ax.set_ylabel("Units")
    ax.set_title("Narrow (<0.4 ms) vs wide-spiking units", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["wave"] = fig_to_image(fig, tmp, "wave", 12)

    # 3. Psychometric curve: P(left choice) vs contrast difference.
    t = trials[trials.included.astype(bool)].copy()
    t["cdiff"] = (t.visual_stimulus_right_contrast - t.visual_stimulus_left_contrast).round(2)
    g = t.groupby("cdiff").response_choice
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = g.size().index
    ax.plot(x, g.apply(lambda s: (s == 1).mean()), "o-", color="#2f5d8a", label="left")
    ax.plot(x, g.apply(lambda s: (s == -1).mean()), "o-", color="#c0504d", label="right")
    ax.plot(x, g.apply(lambda s: (s == 0).mean()), "o-", color="#888888", label="no-go")
    ax.set_xlabel("Contrast right − left")
    ax.set_ylabel("P(choice)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Behaviour, all sessions (included trials)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["psy"] = fig_to_image(fig, tmp, "psy", 12)
    return imgs


def build(out_dir: Path, pdf: Path) -> None:
    tables = out_dir / "tables"
    sessions = pd.read_csv(tables / "sessions.csv")
    units = pd.read_csv(tables / "units.csv")
    trials = pd.read_csv(tables / "trials.csv")
    regsum = pd.read_csv(tables / "region_summary.csv")
    good = units[units.good]
    age_days = sessions.age.str.extract(r"(\d+)")[0].astype(int)

    doc = SimpleDocTemplate(
        str(pdf), pagesize=landscape(A4),
        leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm,
        title="DANDI:000017 export: parameters and data",
    )
    W = landscape(A4)[0] - 3 * cm
    story = []
    tmp = Path(tempfile.mkdtemp())
    imgs = figures(units, trials, regsum, tmp)

    # ---- Overview ---------------------------------------------------------
    story += [
        Paragraph("DANDI:000017: Parameters and Data", H1),
        Paragraph(
            "<b>Distributed coding of choice, action and engagement across the mouse brain</b> "
            "(Steinmetz, Zatka-Haas, Carandini &amp; Harris, <i>Nature</i> 576, 266–273, 2019). "
            "Dandiset version 0.240329.1926, converted from NWB to CSV/Parquet with "
            "<font face='Courier'>dandi_export/export_000017.py</font>.",
            BODY,
        ),
        Spacer(1, 6),
    ]
    zi = good[good.region == "ZI"]
    lh = good[good.region == "LH"]
    overview = [
        ["Quantity", "Value"],
        ["Sessions / mice", f"{len(sessions)} / {sessions.subject_id.nunique()}"],
        ["Trials (included)", f"{len(trials):,} ({int(trials.included.astype(bool).sum()):,})"],
        ["Sorted units (good, annotation ≥ 2)", f"{len(units):,} ({len(good):,})"],
        ["Brain regions (excluding 'root')", str(units.loc[units.region != 'root', 'region'].nunique())],
        ["Total spikes", f"{int(units.n_spikes.sum()):,}"],
        ["Total recording time", f"{sessions.duration_s.sum() / 3600:.1f} h"],
        ["ZI good units (sessions)", f"{len(zi)} ({zi.session_id.nunique()})"],
        ["LH good units (sessions)", f"{len(lh)} ({lh.session_id.nunique()})"],
    ]
    story.append(table(header_white(overview), [8 * cm, 8 * cm]))

    # ---- Parameters -------------------------------------------------------
    story.append(Paragraph("1. Parameters", H2))
    params = [
        ["Group", "Parameter", "Value"],
        ["Subjects", "Species / strain", "Mus musculus; C57Bl6/J background"],
        ["Subjects", "Sex", ", ".join(f"{k}: {v}" for k, v in sessions.drop_duplicates('subject_id').sex.value_counts().items())],
        ["Subjects", "Genotypes", "; ".join(sorted(sessions.genotype.dropna().unique()))],
        ["Subjects", "Age range", f"{age_days.min()} – {age_days.max()} days"],
        ["Recording", "Probes", "Neuropixels Phase 3A, 1–3 probes per session, 384 channels each"],
        ["Recording", "Spike sampling rate", "30 kHz"],
        ["Recording", "Spike sorting", "Kilosort + manual curation in Phy (0 noise excluded; 1 MUA; 2 good; 3 unsorted)"],
        ["Recording", "Anatomy", "Probe tracks registered to the Allen Common Coordinate Framework (CCF); region = CCF acronym at the unit's peak channel"],
        ["Task", "Paradigm", "2AFC visual contrast discrimination with no-go option; mouse turns a wheel to bring the higher-contrast grating to centre"],
        ["Task", "Contrasts (each side)", ", ".join(f"{c:g}" for c in sorted(trials.visual_stimulus_left_contrast.unique()))],
        ["Task", "Trial events", "visual stimulus onset → auditory go cue → wheel response → feedback (water reward or white-noise burst)"],
        ["Task", "Choice coding", "+1 left, −1 right, 0 no-go; feedback +1 reward, −1 noise"],
        ["Spontaneous", "Definition", "Intervals with no task or stimulus presentation running"],
        ["Spontaneous", "Duration per session", f"median {sessions.spontaneous_duration_s.median():.0f} s (range {sessions.spontaneous_duration_s.min():.0f}–{sessions.spontaneous_duration_s.max():.0f} s)"],
        ["Behaviour", "Signals", f"wheel position (2500 Hz → downsampled to {WHEEL_DOWNSAMPLE_HZ:g} Hz), wheel movements, licks, pupil area/position, face motion energy"],
    ]
    for w, (ev, lo, hi) in WINDOWS.items():
        params.append(["Export", f"Spike-count window '{w}'", f"{lo:+g} to {hi:+g} s relative to {ev}"])
    params += [
        ["Export", "Firing rates", "mean = spikes / whole recording; task = within first-trial-start … last-trial-end; spontaneous = within spontaneous intervals"],
        ["Export", "Waveform duration", "trough-to-peak of the mean extracellular waveform (samples / 30 kHz). Not equivalent to intracellular AP half-width"],
    ]
    story.append(table(header_white(params), [2.5 * cm, 5 * cm, W - 7.5 * cm]))

    # ---- Figures ---------------------------------------------------------
    story += [PageBreak(), Paragraph("2. Overview figures", H2), imgs["spont"], Spacer(1, 6)]
    story.append(Table([[imgs["wave"], imgs["psy"]]], colWidths=[W / 2, W / 2]))

    # ---- Region summary --------------------------------------------------
    story += [PageBreak(), Paragraph("3. Firing rates by brain region (good units)", H2),
              Paragraph("Rows for regions studied in the literature extraction (ZI, LH) are highlighted. "
                        "Spontaneous rate is the value most directly comparable to the literature "
                        "'Spontaneous firing rate in Hz' field. Note that these are extracellular recordings in awake behaving mice, "
                        "not slice patch-clamp.", SMALL), Spacer(1, 4)]
    cols = ["region", "n_units", "n_sessions", "spont_rate_mean_hz", "spont_rate_sd_hz",
            "spont_rate_median_hz", "mean_rate_mean_hz", "waveform_duration_ms_median"]
    rows = [["Region", "Units", "Sessions", "Spont. mean (Hz)", "Spont. SD (Hz)",
             "Spont. median (Hz)", "Overall mean (Hz)", "Waveform dur. median (ms)"]]
    hl = []
    for i, r in enumerate(regsum[cols].itertuples(index=False), start=1):
        rows.append([fmt(v) for v in r])
        if r.region in LIT_REGIONS:
            hl.append(i)
    story.append(table(header_white(rows), [W / 8] * 8, highlight=hl))

    # ---- ZI / LH unit detail --------------------------------------------
    lit = good[good.region.isin(LIT_REGIONS)].sort_values(["region", "session_id", "unit_id"])
    if len(lit):
        story += [PageBreak(), Paragraph("4. Unit-level data: ZI and LH", H2)]
        ucols = ["region", "session_id", "unit_id", "ccf_ap", "ccf_dv", "ccf_lr", "depth_um",
                 "waveform_duration_ms", "n_spikes", "mean_rate_hz", "task_rate_hz", "spontaneous_rate_hz"]
        rows = [["Region", "Session", "Unit", "CCF AP", "CCF DV", "CCF LR", "Depth (µm)",
                 "Wave dur. (ms)", "Spikes", "Mean (Hz)", "Task (Hz)", "Spont. (Hz)"]]
        for r in lit[ucols].itertuples(index=False):
            rows.append([fmt(v, 1) if isinstance(v, float) and k in (3, 4, 5, 6) else fmt(v)
                         for k, v in enumerate(r)])
        story.append(table(header_white(rows), [1.5 * cm, 3.4 * cm] + [(W - 4.9 * cm) / 10] * 10))

    # ---- Sessions --------------------------------------------------------
    story += [PageBreak(), Paragraph("5. Sessions", H2)]
    rows = [["Session", "Sex", "Age", "Genotype", "Duration (s)", "Trials", "Units", "Good", "Regions"]]
    for s in sessions.itertuples(index=False):
        rows.append([s.session_id, s.sex, s.age, s.genotype, f"{s.duration_s:.0f}", s.n_trials,
                     s.n_units, s.n_good_units, s.regions.replace(";", ", ")])
    story.append(table(header_white(rows),
                       [3.2 * cm, 1 * cm, 1.6 * cm, 3.6 * cm, 1.8 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, W - 15.1 * cm]))

    # ---- Per-session behaviour ------------------------------------------
    story += [Spacer(1, 8), Paragraph("6. Behavioural performance per session", H2)]
    t = trials.copy()
    t["correct"] = t.feedback_type == 1
    b = t.groupby("session_id").agg(
        trials=("trial_id", "size"),
        included=("included", lambda s: int(s.astype(bool).sum())),
        pct_correct=("correct", "mean"),
        pct_nogo=("response_choice", lambda s: (s == 0).mean()),
    )
    rt = (t.response_time - t.go_cue).groupby(t.session_id).median()
    rows = [["Session", "Trials", "Included", "% rewarded", "% no-go", "Median response − go cue (s)"]]
    for sid, r in b.iterrows():
        rows.append([sid, r.trials, r.included, f"{100 * r.pct_correct:.1f}", f"{100 * r.pct_nogo:.1f}", f"{rt[sid]:.3f}"])
    story.append(table(header_white(rows), [4 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 5 * cm]))

    # ---- Data dictionary -------------------------------------------------
    story += [PageBreak(), Paragraph("7. Data files and columns", H2)]
    dd = [
        ["File", "Rows", "Columns"],
        ["tables/sessions.csv", f"{len(sessions)}", ", ".join(sessions.columns)],
        ["tables/units.csv", f"{len(units):,}", ", ".join(units.columns)],
        ["tables/trials.csv", f"{len(trials):,}", ", ".join(trials.columns)],
        ["tables/region_summary.csv", f"{len(regsum)}", ", ".join(regsum.columns)],
        ["trial_unit_counts.parquet", "trials × units per session",
         "session_id, trial_id, unit_id, " + ", ".join(f"count_{w}" for w in WINDOWS)],
        ["spikes/&lt;session&gt;.parquet", f"{int(units.n_spikes.sum()):,} total", "unit_id, time (s), amp, depth (µm)"],
        ["behavior/&lt;session&gt;__&lt;signal&gt;.parquet", "per signal",
         "wheel_position, wheel_moves (start_time, stop_time, type), lick_times, eye_area, eye_xy_positions, face_motion_energy"],
    ]
    story.append(table(header_white(dd), [5.5 * cm, 3.5 * cm, W - 9 * cm]))
    story += [Spacer(1, 6), Paragraph(
        "Keys: <font face='Courier'>session_id</font> joins every table; "
        "<font face='Courier'>(session_id, unit_id)</font> joins units ↔ spikes ↔ trial counts; "
        "<font face='Courier'>(session_id, trial_id)</font> joins trials ↔ trial counts. "
        "All times are seconds on the session clock. Column descriptions: dandi_export/README.md.", BODY)]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.5 * cm, 0.7 * cm, "DANDI:000017 v0.240329.1926 · Steinmetz et al. 2019")
        canvas.drawRightString(landscape(A4)[0] - 1.5 * cm, 0.7 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", type=Path, help="export_000017.py output folder")
    ap.add_argument("pdf", type=Path)
    a = ap.parse_args(argv)
    build(a.out_dir, a.pdf)
    print(f"wrote {a.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
