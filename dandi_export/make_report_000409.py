"""Build a PDF report of the DANDI:000409 (IBL Brain Wide Map) export.

Usage:
    python dandi_export/make_report_000409.py dandi_export/output_000409 \
        dandi_export/DANDI_000409_report.pdf
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
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

sys.path.insert(0, str(Path(__file__).parent))
from export_000409 import VERSION, WINDOWS  # noqa: E402
from make_report import BODY, H1, H2, SMALL, fig_to_image, fmt, header_white, table  # noqa: E402

# Zona incerta + lateral hypothalamic area, as in otto_reextraction/.
# (Allen "LH" is the lateral habenula.)
LIT_REGIONS = {"ZI", "LHA"}
MAX_UNIT_ROWS = 600


def figures(units, trials, regsum, tmp):
    imgs = {}
    good = units[units.good & units.region.isin(regsum.region)]

    top = regsum[regsum.n_units >= 20].nlargest(70, "n_units")
    top = pd.concat([top, regsum[regsum.region.isin(LIT_REGIONS) & ~regsum.region.isin(top.region)]])
    top = top.sort_values("prestim_rate_median_hz")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    data = [good.loc[good.region == r, "prestim_rate_hz"].values for r in top.region]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, r in zip(bp["boxes"], top.region):
        patch.set_facecolor("#e8a33d" if r in LIT_REGIONS else "#9db8d3")
    ax.set_xticks(range(1, len(top) + 1), top.region, rotation=90, fontsize=5.5)
    ax.set_ylabel("Pre-stimulus rate (Hz)")
    ax.set_title("Pre-stimulus (quiescent) firing rate by region: good units, 70 best-sampled regions plus ZI/LHA (orange)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["rate"] = fig_to_image(fig, tmp, "rate", 24)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(good.peak_to_trough_ms.dropna(), bins=np.arange(0, 1.6, 0.0333), color="#5b85aa")
    ax.axvline(0.4, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Peak-to-trough waveform duration (ms)")
    ax.set_ylabel("Units")
    ax.set_title("Narrow (<0.4 ms) vs wide-spiking units", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["wave"] = fig_to_image(fig, tmp, "wave", 12)

    t = trials.copy()
    sign = np.where(t.gabor_stimulus_side == "right", 1, -1)
    t["signed"] = t.gabor_stimulus_contrast * sign
    t["ccw"] = t.mouse_wheel_choice == "counter_clockwise"
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for p, c in [(0.2, "#c0504d"), (0.5, "#555555"), (0.8, "#2f5d8a")]:
        g = t[np.isclose(t.probability_left, p)].groupby("signed").ccw.mean()
        ax.plot(g.index, g.values, "o-", ms=3, color=c, label=f"P(left stim) = {p}")
    ax.set_xlabel("Signed contrast (%; + = right)")
    ax.set_ylabel("P(counter-clockwise turn)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Behaviour by block prior, all sessions", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["psy"] = fig_to_image(fig, tmp, "psy", 12)
    return imgs


def build(out_dir: Path, pdf: Path, note: str | None = None) -> None:
    tables = out_dir / "tables"
    sessions = pd.read_csv(tables / "sessions.csv")
    units = pd.read_csv(tables / "units.csv", keep_default_na=False, na_values=[""], low_memory=False)
    units["region"] = units.region.fillna("")
    trials = pd.read_csv(tables / "trials.csv", low_memory=False)
    regsum = pd.read_csv(tables / "region_summary.csv", keep_default_na=False, na_values=[""])
    good = units[units.good]

    doc = SimpleDocTemplate(
        str(pdf), pagesize=landscape(A4),
        leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm,
        title="DANDI:000409 IBL Brain Wide Map: parameters and data",
    )
    W = landscape(A4)[0] - 3 * cm
    story = []
    tmp = Path(tempfile.mkdtemp())
    imgs = figures(units, trials, regsum, tmp)

    story += [
        Paragraph("DANDI:000409: IBL Brain Wide Map, Parameters and Data", H1),
        Paragraph(
            "International Brain Laboratory, <i>A brain-wide map of neural activity during complex behaviour</i> "
            f"(Nature 2025). Dandiset version {VERSION}. Only the processed "
            "<font face='Courier'>desc-processed_behavior+ecephys.nwb</font> files were used "
            "(spike-sorted units, trials, behaviour); the ~43 TB of raw voltage and video was not downloaded. "
            "Converted with <font face='Courier'>dandi_export/export_000409.py</font>.", BODY),
        Spacer(1, 6),
    ]
    if note:
        story += [Paragraph(f"<b>{note}</b>", BODY), Spacer(1, 6)]
    zi, lha = good[good.region == "ZI"], good[good.region == "LHA"]
    overview = [
        ["Quantity", "Value"],
        ["Sessions / mice / labs", f"{len(sessions)} / {sessions.subject_id.nunique()} / {sessions.lab.nunique()}"],
        ["Probe insertions", f"{int(sessions.n_probes.sum())}"],
        ["Trials", f"{len(trials):,}"],
        ["Units (good: IBL quality score = 1)", f"{len(units):,} ({len(good):,})"],
        ["Brain regions with good units", f"{len(regsum)}"],
        ["Total spikes", f"{int(units.spike_count.clip(lower=0).sum()):,}"],
        ["ZI (zona incerta) good units / sessions / mice",
         f"{len(zi)} / {zi.session_id.nunique()} / {zi.subject_id.nunique()}"],
        ["LHA (lateral hypothalamic area) good units / sessions / mice",
         f"{len(lha)} / {lha.session_id.nunique()} / {lha.subject_id.nunique()}"],
    ]
    story.append(table(header_white(overview), [9 * cm, 8 * cm]))

    story.append(Paragraph("1. Parameters", H2))
    contrasts = ", ".join(f"{c:g}" for c in sorted(trials.gabor_stimulus_contrast.dropna().unique()))
    priors = ", ".join(f"{p:g}" for p in sorted(trials.probability_left.dropna().unique()))
    labs = ", ".join(f"{k} ({v})" for k, v in sessions.lab.value_counts().items())
    params = [
        ["Group", "Parameter", "Value"],
        ["Subjects", "Species / sex", f"Mus musculus (C57BL/6J); " + ", ".join(f"{k}: {v}" for k, v in sessions.drop_duplicates('subject_id').sex.value_counts().items())],
        ["Subjects", "Age at recording", f"median {sessions.age_days.median():.0f} days (range {sessions.age_days.min():.0f}–{sessions.age_days.max():.0f})"],
        ["Subjects", "Labs (sessions)", labs],
        ["Recording", "Probes", "Neuropixels 1.0 (384 channels), 1–2 probes per session; 30 kHz AP band"],
        ["Recording", "Spike sorting", "IBL pykilosort pipeline. 'good' = ibl_quality_score 1.0, i.e. passes all three IBL metrics (sliding refractory-period, noise cutoff, median amplitude > 50 µV)"],
        ["Recording", "Anatomy", "Histology-aligned channel locations in Allen CCF; region = Allen structure at each unit's max-amplitude electrode (acronym mapped from the full name)"],
        ["Task", "Paradigm", "IBL choice world: Gabor patch appears left or right; mouse turns a wheel to centre it. The wheel must be held still for a quiescence period before the stimulus "
         f"(observed {trials.quiescence_period.min():.2f}–{trials.quiescence_period.max():.2f} s, median {trials.quiescence_period.median():.2f} s)"],
        ["Task", "Contrasts (%)", contrasts],
        ["Task", "Block priors P(left)", f"{priors}: first 90 trials unbiased, then alternating 0.2/0.8 blocks"],
        ["Task", "Events", "stimulus onset + go cue (5 kHz, 100 ms) → wheel movement onset → choice registration (±35°) → feedback (water, or white noise + 2 s timeout)"],
        ["Task", "Choice coding", "mouse_wheel_choice: counter_clockwise / clockwise / no-go; is_mouse_rewarded True/False"],
        ["Protocol", "Task version(s)", "; ".join(sorted(sessions.protocol.dropna().unique()))[:400]],
    ]
    for w, (ev, lo, hi) in WINDOWS.items():
        params.append(["Export", f"Spike-count window '{w}'", f"{lo:+g} to {hi:+g} s relative to {ev}"])
    params += [
        ["Export", "Firing rates", "firing_rate = IBL's whole-recording average; task = first trial start to last trial end; "
         "prestim = mean over trials of −0.4…−0.1 s before stimulus (mouse quiescent: the closest analogue to a spontaneous/baseline rate)"],
        ["Export", "Waveform", "peak_to_trough_ms = trough-to-peak of mean extracellular waveform (not equivalent to intracellular AP half-width)"],
    ]
    story.append(table(header_white(params), [2.3 * cm, 4.2 * cm, W - 6.5 * cm]))

    story += [PageBreak(), Paragraph("2. Overview figures", H2), imgs["rate"], Spacer(1, 6),
              Table([[imgs["wave"], imgs["psy"]]], colWidths=[W / 2, W / 2])]

    story += [PageBreak(), Paragraph("3. Firing rates by brain region (good units)", H2),
              Paragraph("ZI (zona incerta) and LHA (lateral hypothalamic area), the regions in the literature extraction, are highlighted. "
                        "These are extracellular recordings in awake, head-fixed mice performing a task, not slice patch-clamp. "
                        "Sorted by number of units.", SMALL), Spacer(1, 4)]
    cols = ["region", "region_name", "n_units", "n_sessions", "n_mice", "firing_rate_mean_hz", "firing_rate_sd_hz",
            "firing_rate_median_hz", "prestim_rate_mean_hz", "prestim_rate_median_hz", "stim_rate_mean_hz",
            "peak_to_trough_ms_median"]
    rows = [["Acronym", "Region", "Units", "Sessions", "Mice", "Rate mean (Hz)", "Rate SD (Hz)", "Rate median (Hz)",
             "Pre-stim mean (Hz)", "Pre-stim median (Hz)", "Stim mean (Hz)", "Peak-trough median (ms)"]]
    hl = []
    for i, r in enumerate(regsum[cols].itertuples(index=False), start=1):
        rows.append([fmt(v) for v in r])
        if r.region in LIT_REGIONS:
            hl.append(i)
    story.append(table(header_white(rows), [1.6 * cm, 6 * cm] + [(W - 7.6 * cm) / 10] * 10, highlight=hl))

    lit = good[good.region.isin(LIT_REGIONS)]
    if len(lit):
        story += [PageBreak(), Paragraph("4. ZI and LHA by session", H2)]
        g = lit.groupby(["region", "session_id", "subject_id"]).agg(
            n=("unit_id", "size"), fr=("firing_rate_hz", "mean"), pre=("prestim_rate_hz", "mean"),
            stim=("stim_rate_hz", "mean"), ptt=("peak_to_trough_ms", "median")).reset_index()
        rows = [["Region", "Session", "Mouse", "Good units", "Rate mean (Hz)", "Pre-stim mean (Hz)", "Stim mean (Hz)", "Peak-trough median (ms)"]]
        rows += [[r.region, r.session_id, r.subject_id, r.n, fmt(r.fr), fmt(r.pre), fmt(r.stim), fmt(r.ptt)]
                 for r in g.itertuples(index=False)]
        story.append(table(header_white(rows), [1.6 * cm, 4.5 * cm, 3.5 * cm] + [(W - 9.6 * cm) / 5] * 5))

        story += [PageBreak(), Paragraph("5. Unit-level data: ZI and LHA (good units)", H2)]
        shown = lit.sort_values(["region", "session_id", "unit_id"])
        if len(shown) > MAX_UNIT_ROWS:
            story.append(Paragraph(f"First {MAX_UNIT_ROWS} of {len(shown)} units shown; all are in tables/units.csv.", SMALL))
            shown = shown.head(MAX_UNIT_ROWS)
        ucols = ["region", "session_id", "unit_id", "ccf_x", "ccf_y", "ccf_z", "peak_to_trough_ms",
                 "median_amplitude_uV", "spike_count", "firing_rate_hz", "prestim_rate_hz", "stim_rate_hz"]
        rows = [["Region", "Session", "Unit", "CCF x", "CCF y", "CCF z", "Peak-trough (ms)", "Amp. (µV)",
                 "Spikes", "Rate (Hz)", "Pre-stim (Hz)", "Stim (Hz)"]]
        for r in shown[ucols].itertuples(index=False):
            rows.append([r.region, r.session_id, r.unit_id, fmt(r.ccf_x, 0), fmt(r.ccf_y, 0), fmt(r.ccf_z, 0),
                         fmt(r.peak_to_trough_ms), fmt(r.median_amplitude_uV, 1), r.spike_count,
                         fmt(r.firing_rate_hz), fmt(r.prestim_rate_hz), fmt(r.stim_rate_hz)])
        story.append(table(header_white(rows), [1.5 * cm, 4.2 * cm] + [(W - 5.7 * cm) / 10] * 10))

    story += [PageBreak(), Paragraph("6. Mice", H2)]
    t = trials.merge(sessions[["session_id", "subject_id"]], on="session_id")
    perf = t.groupby("subject_id").is_mouse_rewarded.mean()
    m = sessions.groupby("subject_id").agg(
        lab=("lab", "first"), sex=("sex", "first"), sessions=("session_id", "size"),
        age=("age_days", "median"), trials=("n_trials", "sum"), units=("n_units", "sum"),
        good=("n_good_units", "sum")).reset_index()
    rows = [["Mouse", "Lab", "Sex", "Sessions", "Median age (d)", "Trials", "% rewarded", "Units", "Good units"]]
    for r in m.itertuples(index=False):
        rows.append([r.subject_id, r.lab, r.sex, r.sessions, fmt(r.age, 0), r.trials,
                     f"{100 * perf.get(r.subject_id, np.nan):.1f}", r.units, r.good])
    story.append(table(header_white(rows), [3.5 * cm, 4 * cm, 1.2 * cm] + [(W - 8.7 * cm) / 6] * 6))

    story += [PageBreak(), Paragraph("7. Data files and columns", H2)]
    dd = [
        ["File", "Rows", "Columns"],
        ["tables/sessions.csv", f"{len(sessions)}", ", ".join(sessions.columns)],
        ["tables/units.csv", f"{len(units):,}", ", ".join(units.columns)],
        ["tables/trials.csv", f"{len(trials):,}", ", ".join(trials.columns)],
        ["tables/region_summary.csv", f"{len(regsum)}", ", ".join(regsum.columns)],
        ["trial_unit_counts.parquet", "trials × units per session",
         "session_id, trial_id, unit_id, " + ", ".join(f"count_{w}" for w in WINDOWS)],
    ]
    story.append(table(header_white(dd), [5 * cm, 3.5 * cm, W - 8.5 * cm]))
    story += [Spacer(1, 6), Paragraph(
        "Keys: <font face='Courier'>(session_id, unit_id)</font> joins units ↔ trial counts; "
        "<font face='Courier'>(session_id, trial_id)</font> joins trials ↔ trial counts; "
        "<font face='Courier'>cluster_uuid</font> links a unit to the IBL database (ONE API). "
        "All times are seconds on the session clock. Column descriptions: dandi_export/README_000409.md.", BODY)]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.5 * cm, 0.7 * cm, f"DANDI:000409 v{VERSION} · IBL Brain Wide Map")
        canvas.drawRightString(landscape(A4)[0] - 1.5 * cm, 0.7 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--note", help="highlighted note under the title (e.g. partial export)")
    a = ap.parse_args(argv)
    build(a.out_dir, a.pdf, a.note)
    print(f"wrote {a.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
