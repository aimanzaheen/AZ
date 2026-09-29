"""Build a PDF report of the DANDI:000409 (IBL Brain Wide Map) export.

The report quotes the dataset's own published metadata and NWB descriptions,
then summarises the exported data. No brain region is singled out; every
region is summarised the same way. Numbers computed by the export are
labelled as such.

Usage:
    python dandi_export/make_report_000409.py dandi_export/output_000409 \
        dandi_export/DANDI_000409_report.pdf [--note "..."]
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
from export_000409 import VERSION, WINDOWS  # noqa: E402
from report_common import (  # noqa: E402
    BAR, BODY, H1, H2, NOTE, SMALL, SRC_EXPORT, SRC_NWB, W, descriptions_section, doc, esc,
    fig_to_image, fmt, legend_para, load_metadata, published_section, table,
)

N_FIG_REGIONS = 70


def figures(units, trials, regsum, tmp):
    imgs = {}
    good = units[units.good & units.region.isin(regsum.region)]

    top = regsum.nlargest(N_FIG_REGIONS, "n_units").sort_values("firing_rate_median_hz")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    data = [good.loc[good.region == r, "firing_rate_hz"].values for r in top.region]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9db8d3")
    ax.set_xticks(range(1, len(top) + 1), top.region, rotation=90, fontsize=5.5)
    ax.set_ylabel("firing_rate (Hz)")
    ax.set_title(f"NWB units.firing_rate by region: units with ibl_quality_score = 1, the {len(top)} regions with the most such units "
                 "(outliers hidden)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["rate"] = fig_to_image(fig, tmp, "rate", 24)

    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.hist(good.peak_to_trough_ms.dropna(), bins=np.arange(0, 1.6, 0.0333), color=BAR)
    ax.set_xlabel("peak_to_trough_duration_ms")
    ax.set_ylabel("Units")
    ax.set_title("Distribution of peak_to_trough_duration_ms", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["wave"] = fig_to_image(fig, tmp, "wave", 12)

    t = trials.copy()
    sign = np.where(t.gabor_stimulus_side == "right", 1, -1)
    t["signed"] = t.gabor_stimulus_contrast * sign
    t["ccw"] = t.mouse_wheel_choice == "counter_clockwise"
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for p, c in [(0.2, "#c0504d"), (0.5, "#555555"), (0.8, "#2f5d8a")]:
        g = t[np.isclose(t.probability_left, p)].groupby("signed").ccw.mean()
        ax.plot(g.index, g.values, "o-", ms=3, color=c, label=f"probability_left = {p}")
    ax.set_xlabel("gabor_stimulus_contrast (%), signed: + right, − left")
    ax.set_ylabel("Fraction 'counter_clockwise'")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("mouse_wheel_choice by contrast and block, all sessions", fontsize=8.5)
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
    dm, fm, desc = load_metadata(out_dir)
    good = units[units.good]
    tmp = Path(tempfile.mkdtemp())
    imgs = figures(units, trials, regsum, tmp)

    d, footer = doc(pdf, f"DANDI:000409 {dm['name']}", f"DANDI:000409 v{VERSION} · {dm['name']}")
    story = [Paragraph(f"DANDI:000409: {esc(dm['name'])}", H1)]
    if note:
        story.append(Paragraph(f"<b>{esc(note)}</b>", NOTE))
    story += [
        Paragraph("Scope of this export (computed by this export): only the 459 "
                  "<font face='Courier'>desc-processed_behavior+ecephys.nwb</font> files were used (spike-sorted units, "
                  "trials and behaviour, ~0.5 TB). The raw electrophysiology and video files (~49 TB) were not downloaded.", BODY),
        Spacer(1, 4), legend_para(), Spacer(1, 4),
    ]
    story += published_section(dm, fm, "1")

    story += [Spacer(1, 6), Paragraph("2. Contents of the processed files", H2)]
    ages = sessions.age_days.dropna()
    q = trials.quiescence_period
    rows = [["Quantity", "Value", "Source"],
            ["Sessions / subjects / labs", f"{len(sessions)} / {sessions.subject_id.nunique()} / {sessions.lab.nunique()}", SRC_NWB],
            ["Sessions per lab (NWB lab field)", ", ".join(f"{k} ({v})" for k, v in sessions.lab.value_counts().items()), SRC_NWB],
            ["Sex, by subject", ", ".join(f"{k}: {v}" for k, v in sessions.drop_duplicates('subject_id').sex.value_counts().items()), SRC_NWB],
            ["Age at session (session start − subject.date_of_birth)",
             f"median {ages.median():.0f} days (range {ages.min():.0f}–{ages.max():.0f})" if len(ages) else "n/a", SRC_EXPORT],
            ["Probes per session", ", ".join(f"{k}: {v} sessions" for k, v in sessions.n_probes.value_counts().sort_index().items()), SRC_NWB],
            ["Probe device description (example file)", "; ".join(f"{k}: {v}" for k, v in fm.get("devices", {}).items()), SRC_NWB],
            ["Task protocol(s) (NWB protocol field)", "; ".join(sorted(sessions.protocol.dropna().unique())), SRC_NWB],
            ["Trials", f"{len(trials):,}", SRC_NWB],
            ["gabor_stimulus_contrast values (%)", ", ".join(f"{c:g}" for c in sorted(trials.gabor_stimulus_contrast.dropna().unique())), SRC_NWB],
            ["probability_left values", ", ".join(f"{p:g}" for p in sorted(trials.probability_left.dropna().unique())), SRC_NWB],
            ["mouse_wheel_choice values", ", ".join(f"{k} ({v:,})" for k, v in trials.mouse_wheel_choice.value_counts().items()), SRC_NWB],
            ["quiescence_period range", f"{q.min():.2f}–{q.max():.2f} s (median {q.median():.2f} s)", SRC_NWB],
            ["Units (ibl_quality_score = 1)", f"{len(units):,} ({len(good):,})", SRC_NWB],
            ["ibl_quality_score counts", ", ".join(f"{k:.2f}: {v:,}" for k, v in units.ibl_quality_score.round(2).value_counts().sort_index().items()), SRC_NWB],
            ["kilosort2_label counts", ", ".join(f"{k}: {v:,}" for k, v in units.kilosort2_label.value_counts().items()), SRC_NWB],
            ["Distinct electrode location labels at units' max_electrode", f"{units.region_name.nunique():,}", SRC_NWB],
            ["Total spikes (sum of spike_count)", f"{int(units.spike_count.clip(lower=0).sum()):,}", SRC_NWB]]
    story.append(table(rows, [6.5 * cm, W - 10.5 * cm, 4 * cm]))
    story.append(Paragraph("Definitions of ibl_quality_score, the trial columns and the electrode columns are quoted in section 8.", SMALL))

    story += [Paragraph("3. Export parameters", H2)]
    rows = [["Parameter", "Definition", "Source"],
            ["region_name", "electrodes.location at the unit's max_electrode", SRC_NWB],
            ["region", "Allen acronym matched to region_name (punctuation-insensitive) using allen_structures.csv, a snapshot of the Allen Mouse Brain Atlas ontology", SRC_EXPORT],
            ["good", "ibl_quality_score = 1.0 (the NWB description: '1.0 = all three passed')", SRC_NWB],
            ["firing_rate_hz", "units.firing_rate as stored in the NWB file", SRC_NWB],
            ["peak_to_trough_ms", "units.peak_to_trough_duration_ms (values of −1 treated as missing)", SRC_NWB],
            ["task_rate_hz", "spikes between first trial start and last trial stop / that duration", SRC_EXPORT],
            ["prestim_rate_hz, stim_rate_hz", "mean over trials of count_prestim / 0.3 s and count_stim / 0.25 s", SRC_EXPORT]]
    for w, (ev, lo, hi) in WINDOWS.items():
        rows.append([f"count_{w}", f"spikes from {lo:+g} to {hi:+g} s relative to trials.{ev}", SRC_EXPORT])
    story.append(table(rows, [4 * cm, W - 8 * cm, 4 * cm]))

    story += [PageBreak(), Paragraph(f"4. Overview figures ({SRC_EXPORT})", H2), imgs["rate"], Spacer(1, 4),
              Table([[imgs["wave"], imgs["psy"]]], colWidths=[W / 2, W / 2])]

    story += [PageBreak(), Paragraph(f"5. All regions ({SRC_EXPORT})", H2),
              Paragraph("One row per region for units with ibl_quality_score = 1, excluding unassigned labels ('root', 'void', "
                        "or names not found in the Allen ontology). Sorted by number of units. firing_rate is the NWB value; "
                        "pre-stimulus and stimulus rates are computed by this export (section 3).", SMALL), Spacer(1, 4)]
    cols = ["region", "region_name", "n_units", "n_sessions", "n_mice", "firing_rate_mean_hz", "firing_rate_sd_hz",
            "firing_rate_median_hz", "prestim_rate_mean_hz", "prestim_rate_median_hz", "stim_rate_mean_hz",
            "peak_to_trough_ms_median"]
    rows = [["Acronym", "Region (NWB location)", "Units", "Sessions", "Mice", "firing_rate mean (Hz)", "SD (Hz)",
             "Median (Hz)", "Pre-stim mean (Hz)", "Pre-stim median (Hz)", "Stim mean (Hz)", "peak_to_trough median (ms)"]]
    rows += [[fmt(v) for v in r] for r in regsum[cols].itertuples(index=False)]
    story.append(table(rows, [1.6 * cm, 6 * cm] + [(W - 7.6 * cm) / 10] * 10))

    unmatched = units.loc[units.region == "", "region_name"].value_counts()
    story += [PageBreak(), Paragraph("6. Subjects", H2)]
    t = trials.merge(sessions[["session_id", "subject_id"]], on="session_id")
    perf = t.groupby("subject_id").is_mouse_rewarded.mean()
    m = sessions.groupby("subject_id").agg(
        lab=("lab", "first"), sex=("sex", "first"), n=("session_id", "size"), age=("age_days", "median"),
        trials=("n_trials", "sum"), units=("n_units", "sum"), good=("n_good_units", "sum")).reset_index()
    rows = [["Subject", "Lab", "Sex", "Sessions", "Median age (d)", "Trials", "is_mouse_rewarded (%)", "Units", "Score = 1"]]
    for r in m.itertuples(index=False):
        rows.append([r.subject_id, r.lab, r.sex, r.n, fmt(r.age, 0), r.trials,
                     f"{100 * perf.get(r.subject_id, np.nan):.1f}", r.units, r.good])
    story.append(table(rows, [3.5 * cm, 4 * cm, 1.2 * cm] + [(W - 8.7 * cm) / 6] * 6))

    story += [PageBreak(), Paragraph("7. Data checks performed by this export", H2),
              Paragraph(f"<b>Region labels.</b> {len(unmatched)} distinct location names ({int(unmatched.sum()):,} units) did not "
                        "match an Allen ontology name and have an empty <i>region</i> acronym; their <i>region_name</i> is kept. "
                        + ", ".join(f"'{esc(k)}' ({v:,})" for k, v in unmatched.head(15).items()), NOTE),
              Paragraph(f"<b>peak_to_trough_duration_ms.</b> {int(units.peak_to_trough_ms.isna().sum()):,} units have the value "
                        "−1 in the NWB file; they are treated as missing.", NOTE)]
    story += [PageBreak()]
    story += descriptions_section(desc, "8", fm.get("source_file"))

    story += [PageBreak(), Paragraph("9. Exported files", H2)]
    dd = [["File", "Rows", "Columns"],
          ["tables/sessions.csv", f"{len(sessions)}", ", ".join(sessions.columns)],
          ["tables/units.csv", f"{len(units):,}", ", ".join(units.columns)],
          ["tables/trials.csv", f"{len(trials):,}", ", ".join(trials.columns)],
          ["tables/region_summary.csv", f"{len(regsum)}", ", ".join(regsum.columns)],
          ["tables/dataset_metadata.json, nwb_file_metadata.json, nwb_descriptions.csv", "", "the published metadata quoted in sections 1 and 8"],
          ["trial_unit_counts.parquet", "trials × units per session", "session_id, trial_id, unit_id, " + ", ".join(f"count_{w}" for w in WINDOWS)]]
    story.append(table(dd, [6 * cm, 3.5 * cm, W - 9.5 * cm]))
    d.build(story, onFirstPage=footer, onLaterPages=footer)


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
