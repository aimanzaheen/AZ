"""Build a PDF report of the DANDI:000017 export.

The report quotes the dataset's own published metadata and NWB descriptions,
then summarises the exported data. No brain region is singled out; every
region is summarised the same way. Numbers computed by the export are
labelled as such.

Usage:
    python dandi_export/make_report.py dandi_export/output dandi_export/DANDI_000017_report.pdf
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
from export_000017 import WHEEL_DOWNSAMPLE_HZ, WINDOWS  # noqa: E402
from report_common import (  # noqa: E402
    BAR, BODY, H1, H2, NOTE, SMALL, SRC_EXPORT, SRC_NWB, W, descriptions_section, doc, esc,
    fig_to_image, fmt, legend_para, load_metadata, published_section, table,
)

VERSION = "0.240329.1926"
MIN_UNITS_FIG = 20


def figures(units, trials, regsum, tmp):
    imgs = {}
    good = units[units.good & (units.region != "root")]

    top = regsum[regsum.n_units >= MIN_UNITS_FIG].sort_values("spont_rate_median_hz")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    data = [good.loc[good.region == r, "spontaneous_rate_hz"].values for r in top.region]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9db8d3")
    ax.set_xticks(range(1, len(top) + 1), top.region, rotation=90, fontsize=6.5)
    ax.set_ylabel("Rate in 'spontaneous' intervals (Hz)")
    ax.set_title(f"Firing rate during the NWB 'spontaneous' intervals, by region "
                 f"(units with phy_annotation ≥ 2; all {len(top)} regions with ≥{MIN_UNITS_FIG} such units; outliers hidden)",
                 fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["spont"] = fig_to_image(fig, tmp, "spont", 25)

    fig, ax = plt.subplots(figsize=(5, 3.0))
    ax.hist(good.waveform_duration_ms, bins=np.arange(0, 1.6, 0.0333), color=BAR)
    ax.set_xlabel("waveform_duration (ms)")
    ax.set_ylabel("Units")
    ax.set_title("Distribution of trough-to-peak waveform duration", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["wave"] = fig_to_image(fig, tmp, "wave", 12)

    t = trials[trials.included.astype(bool)].copy()
    t["cdiff"] = (t.visual_stimulus_right_contrast - t.visual_stimulus_left_contrast).round(2)
    g = t.groupby("cdiff").response_choice
    fig, ax = plt.subplots(figsize=(5, 3.0))
    x = g.size().index
    for val, c, lab in [(1, "#2f5d8a", "+1 (left choice)"), (-1, "#c0504d", "−1 (right choice)"), (0, "#888888", "0 (NoGo)")]:
        ax.plot(x, g.apply(lambda s, v=val: (s == v).mean()), "o-", ms=3, color=c, label=lab)
    ax.set_xlabel("Right contrast − left contrast")
    ax.set_ylabel("Fraction of trials")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, title="response_choice", title_fontsize=7)
    ax.set_title("response_choice by contrast difference (included trials, all sessions)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    imgs["psy"] = fig_to_image(fig, tmp, "psy", 12)
    return imgs


def build(out_dir: Path, pdf: Path) -> None:
    tables = out_dir / "tables"
    sessions = pd.read_csv(tables / "sessions.csv")
    units = pd.read_csv(tables / "units.csv")
    trials = pd.read_csv(tables / "trials.csv")
    regsum = pd.read_csv(tables / "region_summary.csv")
    dm, fm, desc = load_metadata(out_dir)
    good = units[units.good]
    age_days = sessions.age.str.extract(r"(\d+)")[0].astype(int)
    tmp = Path(tempfile.mkdtemp())
    imgs = figures(units, trials, regsum, tmp)

    d, footer = doc(pdf, f"DANDI:000017 {dm['name']}", f"DANDI:000017 v{VERSION} · {dm['name']}")
    story = [
        Paragraph(f"DANDI:000017: {esc(dm['name'])}", H1),
        Paragraph(f"{esc(dm.get('citation'))}", BODY),
        Spacer(1, 4),
        legend_para(),
        Spacer(1, 4),
    ]
    story += published_section(dm, fm, "1")

    # ---- 2. Contents --------------------------------------------------------
    story += [Spacer(1, 6), Paragraph("2. Contents of the files", H2)]
    ptypes = sorted({g for g in units.probe.unique()})
    rows = [["Quantity", "Value", "Source"],
            ["Sessions / subjects", f"{len(sessions)} / {sessions.subject_id.nunique()}", SRC_NWB],
            ["Subjects (sessions each)", ", ".join(f"{k} ({v})" for k, v in sessions.subject_id.value_counts().sort_index().items()), SRC_NWB],
            ["Species", "Mus musculus (subject.species)", SRC_NWB],
            ["Strain (subject.description)", "; ".join(sorted(sessions.strain.dropna().unique())), SRC_NWB],
            ["Genotypes (subject.genotype)", "; ".join(sorted(sessions.genotype.dropna().unique())), SRC_NWB],
            ["Sex, by subject", ", ".join(f"{k}: {v}" for k, v in sessions.drop_duplicates('subject_id').sex.value_counts().items()), SRC_NWB],
            ["Age at session (subject.age)", f"{age_days.min()}–{age_days.max()} days", SRC_NWB],
            ["Electrode groups", f"{', '.join(ptypes)}; description '{fm['electrode_groups'][next(iter(fm['electrode_groups']))]['description']}'", SRC_NWB],
            ["Unit sampling_rate", ", ".join(f"{v:g} Hz" for v in sorted(units.get('sampling_rate', pd.Series([30000])).unique())) if 'sampling_rate' in units else "30000 Hz", SRC_NWB],
            ["Trials (included == True)", f"{len(trials):,} ({int(trials.included.astype(bool).sum()):,})", SRC_NWB],
            ["Visual contrast values", ", ".join(f"{c:g}" for c in sorted(trials.visual_stimulus_left_contrast.unique())), SRC_NWB],
            ["response_choice values", ", ".join(str(int(v)) for v in sorted(trials.response_choice.unique())), SRC_NWB],
            ["feedback_type values", ", ".join(str(int(v)) for v in sorted(trials.feedback_type.unique())), SRC_NWB],
            ["Units (phy_annotation ≥ 2)", f"{len(units):,} ({len(good):,})", SRC_NWB],
            ["phy_annotation counts", ", ".join(f"{k}: {v:,}" for k, v in units.phy_annotation.value_counts().sort_index().items()), SRC_NWB],
            ["Distinct electrode 'location' labels at unit peak channels", f"{units.region.nunique()} (including 'root')", SRC_NWB],
            ["Total spikes", f"{int(units.n_spikes.sum()):,}", SRC_NWB],
            ["'spontaneous' interval time per session", f"median {sessions.spontaneous_duration_s.median():.0f} s (range {sessions.spontaneous_duration_s.min():.0f}–{sessions.spontaneous_duration_s.max():.0f} s)", SRC_EXPORT],
            ["Recording duration", f"{sessions.duration_s.sum() / 3600:.1f} h total", SRC_EXPORT]]
    story.append(table(rows, [6 * cm, W - 10 * cm, 4 * cm]))
    story.append(Paragraph("Definitions of phy_annotation, the trial columns, the 'spontaneous' intervals and the electrode "
                           "columns are quoted in section 8.", SMALL))

    story += [Paragraph("3. Export parameters", H2)]
    rows = [["Parameter", "Definition", "Source"],
            ["Unit region", "electrodes.location at row peak_channel − 1 (peak_channel is 1-based; the first electrode listed for each unit is always peak_channel − 1)", SRC_EXPORT],
            ["good", "phy_annotation ≥ 2 (the inclusion rule stated in the phy_annotations description)", SRC_NWB],
            ["mean_rate_hz", "n_spikes / recording duration", SRC_EXPORT],
            ["task_rate_hz", "spikes between first trial start and last trial stop / that duration", SRC_EXPORT],
            ["spontaneous_rate_hz", "spikes inside the intervals/spontaneous table / total duration of those intervals", SRC_EXPORT],
            ["waveform_duration_ms", "waveform_duration (samples) / sampling_rate × 1000; see section 7 for the sessions where it was recomputed", SRC_EXPORT]]
    for w, (ev, lo, hi) in WINDOWS.items():
        rows.append([f"count_{w}", f"spikes from {lo:+g} to {hi:+g} s relative to trials.{ev}", SRC_EXPORT])
    rows.append(["wheel_position export", f"downsampled from 2500 Hz to {WHEEL_DOWNSAMPLE_HZ:g} Hz (every 25th sample)", SRC_EXPORT])
    story.append(table(rows, [4 * cm, W - 8 * cm, 4 * cm]))

    # ---- 4. Figures ---------------------------------------------------------
    story += [PageBreak(), Paragraph(f"4. Overview figures ({SRC_EXPORT})", H2), imgs["spont"], Spacer(1, 4),
              Table([[imgs["wave"], imgs["psy"]]], colWidths=[W / 2, W / 2])]

    # ---- 5. Regions ---------------------------------------------------------
    story += [PageBreak(), Paragraph(f"5. All regions ({SRC_EXPORT})", H2),
              Paragraph("One row per electrode 'location' label, for units with phy_annotation ≥ 2, excluding 'root'. "
                        "Sorted by number of units. Region labels are exactly as stored in the NWB electrodes table.", SMALL),
              Spacer(1, 4)]
    cols = ["region", "n_units", "n_sessions", "spont_rate_mean_hz", "spont_rate_sd_hz",
            "spont_rate_median_hz", "mean_rate_mean_hz", "waveform_duration_ms_median"]
    rows = [["Region (location)", "Units", "Sessions", "Spontaneous-interval rate mean (Hz)", "SD (Hz)",
             "Median (Hz)", "Whole-recording rate mean (Hz)", "waveform_duration median (ms)"]]
    rows += [[fmt(v) for v in r] for r in regsum[cols].itertuples(index=False)]
    story.append(table(rows, [W / 8] * 8))

    # ---- 6. Sessions --------------------------------------------------------
    story += [PageBreak(), Paragraph("6. Sessions", H2)]
    t = trials.copy()
    rt = (t.response_time - t.go_cue).groupby(t.session_id).median()
    perf = t.groupby("session_id").agg(rew=("feedback_type", lambda s: (s == 1).mean()),
                                       nogo=("response_choice", lambda s: (s == 0).mean()))
    rows = [["Session", "Sex", "Age", "Genotype", "Duration (s)", "Trials", "Units", "phy ≥ 2",
             "feedback_type = 1 (%)", "NoGo (%)", "Median response_time − go_cue (s)", "Regions"]]
    for s in sessions.itertuples(index=False):
        rows.append([s.session_id, s.sex, s.age, s.genotype, f"{s.duration_s:.0f}", s.n_trials, s.n_units,
                     s.n_good_units, f"{100 * perf.rew[s.session_id]:.1f}", f"{100 * perf.nogo[s.session_id]:.1f}",
                     f"{rt[s.session_id]:.3f}", s.regions.replace(";", ", ")])
    story.append(table(rows, [3 * cm, 0.8 * cm, 1.4 * cm, 3 * cm, 1.5 * cm, 1.1 * cm, 1.1 * cm, 1.2 * cm,
                              1.6 * cm, 1.3 * cm, 2 * cm, W - 18 * cm]))
    story.append(Paragraph("Subject, genotype, age and counts: NWB file. Percentages and response times: computed by this export.", SMALL))

    # ---- 7. Data checks -----------------------------------------------------
    n_bad = units.loc[units.waveform_duration_source != "nwb", "session_id"].nunique()
    story += [PageBreak(), Paragraph("7. Data checks performed by this export", H2), Paragraph(
        f"<b>waveform_duration.</b> In {n_bad} of {len(sessions)} sessions the NWB column <i>waveform_duration</i> "
        "contains values above 100 samples (&gt; 3 ms; up to ~3800 samples, ~127 ms), which is outside the range of "
        "spike waveforms. In the session examined in detail (Hench 2017-06-15) these values match <i>cluster_depths</i> "
        "(r &gt; 0.9999). For those sessions the export recomputes trough-to-peak duration from <i>waveform_mean</i> "
        "on the highest-amplitude channel. Where the stored value is plausible (checked on Cori 2016-12-14), this "
        "recomputation agrees with it at r ≈ 0.96. units.csv records which source was used "
        "(<font face='Courier'>waveform_duration_source</font>).", NOTE),
        Paragraph("<b>Sampling-rate field.</b> The wheel_position series stores 0.0004 in its <i>rate</i> field, "
                  "which is the sampling period (2500 Hz). The export treats rate values &lt; 1 as periods.", NOTE),
        Paragraph("<b>Pupil timestamps.</b> In some sessions the eye tracking series stores only two timestamps "
                  "(first and last frame). The export spaces the frames evenly between them.", NOTE)]
    story += [PageBreak()]
    story += descriptions_section(desc, "8", fm.get("source_file"))

    story += [PageBreak(), Paragraph("9. Exported files", H2)]
    dd = [["File", "Rows", "Columns"],
          ["tables/sessions.csv", f"{len(sessions)}", ", ".join(sessions.columns)],
          ["tables/units.csv", f"{len(units):,}", ", ".join(units.columns)],
          ["tables/trials.csv", f"{len(trials):,}", ", ".join(trials.columns)],
          ["tables/region_summary.csv", f"{len(regsum)}", ", ".join(regsum.columns)],
          ["tables/dataset_metadata.json, nwb_file_metadata.json, nwb_descriptions.csv", "", "the published metadata quoted in sections 1 and 8"],
          ["trial_unit_counts.parquet", "trials × units per session", "session_id, trial_id, unit_id, " + ", ".join(f"count_{w}" for w in WINDOWS)],
          ["spikes/<session>.parquet", f"{int(units.n_spikes.sum()):,} total", "unit_id, time (s), amp, depth"],
          ["behavior/<session>__<signal>.parquet", "per signal", "wheel_position, wheel_moves (start_time, stop_time, type), lick_times, eye_area, eye_xy_positions, face_motion_energy"]]
    story.append(table(dd, [6 * cm, 3.5 * cm, W - 9.5 * cm]))

    d.build(story, onFirstPage=footer, onLaterPages=footer)


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
