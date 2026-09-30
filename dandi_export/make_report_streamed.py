"""Build a neutral PDF report for a Dandiset exported with export_streamed.py.

Usage:
    python dandi_export/make_report_streamed.py 001417 dandi_export/output_001417 \
        dandi_export/DANDI_001417_report.pdf [--note "..."]

The report quotes the published metadata and NWB descriptions verbatim,
lists what the files contain, and summarises every region the same way
(sorted by unit count). Unit-quality columns are reported as the authors
provide them; no filter is applied unless stated.
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
from reportlab.platypus import PageBreak, Paragraph, Spacer

sys.path.insert(0, str(Path(__file__).parent))
from report_common import (  # noqa: E402
    BODY, H1, H2, NOTE, SMALL, SRC_EXPORT, SRC_NWB, W, descriptions_section, doc, esc,
    fig_to_image, fmt, legend_para, load_metadata, published_section, table,
)

# Units columns that commonly carry the authors' own quality labels.
QUALITY_COLS = ["quality", "default_qc", "decoder_label", "kilosort2_label", "KSLabel", "label", "bc_unitType"]
N_FIG = 60


def read_csv(p):
    return pd.read_csv(p, keep_default_na=False, na_values=[""])


def txt(v, sep=None, joiner=", "):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).replace(sep, joiner) if sep else str(v)


def build(dandiset: str, out: Path, pdf: Path, note: str | None) -> None:
    t = out / "tables"
    version = (out / "VERSION").read_text().strip()
    sessions = read_csv(t / "sessions.csv")
    units = pd.read_parquet(t / "units.parquet")
    regsum = read_csv(t / "region_summary.csv")
    regsum["region"] = regsum.region.astype(str)
    isum = read_csv(t / "intervals_summary.csv") if (t / "intervals_summary.csv").stat().st_size > 1 else pd.DataFrame()
    dm, fm, desc = load_metadata(out)
    tmp = Path(tempfile.mkdtemp())

    d, footer = doc(pdf, f"DANDI:{dandiset} {dm['name']}", f"DANDI:{dandiset} v{version} · {dm['name'][:90]}")
    story = [Paragraph(f"DANDI:{dandiset}: {esc(dm['name'])}", H1)]
    if note:
        story.append(Paragraph(f"<b>{esc(note)}</b>", NOTE))
    story += [Paragraph("Scope of this export (computed by this export): the units, electrodes and intervals tables of every "
                        f"NWB file{' (session files; the per-probe *_ecephys.nwb LFP files were skipped)' if dandiset == '001051' else ''} "
                        "were read remotely from the DANDI Archive; voltage, LFP, video and other time series were not downloaded.", BODY),
              Spacer(1, 4), legend_para(), Spacer(1, 4)]
    story += published_section(dm, fm, "1")

    # ---- 2. Contents --------------------------------------------------------
    story += [Spacer(1, 6), Paragraph("2. Contents of the files", H2)]
    def vc(col):
        return ", ".join(f"{k} ({v})" for k, v in sessions[col].astype(str).value_counts().items()) if col in sessions else ""
    rows = [["Quantity", "Value", "Source"],
            ["NWB files read / subjects", f"{len(sessions)} / {sessions.subject_id.nunique()}", SRC_NWB],
            ["Species", vc("species"), SRC_NWB],
            ["Sex (files)", vc("sex"), SRC_NWB],
            ["Genotype (files)", vc("genotype"), SRC_NWB],
            ["Strain (files)", vc("strain"), SRC_NWB],
            ["Age (files)", vc("age"), SRC_NWB],
            ["Lab / institution", f"{vc('lab')} / {vc('institution')}", SRC_NWB],
            ["Units (total)", f"{len(units):,}", SRC_NWB],
            ["Units per file", f"median {sessions.n_units.median():.0f} (range {sessions.n_units.min()}–{sessions.n_units.max()})", SRC_NWB],
            ["Spikes (sum of spike_times_index lengths)", f"{int(units.n_spikes.sum()):,}" if "n_spikes" in units else "", SRC_NWB],
            ["How region was assigned to units", "; ".join(sorted(sessions.region_source.dropna().astype(str).unique())), SRC_EXPORT],
            ["Distinct region labels", f"{units.region.astype(str).nunique():,}", SRC_NWB]]
    for c in QUALITY_COLS:
        if c in units:
            rows.append([f"units.{c} values", ", ".join(f"{k}: {v:,}" for k, v in units[c].astype(str).value_counts().items()), SRC_NWB])
    story.append(table(rows, [6 * cm, W - 10 * cm, 4 * cm]))
    story.append(Paragraph("The meaning of each units column, including any quality label, is quoted from the NWB file in section 7.", SMALL))

    # ---- 3. Recording content per file --------------------------------------
    story += [PageBreak(), Paragraph("3. NWB files", H2)]
    rows = [["File (session_id)", "Subject", "Sex", "Age", "Size (GB)", "Units", "Electrodes", "Electrode groups", "Interval tables (rows)"]]
    for s in sessions.itertuples(index=False):
        rows.append([s.session_id, txt(s.subject_id), txt(s.sex), txt(s.age), fmt(s.size_gb), s.n_units, s.n_electrodes,
                     txt(s.electrode_groups, ";"), txt(s.intervals, ";", "; ")])
    story.append(table(rows, [5.2 * cm, 1.8 * cm, 0.9 * cm, 1.6 * cm, 1.3 * cm, 1.2 * cm, 1.5 * cm, 4 * cm, W - 17.5 * cm]))
    other = [["File (session_id)", "Acquisition", "Processing modules: interfaces"]]
    for s in sessions.itertuples(index=False):
        other.append([s.session_id, txt(s.acquisition, ";"), txt(s.processing, ";", "; ")])
    story += [Spacer(1, 6), Paragraph("Other data in each file (not exported)", SMALL),
              table(other, [5.2 * cm, 7 * cm, W - 12.2 * cm])]

    # ---- 4. Figures ---------------------------------------------------------
    story += [PageBreak(), Paragraph(f"4. Units per region ({SRC_EXPORT})", H2)]
    top = regsum.nlargest(N_FIG, "n_units")
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(range(len(top)), top.n_units, color="#9db8d3", edgecolor="#5b85aa", lw=0.5)
    ax.set_xticks(range(len(top)), top.region, rotation=90, fontsize=5.5)
    ax.set_ylabel("Units")
    ax.set_title(f"All units by region label, the {len(top)} labels with the most units (no quality filter)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    story.append(fig_to_image(fig, tmp, "count", 24))
    if "firing_rate" in units:
        fr = pd.to_numeric(units.firing_rate, errors="coerce")
        fig, ax = plt.subplots(figsize=(11, 3.2))
        tr = top.sort_values("firing_rate_median_hz") if "firing_rate_median_hz" in top else top
        ax.boxplot([fr[units.region.astype(str) == r].dropna().values for r in tr.region], showfliers=False,
                   patch_artist=True, widths=0.6, boxprops=dict(facecolor="#9db8d3"))
        ax.set_xticks(range(1, len(tr) + 1), tr.region, rotation=90, fontsize=5.5)
        ax.set_ylabel("units.firing_rate (Hz)")
        ax.set_title("units.firing_rate (NWB value) by region, same labels, all units, outliers hidden", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        story.append(fig_to_image(fig, tmp, "fr", 24))

    # ---- 5. Region table ----------------------------------------------------
    story += [PageBreak(), Paragraph(f"5. All region labels ({SRC_EXPORT})", H2),
              Paragraph("One row per region label exactly as stored in the NWB files, all units (no quality filter), sorted by "
                        "unit count. Firing-rate columns summarise the NWB units.firing_rate values where the dataset provides them.", SMALL),
              Spacer(1, 4)]
    cols = [c for c in ["region", "n_units", "n_sessions", "n_subjects", "firing_rate_mean_hz", "firing_rate_median_hz"] if c in regsum]
    names = {"region": "Region label", "n_units": "Units", "n_sessions": "Files", "n_subjects": "Subjects",
             "firing_rate_mean_hz": "firing_rate mean (Hz)", "firing_rate_median_hz": "firing_rate median (Hz)"}
    rows = [[names[c] for c in cols]] + [[fmt(v) for v in r] for r in regsum[cols].itertuples(index=False)]
    story.append(table(rows, [W / len(cols)] * len(cols)))

    # ---- 6. Intervals -------------------------------------------------------
    story += [PageBreak(), Paragraph("6. Intervals tables (stimuli, trials, epochs)", H2),
              Paragraph("Each NWB intervals table, with the number of files containing it, total rows, and its columns as stored.", SMALL)]
    if len(isum):
        rows = [["Table", "Files", "Rows", "Columns"]] + [[r.table, r.n_sessions, f"{r.n_rows:,}", r.columns]
                                                         for r in isum.itertuples(index=False)]
        story.append(table(rows, [6 * cm, 1.5 * cm, 2 * cm, W - 9.5 * cm]))
    else:
        story.append(Paragraph("No intervals tables were found.", BODY))

    story += [PageBreak()]
    story += descriptions_section(desc, "7", fm.get("source_file"))

    story += [PageBreak(), Paragraph("8. Exported files", H2)]
    dd = [["File", "Rows", "Columns"],
          ["tables/sessions.csv", str(len(sessions)), ", ".join(sessions.columns)],
          ["tables/units.parquet", f"{len(units):,}", ", ".join(units.columns)],
          ["tables/region_summary.csv", str(len(regsum)), ", ".join(regsum.columns)],
          ["tables/intervals_summary.csv", str(len(isum)), "table, n_sessions, n_rows, columns"],
          ["intervals/<table>.parquet", "one per intervals table", "session_id + the table's columns"],
          ["tables/dataset_metadata.json, nwb_file_metadata.json, nwb_descriptions.csv", "", "published metadata quoted in sections 1 and 7"]]
    story.append(table(dd, [6 * cm, 3 * cm, W - 9 * cm]))
    d.build(story, onFirstPage=footer, onLaterPages=footer)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dandiset")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--note")
    a = ap.parse_args(argv)
    build(a.dandiset, a.out_dir, a.pdf, a.note)
    print(f"wrote {a.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
