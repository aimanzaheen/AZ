"""Shared building blocks for the DANDI export PDF reports.

Every report separates three kinds of content, and labels each:
  * DANDI metadata: the Dandiset's published metadata, quoted verbatim
  * NWB file: fields and descriptions written in the NWB files
  * Computed by this export: numbers derived by the export scripts
No brain region, condition or subject is singled out.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ACCENT = colors.HexColor("#2f5d8a")
BAR = "#5b85aa"
PAGE = landscape(A4)
W = PAGE[0] - 3 * cm

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=ACCENT, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=ACCENT, spaceBefore=8)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], textColor=ACCENT, spaceBefore=6, fontSize=10)
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=9.5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7, leading=8.5)
QUOTE = ParagraphStyle("quote", parent=BODY, fontSize=8.5, leading=11.5, leftIndent=10,
                       borderColor=colors.HexColor("#b8c7d6"), borderWidth=0, borderPadding=0,
                       textColor=colors.HexColor("#222222"))
NOTE = ParagraphStyle("note", parent=BODY, backColor=colors.HexColor("#f3f6f9"),
                      borderColor=colors.HexColor("#9db8d3"), borderWidth=0.6, borderPadding=6,
                      spaceBefore=4, spaceAfter=12)

SRC_DANDI, SRC_NWB, SRC_EXPORT = "DANDI metadata", "NWB file", "Computed by this export"


def esc(x) -> str:
    return escape("" if x is None else str(x))


def table(rows, col_widths=None, header_rows=1, raw=False):
    """Rows of cells; header row is styled. Cells are escaped unless raw=True."""
    data = []
    for i, r in enumerate(rows):
        cells = []
        for c in r:
            txt = str(c) if (raw or i < header_rows) else esc(c)
            if i < header_rows:
                txt = f'<font color="white"><b>{txt}</b></font>'
            cells.append(Paragraph(txt, CELL))
        data.append(cells)
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#f3f6f9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return t


def fmt(x, nd=2):
    if x is None:
        return ""
    if isinstance(x, (float, np.floating)):
        return "" if np.isnan(x) else f"{x:.{nd}f}"
    return str(x)


def fig_to_image(fig, tmp: Path, name: str, width_cm: float) -> Image:
    p = tmp / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    w, h = fig.get_size_inches()
    return Image(str(p), width=width_cm * cm, height=width_cm * cm * h / w)


def load_metadata(out_dir: Path) -> tuple[dict, dict, pd.DataFrame]:
    t = out_dir / "tables"
    return (json.loads((t / "dataset_metadata.json").read_text()),
            json.loads((t / "nwb_file_metadata.json").read_text()),
            pd.read_csv(t / "nwb_descriptions.csv", keep_default_na=False))


def _names(items):
    out = []
    for x in items or []:
        if isinstance(x, dict):
            out.append(x.get("name") or x.get("url") or x.get("identifier") or "")
        else:
            out.append(str(x))
    return "; ".join(o for o in out if o)


def published_section(dm: dict, fm: dict, number: str) -> list:
    """Section quoting the Dandiset metadata and NWB file-level fields verbatim."""
    story = [Paragraph(f"{number}. The dataset as published", H2),
             Paragraph("Everything in this section is quoted from the Dandiset's published metadata or, "
                       f"for file-level fields, from <font face='Courier'>{esc(fm.get('source_file'))}</font>.", SMALL),
             Spacer(1, 4)]
    a = dm.get("assetsSummary", {})
    contrib = "; ".join(
        f"{c['name']}" + (f" ({', '.join(r.removeprefix('dcite:') for r in c.get('roleName', []))})" if c.get("roleName") else "")
        for c in dm.get("contributor", []))
    rows = [["Field", "Value (DANDI metadata)"],
            ["Name", dm.get("name")],
            ["Identifier / version", f"DANDI:{dm.get('identifier')} / {dm.get('version')}"],
            ["Citation", dm.get("citation")],
            ["DOI", dm.get("doi") or ""],
            ["Contributors (roles)", contrib],
            ["License", ", ".join(x.removeprefix("spdx:") for x in dm.get("license", []))],
            ["Keywords", ", ".join(dm.get("keywords") or [])],
            ["Study target", "; ".join(dm.get("studyTarget") or [])],
            ["Protocol", "; ".join(dm.get("protocol") or [])],
            ["Ethics approval", _names(dm.get("ethicsApproval"))],
            ["Related resources", "; ".join(
                f"{r.get('name') or ''} {r.get('identifier') or r.get('url') or ''} ({r.get('relation', '').removeprefix('dcite:')})".strip()
                for r in dm.get("relatedResource") or [])],
            ["Anatomy (about)", _names(dm.get("about"))],
            ["Files / size / subjects", f"{a.get('numberOfFiles')} files, {a.get('numberOfBytes', 0) / 1e9:,.2f} GB, "
                                        f"{a.get('numberOfSubjects')} subjects"],
            ["Species", _names(a.get("species"))],
            ["Approach", _names(a.get("approach"))],
            ["Measurement technique", _names(a.get("measurementTechnique"))],
            ["Variables measured", ", ".join(map(str, a.get("variableMeasured") or []))],
            ["Date published", dm.get("datePublished") or ""]]
    rows = [r for r in rows if r[1] not in (None, "", "None")]
    story.append(table(rows, [4.5 * cm, W - 4.5 * cm]))
    story += [Paragraph("Description (DANDI metadata, verbatim)", H3)]
    for para in str(dm.get("description", "")).split("\n"):
        if para.strip():
            story.append(Paragraph(esc(para.strip()), QUOTE))
    story.append(Paragraph(f"File-level fields (NWB file, verbatim)", H3))
    rows = [["Field", "Value (NWB file)"]]
    for k, v in fm.items():
        if k == "source_file":
            continue
        if isinstance(v, dict):
            def flat(x):
                if isinstance(x, dict):
                    return ", ".join(f"{k2} = '{v2}'" for k2, v2 in x.items() if v2 not in (None, "", []))
                return x
            v = "; ".join(f"{kk}: {flat(vv)}" for kk, vv in v.items() if vv not in (None, "", [], {}))
        elif isinstance(v, list):
            v = "; ".join(map(str, v))
        if v not in (None, ""):
            rows.append([k, v])
    story.append(table(rows, [4.5 * cm, W - 4.5 * cm]))
    return story


def descriptions_section(desc: pd.DataFrame, number: str, source_file: str) -> list:
    rows = [["Location in NWB file", "Name", "Description (as written in the NWB file)"]]
    for r in desc.itertuples(index=False):
        rows.append([r.location, r.name, r.description])
    return [Paragraph(f"{number}. Column and signal descriptions from the NWB files", H2),
            Paragraph(f"Quoted verbatim from <font face='Courier'>{esc(source_file)}</font>; "
                      "all files in the Dandiset share this structure.", SMALL),
            Spacer(1, 4),
            table(rows, [4.8 * cm, 4.6 * cm, W - 9.4 * cm])]


def doc(pdf: Path, title: str, footer_text: str) -> tuple[SimpleDocTemplate, callable]:
    d = SimpleDocTemplate(str(pdf), pagesize=PAGE, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                          topMargin=1.3 * cm, bottomMargin=1.3 * cm, title=title)

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(1.5 * cm, 0.7 * cm, footer_text)
        canvas.drawRightString(PAGE[0] - 1.5 * cm, 0.7 * cm, f"Page {doc_.page}")
        canvas.restoreState()

    return d, footer


def legend_para() -> Paragraph:
    return Paragraph(
        f"Source labels: <b>{SRC_DANDI}</b> = the Dandiset's published metadata; <b>{SRC_NWB}</b> = values or text "
        f"stored in the NWB files; <b>{SRC_EXPORT}</b> = derived by the export scripts in "
        "<font face='Courier'>dandi_export/</font> (definitions given where used).", SMALL)
