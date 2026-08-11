#!/usr/bin/env python3
"""
IHC primary-antibody dilution/volume survey for calretinin, parvalbumin, and
somatostatin, plus mouse-brain positive-control regions for a validation run.

Run locally with no third-party dependencies:

    python3 ihc_antibody_volumes_table.py

Writes two CSVs to ./output/ and prints formatted tables to stdout.

DATA SOURCE / IMPORTANT CAVEAT
-------------------------------
Published IHC methods sections almost never report a literal applied volume
(uL) of primary antibody per section/slide -- that number depends on each
lab's slide size, chamber, and free-floating vs. mounted protocol. What is
reliably reported is the working DILUTION RATIO of the primary antibody
(e.g. 1:1000), which is what actually determines antibody concentration/
"strength" regardless of the physical volume pipetted. This script therefore
uses dilution ratio as the comparable proxy for antibody "volume/amount" the
user asked about, and reports the LEAST DILUTE (lowest ratio = most
concentrated = literature-derived maximum) value found per marker as the
upper-bound reference point for a dilution/volume optimization series.

Every row below is a real, individually verified literature entry (PubMed,
accessed via PMC full text where available). Rows where the working dilution
could not be recovered from the accessible text are marked accordingly
rather than guessed.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Optional

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# The user's own primary antibodies for this validation run.
USER_ANTIBODIES = {
    "Calretinin": "Goat",
    "Parvalbumin": "Mouse",
    "Somatostatin": "Rabbit",
}


@dataclass
class LitEntry:
    marker: str
    host_species: str          # host species of the antibody used IN THE PAPER
    clonality: str
    dilution_text: str         # e.g. "1:2000" or "not reported"
    dilution_factor: Optional[int]  # numeric denominator, None if not reported
    catalog: str
    tissue: str
    citation: str
    pmid: str
    doi: str
    note: str = ""
    is_mouse_tissue: bool = True  # False for cross-species reference entries

    @property
    def matches_user_host(self) -> bool:
        return self.host_species.lower() == USER_ANTIBODIES[self.marker].lower()


@dataclass
class ControlRegion:
    marker: str
    gene_symbol: str
    region: str
    justification: str


# ---------------------------------------------------------------------------
# Literature: primary-antibody dilution/volume data
# (PMID/DOI verified via PubMed/PMC on 2026-08-11)
# ---------------------------------------------------------------------------
LITERATURE: list[LitEntry] = [
    # ---------------- CALRETININ ----------------
    LitEntry(
        marker="Calretinin",
        host_species="Goat",
        clonality="Polyclonal",
        dilution_text="not reported in accessible text",
        dilution_factor=None,
        catalog="original characterization of the goat anti-calretinin antiserum",
        tissue="mouse, rat, monkey, human brain (IHC/ELISA/WB)",
        citation="Schwaller B, Brückner G, Celio MR, Härtig W (1999). A polyclonal "
                 "goat antiserum against the calcium-binding protein calretinin is "
                 "a versatile tool for various immunochemical techniques. J "
                 "Neurosci Methods 92(1-2):137-44.",
        pmid="10595711",
        doi="10.1016/s0165-0270(99)00106-5",
        note="Foundational validation paper for the goat anti-calretinin reagent "
             "(same host species/clonality as your antibody); confirms specificity "
             "in mouse brain but does not state a working dilution in the "
             "abstract/available text.",
    ),
    LitEntry(
        marker="Calretinin",
        host_species="Rabbit",
        clonality="Polyclonal",
        dilution_text="1:2000",
        dilution_factor=2000,
        catalog="Sigma-Aldrich C7479",
        tissue="mouse cerebral cortex, chromogenic IHC",
        citation="Kim M, Soontornniyomkij V, Ji B, Zhou X (2012). System-wide "
                 "immunohistochemical analysis of protein co-localization. PLoS "
                 "ONE 7(2):e32043.",
        pmid="22363794",
        doi="10.1371/journal.pone.0032043",
    ),
    LitEntry(
        marker="Calretinin",
        host_species="Mouse",
        clonality="Monoclonal",
        dilution_text="1:5000",
        dilution_factor=5000,
        catalog="Millipore MAB1568",
        tissue="mouse olfactory bulb, free-floating immunofluorescence",
        citation="Byrne DJ, Lipovsek M, Crespo A, Grubb MS (2022). Brief sensory "
                 "deprivation triggers plasticity of dopamine-synthesising enzyme "
                 "expression in genetically labelled olfactory bulb dopaminergic "
                 "neurons. Eur J Neurosci 56(1):3591-3612.",
        pmid="35510299",
        doi="10.1111/ejn.15684",
    ),
    LitEntry(
        marker="Calretinin",
        host_species="Rabbit",
        clonality="Polyclonal",
        dilution_text="1:5000",
        dilution_factor=5000,
        catalog="Swant 7699/3H",
        tissue="mouse olfactory bulb, free-floating immunofluorescence",
        citation="Byrne DJ, Lipovsek M, Crespo A, Grubb MS (2022). Eur J Neurosci "
                 "56(1):3591-3612.",
        pmid="35510299",
        doi="10.1111/ejn.15684",
    ),
    LitEntry(
        marker="Calretinin",
        host_species="Rabbit",
        clonality="Monoclonal",
        dilution_text="1:100",
        dilution_factor=100,
        catalog="Invitrogen/Life Technologies",
        tissue="human fetal/infant frontal cortex, paraffin + microwave antigen "
               "retrieval",
        citation="Marguet F, Friocourt G, Brosolo M, et al. (2020). Prenatal "
                 "alcohol exposure is a leading cause of interneuronopathy in "
                 "humans. Acta Neuropathol Commun 8:208.",
        pmid="33256853",
        doi="10.1186/s40478-020-01089-z",
        note="Human paraffin tissue with antigen retrieval, not mouse "
             "free-floating -- included because it is the single most "
             "concentrated CR dilution found in this search; antigen retrieval "
             "protocols commonly tolerate/require more concentrated antibody "
             "than frozen free-floating sections.",
        is_mouse_tissue=False,
    ),

    # ---------------- PARVALBUMIN ----------------
    LitEntry(
        marker="Parvalbumin",
        host_species="Mouse",
        clonality="Monoclonal",
        dilution_text="1:4000",
        dilution_factor=4000,
        catalog="Sigma-Aldrich P3088",
        tissue="mouse cerebral cortex, chromogenic IHC",
        citation="Kim M, Soontornniyomkij V, Ji B, Zhou X (2012). PLoS ONE "
                 "7(2):e32043.",
        pmid="22363794",
        doi="10.1371/journal.pone.0032043",
    ),
    LitEntry(
        marker="Parvalbumin",
        host_species="not stated in accessible excerpt",
        clonality="unstated",
        dilution_text="1:1000",
        dilution_factor=1000,
        catalog="Synaptic Systems #195 004",
        tissue="mouse primary visual cortex, immunofluorescence",
        citation="Lilja A, Didio G, Hong J, Heo WD, Castrén E, Umemori J (2022). "
                 "Optical Activation of TrkB (E281A) in Excitatory and Inhibitory "
                 "Neurons of the Mouse Visual Cortex. Int J Mol Sci 23(18):10249.",
        pmid="36142154",
        doi="10.3390/ijms231810249",
    ),
    LitEntry(
        marker="Parvalbumin",
        host_species="Rabbit",
        clonality="Polyclonal",
        dilution_text="1:1000",
        dilution_factor=1000,
        catalog="Abcam ab11427",
        tissue="mouse cerebral cortex, free-floating immunofluorescence",
        citation="Huh Y, Jung D, Seo T, et al. (2018). Brain stimulation patterns "
                 "emulating endogenous thalamocortical input to "
                 "parvalbumin-expressing interneurons reduce nociception in mice. "
                 "Brain Stimul 11(5):1151-1160.",
        pmid="29784588",
        doi="10.1016/j.brs.2018.05.007",
    ),
    LitEntry(
        marker="Parvalbumin",
        host_species="Goat",
        clonality="Polyclonal",
        dilution_text="1:500",
        dilution_factor=500,
        catalog="Swant PVG-214",
        tissue="mouse presubiculum, free-floating immunofluorescence",
        citation="Nassar M, Simonnet J, Lofredi R, et al. (2015). Diversity and "
                 "overlap of parvalbumin and somatostatin expressing interneurons "
                 "in mouse presubiculum. Front Neural Circuits 9:20.",
        pmid="26005406",
        doi="10.3389/fncir.2015.00020",
        note="Most concentrated PV dilution found in this search.",
    ),

    # ---------------- SOMATOSTATIN ----------------
    LitEntry(
        marker="Somatostatin",
        host_species="Rat",
        clonality="Monoclonal",
        dilution_text="1:200",
        dilution_factor=200,
        catalog="Chemicon/Millipore MAB357",
        tissue="mouse presubiculum, free-floating immunofluorescence",
        citation="Nassar M, Simonnet J, Lofredi R, et al. (2015). Front Neural "
                 "Circuits 9:20.",
        pmid="26005406",
        doi="10.3389/fncir.2015.00020",
        note="Most concentrated SST dilution found in this search; only one "
             "quantified data point was recovered (see caveats below).",
    ),
    LitEntry(
        marker="Somatostatin",
        host_species="not stated in accessible text",
        clonality="unstated",
        dilution_text="not reported in accessible text",
        dilution_factor=None,
        catalog="n/a",
        tissue="mouse neocortex (X94/X98/GIN transgenic lines)",
        citation="Ma Y, Hu H, Berrebi AS, Mathers PH, Agmon A (2006). Distinct "
                 "subtypes of somatostatin-containing neocortical interneurons "
                 "revealed in transgenic mice. J Neurosci 26(19):5069-82.",
        pmid="16687498",
        doi="10.1523/JNEUROSCI.0661-06.2006",
        note="Landmark SST interneuron classification paper in mouse cortex; "
             "full-text methods (with dilution) were not retrievable in this "
             "search -- included for context/citation only.",
    ),
    LitEntry(
        marker="Somatostatin",
        host_species="not stated in abstract",
        clonality="unstated",
        dilution_text="not reported in accessible text",
        dilution_factor=None,
        catalog="n/a",
        tissue="mouse hippocampus, cortex, striatum (BDNF-knockout study)",
        citation="Grosse G, Djalali S, Deng DR, et al. (2005). Area-specific "
                 "effects of brain-derived neurotrophic factor (BDNF) genetic "
                 "ablation on various neuronal subtypes of the mouse brain. Brain "
                 "Res Dev Brain Res 156(2):111-26.",
        pmid="16099299",
        doi="10.1016/j.devbrainres.2004.12.012",
    ),
]


# ---------------------------------------------------------------------------
# Allen Mouse Brain Atlas (Lein et al. 2007, Nature 445:168-176; ISH data at
# mouse.brain-map.org) positive-control regions for each marker.
# ---------------------------------------------------------------------------
CONTROL_REGIONS: list[ControlRegion] = [
    ControlRegion(
        marker="Parvalbumin",
        gene_symbol="Pvalb",
        region="Reticular nucleus of the thalamus (RT)",
        justification="Virtually all RT neurons are GABAergic and PV+; one of "
                       "the strongest, most uniform Pvalb ISH signals in the "
                       "Allen Mouse Brain Atlas -- a very reliable positive "
                       "control.",
    ),
    ControlRegion(
        marker="Parvalbumin",
        gene_symbol="Pvalb",
        region="Cerebellum, Purkinje cell layer",
        justification="Purkinje cells express PV strongly and uniformly across "
                       "the entire cerebellar cortex, giving an easily "
                       "recognized, anatomically distinct positive band.",
    ),
    ControlRegion(
        marker="Parvalbumin",
        gene_symbol="Pvalb",
        region="Globus pallidus (external segment)",
        justification="Dense, high-intensity Pvalb expression in pallidal "
                       "projection neurons; a good secondary/confirmatory "
                       "region alongside RT and cerebellum.",
    ),
    ControlRegion(
        marker="Calretinin",
        gene_symbol="Calb2",
        region="Olfactory bulb (periglomerular and mitral/tufted cells)",
        justification="Among the highest Calb2 ISH signal anywhere in the "
                       "Allen Mouse Brain Atlas; periglomerular interneurons "
                       "are strongly and consistently Calb2+.",
    ),
    ControlRegion(
        marker="Calretinin",
        gene_symbol="Calb2",
        region="Hippocampal hilus / dentate gyrus interneurons",
        justification="A well-characterized population of CR+ interneurons "
                       "and mossy cells gives a reliable, easy-to-locate "
                       "positive signal within the hippocampal formation.",
    ),
    ControlRegion(
        marker="Calretinin",
        gene_symbol="Calb2",
        region="Neocortical layer I / superficial layers (CGE-derived "
               "bipolar interneurons)",
        justification="Layer I and upper layer II/III contain a dense band "
                       "of CR+ bipolar interneurons, useful as a "
                       "region-matched control if your experimental tissue "
                       "is also cortex.",
    ),
    ControlRegion(
        marker="Somatostatin",
        gene_symbol="Sst",
        region="Hypothalamus: periventricular and arcuate nuclei",
        justification="Classic SST-producing neuroendocrine neurons; among "
                       "the strongest Sst ISH signals in the Allen Mouse "
                       "Brain Atlas.",
    ),
    ControlRegion(
        marker="Somatostatin",
        gene_symbol="Sst",
        region="Neocortex, layer V/VI (Martinotti and non-Martinotti "
               "SST+ interneurons)",
        justification="A reliable, moderately dense SST+ interneuron "
                       "population; useful region-matched control if your "
                       "tissue of interest is cortex.",
    ),
    ControlRegion(
        marker="Somatostatin",
        gene_symbol="Sst",
        region="Central amygdala / bed nucleus of the stria terminalis "
               "(BNST)",
        justification="Both nuclei show dense, high-confidence Sst "
                       "expression in the Allen Mouse Brain Atlas and are "
                       "commonly used SST positive controls in the "
                       "literature.",
    ),
]


def _fmt_dilution(entry: LitEntry) -> str:
    return entry.dilution_text


def print_literature_table() -> None:
    col_widths = (13, 34, 12, 40, 8)
    header = ("Marker", "Host (paper) / Catalog", "Dilution", "Citation", "PMID")
    _print_row(header, col_widths, header=True)
    for marker in USER_ANTIBODIES:
        rows = [e for e in LITERATURE if e.marker == marker]
        print(f"\n--- {marker} "
              f"(your antibody host: {USER_ANTIBODIES[marker]}) ---")
        for e in rows:
            host_cat = f"{e.host_species} / {e.catalog}"
            citation_short = e.citation.split(".")[0] + "."
            match_flag = " [SAME HOST]" if e.matches_user_host else ""
            _print_row(
                (marker, host_cat + match_flag, _fmt_dilution(e), citation_short, e.pmid),
                col_widths,
            )
        quantified = [e for e in rows if e.dilution_factor is not None]
        mouse_quantified = [e for e in quantified if e.is_mouse_tissue]
        if mouse_quantified:
            best = min(mouse_quantified, key=lambda e: e.dilution_factor)
            print(f"  -> Literature-derived MAXIMUM in MOUSE tissue for "
                  f"{marker}: {best.dilution_text} "
                  f"({best.host_species} host, {best.catalog}, PMID {best.pmid})")
        elif quantified:
            best = min(quantified, key=lambda e: e.dilution_factor)
            print(f"  -> No mouse-tissue dilution recovered for {marker}; "
                  f"most concentrated non-mouse reference: {best.dilution_text} "
                  f"({best.tissue}, PMID {best.pmid})")
        else:
            print(f"  -> No quantified dilution recovered for {marker} in this "
                  f"search; see notes column in the CSV.")
        outliers = [e for e in quantified if not e.is_mouse_tissue]
        for o in outliers:
            print(f"  -- FYI, non-mouse-tissue reference point: {o.dilution_text} "
                  f"in {o.tissue} (PMID {o.pmid}); not directly comparable, see "
                  f"note in CSV.")


def _print_row(cols, widths, header=False):
    line = " | ".join(str(c)[:w].ljust(w) for c, w in zip(cols, widths))
    print(line)
    if header:
        print("-" * (sum(widths) + 3 * (len(widths) - 1)))


def print_control_region_table() -> None:
    print("\n\nPositive control brain regions (mouse), Allen Mouse Brain Atlas")
    print("(Lein et al. 2007, Nature 445:168-176; mouse.brain-map.org)")
    col_widths = (13, 8, 38, 60)
    _print_row(("Marker", "Gene", "Region", "Why it's a good control"), col_widths, header=True)
    for c in CONTROL_REGIONS:
        _print_row((c.marker, c.gene_symbol, c.region, c.justification), col_widths)


def write_csvs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lit_path = os.path.join(OUTPUT_DIR, "ihc_antibody_dilution_literature.csv")
    with open(lit_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "marker", "your_antibody_host", "paper_host_species", "clonality",
            "dilution", "dilution_denominator", "same_host_as_yours",
            "mouse_tissue", "catalog", "tissue", "citation", "pmid", "doi", "notes",
        ])
        for e in LITERATURE:
            writer.writerow([
                e.marker, USER_ANTIBODIES[e.marker], e.host_species, e.clonality,
                e.dilution_text, e.dilution_factor if e.dilution_factor else "",
                "yes" if e.matches_user_host else "no",
                "yes" if e.is_mouse_tissue else "no",
                e.catalog, e.tissue, e.citation, e.pmid, e.doi, e.note,
            ])

    ctrl_path = os.path.join(OUTPUT_DIR, "positive_control_regions.csv")
    with open(ctrl_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["marker", "gene_symbol", "region", "justification"])
        for c in CONTROL_REGIONS:
            writer.writerow([c.marker, c.gene_symbol, c.region, c.justification])

    print(f"\nWrote:\n  {lit_path}\n  {ctrl_path}")


def print_summary_recommendation() -> None:
    print("\n\nSuggested starting point for your dilution/volume optimization "
          "series")
    print("(bracket the literature maximum -- do not exceed it as your most "
          "concentrated test point):")
    for marker in USER_ANTIBODIES:
        quantified = [e for e in LITERATURE
                      if e.marker == marker and e.dilution_factor is not None]
        mouse_quantified = [e for e in quantified if e.is_mouse_tissue]
        pool = mouse_quantified or quantified
        if not pool:
            print(f"  {marker}: no quantified literature value found -- start "
                  f"conservatively (e.g. 1:1000) and titrate.")
            continue
        best = min(pool, key=lambda e: e.dilution_factor)
        series = [best.dilution_factor, best.dilution_factor * 2,
                  best.dilution_factor * 4, best.dilution_factor * 8]
        series_txt = ", ".join(f"1:{v}" for v in series)
        tag = "" if mouse_quantified else " (non-mouse tissue, treat with caution)"
        print(f"  {marker}: literature max{tag} = 1:{best.dilution_factor} "
              f"-> try {series_txt}")


def main() -> None:
    print("=" * 100)
    print("IHC primary-antibody dilution/volume literature survey")
    print("Calretinin (goat), Parvalbumin (mouse), Somatostatin (rabbit)")
    print("=" * 100)
    print_literature_table()
    print_control_region_table()
    print_summary_recommendation()
    write_csvs()


if __name__ == "__main__":
    main()
