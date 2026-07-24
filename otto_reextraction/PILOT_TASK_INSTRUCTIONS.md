# Otto re-extraction + verification task (standing in for the Anthropic API)

You are doing scientific data extraction on a neuroscience paper, standing in for what would
normally be an Anthropic API call in the `otto_reextraction` pipeline (that pipeline has no
API key available right now, so a human is having you do this extraction pass directly instead).

CONTEXT: This is part of a project auditing "Otto" (an AI research-extraction tool) against 71
papers about the zona incerta (ZI) brain region. You will be given ONE paper (its Reference ID,
e.g. `AIM-12`). For that paper you must (1) independently re-extract structured data using 4
fixed prompt templates, writing each as a JSON file, and (2) verify whether Otto's own
previously-recorded values for this paper are accurate given the actual paper text.

PAPER TEXT: Read `/home/user/AZ/otto_reextraction/data/cache/<PAPER_ID>.json` for the full paper
text — it's in the `"text"` field (a JSON string). This is the ONLY source of truth for the
paper's content; don't use outside knowledge about this paper.

OTTO'S ORIGINAL VALUES: Read `/home/user/AZ/otto_reextraction/data/otto_values/<PAPER_ID>.json`
for Otto's previously-recorded field values for this paper (Title, Anatomical connections,
Verification of Zona Incerta Targeting, Photometry Data, Electrophysiology Data, Purpose of this
circuit, Necessity vs. sufficiency or Both). Empty string means Otto recorded nothing for that
field.

## TASK 1 — Re-extraction (produce 4 files)

For each of the 4 modalities below, apply the given extraction instructions to the paper text,
and write the result as a JSON file at
`/home/user/AZ/otto_reextraction/data/reextracted_raw/<PAPER_ID>__<modality>.json`.
Follow the instructions literally, including row/table structure. If a value is not explicitly
stated in the paper text, write "not explicitly stated" for it — never estimate or infer a
number that isn't given.

Each output JSON file must have exactly this shape:
```json
{
  "rows": [
    {
      "row_label": "short human-readable label for what this row covers",
      "figure_ref": "figure/table reference, or 'not explicitly stated'",
      "source_quote": "a short verbatim quote from the paper text supporting this row, or 'not explicitly stated'",
      "fields": {"<field name from the instructions>": "<value>", "...": "..."}
    }
  ],
  "summary_paragraph": "the free-text summary paragraph the instructions ask for, or the direct answer if the instructions pose a single question rather than a table"
}
```
`rows` should have one entry per distinct item the instructions ask you to enumerate (e.g. one
row per traced pathway, per population-per-epoch, or per cell-type group). Leave `rows` as an
empty list ONLY if the instructions ask a single question with no table (this applies to the
"functional" modality, which is just one question — put your answer directly in
`summary_paragraph` and use `"rows": []`).

### Modality: anatomical → `<PAPER_ID>__anatomical.json`
Instructions:
"""
The anatomical circuitry(s) discussed in the paper:

You are extracting anatomical circuit tracing data from a neuroscience paper involving the zona
incerta (ZI) or a directly connected region. This includes anterograde tracing (AAV, PHAL, BDA),
retrograde tracing (CTB, retrobeads, retrograde AAV/rAAV), monosynaptic rabies tracing, and
transsynaptic/polysynaptic labeling (e.g., HSV, pseudorabies, TRIO/cTRIO). Identify every
distinct traced pathway in the paper — a single paper may report several, so make sure to report
all of the circuits present in the paper (e.g., ZI→PAG, PL→ZI, ZI→RE). Step 1 — Identify each
distinct traced pathway/circuit as the authors define it (source region, target region, and
direction of tracing: anterograde or retrograde). Note if a pathway is cell-type specific (e.g.,
"Vgat+ ZI→PVT" rather than just "ZI→PVT"). Step 2 — For each pathway, extract (where reported):
(1) Source region (and specific subregion, e.g., rostral/caudal/dorsal/ventral ZI) (2) Target
region (3) Tracing direction (anterograde/retrograde/transsynaptic/monosynaptic rabies) (4)
Tracer/virus used (e.g., AAV-DIO-ChR2, CTB-647, rabies-EnvA-ΔG, PRV) (5) Cre-driver line or
promoter used for cell-type specificity (e.g., Vgat-Cre, Vglut2-Cre, TH-Cre) (6) Injection
coordinates (AP/ML/DV) for source and/or target, if given (7) Injection volume (nL) and infusion
rate, if given (8) Survival time post-injection (days/weeks) (9) N animals used for tracing
quantification (10) Quantification metric used (e.g., % of labeled cells, fluorescence
intensity, fiber density, number of labeled boutons) (11) Quantitative result (the actual
number/percentage reported) (12) Co-localization data (e.g., % overlap with a molecular marker
such as Vgat, PV, SST, TH) (13) Whether functional connectivity was also tested (e.g.,
optogenetically-evoked EPSC/IPSC in target neurons) and the result (14) Statistical test and
p-value for any quantitative comparison between groups/pathways (15) Figure/table reference, and
whether main text, figure, or supplement (16) Any caveats the authors note about tracer spread,
off-target labeling, or specificity limitations. Step 3 — Extend the column set for
pathway-specific details not listed above (e.g., laterality — ipsilateral vs. contralateral,
layer-specific cortical origin such as L5, collateral branching patterns). Step 4 — If a value is
not explicitly stated, write "not explicitly stated." Never estimate coordinates, volumes, or
percentages. Step 5 — If the same source-target pathway is examined with more than one tracing
method in the same paper (e.g., anterograde AAV plus retrograde CTB), create separate rows for
each method rather than merging results. Step 6 — After the table, write a short paragraph
mapping each traced pathway onto a simple "region A → region B (cell type, direction of effect
if functionally tested)" notation, since this format will be most directly usable for building
your connectome/circuit-prediction model.
"""
If this paper has NO anatomical circuit tracing data at all, write `"rows": []` and a one-line
`summary_paragraph` saying none was found.

### Modality: photometry → `<PAPER_ID>__photometry.json`
Instructions:
"""
You are extracting fiber photometry / in vivo calcium imaging data from a neuroscience paper
involving the zona incerta (ZI) or a directly connected circuit. Identify every distinct
recorded population and every distinct behavioral/physiological epoch analyzed — do not assume
there is only one of each.

Step 1 — Identify all distinct recorded populations (by region, marker, or projection target,
exactly as named by the authors — e.g., "ZI Vgat+ neurons," "ZI Lhx6+ neurons," "A13
dopaminergic neurons," "ZIr-PVT projection neurons").

Step 2 — Identify all distinct behavioral states or epochs the paper aligns signal to (e.g.,
locomotion onset, wake/NREM/REM transitions, cue presentation, reward consumption, nociceptive
stimulus, seizure onset, anesthesia induction/emergence). If a population is recorded across
multiple states, create one row per population-per-state combination.

Step 3 — For each row, extract (where reported): (1) Population/cell type recorded (verbatim
from paper) (2) Indicator used (e.g., GCaMP6, GCaMP6M, GCaMP8) and delivery method (viral
construct, Cre-line, microendoscope/GRIN lens vs. fiber photometry) (3) Behavioral/physiological
state or epoch analyzed (4) Direction of signal change (increase/decrease/no change) relative to
baseline or to the comparator state (5) Peak ΔF/F (%) or z-score, with ±SD/SEM (6) N animals (7)
N neurons (if single-cell resolution via microendoscopy) or N trials/events (if bulk fiber
photometry) (8) Latency to peak or onset of change (s) (9) Duration of response (s) (10)
Classification of neurons by dominant state (e.g., "wake-max," "REM-max," "NREM-max," if paper
uses this framework) (11) Percentage of neurons in each classification (if single-cell
resolution) (12) Correlation with behavioral variable (r, R², or regression coefficient) if
reported (13) p-value for signal change vs. baseline (14) p-value for between-group/between-
condition comparison (15) Statistical test used (16) Control signal/condition used (isosbestic
control, GFP-only virus, non-Cre control) (17) Effect of pharmacological/optogenetic/
chemogenetic manipulation on the recorded signal, if co-tested (18) Recording system/hardware,
if specified (19) Figure/table reference, and whether value is from main text, figure, or
supplement

Step 4 — Extend the column set for anything paper-specific not listed (e.g., cross-correlation
with EEG delta power, cFos co-labeling percentage, causal ablation results paired with
photometry).

Step 5 — If a value is not explicitly stated, write "not explicitly stated." Never estimate.

Step 6 — If the paper reports state classifications as percentages of a population (e.g., "51%
REM-max, 24% wake-max"), preserve these exactly and do not average or collapse across papers even
if they used similar terminology — paper-specific percentages must stay attributed to that paper
alone.

Step 7 — After the table, write a short paragraph noting whether the recorded population's
photometry signal is described as necessary, sufficient, or merely correlated with the
behavior/state (this distinction matters for your movement/behavior-prediction model, since
correlation-only findings should be weighted differently than causal findings). Return a single
flat table, one row per population-per-behavioral-epoch, with "Paper Citation" as the first
column. Do not merge multiple epochs into a single cell — always split into separate rows.
Append paper-specific extra columns at the end rather than inserting them mid-table. After the
table, include the necessity/sufficiency/correlation summary paragraph from Step 7 as plain
text, not a table row.
"""
If this paper has NO fiber photometry / calcium imaging data at all, write `"rows": []` and a
one-line `summary_paragraph` saying none was found.

### Modality: ephys → `<PAPER_ID>__ephys.json`
Instructions:
"""
Extract all the electrophysiology data present in the paper.
(1) Cell Type / Group (2) Mean Cm in pF with ±SD (3) N neurons/N animals for Cm (4) Mean Rin in
MΩ with ±SD (5) N neurons/N animals for Rin (6) AP half-width in ms (7) Upstroke:Downstroke
ratio (8) Depolarization blockade (Yes/No) (9) Current at blockade in pA if applicable (10)
Resting membrane potential in mV (11) AP threshold in mV (12) Spontaneous firing rate in Hz (13)
Voltage sag present (Yes/No) (14) Voltage sag magnitude in % (15) Rebound spiking (Yes/No) (16)
Recording temperature in °C (17) Internal/external solution composition (18) p-value for Cm
comparison within group/region (19) p-value for Rin comparison within group/region (20) p-value
for AP kinetics comparison (21) Statistical test used for each comparison (22) Figure/table
reference where each data point appears (23) Functional classification (if author mentioned)
"""
One row per distinct cell type/group recorded. If this paper has NO electrophysiology/patch-clamp
data at all, write `"rows": []` and a one-line `summary_paragraph` saying none was found.

### Modality: functional → `<PAPER_ID>__functional.json`
Instructions: "What purpose does this circuit serve?"
This is a single question, not a table — use `"rows": []` and put your answer (a few sentences,
grounded in what the paper actually says the circuit does) in `summary_paragraph`. Additionally,
in that same summary_paragraph, state whether the paper's evidence establishes this as a
NECESSITY finding (loss-of-function: lesion/inhibition/ablation shows the circuit is required), a
SUFFICIENCY finding (gain-of-function: activation is enough to drive the effect), BOTH, or
NEITHER/unclear — and briefly say why.

## TASK 2 — Verify Otto's original values (produce 4 files)

For each field in the otto_values file, decide whether Otto's value is accurate and reasonably
complete given the actual paper text you just read. Use your Task 1 re-extraction as a helpful
cross-check, but base your verdict on the paper text itself — your re-extraction can also be
wrong, so don't treat it as ground truth either. An empty/blank Otto value is "match" only if
the paper genuinely reports nothing for that field; otherwise it's "mismatch" or "partial".

Verdict options: "match" (accurate and reasonably complete), "partial" (correct as far as it goes
but missing something the paper reports), "mismatch" (wrong, unsupported, or contradicted by the
text), "unverifiable" (not enough info in the text to judge — including cases where a number
might live only in a figure/table that didn't extract as text; say so in the explanation rather
than calling it a hard mismatch).

Write 4 JSON files, each shaped like:
```json
{"verdicts": [{"otto_field": "<exact field name>", "verdict": "match|partial|mismatch|unverifiable", "confidence": "high|medium|low", "explanation": "one or two sentences justifying the verdict", "evidence_quote": "a short verbatim quote from the paper text, or 'not explicitly stated'"}]}
```
- `data/verification_raw/<PAPER_ID>__anatomical.json` → verdicts for BOTH "Anatomical
  connections" and "Verification of Zona Incerta Targeting" (2 entries)
- `data/verification_raw/<PAPER_ID>__photometry.json` → verdict for "Photometry Data" (1 entry)
- `data/verification_raw/<PAPER_ID>__ephys.json` → verdict for "Electrophysiology Data" (1 entry)
- `data/verification_raw/<PAPER_ID>__functional.json` → verdicts for BOTH "Purpose of this
  circuit" and "Necessity vs. sufficiency or Both" (2 entries)

Write all 8 files (4 re-extraction + 4 verification) as valid, parseable JSON (use the Write
tool), with `<PAPER_ID>` replaced by the actual Reference ID throughout all paths. Do not add
commentary outside the files. When done, reply with a one-paragraph summary of what you found
(any notable mismatches) — under 150 words.
