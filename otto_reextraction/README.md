# Otto Re-extraction Pipeline

Re-runs your 4 extraction prompt templates against freshly fetched paper
text, lines the results up next to Otto's original `study_level.csv`
output for manual review, and (via `verify.py`) has the model actively
audit whether each of Otto's original values is correct against the
source text — so you can check Otto's accuracy without eyeballing all
400+ fields at equal weight.

```
data/otto_output.csv (Otto's original extraction, one row per paper)
        │
1. fetch_papers.py   → data/cache/<paper_id>.json (+ data/fetch_manifest.csv)
        │               DOI → PMC/PubMed full text, abstract fallback, cached
2. reextract.py       → data/reextracted_raw/<paper_id>__<modality>.json
        │               + data/reextracted.csv                              re-run the 4 prompts via the
        │                                                                    Anthropic API against fetched text
        ├──────────────────────┬─────────────────────────────
        │                      │
3a. compare.py                 3b. verify.py
        │  → data/side_by_side.csv     │  → data/verification_raw/<paper_id>__<modality>.json
        │    [otto_value vs.            │  → data/verification.csv
        │     reextracted_value,        │    [paper_id, otto_field, verdict, confidence,
        │     for eyeballing]           │     explanation, evidence_quote] - model judges Otto
        │                               │     against the paper text (not just vs. reextraction)
        │                               │
4a. render_review.py           4b. render_verification.py
        → data/review.html             → data/verification.html
          per-paper side-by-side          flat, sortable/filterable triage table,
          browsing view                   mismatches-first
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export NCBI_API_KEY=...   # optional but recommended: raises the E-utilities rate limit from ~3/s to ~10/s
```

## Running

```bash
cd otto_reextraction

python fetch_papers.py                 # fetch + cache paper text for every row in data/otto_output.csv
python reextract.py                    # re-run the 4 prompts for every paper with cached text

python compare.py                      # join otto vs. reextracted -> data/side_by_side.csv
python render_review.py                # data/side_by_side.csv -> data/review.html

python verify.py                       # audit otto vs. paper text (+ reextraction as a cross-check) -> data/verification.csv
python render_verification.py          # data/verification.csv -> data/verification.html
```

`verify.py` is the "did Otto get this right" check: for each paper and
Otto field, it shows the model the paper text, Otto's original value, and
the independent re-extraction, and asks for a verdict — `match`,
`partial` (correct but incomplete), `mismatch` (wrong/unsupported), or
`unverifiable` (not enough text to judge, e.g. abstract-only) — plus a
one-line explanation and a supporting quote. It's a second opinion, not
ground truth: sort `data/verification.csv` by `verdict` (mismatches are
sorted first) or open `data/verification.html` and uncheck "match" to see
only what needs a human look.

Every stage is resumable and cheap to re-run: `fetch_papers.py` skips
papers already in `data/cache/`, and `reextract.py` skips (paper, modality)
pairs already in `data/reextracted_raw/`. Pass `--force` to either to redo
the fetch/API call. Use `--paper-id AIM-50` or `--limit 3` on
`fetch_papers.py`/`reextract.py` to test on a small subset before running
the full ~70-paper batch (each full run is 71 papers × 4 prompts = ~280 API
calls).

## Modality → Otto column mapping

The 4 prompts don't map 1:1 onto Otto's 6 extraction columns — two Otto
columns don't have a dedicated prompt of their own, so they're compared
against their closest-related modality's re-extraction rather than against
a fresh prompt written from scratch for them:

| Modality      | Prompt file             | Otto column(s) compared against it                                    |
|----------------|--------------------------|-------------------------------------------------------------------------|
| `anatomical`   | `prompts/anatomical.txt` | `Anatomical connections`, `Verification of Zona Incerta Targeting`     |
| `photometry`   | `prompts/photometry.txt` | `Photometry Data`                                                       |
| `ephys`        | `prompts/ephys.txt`      | `Electrophysiology Data`                                                |
| `functional`   | `prompts/functional.txt` | `Purpose of this circuit`, `Necessity vs. sufficiency or Both`          |

## How re-extraction is structured

Your prompts ask for a markdown table (one row per pathway / per
population-epoch / per cell-type group) followed by a summary paragraph —
that's the right format for a human reading the chat, but not something a
pipeline can reliably parse back out. So `llm_client.py` sends your prompt
text **verbatim** as the analysis instructions, wrapped with a tool-use
(function-calling) contract that forces the model to return the same
content as structured JSON instead of freeform markdown:

```json
{
  "rows": [
    {"row_label": "...", "figure_ref": "...", "source_quote": "...", "fields": {"<field from your prompt>": "value", ...}}
  ],
  "summary_paragraph": "..."
}
```

`fields` holds every itemized sub-field your prompt asked for for that row
(e.g. all 16 anatomical sub-fields), keyed by your prompt's own field
names — nothing about the analytical instructions changes, only the output
format. `figure_ref` and `source_quote` are pulled to the top level of each
row because `compare.py` needs them as their own columns.

`compare.py` then emits, per paper and per Otto field: one `(overall
summary)` row (the model's summary paragraph, most directly comparable to
Otto's single condensed cell) plus one row per itemized item (pathway /
population-epoch / cell-type group) with its own quote and figure
reference — so you can check the coarse summary and drill into individual
data points from the same CSV.

## Known limitation: NCBI access in this sandbox

`fetch_papers.py` calls NCBI E-utilities directly (`eutils.ncbi.nlm.nih.gov`,
`www.ncbi.nlm.nih.gov`), which is the correct approach for a standalone,
repeatable script — but the network policy for *this particular Claude Code
session* blocks that host (403 at the egress proxy), so I could not run a
live end-to-end smoke test of `fetch_papers.py` here. The DOI→PMCID lookup,
full-text `efetch`, and abstract-fallback logic are covered by unit tests
against mocked HTTP responses (`tests/test_ncbi.py`), but you should run
`python fetch_papers.py --limit 3` yourself first (from an environment that
can reach NCBI) to confirm live behavior before kicking off the full batch.

## Copyright note

Fetched paper text (`data/cache/`) and everything derived from it
(`data/reextracted_raw/`, `data/reextracted.csv`, `data/side_by_side.csv`,
`data/review.html`, `data/verification_raw/`, `data/verification.csv`,
`data/verification.html`) is gitignored — full article text, including
from the PMC open-access subset, generally shouldn't be committed to a
git repo. Only `data/otto_output.csv` (your own extraction) is tracked.

## Tests

```bash
pip install pytest
pytest
```

30 tests cover CSV loading, DOI normalization, JATS XML parsing (including a
regression test for double-counting `<caption><p>` text), the
flatten/render logic, the otto/reextraction join in `compare.py`, the
verification flatten/HTML rendering, and the Anthropic call wrapper's
retry/backoff behavior for both extraction and verification calls — all
against fixtures or fake clients, no network or API key required.
