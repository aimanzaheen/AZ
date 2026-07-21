# Otto Re-extraction Pipeline

Re-runs your 4 extraction prompt templates against paper text you supply
locally, lines the results up next to Otto's original `study_level.csv`
output for manual review, and (via `verify.py`) has the model actively
audit whether each of Otto's original values is correct against the
source text — so you can check Otto's accuracy without eyeballing all
400+ fields at equal weight.

```
data/otto_output.csv (Otto's original extraction, one row per paper)
papers/*.pdf (you supply these)
        │
1. scrape_papers.py  → data/cache/<paper_id>.json (+ data/scrape_manifest.csv)
        │               matches each file to its Reference ID, extracts text, caches it
2. reextract.py       → data/reextracted_raw/<paper_id>__<modality>.json
        │               + data/reextracted.csv                              re-run the 4 prompts via the
        │                                                                    Anthropic API against that text
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

An alternate `fetch_papers.py` (DOI → PMC/PubMed via NCBI E-utilities) is
also included for papers you don't have a local copy of — see "Fetching
instead of supplying papers" below. Both write the same
`data/cache/<paper_id>.json` schema, so `reextract.py`/`verify.py` don't
care which one produced it.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Running

```bash
cd otto_reextraction

# 1. Drop your paper files into papers/ (PDF, TXT, or MD).
#    Naming a file after its Reference ID (e.g. AIM-50.pdf) guarantees a
#    correct match; otherwise scrape_papers.py falls back to matching by
#    DOI found in the text, then fuzzy title match.
cp /path/to/your/pdfs/*.pdf papers/

python scrape_papers.py                # match + cache text -> data/cache/, data/scrape_manifest.csv
python reextract.py                    # re-run the 4 prompts for every paper with cached text

python compare.py                      # join otto vs. reextracted -> data/side_by_side.csv
python render_review.py                # data/side_by_side.csv -> data/review.html

python verify.py                       # audit otto vs. paper text (+ reextraction as a cross-check) -> data/verification.csv
python render_verification.py          # data/verification.csv -> data/verification.html
```

**Always check `data/scrape_manifest.csv` (also printed at the end of the
run) before trusting anything downstream** — it lists every file's match
method and confidence, every file that didn't match anything, and every
`otto_output.csv` row that had no matching file. A `title`-method match
means the filename/DOI didn't line up and the model matched based on
title-word overlap; spot-check those. A `duplicate_match` means two files
matched the same paper — the higher-confidence one won and the other was
skipped.

`verify.py` is the "did Otto get this right" check: for each paper and
Otto field, it shows the model the paper text, Otto's original value, and
the independent re-extraction, and asks for a verdict — `match`,
`partial` (correct but incomplete), `mismatch` (wrong/unsupported), or
`unverifiable` (not enough text to judge, e.g. abstract-only) — plus a
one-line explanation and a supporting quote. It's a second opinion, not
ground truth: sort `data/verification.csv` by `verdict` (mismatches are
sorted first) or open `data/verification.html` and uncheck "match" to see
only what needs a human look.

Every stage is resumable and cheap to re-run: `scrape_papers.py` re-derives
the manifest each run but only re-caches files whose match changed;
`reextract.py`/`verify.py` skip (paper, modality) pairs already in
`data/reextracted_raw/`/`data/verification_raw/`. Pass `--force` to
`reextract.py`/`verify.py` to redo an API call. Use `--paper-id AIM-50` or
`--limit 3` on `reextract.py`/`verify.py` to test on a small subset before
running the full ~70-paper batch (each full run is 71 papers × 4 prompts =
~280 API calls per stage).

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

## How paper matching works (`scrape_papers.py`)

For each file in `papers/`, in order of confidence:

1. **Filename** (minus extension) equals a `Reference ID`, e.g. `AIM-50.pdf` → `AIM-50` (confidence 1.0).
2. **DOI**: a DOI found anywhere in the extracted text matches a row's `DOI` column (confidence 0.95).
3. **Title**: word-overlap between the extracted text's first ~4000 characters and a row's `Title`, if it clears a 0.5 Jaccard threshold (confidence = the overlap score).

If two files match the same paper, the higher-confidence one wins and the
other is recorded as `duplicate_match` in the manifest rather than silently
overwritten. Files that don't clear any of the three thresholds are
recorded as `unmatched`. Rows in `otto_output.csv` with no matching file
are listed at the end of the run so you know exactly what's missing before
`reextract.py`/`verify.py` silently skip them.

## Fetching instead of supplying papers

If you'd rather have papers fetched automatically instead of supplying
files, `fetch_papers.py` looks each row's DOI up on PMC/PubMed via NCBI
E-utilities and falls back to an abstract if full text isn't open-access,
writing the same cache format `scrape_papers.py` does. It calls
`eutils.ncbi.nlm.nih.gov`/`www.ncbi.nlm.nih.gov` directly, which this
particular sandbox's network policy blocks (403 at the egress proxy), so
it's covered by unit tests against mocked HTTP responses
(`tests/test_ncbi.py`) rather than a live smoke test here — run
`python fetch_papers.py --limit 3` yourself first from an environment that
can reach NCBI to confirm live behavior.

## Copyright note

Both `papers/` (your supplied files) and everything derived from paper
text (`data/cache/`, `data/reextracted_raw/`, `data/reextracted.csv`,
`data/side_by_side.csv`, `data/review.html`, `data/verification_raw/`,
`data/verification.csv`, `data/verification.html`,
`data/scrape_manifest.csv`) are gitignored — full article text shouldn't be
committed to a git repo. Only `data/otto_output.csv` (your own extraction)
is tracked.

## Tests

```bash
pip install pytest
pytest
```

44 tests cover CSV loading, DOI normalization, JATS XML parsing (including
a regression test for double-counting `<caption><p>` text), local PDF/TXT
extraction, the filename/DOI/title paper-matching cascade (including
duplicate-match handling), the flatten/render logic, the otto/reextraction
join in `compare.py`, the verification flatten/HTML rendering, and the
Anthropic call wrapper's retry/backoff behavior for both extraction and
verification calls — all against fixtures, fake clients, or generated
sample PDFs, no network or API key required.
