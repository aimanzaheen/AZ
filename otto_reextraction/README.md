# Otto Re-extraction Pipeline

Re-runs your 4 extraction prompt templates against freshly fetched paper
text and lines the results up next to Otto's original `study_level.csv`
output for manual review. No auto-resolution — everything lands side by
side for a human to eyeball.

```
data/otto_output.csv (Otto's original extraction, one row per paper)
        │
1. fetch_papers.py   → data/cache/<paper_id>.json (+ data/fetch_manifest.csv)
        │               DOI → PMC/PubMed full text, abstract fallback, cached
2. reextract.py       → data/reextracted_raw/<paper_id>__<modality>.json
        │               + data/reextracted.csv                              re-run the 4 prompts via the
        │                                                                    Anthropic API against fetched text
3. compare.py         → data/side_by_side.csv
        │               [paper_id, field_name, otto_value, reextracted_value, source_quote, figure_ref, ...]
4. render_review.py   → data/review.html
                        filterable, collapsible page for scanning ~400+ rows across dozens of papers
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
```

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
`data/review.html`) is gitignored — full article text, including from the
PMC open-access subset, generally shouldn't be committed to a git repo.
Only `data/otto_output.csv` (your own extraction) is tracked.

## Tests

```bash
pip install pytest
pytest
```

22 tests cover CSV loading, DOI normalization, JATS XML parsing (including a
regression test for double-counting `<caption><p>` text), the
flatten/render logic, the otto/reextraction join in `compare.py`, and the
Anthropic call wrapper's retry/backoff behavior — all against fixtures or
fake clients, no network or API key required.
