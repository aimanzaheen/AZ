# LH Injection Harvester

Harvests papers describing tracer injections into the mouse brain and compiles them into one
sortable, filterable HTML table with:

- Paper name
- Transmission type (retrograde / anterograde / both)
- Volume injected into LH
- Stereotaxic coordinates
- LHA subdivided? (single/undivided site, or which named subregion was targeted)
- Tracer
- Projections found
- Survival time before perfusion

Configured by default for **tracer injections into the lateral hypothalamus (LH/LHA)**, in
either direction (CTB, fluorogold, retrobeads, retrograde AAV, rabies for retrograde; PHA-L, BDA,
anterograde AAV for anterograde) - override `--query` to target a different injection type,
tracer, or brain region.

```
search_papers.py  → data/search_results.csv     PubMed esearch/esummary for the query
        │                                        (pmid, title, authors, year, journal, doi)
fetch_papers.py    → data/cache/<pmid>.json      PMC full text if open-access, else abstract
        │                                        + data/fetch_manifest.csv
extract.py         → data/extracted_raw/<pmid>.json   Anthropic API call per paper against
        │             + data/extracted.csv             prompts/lh_injection.txt -
        │                                               one row per distinct injection experiment
render_table.py    → data/table.html             the final webpage: sortable/filterable table
```

`run_all.py` runs all four stages in sequence.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
# optional, raises the NCBI unauthenticated rate limit from ~3/sec to ~10/sec:
export NCBI_API_KEY=...
```

## Running

```bash
cd lh_injection_harvester
python run_all.py
```

Open `data/table.html` in a browser.

Or run the stages individually (useful for tuning the query or re-running just extraction):

```bash
python search_papers.py --query '...' --max-results 200   # -> data/search_results.csv
python fetch_papers.py                                     # -> data/cache/, fetch_manifest.csv
python extract.py                                           # -> data/extracted_raw/, extracted.csv
python render_table.py                                       # -> data/table.html
```

Every stage is resumable and cheap to re-run: `fetch_papers.py` skips PMIDs already cached,
`extract.py` skips PMIDs already in `data/extracted_raw/`. Pass `--force` to redo a stage. Use
`--limit N` or `--pmid <id>` on `fetch_papers.py`/`extract.py` to test on a small subset before
running the full batch (each paper costs one Anthropic API call).

## Targeting a different injection type

Edit `--query` (or `common.DEFAULT_QUERY`) to change what `search_papers.py` searches for, and
edit `prompts/lh_injection.txt` (or point `extract.py --prompt-file` at a new file) to change
what gets extracted - e.g. narrow back to retrograde-only, or point at a different target region
entirely. The output schema in `llm_client.py` (`RECORD_LH_INJECTIONS_TOOL`) is generic enough to
reuse as-is; only the prompt wording needs to change to redefine "match" and to explain the new
injection type.

## Papers the search missed a false match on

`extract.py` calls tell the model to first decide whether the paper actually reports a tracer
injection placed IN the LH at all (the keyword search will pull in false positives - e.g. papers
that inject a tracer elsewhere and happen to find LH cells labeled as a downstream result, which
doesn't count). Those are recorded in `data/extracted_raw/<pmid>.json` with
`"lh_injection_present": false` and excluded from the table - check that file's `"notes"` field
if you want to know why a given paper was excluded.

## Copyright note

`data/cache/`, `data/extracted_raw/`, `data/extracted.csv`, `data/search_results.csv`,
`data/fetch_manifest.csv`, and `data/table.html` are all gitignored - they contain fetched paper
text/abstracts and shouldn't be committed to a git repo.

## Network note

This pipeline calls `eutils.ncbi.nlm.nih.gov` / `www.ncbi.nlm.nih.gov` directly, which some
sandboxed environments block at the egress proxy (403). It's covered by unit tests against mocked
HTTP responses (`tests/test_ncbi.py`, `tests/test_search_papers.py`,
`tests/test_fetch_papers.py`) rather than a live smoke test here - run it yourself first from an
environment that can reach NCBI, and confirm live behavior before trusting the harvested table.

## Tests

```bash
pip install pytest
pytest
```
