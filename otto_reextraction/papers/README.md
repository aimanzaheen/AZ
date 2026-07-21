Drop your paper files here (PDF, TXT, or MD).

Naming them after the `Reference ID` column in `data/otto_output.csv`
(e.g. `AIM-50.pdf`) guarantees a correct match. If you don't, `scrape_papers.py`
falls back to matching by DOI found in the text, then by fuzzy title match —
check `data/scrape_manifest.csv` after running it to confirm every file
matched the paper you expected.

This folder's contents are gitignored (see `../.gitignore`) - don't commit
paper PDFs to the repo.
