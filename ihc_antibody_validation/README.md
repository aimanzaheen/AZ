# IHC antibody dilution/volume table

Literature survey of published IHC methods for calretinin, parvalbumin, and
somatostatin, built to support a primary-antibody dilution/volume
optimization run with:

- anti-calretinin, host **goat**
- anti-parvalbumin, host **mouse**
- anti-somatostatin, host **rabbit**

## Run

```bash
python3 ihc_antibody_volumes_table.py
```

No third-party dependencies (standard library only). Prints:

1. A literature table of primary-antibody dilutions per marker, with the
   literature-derived **maximum (most concentrated) dilution** flagged per
   marker as an upper-bound reference for your optimization series.
2. Mouse-brain **positive control regions** per marker, based on Allen Mouse
   Brain Atlas expression data (Lein et al. 2007, *Nature* 445:168-176).
3. A suggested starting dilution series (bracketing the literature max).

Also writes two CSVs to `output/`:

- `ihc_antibody_dilution_literature.csv`
- `positive_control_regions.csv`

## Caveat on "volume"

Published IHC methods sections almost never report a literal applied volume
(µL) of primary antibody per section/slide — that depends on each lab's
slide/chamber setup. What's reliably reported is the working **dilution
ratio**, which is what actually sets antibody concentration regardless of
pipetted volume. This script uses dilution ratio as the comparable proxy,
and calls out the least-dilute (most concentrated) value found per marker as
the literature-derived maximum. See `ihc_antibody_volumes_table.py`'s module
docstring and the CSV `notes` column for per-entry caveats (e.g. one
calretinin data point is from human paraffin tissue with antigen retrieval,
not mouse frozen/free-floating sections, and is excluded from the mouse-only
"maximum" calculation).

All entries are individually verified against PubMed/PMC (PMID + DOI
included in the CSV).
