# Research Article Extractor

A Python tool that extracts structured information from research articles
(PDF, TXT, or Markdown): title, authors, emails, abstract, keywords, DOI,
journal, publication year, section text (introduction, methods, results,
discussion, conclusion, ...), and parsed references.

It uses rule-based heuristics (regex + layout patterns common in academic
papers) — no external API or ML model required — and works fully offline.

## Install

```bash
pip install -r requirements.txt
```

## Usage

Extract a single article to stdout as JSON:

```bash
python -m research_extractor.cli path/to/article.pdf
```

Write the result to a file:

```bash
python -m research_extractor.cli path/to/article.pdf -o article.json
```

Batch-process a directory of articles (one JSON file per article, plus an
optional summary CSV):

```bash
python -m research_extractor.cli path/to/articles/ -o extracted/ --summary-csv summary.csv
```

Or use it as a library:

```python
from research_extractor import parse_article

article = parse_article("path/to/article.pdf")
print(article.title, article.authors, article.doi)
print(article.sections["methods"])
```

## Output fields

| Field               | Description                                          |
|---------------------|-------------------------------------------------------|
| `title`             | Best-guess article title                              |
| `authors`           | List of author names                                  |
| `emails`            | Email addresses found in the document                 |
| `abstract`          | Abstract text                                          |
| `keywords`          | Author-supplied keyword list                           |
| `doi`               | DOI, if present                                        |
| `journal`           | Journal/venue name, if detected                        |
| `publication_year`  | Publication year, if detected                          |
| `sections`          | Dict of section name -> section text (introduction, methods, results, discussion, conclusion, limitations, acknowledgments, ...) |
| `references`        | List of parsed reference/citation strings              |
| `word_count`        | Total word count of the document                       |

## Notes and limitations

This is a heuristic, layout-based extractor tuned for common academic paper
formatting conventions (numbered/titled section headers, a "Keywords:" line,
a DOI string, etc.). It will do best on well-formatted PDFs/text and may miss
fields on articles with unusual layouts, multi-column PDF extraction
artifacts, or non-English text. For production-grade extraction from
large, messy corpora, consider pairing this with a dedicated parsing service
(e.g. GROBID) or an LLM-based extraction pass.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
