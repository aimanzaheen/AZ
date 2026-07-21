"""Command-line interface for extracting information from research articles."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from research_extractor.parser import parse_article
from research_extractor.text_extraction import SUPPORTED_SUFFIXES


def _iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
        )
    raise FileNotFoundError(f"No such file or directory: {input_path}")


def _write_summary_csv(articles: list[dict], csv_path: Path) -> None:
    fields = [
        "source_file",
        "title",
        "authors",
        "doi",
        "journal",
        "publication_year",
        "word_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for article in articles:
            row = {k: article.get(k) for k in fields}
            row["authors"] = "; ".join(article.get("authors") or [])
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-extractor",
        description="Extract structured information (title, authors, abstract, "
        "sections, references, ...) from research article files.",
    )
    parser.add_argument(
        "input",
        help="Path to a research article file (.pdf/.txt/.md) or a directory of them.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write JSON output to this file (single input) or directory "
        "(batch input). Defaults to stdout for a single file.",
    )
    parser.add_argument(
        "--summary-csv",
        help="When processing a directory, also write a summary CSV to this path.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: 2).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)

    try:
        files = _iter_input_files(input_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not files:
        print(f"error: no supported article files found under {input_path}", file=sys.stderr)
        return 1

    is_batch = len(files) > 1 or input_path.is_dir()
    results = []
    exit_code = 0

    for file_path in files:
        try:
            article = parse_article(file_path)
        except Exception as exc:  # noqa: BLE001 - report and continue batch runs
            print(f"error processing {file_path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        results.append(article.to_dict())

    if not results:
        return exit_code

    if is_batch:
        out_dir = Path(args.output) if args.output else Path("extracted")
        out_dir.mkdir(parents=True, exist_ok=True)
        for article in results:
            stem = Path(article["source_file"]).stem
            out_path = out_dir / f"{stem}.json"
            out_path.write_text(json.dumps(article, indent=args.indent), encoding="utf-8")
        print(f"Wrote {len(results)} JSON file(s) to {out_dir}")
        if args.summary_csv:
            _write_summary_csv(results, Path(args.summary_csv))
            print(f"Wrote summary CSV to {args.summary_csv}")
    else:
        payload = json.dumps(results[0], indent=args.indent)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
            print(f"Wrote JSON output to {args.output}")
        else:
            print(payload)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
