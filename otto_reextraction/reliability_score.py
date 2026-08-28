#!/usr/bin/env python3
"""Compute a 0-10 reliability score for Otto's extraction from verification.csv.

The score is a confidence-weighted average of per-field verdicts, with
small-sample shrinkage toward a neutral prior so a handful of audited
fields can't produce a falsely precise-looking headline number.

Design (see README for the full writeup):
  1. Each verdict contributes points: match=1.0, partial=0.5, mismatch=0.0.
     'unverifiable' rows are excluded entirely - they say nothing about
     whether Otto was right or wrong, so they shouldn't move the score
     either direction.
  2. Each verdict is weighted by how confident the audit itself was:
     high=1.0, medium=0.65, low=0.35. A 'high confidence mismatch' should
     hurt more than a 'low confidence mismatch' we're not sure about, and
     symmetrically for matches.
  3. Because the whole corpus is 71 papers x 6 fields = 426 possible
     verdicts, and only a small number will typically be audited (this
     pipeline has no automatic way to audit all of them without an
     Anthropic API key), the raw weighted average is pulled toward a
     neutral prior (score 0.5, i.e. 5/10) by PRIOR_STRENGTH pseudo-verdicts.
     This is the same idea as a Bayesian/IMDB-style weighted rating: with
     few real observations the estimate stays cautious; as more audits
     accumulate, the prior's influence shrinks toward irrelevance.

Both the raw (unshrunk) and shrunk scores are reported so nothing is
hidden - shrinkage is a stated methodological choice, not a way to make
an inconvenient number disappear.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import common

VERDICT_POINTS = {"match": 1.0, "partial": 0.5, "mismatch": 0.0}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.65, "low": 0.35}
PRIOR_STRENGTH = 5.0  # pseudo-verdicts' worth of weight pulling toward 0.5
PRIOR_SCORE = 0.5


def load_verification_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def compute_reliability(rows: list[dict]) -> dict:
    """Compute the reliability score and its breakdown from verification rows.

    Rows with verdict == 'unverifiable' are counted (reported) but excluded
    from the score itself. Rows with an unrecognized verdict/confidence are
    skipped defensively rather than raising, since this reads external CSV
    data that could in principle contain anything.
    """
    verdict_counts = {"match": 0, "partial": 0, "mismatch": 0, "unverifiable": 0}
    field_counts: dict[str, dict[str, int]] = {}
    modality_counts: dict[str, dict[str, int]] = {}
    paper_ids: set[str] = set()

    weighted_points = 0.0
    weighted_weight = 0.0
    scored_n = 0

    for row in rows:
        verdict = row.get("verdict", "")
        confidence = row.get("confidence", "")
        paper_ids.add(row.get("paper_id", ""))

        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

        field = row.get("otto_field", "unknown")
        field_counts.setdefault(field, {"match": 0, "partial": 0, "mismatch": 0, "unverifiable": 0})
        if verdict in field_counts[field]:
            field_counts[field][verdict] += 1

        modality = row.get("modality", "unknown")
        modality_counts.setdefault(modality, {"match": 0, "partial": 0, "mismatch": 0, "unverifiable": 0})
        if verdict in modality_counts[modality]:
            modality_counts[modality][verdict] += 1

        if verdict == "unverifiable" or verdict not in VERDICT_POINTS:
            continue
        weight = CONFIDENCE_WEIGHT.get(confidence)
        if weight is None:
            continue

        weighted_points += VERDICT_POINTS[verdict] * weight
        weighted_weight += weight
        scored_n += 1

    raw_fraction = (weighted_points / weighted_weight) if weighted_weight > 0 else None
    shrunk_fraction = (weighted_points + PRIOR_STRENGTH * PRIOR_SCORE) / (weighted_weight + PRIOR_STRENGTH)

    return {
        "n_field_verdicts": len(rows),
        "n_scored_verdicts": scored_n,
        "n_papers_audited": len({p for p in paper_ids if p}),
        "verdict_counts": verdict_counts,
        "field_counts": field_counts,
        "modality_counts": modality_counts,
        "raw_score_10": round(raw_fraction * 10, 1) if raw_fraction is not None else None,
        "shrunk_score_10": round(shrunk_fraction * 10, 1),
        "weighted_points": weighted_points,
        "weighted_weight": weighted_weight,
    }


def format_report(result: dict, total_papers: int) -> str:
    lines = []
    n_audited = result["n_papers_audited"]
    lines.append(
        f"Otto reliability score: {result['shrunk_score_10']}/10  "
        f"(raw, unshrunk: {result['raw_score_10']}/10)"
    )
    lines.append(
        f"Based on {result['n_scored_verdicts']} scored field-verdicts "
        f"across {n_audited}/{total_papers} papers audited "
        f"({result['n_field_verdicts']} total verdicts including unverifiable)."
    )
    if n_audited < total_papers * 0.2:
        lines.append(
            f"CAUTION: only {n_audited}/{total_papers} papers have been audited so far - "
            "this score is a provisional estimate, not a corpus-wide reliability figure. "
            "It will change, possibly a lot, as more papers are audited."
        )
    lines.append("")
    lines.append("Verdict breakdown: " + ", ".join(f"{k}={v}" for k, v in result["verdict_counts"].items()))
    lines.append("")
    lines.append("By Otto field:")
    for field, counts in sorted(result["field_counts"].items()):
        total = sum(counts.values())
        if total == 0:
            continue
        lines.append(f"  {field}: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    rows = load_verification_rows(Path(args.verification_csv))
    if not rows:
        print("No verification data yet - run verify.py (or manually populate data/verification.csv) first.", file=sys.stderr)
        return 1

    total_papers = len(common.load_otto_rows(args.otto_csv))
    result = compute_reliability(rows)
    print(format_report(result, total_papers))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-csv", default=str(common.DEFAULT_VERIFICATION_CSV))
    parser.add_argument("--otto-csv", default=str(common.DEFAULT_OTTO_CSV))
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
