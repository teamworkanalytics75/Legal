#!/usr/bin/env python3
"""
Verify and normalize SafetyRisk/PublicExposure labels for critical fact groups.

Usage:
    python scripts/verify_risk_labels.py \
        --input case_law_data/top_1000_facts_text_cleaned.csv \
        --output case_law_data/top_1000_facts_risk_verified.csv \
        --report reports/analysis_outputs/risk_labels_verification_report.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


MONKEY_TERMS = ("monkey", "hah", "猴")
RESUME_TERMS = ("resume", "résumé", "employment claims", "cv screenshot")
PLATFORM_TERMS = ("wechat", "zhihu", "sohu", "baidu", "published", "article")
ESUWIKI_TERMS = ("esuwiki", "niu tengyu", "torture", "crackdown", "14 years")
OGC_TERMS = ("ogc", "general counsel")
NON_RESPONSE_TERMS = ("did not respond", "no response", "no reply", "failed to respond", "never responded", "without acknowledgement", "did not acknowledge")
HARM_TERMS = ("torture", "persecution", "harm", "threat", "attack", "retaliation", "detention", "arrest")


@dataclass
class FixStats:
    verified_counts: Dict[str, int]
    corrections: Dict[str, int]


def gather_text(row: pd.Series) -> str:
    fields = [
        "PropositionClean_v2",
        "PropositionClean",
        "Proposition",
        "SourceExcerpt",
        "SourceDocument",
    ]
    values: List[str] = []
    for field in fields:
        value = row.get(field)
        if isinstance(value, str):
            values.append(value)
    return " ".join(values).lower()


def _ensure_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        df[column] = ""


def _log_change(row: pd.Series, column: str, new_value: str, change_log: List[str], label: str) -> None:
    old_value = row.get(column)
    if str(old_value) == new_value:
        return
    row[column] = new_value
    change_log.append(f"{label}:{column}:{old_value}->{new_value}")


def _ensure_exposure(row: pd.Series, target: str, change_log: List[str], label: str) -> None:
    current = (row.get("PublicExposure") or "").strip()
    if current.lower() != target.lower():
        _log_change(row, "PublicExposure", target, change_log, label)


def _ensure_safety(row: pd.Series, target: str, change_log: List[str], label: str) -> None:
    current = (row.get("SafetyRisk") or "").strip().lower()
    target_lower = target.lower()
    priority = {"none": 0, "low": 1, "medium": 2, "high": 3, "extreme": 4}
    if priority.get(current, 0) < priority.get(target_lower, 0):
        _log_change(row, "SafetyRisk", target, change_log, label)


def apply_verifications(df: pd.DataFrame) -> FixStats:
    verified_counts: Dict[str, int] = defaultdict(int)
    correction_counts: Dict[str, int] = defaultdict(int)

    _ensure_column(df, "ClassificationFixed_v3")

    for idx, row in df.iterrows():
        text = gather_text(row)
        changes: List[str] = []

        def record(label: str) -> None:
            verified_counts[label] += 1

        lower_text = text

        if any(term in lower_text for term in MONKEY_TERMS):
            record("monkey")
            _ensure_exposure(row, "already_public", changes, "monkey")
            _ensure_safety(row, "high", changes, "monkey")

        if any(term in lower_text for term in RESUME_TERMS):
            record("resume")
            _ensure_exposure(row, "already_public", changes, "resume")
            _ensure_safety(row, "high", changes, "resume")

        if any(term in lower_text for term in PLATFORM_TERMS):
            record("platform")
            _ensure_exposure(row, "already_public", changes, "platform")
            if any(term in lower_text for term in HARM_TERMS):
                _ensure_safety(row, "medium", changes, "platform")

        if any(term in lower_text for term in ESUWIKI_TERMS):
            record("esuwiki")
            _ensure_safety(row, "extreme", changes, "esuwiki")
            _ensure_exposure(row, "partially_public", changes, "esuwiki")

        if any(term in lower_text for term in OGC_TERMS) and any(term in lower_text for term in NON_RESPONSE_TERMS):
            record("ogc")
            _ensure_safety(row, "high", changes, "ogc")
            _ensure_exposure(row, "partially_public", changes, "ogc")

        if changes:
            existing = row.get("ClassificationFixed_v3")
            prefix = (existing + "; ") if existing else ""
            df.at[idx, "ClassificationFixed_v3"] = prefix + "; ".join(changes)
            for change in changes:
                label = change.split(":", 1)[0]
                correction_counts[label] += 1
            df.iloc[idx] = row

    return FixStats(dict(verified_counts), dict(correction_counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify risk labels for critical facts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    stats = apply_verifications(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        f.write("# Risk Labels Verification Report\n\n")
        f.write(f"- Input: `{args.input}`\n")
        f.write(f"- Output: `{args.output}`\n\n")
        f.write("## Verified Counts\n")
        for label, count in sorted(stats.verified_counts.items()):
            f.write(f"- {label}: {count}\n")
        f.write("\n## Corrections Applied\n")
        if stats.corrections:
            for label, count in sorted(stats.corrections.items()):
                f.write(f"- {label}: {count}\n")
        else:
            f.write("- None\n")


if __name__ == "__main__":
    main()
