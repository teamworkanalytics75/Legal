#!/usr/bin/env python3
"""
Clean proposition text for top facts:
- remove duplicated phrases / OCR artifacts
- strip signature blocks and document headers
- create PropositionClean_v2 column
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v3.csv")
OGC_FIXED = Path("case_law_data/top_1000_facts_ogc_fixed.csv")
OUTPUT_PATH = Path("case_law_data/top_1000_facts_text_cleaned.csv")
REPORT_PATH = Path("reports/analysis_outputs/text_cleaning_report.md")


@dataclass
class CleanResult:
    text: str
    score: int
    reasons: List[str]


DUPLICATE_PATTERNS: List[Tuple[str, str, str]] = [
    (
        r"(in the context of safety risks)(?:[, ]+\1)+",
        r"\1",
        "duplicate_safety_risk",
    ),
    (
        r"(you can paste in the)\s+\1",
        r"\1",
        "duplicate_phrase",
    ),
]


def _replace_ocr_artifacts(text: str, reasons: List[str]) -> str:
    replacements = {
        "##ation": "defamation",
        "The materials mention of it. this statement": "This statement was uploaded",
    }
    for needle, repl in replacements.items():
        if needle in text:
            text = text.replace(needle, repl)
            reasons.append("ocr_artifact")
    return text


def _remove_signature_blocks(text: str, reasons: List[str]) -> str:
    patterns = [
        r"\(Signature of Plaintiff\)[^\n]*",
        r"Flat C, 25/F, Tower 7[^\n]*",
        r"\+\d[\d\-\s\(\)]{6,}",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"WEIQI\s+ZHANG[^,]+STATEMENT OF CLAIM",
        r"Defendants STATEMENT OF CLAIM",
    ]
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            reasons.append("signature_block")
    return text


def _normalize_materials_phrase(text: str, reasons: List[str]) -> str:
    if re.search(r"^The materials mention", text, flags=re.IGNORECASE):
        text = re.sub(
            r"^The materials mention(?: of)?",
            "The materials state",
            text,
            flags=re.IGNORECASE,
        )
        reasons.append("materials_phrase")
    return text


def _clean_text(text: str) -> CleanResult:
    original = text or ""
    working = original
    score = 0
    reasons: List[str] = []

    for pattern, repl, tag in DUPLICATE_PATTERNS:
        if re.search(pattern, working, flags=re.IGNORECASE):
            working = re.sub(pattern, repl, working, flags=re.IGNORECASE)
            score += 1
            reasons.append(tag)

    before = working
    working = _replace_ocr_artifacts(working, reasons)
    if working != before:
        score += 1

    before = working
    working = _remove_signature_blocks(working, reasons)
    if working != before:
        score += 1

    before = working
    working = _normalize_materials_phrase(working, reasons)
    if working != before:
        score += 1

    working = re.sub(r"\s{2,}", " ", working)
    working = working.strip()

    if not working:
        working = original.strip()

    return CleanResult(text=working, score=score, reasons=reasons)


def _select_input(provided: Path | None = None) -> Path:
    if provided:
        return provided
    if OGC_FIXED.exists():
        return OGC_FIXED
    return DEFAULT_INPUT


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean proposition text.")
    parser.add_argument("--input", type=Path, help="Input CSV (defaults to v3 or ogc-fixed).")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output CSV path.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH, help="Report markdown path.")
    args = parser.parse_args()

    input_path = _select_input(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    source_col = "PropositionClean" if "PropositionClean" in df.columns else "Proposition"
    cleaned_col = "PropositionClean_v2"

    cleaned_texts: List[str] = []
    issue_scores: List[int] = []
    reason_counter: Counter[str] = Counter()
    samples: List[Tuple[str, str, List[str]]] = []

    for _, row in df.iterrows():
        base_text = str(row.get(cleaned_col) or row[source_col] or "")
        result = _clean_text(base_text)
        cleaned_texts.append(result.text)
        issue_scores.append(result.score)
        for reason in result.reasons:
            reason_counter[reason] += 1
        if result.text != base_text and len(samples) < 15:
            samples.append((base_text, result.text, result.reasons))

    df[cleaned_col] = cleaned_texts
    changed_count = sum(1 for before, after in zip(df[source_col], cleaned_texts) if str(before) != str(after))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    worst_indices = sorted(
        range(len(issue_scores)),
        key=lambda idx: issue_scores[idx],
        reverse=True,
    )[:25]

    lines = [
        "# Text Cleaning Report",
        "",
        f"- Input file: {input_path}",
        f"- Output file: {args.output}",
        f"- Facts cleaned: {changed_count}",
        "",
        "## Issue Breakdown",
    ]
    for reason, count in reason_counter.most_common():
        lines.append(f"- {reason}: {count}")
    lines.append("")
    lines.append("## Sample Before/After")
    for before, after, reasons in samples:
        lines.append(f"- **Reasons**: {', '.join(reasons) or 'general'}")
        lines.append(f"  - Before: {before}")
        lines.append(f"  - After: {after}")
    lines.append("")
    lines.append("## Worst 25 Facts (by issues)")
    for idx in worst_indices:
        if issue_scores[idx] == 0:
            continue
        lines.append(f"- FactID {df.loc[idx, 'FactID']}: score={issue_scores[idx]}, text={cleaned_texts[idx]}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Cleaned {changed_count} facts. Output: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
