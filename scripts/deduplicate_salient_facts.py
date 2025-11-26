#!/usr/bin/env python3
"""
Deduplicate and clean the salient facts table.

Usage:
    python scripts/deduplicate_salient_facts.py \
        --input case_law_data/facts_truth_table_salient.csv \
        --output case_law_data/facts_truth_table_salient_deduplicated.csv \
        --report reports/analysis_outputs/deduplication_report.md
"""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List

import pandas as pd

DEFAULT_INPUT = Path("case_law_data/facts_truth_table_salient.csv")
DEFAULT_OUTPUT = Path("case_law_data/facts_truth_table_salient_deduplicated.csv")
DEFAULT_REPORT = Path("reports/analysis_outputs/deduplication_report.md")
LEGAL_KEYWORDS = {
    "defamation",
    "harassment",
    "retaliation",
    "privacy",
    "seal",
    "torture",
    "persecution",
    "wechat",
    "statement",
    "ogc",
    "zhihu",
}
SIGNATURE_MARKERS = (
    "sincerely",
    "best regards",
    "yours truly",
    "respectfully submitted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate salient facts file.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to salient facts CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write deduplicated CSV.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Path to write markdown report.")
    parser.add_argument("--similarity", type=float, default=0.85, help="Similarity ratio threshold.")
    return parser.parse_args()


def normalize_proposition(text: str) -> str:
    """Lowercase, remove punctuation, normalize whitespace for dedup comparisons."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def clean_proposition(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    lowered = cleaned.lower()

    # Remove boilerplate prefixes
    if lowered.startswith("the document refers to"):
        cleaned = cleaned.split(":", 1)[-1].strip()

    # Normalize email headers
    cleaned = re.sub(r"From:\s*([^<\n]+).*?To:\s*([^\n]+)", r"Email from \1 to \2", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.replace("↔", "↔ ")

    # Remove signature blocks
    if any(marker in lowered for marker in SIGNATURE_MARKERS):
        if len(cleaned) < 160:
            return ""

    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_fragment(text: str) -> bool:
    lowered = text.lower()
    if len(text) < 30 and not any(keyword in lowered for keyword in LEGAL_KEYWORDS):
        if not re.search(r"\b(is|was|were|has|have|sent|filed|warned|wrote|notified|ordered|published)\b", lowered):
            return True
    return False


def fuzzy_merge(records: List[dict], similarity_threshold: float) -> List[dict]:
    sorted_rows = sorted(
        records,
        key=lambda r: (
            float(r.get("_sort_score", 0) or 0),
            -float(r.get("ConfidenceTier", 0) or 0),
        ),
        reverse=True,
    )
    kept: List[dict] = []
    used = set()
    for i, row in enumerate(sorted_rows):
        if i in used:
            continue
        normalized = row["normalized_prop"]
        group_indices = [i]
        for j in range(i + 1, len(sorted_rows)):
            if j in used:
                continue
            other_norm = sorted_rows[j]["normalized_prop"]
            if not normalized or not other_norm:
                continue
            ratio = SequenceMatcher(None, normalized, other_norm).ratio()
            if ratio >= similarity_threshold:
                group_indices.append(j)
                used.add(j)
        kept.append(row)
        used.update(group_indices[1:])
    return kept


def deduplicate_dataframe(df: pd.DataFrame, similarity_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["Proposition"] = df["Proposition"].apply(clean_proposition)
    df = df[df["Proposition"].astype(bool)]
    df = df[~df["Proposition"].apply(is_fragment)]
    df["normalized_prop"] = df["Proposition"].apply(normalize_proposition)
    score_col = next(
        (c for c in ("CausalSalienceScore", "SalienceScore", "ImportanceScore", "ExtractionConfidence") if c in df.columns),
        None,
    )
    if score_col:
        df["_sort_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    else:
        df["_sort_score"] = 0.0

    df = df.sort_values(["normalized_prop", "_sort_score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["normalized_prop"], keep="first")

    deduped_rows = fuzzy_merge(df.to_dict("records"), similarity_threshold=similarity_threshold)
    result = pd.DataFrame(deduped_rows)
    return result.drop(columns=["normalized_prop", "_sort_score"], errors="ignore")


def write_report(path: Path, initial_count: int, final_count: int, dropped: int, fragment_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("## Salient Facts Deduplication Report\n\n")
        handle.write(f"- Total rows before cleanup: **{initial_count}**\n")
        handle.write(f"- Removed duplicates/fragments: **{dropped}**\n")
        handle.write(f"- Removed fragment rows: **{fragment_count}**\n")
        handle.write(f"- Rows after cleanup: **{final_count}**\n")
        reduction = (dropped / initial_count * 100) if initial_count else 0
        handle.write(f"- Reduction: **{reduction:.1f}%**\n")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    initial_count = len(df)
    fragments = df["Proposition"].apply(lambda text: is_fragment(str(text) if isinstance(text, str) else "")).sum()
    deduped = deduplicate_dataframe(df, args.similarity)
    final_count = len(deduped)
    dropped = initial_count - final_count

    args.output.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(args.output, index=False)
    write_report(args.report, initial_count, final_count, dropped, fragments)
    print(f"Deduplicated facts written to {args.output}")
    print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
