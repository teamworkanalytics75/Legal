#!/usr/bin/env python3
"""
Remove KG export blob rows from the top facts CSV and emit a clean copy plus audit report.

Usage:
    python scripts/remove_kg_blobs.py \
        --input case_law_data/top_1000_facts_for_chatgpt_v2.csv \
        --clean case_law_data/top_1000_facts_no_kg_blobs.csv \
        --removed case_law_data/kg_edges_export_removed.csv \
        --report reports/analysis_outputs/kg_blobs_removal_report.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v2.csv")
DEFAULT_CLEAN = Path("case_law_data/top_1000_facts_no_kg_blobs.csv")
DEFAULT_REMOVED = Path("case_law_data/kg_edges_export_removed.csv")
DEFAULT_REPORT = Path("reports/analysis_outputs/kg_blobs_removal_report.md")

PATTERNS = [
    re.compile(r"Source,Target,Relation", re.IGNORECASE),
    re.compile(r"HUB_|ART_|ENT_|PLA_", re.IGNORECASE),
    re.compile(r",Directed,|,contains,", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove KG export blob rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN, help="Output CSV without KG blobs.")
    parser.add_argument("--removed", type=Path, default=DEFAULT_REMOVED, help="CSV storing removed KG rows.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Markdown report path.")
    parser.add_argument("--length-threshold", type=int, default=2000, help="Length threshold for blob detection.")
    return parser.parse_args()


def is_kg_blob(row: pd.Series, length_threshold: int) -> bool:
    fact_id = str(row.get("FactID", ""))
    proposition = str(row.get("Proposition", ""))
    if fact_id.startswith("MISSING_"):
        return True
    if len(proposition) > length_threshold and proposition.count(",") > 10:
        return True
    for pattern in PATTERNS:
        if pattern.search(proposition):
            return True
    return False


def write_report(report_path: Path, removed_df: pd.DataFrame, final_count: int, original_count: int) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    removed_count = len(removed_df)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("## KG Blob Removal Report\n\n")
        handle.write(f"- Original rows: **{original_count}**\n")
        handle.write(f"- Rows removed (KG blobs): **{removed_count}**\n")
        handle.write(f"- Final rows after removal: **{final_count}**\n")
        if removed_count:
            percent = removed_count / original_count * 100 if original_count else 0
            handle.write(f"- Percentage removed: **{percent:.2f}%**\n\n")
            handle.write("### Removed FactIDs\n")
            handle.write(", ".join(map(str, removed_df["FactID"].tolist())) + "\n\n")
            handle.write("### Sample Removed Rows\n")
            sample = removed_df.head(3)
            handle.write(sample[["FactID", "Proposition"]].to_string(index=False))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    original_count = len(df)
    df["is_kg_blob"] = df.apply(lambda row: is_kg_blob(row, args.length_threshold), axis=1)

    removed_df = df[df["is_kg_blob"]].drop(columns=["is_kg_blob"])
    clean_df = df[~df["is_kg_blob"]].drop(columns=["is_kg_blob"])

    args.removed.parent.mkdir(parents=True, exist_ok=True)
    args.clean.parent.mkdir(parents=True, exist_ok=True)

    removed_df.to_csv(args.removed, index=False)
    clean_df.to_csv(args.clean, index=False)

    write_report(args.report, removed_df, len(clean_df), original_count)

    print(f"Removed {len(removed_df)} KG blob rows.")
    print(f"Clean CSV written to {args.clean}")
    print(f"Removed rows stored in {args.removed}")
    print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
