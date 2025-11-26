#!/usr/bin/env python3
"""
Compare two truth tables and summarize what changed.

Usage:
    python scripts/compare_truth_tables.py \
        --old case_law_data/facts_truth_table.csv \
        --new case_law_data/facts_truth_table_v2.csv \
        --out removed_rows.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "proposition" not in df.columns:
        for alt in ("Proposition", "PROPOSITION"):
            if alt in df.columns:
                df["proposition"] = df[alt]
                break
        else:
            raise ValueError(f"{path} missing 'proposition' column.")
    df["proposition_norm"] = df["proposition"].astype(str).str.strip().str.lower()
    if "fact_id" not in df.columns and "FactID" in df.columns:
        df["fact_id"] = df["FactID"]
    df["fact_id"] = df.get("fact_id", range(1, len(df) + 1)).astype(str)
    return df


def compare_tables(old_df: pd.DataFrame, new_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    old_norm = set(old_df["proposition_norm"])
    new_norm = set(new_df["proposition_norm"])

    removed_norm = old_norm - new_norm
    added_norm = new_norm - old_norm

    removed_rows = old_df[old_df["proposition_norm"].isin(removed_norm)].copy()
    removed_rows["status"] = "removed"
    added_rows = new_df[new_df["proposition_norm"].isin(added_norm)].copy()
    added_rows["status"] = "added"

    summary = {
        "old_rows": len(old_df),
        "new_rows": len(new_df),
        "removed_rows": len(removed_rows),
        "added_rows": len(added_rows),
        "retained_rows": len(new_df) - len(added_rows),
        "retention_pct": (len(new_df) / len(old_df) * 100) if len(old_df) else 0,
    }

    return pd.concat([removed_rows, added_rows], ignore_index=True), summary


def bucket_counts(df: pd.DataFrame, column: str, max_rows: int = 10) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    counts = (
        df[column]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": column})
    )
    return counts.head(max_rows)


def format_counts(df: pd.DataFrame, column: str) -> str:
    if df.empty:
        return "None"
    lines = [f"- {row[column]}: {row['count']}" for _, row in df.iterrows()]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two truth tables and report differences.")
    parser.add_argument("--old", type=Path, required=True, help="Path to legacy truth table CSV.")
    parser.add_argument("--new", type=Path, required=True, help="Path to new truth table CSV.")
    parser.add_argument("--out", type=Path, default=Path("case_law_data/truth_table_diff.csv"))
    parser.add_argument("--report", type=Path, default=Path("case_law_data/truth_table_diff.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_df = load_table(args.old)
    new_df = load_table(args.new)

    diff_df, summary = compare_tables(old_df, new_df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    diff_df.to_csv(args.out, index=False)

    removed_df = diff_df[diff_df["status"] == "removed"]
    added_df = diff_df[diff_df["status"] == "added"]

    lines = [
        f"# Truth Table Comparison",
        f"- Old rows: {summary['old_rows']}",
        f"- New rows: {summary['new_rows']}",
        f"- Retention: {summary['retained_rows']} ({summary['retention_pct']:.1f}%)",
        f"- Removed rows: {summary['removed_rows']}",
        f"- Added rows: {summary['added_rows']}",
        "",
        "## Removed rows by evidence_type",
        format_counts(bucket_counts(removed_df, "evidence_type"), "evidence_type"),
        "",
        "## Removed rows by speaker",
        format_counts(bucket_counts(removed_df, "speaker"), "speaker"),
        "",
        "## Removed rows containing Xi/WeChat/OGC",
    ]

    keywords = ["xi", "wechat", "ogc"]
    for keyword in keywords:
        count = removed_df["proposition"].str.contains(keyword, case=False, na=False).sum()
        lines.append(f"- {keyword}: {count}")

    lines += [
        "",
        "## Added rows (top evidence types)",
        format_counts(bucket_counts(added_df, "evidence_type"), "evidence_type"),
    ]

    args.report.write_text("\n".join(lines))
    print(f"Wrote diff CSV to {args.out}")
    print(f"Wrote summary report to {args.report}")


if __name__ == "__main__":
    main()
