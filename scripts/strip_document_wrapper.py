#!/usr/bin/env python3
"""
Strip "The document refers to ..." wrapper from propositions and add
PropositionClean column with summary stats.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd

PREFIX = "the document refers to"


def strip_wrapper(value: str) -> tuple[str, bool]:
    if not isinstance(value, str):
        return "", False
    lowered = value.lower()
    if lowered.startswith(PREFIX):
        cleaned = value[len(PREFIX):]
        cleaned = cleaned.lstrip(" :,-")
        return cleaned.strip(), True
    return value.strip(), False


def format_sample(before: str, after: str) -> str:
    return textwrap.dedent(
        f"""\
        - before: {before[:200]}
          after: {after[:200] or '[empty]'}"""
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip 'The document refers to' wrapper text.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("case_law_data/top_1000_facts_no_kg_blobs.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("case_law_data/top_1000_facts_with_clean_propositions.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/analysis_outputs/wrapper_stripping_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    if "Proposition" not in df.columns:
        raise SystemExit("Input CSV missing 'Proposition' column.")

    before_lengths = df["Proposition"].fillna("").astype(str).str.len()

    clean_values: list[str] = []
    stripped_flags: list[bool] = []
    for prop in df["Proposition"]:
        value = "" if pd.isna(prop) else str(prop)
        cleaned, stripped = strip_wrapper(value)
        clean_values.append(cleaned)
        stripped_flags.append(stripped)
    df.insert(df.columns.get_loc("Proposition") + 1, "PropositionClean", clean_values)
    stripped_mask = pd.Series(stripped_flags)

    total = len(df)
    stripped_count = int(stripped_mask.sum())
    percent = (stripped_count / total * 100) if total else 0.0
    after_lengths = pd.Series(clean_values).str.len()
    reductions = (before_lengths - after_lengths).where(stripped_mask)
    avg_reduction = float(reductions.dropna().mean() or 0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    sample_pairs = []
    if stripped_count:
        matches = df[stripped_mask].head(5)
        for _, row in matches.iterrows():
            before = str(row["Proposition"])
            after = str(row["PropositionClean"])
            sample_pairs.append(format_sample(before, after))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Wrapper Stripping Report",
        "",
        f"- Input: `{args.input}`",
        f"- Output: `{args.output}`",
        "",
        f"Total rows: {total}",
        f"Wrappers stripped: {stripped_count} ({percent:.1f}%)",
        f"Average length reduction (stripped rows): {avg_reduction:.1f} characters",
        "",
        "## Samples",
    ]
    if sample_pairs:
        report_lines.extend(sample_pairs)
    else:
        report_lines.append("- No wrappers detected.")

    args.report.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrappers stripped: {stripped_count}/{total} ({percent:.1f}%)")
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
