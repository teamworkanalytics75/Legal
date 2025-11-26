#!/usr/bin/env python3
"""Audit and fix OGC non-response facts in the top facts CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

DEFAULT_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v3.csv")
DEFAULT_MISSING = Path("case_law_data/missing_critical_facts.csv")
DEFAULT_OUTPUT = Path("case_law_data/top_1000_facts_ogc_fixed.csv")
DEFAULT_REPORT = Path("reports/analysis_outputs/ogc_facts_audit_report.md")

OGC_KEYWORDS = [r"OGC", r"Office of General Counsel", r"General Counsel"]
NON_RESPONSE_PHRASES = [
    r"did not respond",
    r"no response",
    r"no reply",
    r"failed to respond",
    r"never responded",
    r"did not acknowledge",
    r"no acknowledgement",
    r"no acknowledgment",
    r"silence",
]
DATE_PATTERN = re.compile(
    r"(\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and fix OGC non-response facts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path.")
    parser.add_argument("--missing", type=Path, default=DEFAULT_MISSING, help="Missing facts CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Audit markdown report path.")
    parser.add_argument("--ogc-min", type=int, default=8, help="Minimum desired number of OGC facts.")
    return parser.parse_args()


def contains_patterns(series: pd.Series, patterns: Sequence[str]) -> pd.Series:
    regex = "|".join(patterns)
    return series.fillna("").str.contains(regex, case=False, regex=True)


def extract_date(*texts: str) -> Optional[str]:
    for text in texts:
        if not text:
            continue
        match = DATE_PATTERN.search(text)
        if match:
            return match.group(1)
    return None


def normalize_prop(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return cleaned


def apply_ogc_fix(row: pd.Series) -> pd.Series:
    row = row.copy()
    row["TruthStatus"] = "True"
    if not str(row.get("EventType", "")).strip():
        row["EventType"] = "Communication"
    row["ActorRole"] = "Harvard"
    row["Speaker"] = "Harvard Office of General Counsel"
    if str(row.get("SafetyRisk", "")).strip().lower() in {"", "none"}:
        row["SafetyRisk"] = "medium"
    if str(row.get("PublicExposure", "")).strip().lower() in {"", "not_public"}:
        row["PublicExposure"] = "already_public"
    row["EvidenceType"] = "Email"
    date = extract_date(
        str(row.get("EventDate", "")),
        str(row.get("PropositionClean", "")),
        str(row.get("Proposition", "")),
        str(row.get("SourceDocument", "")),
        str(row.get("SourceExcerpt", "")),
    )
    if date:
        row["EventDate"] = date
    existing_flag = str(row.get("ClassificationFixed_v2", "")).strip()
    note = "OGC_FIX"
    if note not in existing_flag.split(";"):
        row["ClassificationFixed_v2"] = note if not existing_flag else f"{existing_flag};{note}"
    return row


def select_missing_ogc_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    text_cols = ["Proposition", "SourceDocument", "SourceExcerpt", "Notes"]
    ogc_mask = pd.Series(False, index=df.index)
    non_resp_mask = pd.Series(False, index=df.index)
    for col in text_cols:
        ogc_mask |= contains_patterns(df[col], OGC_KEYWORDS)
        non_resp_mask |= contains_patterns(df[col], NON_RESPONSE_PHRASES)
    return df[ogc_mask & non_resp_mask].copy()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    text_cols = ["PropositionClean", "Proposition", "SourceDocument", "SourceExcerpt"]
    ogc_mask = pd.Series(False, index=df.index)
    non_resp_mask = pd.Series(False, index=df.index)
    for col in text_cols:
        ogc_mask |= contains_patterns(df[col], OGC_KEYWORDS)
        non_resp_mask |= contains_patterns(df[col], NON_RESPONSE_PHRASES)

    ogc_indices = df.index[ogc_mask & non_resp_mask].tolist()
    original_ogc_count = len(ogc_indices)

    for idx in ogc_indices:
        df.loc[idx] = apply_ogc_fix(df.loc[idx])

    sample_df = df.loc[ogc_indices, ["FactID", "Proposition", "EventDate", "TruthStatus", "SafetyRisk", "PublicExposure"]].copy()

    added_count = 0
    added_fact_ids: list[str] = []
    needed = max(0, args.ogc_min - original_ogc_count)
    if needed > 0 and args.missing.exists():
        missing_df = pd.read_csv(args.missing)
        candidates = select_missing_ogc_rows(missing_df)
        existing_props = set(df["Proposition"].map(normalize_prop))
        new_rows = []
        for _, row in candidates.iterrows():
            if len(new_rows) >= needed:
                break
            if normalize_prop(row.get("Proposition", "")) in existing_props:
                continue
            fixed_row = apply_ogc_fix(row)
            fixed_row["PropositionClean"] = row.get("Proposition")
            if not str(fixed_row.get("FactID", "")).strip():
                fixed_row["FactID"] = f"MISSING_OGC_{len(new_rows)+1:03d}"
            new_rows.append(fixed_row)
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            added_count = len(new_rows)
            added_fact_ids = new_df["FactID"].astype(str).tolist()

    df = df.sort_values(by="causal_salience_score", ascending=False).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    report_lines = [
        "## OGC Facts Audit Report\n",
        f"- Total rows after processing: **{len(df)}**",
        f"- OGC non-response facts found: **{original_ogc_count}**",
        f"- OGC facts fixed: **{original_ogc_count}**",
        f"- Missing OGC facts added: **{added_count}**",
    ]
    if not sample_df.empty:
        report_lines.append("\n### Sample Fixed OGC Facts\n")
        report_lines.append(sample_df.head(5).to_string(index=False))
        report_lines.append("\n")
    if added_fact_ids:
        report_lines.append("### Added OGC FactIDs\n")
        report_lines.append(", ".join(added_fact_ids))
        report_lines.append("\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as rep:
        rep.write("\n".join(report_lines))

    print(f"OGC facts found: {original_ogc_count}")
    print(f"Missing OGC facts added: {added_count}")
    print(f"Updated CSV written to {args.output}")
    print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
