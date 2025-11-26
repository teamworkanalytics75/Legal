#!/usr/bin/env python3
"""
Fix classification issues (PublicExposure, SafetyRisk, TruthStatus, ActorRole)
for the deduplicated salient fact set.

Usage:
    python scripts/fix_fact_classifications.py \
        --input case_law_data/facts_truth_table_salient_deduplicated.csv \
        --output case_law_data/facts_truth_table_salient_classified.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


PUBLIC_KEYWORDS = (
    "statement 1",
    "statement 2",
    "wechat",
    "zhihu",
    "published",
    "article",
    "post",
    "website",
    "official account",
    "press release",
    "media release",
)
HIGH_RISK_KEYWORDS = {
    "extreme": ("torture", "detention", "arrest", "kidnap", "execution"),
    "high": ("surveillance", "doxxing", "harassment", "persecution", "retaliation"),
    "medium": ("endangerment", "harm", "threat", "intimidation"),
}
TRUTH_TRUE_KEYWORDS = (
    "published on",
    "issued on",
    "dated",
    "exhibit",
    "schedule",
    "filed on",
    "email",
    "wechat",
    "zhihu",
    "did not respond",
    "no response",
    "no reply",
    "acknowledgment",
)
ROLE_KEYWORDS = {
    "Harvard": ("harvard club", "hcbeijing", "harvard alumni", "haa", "hcs", "harvardhk", "harvardsh"),
    "StateActor": ("prc", "people's republic of china", "chinese government", "ccp", "party-state", "state council"),
    "WeChatPublisher": ("wechat official account", "wechat article", "weibo", "zhihu"),
}


@dataclass
class ClassificationResult:
    dataframe: pd.DataFrame
    stats_before: Dict[str, Dict[str, int]] = field(default_factory=dict)
    stats_after: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _keyword_hit(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _safety_from_text(text: str) -> str | None:
    lowered = text.lower()
    for level, keywords in HIGH_RISK_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return level
    return None


def _truth_from_text(text: str) -> str | None:
    lowered = text.lower()
    if any(keyword in lowered for keyword in TRUTH_TRUE_KEYWORDS):
        return "True"
    return None


def _actor_role_from_text(text: str, current_role: str) -> str | None:
    lowered = text.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return role
    if "plaintiff" in lowered and current_role in (None, "", "Unknown", "Other"):
        return "Plaintiff"
    if "court" in lowered and current_role in (None, "", "Unknown", "Other"):
        return "Court"
    if "ogc" in lowered and current_role not in ("Harvard", "Court"):
        return "Harvard"
    return None


def _collect_stats(df: pd.DataFrame, column: str) -> Dict[str, int]:
    if column not in df.columns:
        return {}
    return df[column].fillna("Missing").value_counts().to_dict()


def apply_classification_fixes(df: pd.DataFrame) -> ClassificationResult:
    string_columns = [col for col in df.columns if df[col].dtype == object]

    def gather_text(row: pd.Series) -> str:
        values = []
        for column in string_columns:
            value = row.get(column)
            if isinstance(value, str):
                values.append(value)
        return " ".join(values)

    stats_before = {
        "PublicExposure": _collect_stats(df, "PublicExposure"),
        "SafetyRisk": _collect_stats(df, "SafetyRisk"),
        "TruthStatus": _collect_stats(df, "TruthStatus"),
        "ActorRole": _collect_stats(df, "ActorRole"),
    }

    fixed_notes: List[List[Tuple[str, str, str]]] = [[] for _ in range(len(df))]

    for idx, row in df.iterrows():
        text = gather_text(row)
        notes: List[Tuple[str, str, str]] = []

        exposure = row.get("PublicExposure")
        if _keyword_hit(text, PUBLIC_KEYWORDS):
            new_exposure = "already_public" if "published" in text.lower() or "wechat" in text.lower() else "partially_public"
            if exposure != new_exposure:
                df.at[idx, "PublicExposure"] = new_exposure
                notes.append(("PublicExposure", exposure, new_exposure))

        safety = row.get("SafetyRisk")
        inferred_safety = _safety_from_text(text)
        if inferred_safety and inferred_safety != safety:
            df.at[idx, "SafetyRisk"] = inferred_safety
            notes.append(("SafetyRisk", safety, inferred_safety))

        truth = row.get("TruthStatus")
        inferred_truth = _truth_from_text(text)
        if inferred_truth and truth != inferred_truth:
            df.at[idx, "TruthStatus"] = inferred_truth
            notes.append(("TruthStatus", truth, inferred_truth))

        actor_role = row.get("ActorRole")
        inferred_role = _actor_role_from_text(text, actor_role)
        if inferred_role and inferred_role != actor_role:
            df.at[idx, "ActorRole"] = inferred_role
            notes.append(("ActorRole", actor_role, inferred_role))

        fixed_notes[idx] = notes

    df["ClassificationFixed"] = [
        "; ".join(f"{column}:{old}->{new}" for column, old, new in notes) if notes else ""
        for notes in fixed_notes
    ]

    stats_after = {
        "PublicExposure": _collect_stats(df, "PublicExposure"),
        "SafetyRisk": _collect_stats(df, "SafetyRisk"),
        "TruthStatus": _collect_stats(df, "TruthStatus"),
        "ActorRole": _collect_stats(df, "ActorRole"),
    }

    return ClassificationResult(df, stats_before, stats_after)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix fact classifications in the salient dataset.")
    parser.add_argument("--input", type=Path, required=True, help="Path to deduplicated salient CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the classified CSV.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    if df.empty:
        raise SystemExit("Input CSV is empty.")

    result = apply_classification_fixes(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)

    print("Classification fixes applied.")
    for column, stats in result.stats_before.items():
        before = stats
        after = result.stats_after.get(column, {})
        print(f"{column}: before={before} after={after}")


if __name__ == "__main__":
    main(sys.argv[1:])
