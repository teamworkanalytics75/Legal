#!/usr/bin/env python3
"""Regenerate the top-1000 facts CSV with deduped, enriched data."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

SALIENT_PATH = Path("case_law_data/facts_truth_table_salient.csv")
TOP_PATH = Path("case_law_data/top_1000_facts_for_chatgpt.csv")
MAX_ROWS = 1000

PUBLIC_KEYWORDS = [
    "statement",
    "press release",
    "published",
    "wechat",
    "website",
    "blog",
    "public",
    "article",
    "media",
    "news",
    "zhihu",
]

RISK_HIGH = [
    "torture",
    "detention",
    "detained",
    "arrest",
    "arrested",
    "abduction",
    "kidnap",
    "threatened",
    "surveillance",
    "persecution",
    "physical harm",
    "danger",
    "risk of death",
    "security threat",
]

RISK_MEDIUM = [
    "intimidation",
    "harassment",
    "threat",
    "fear",
    "danger",
    "risk",
]


def sanitize(text: str | float | None) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return str(text).strip()


def derive_subject(
    proposition: str,
    speaker: str,
    actor_role: str,
    source_document: str,
) -> str:
    speaker = sanitize(speaker)
    if speaker and speaker.lower() not in {"unknown", "n/a", "unspecified"}:
        return speaker

    prop = sanitize(proposition)
    from_match = re.search(r"from:\s*([^<\\|\\n]+)", prop, re.IGNORECASE)
    if from_match:
        name = from_match.group(1).strip().strip('"')
        if name:
            return name

    if "harvard" in prop.lower():
        return "Harvard"

    actor = sanitize(actor_role)
    if actor and actor.lower() not in {"", "unknown"}:
        return actor

    source = sanitize(source_document)
    if source:
        # Use filename-ish token for traceability.
        return source.split("/")[-1].split("\\")[-1]

    return "Unknown"


def fix_public_exposure(value: str, proposition: str) -> str:
    value = sanitize(value) or "unknown"
    prop = proposition.lower()
    if value != "not_public":
        return value
    if any(keyword in prop for keyword in PUBLIC_KEYWORDS):
        return "public"
    return value


def fix_safety_risk(value: str, proposition: str) -> str:
    value = sanitize(value) or "none"
    prop = proposition.lower()
    if any(keyword in prop for keyword in RISK_HIGH):
        return "high"
    if value in {"none", ""} and any(keyword in prop for keyword in RISK_MEDIUM):
        return "medium"
    return value


def select_top_unique(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__sort_score"] = df["confidence_tier"].fillna(3)
    df = df.sort_values(
        by=["__sort_score", "causal_salience_score"],
        ascending=[True, False],
    )
    df = df.drop_duplicates(subset=["Proposition"], keep="first")
    return df.head(MAX_ROWS).drop(columns="__sort_score")


def regenerate_top_csv() -> None:
    if not SALIENT_PATH.exists():
        raise FileNotFoundError(f"Missing {SALIENT_PATH}")

    df = pd.read_csv(SALIENT_PATH)
    top = select_top_unique(df)

    top["Subject"] = [
        derive_subject(prop, speaker, actor, src)
        for prop, speaker, actor, src in zip(
            top["Proposition"],
            top["Speaker"],
            top["ActorRole"],
            top["SourceDocument"],
        )
    ]
    top["PublicExposure"] = [
        fix_public_exposure(value, proposition)
        for value, proposition in zip(top["PublicExposure"], top["Proposition"])
    ]
    top["SafetyRisk"] = [
        fix_safety_risk(value, proposition)
        for value, proposition in zip(top["SafetyRisk"], top["Proposition"])
    ]

    columns = [
        "Proposition",
        "FactID",
        "Subject",
        "Verb",
        "Object",
        "EventType",
        "EventDate",
        "EventLocation",
        "ActorRole",
        "Speaker",
        "TruthStatus",
        "EvidenceType",
        "SourceDocument",
        "SourceExcerpt",
        "SafetyRisk",
        "PublicExposure",
        "RiskRationale",
        "confidence_tier",
        "causal_salience_score",
        "confidence_reason",
        "causal_salience_reason",
    ]
    missing_cols = [col for col in columns if col not in top.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in salient dataset: {missing_cols}")

    TOP_PATH.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(TOP_PATH, index=False, columns=columns)


if __name__ == "__main__":
    regenerate_top_csv()
