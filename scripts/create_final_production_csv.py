#!/usr/bin/env python3
"""Create the final production-ready CSV for ChatGPT review."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

VERIFIED_PATH = Path("case_law_data/top_1000_facts_risk_verified.csv")
TEXT_CLEANED_PATH = Path("case_law_data/top_1000_facts_text_cleaned.csv")
OGC_FIXED_PATH = Path("case_law_data/top_1000_facts_ogc_fixed.csv")
MISSING_FACTS_PATH = Path("case_law_data/missing_critical_facts.csv")
OUTPUT_PATH = Path("case_law_data/top_1000_facts_for_chatgpt_v4_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/final_production_csv_report.md")


def load_base_dataframe() -> pd.DataFrame:
    if VERIFIED_PATH.exists():
        return pd.read_csv(VERIFIED_PATH)
    if TEXT_CLEANED_PATH.exists():
        return pd.read_csv(TEXT_CLEANED_PATH)
    raise FileNotFoundError("Missing risk-verified or text-cleaned CSV.")


def ensure_proposition_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "PropositionClean_v2" not in df.columns:
        if "PropositionClean" in df.columns:
            df.insert(0, "PropositionClean_v2", df["PropositionClean"])
        else:
            df.insert(0, "PropositionClean_v2", df["Proposition"])
    return df


def normalize_columns(df: pd.DataFrame, template_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in template_columns:
        if column not in df.columns:
            df[column] = ""
    return df[template_columns]


def merge_ogc_rows(base_df: pd.DataFrame) -> pd.DataFrame:
    if not OGC_FIXED_PATH.exists():
        return base_df

    ogc_df = pd.read_csv(OGC_FIXED_PATH)
    ogc_df = ensure_proposition_clean(ogc_df)
    template_columns = list(base_df.columns)
    ogc_df = normalize_columns(ogc_df, template_columns)

    combined = pd.concat([base_df, ogc_df], ignore_index=True)
    combined = combined.sort_values(by="causal_salience_score", ascending=False)
    if "FactID" in combined.columns:
        combined = combined.drop_duplicates(subset=["FactID", "PropositionClean_v2"], keep="first")
    else:
        combined = combined.drop_duplicates(subset=["PropositionClean_v2"], keep="first")
    return combined


def ensure_ogc_coverage(df: pd.DataFrame, target: int = 8) -> pd.DataFrame:
    df = df.copy()
    current_mask = df["PropositionClean_v2"].map(lambda text: contains_keywords(text, OGC_KEYWORDS))
    current_count = int(current_mask.sum())
    if current_count >= target or not MISSING_FACTS_PATH.exists():
        return df

    missing_df = pd.read_csv(MISSING_FACTS_PATH)
    ogc_candidates = missing_df[
        missing_df["Proposition"].str.contains("ogc|general counsel", case=False, na=False)
    ].copy()
    if ogc_candidates.empty:
        return df

    ogc_candidates["PropositionClean_v2"] = ogc_candidates["Proposition"]
    template_columns = list(df.columns)
    ogc_candidates = normalize_columns(ogc_candidates, template_columns)

    # Drop candidates already present (by FactID or text)
    if "FactID" in df.columns:
        ogc_candidates = ogc_candidates[~ogc_candidates["FactID"].isin(df["FactID"])]
    ogc_candidates = ogc_candidates[~ogc_candidates["PropositionClean_v2"].isin(df["PropositionClean_v2"])]

    needed = target - current_count
    if needed <= 0:
        return df

    supplemental = ogc_candidates.head(needed)
    if supplemental.empty:
        return df

    combined = pd.concat([df, supplemental], ignore_index=True)
    return combined


def sort_and_trim(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["causal_salience_score"] = pd.to_numeric(df["causal_salience_score"], errors="coerce").fillna(0.0)
    df = df.sort_values(by="causal_salience_score", ascending=False)
    if len(df) > 1000:
        df = df.head(1000)
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    desired_columns = [
        "PropositionClean_v2",
        "Proposition",
        "FactID",
        "EventType",
        "EventDate",
        "EventLocation",
        "ActorRole",
        "Subject",
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
        "ClassificationFixed",
        "ClassificationFixed_v2",
        "ClassificationFixed_v3",
    ]
    for column in desired_columns:
        if column not in df.columns:
            df[column] = ""
    return df[desired_columns]


OGC_KEYWORDS = ["ogc", "office of general counsel", "general counsel"]
MONKEY_KEYWORDS = ["monkey", "résumé", "resume"]
WECHAT_KEYWORDS = ["wechat", "zhihu", "sohu", "baidu", "published", "article"]
ESUWIKI_KEYWORDS = ["esuwiki", "niu tengyu", "torture", "crackdown", "14 years"]


def contains_keywords(text: str, keywords: Iterable[str]) -> bool:
    if not isinstance(text, str):
        return False
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def apply_category_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def set_value(idx: int, column: str, value: str) -> None:
        current = str(df.at[idx, column]).strip()
        if current.lower() != value.lower():
            df.at[idx, column] = value

    for idx, text in df["PropositionClean_v2"].items():
        lower_text = text.lower() if isinstance(text, str) else ""

        if contains_keywords(lower_text, OGC_KEYWORDS):
            set_value(idx, "TruthStatus", "True")
            set_value(idx, "EventType", "Communication")
            date_value = str(df.at[idx, "EventDate"]).strip().lower()
            if date_value in {"", "nan", "none"}:
                set_value(idx, "EventDate", "Unknown")
            set_value(idx, "SafetyRisk", "high")
            set_value(idx, "PublicExposure", "already_public")
            continue

        if contains_keywords(lower_text, MONKEY_KEYWORDS):
            set_value(idx, "SafetyRisk", "high")
            set_value(idx, "PublicExposure", "already_public")
            continue

        if contains_keywords(lower_text, ESUWIKI_KEYWORDS):
            set_value(idx, "SafetyRisk", "extreme")
            exposure = (str(df.at[idx, "PublicExposure"] or "") or "").lower()
            if exposure not in {"already_public", "partially_public"}:
                set_value(idx, "PublicExposure", "partially_public")
            continue

        if contains_keywords(lower_text, WECHAT_KEYWORDS):
            set_value(idx, "PublicExposure", "already_public")

    return df


def create_report(df: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ogc_mask = df["PropositionClean_v2"].map(lambda x: contains_keywords(x, OGC_KEYWORDS))
    ogc_facts = df[ogc_mask]

    truth_ok = bool(
        ogc_facts["TruthStatus"].fillna("").astype(str).str.strip().str.lower().eq("true").all()
    ) if not ogc_facts.empty else False
    event_ok = bool(
        ogc_facts["EventDate"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .ne("")
        .all()
    ) if not ogc_facts.empty else False

    monkey_ok = bool(
        df[df["PropositionClean_v2"].map(lambda x: contains_keywords(x, MONKEY_KEYWORDS))]["SafetyRisk"]
        .str.lower()
        .isin({"high", "extreme"})
        .all()
    )
    esuwiki_ok = bool(
        df[df["PropositionClean_v2"].map(lambda x: contains_keywords(x, ESUWIKI_KEYWORDS))]["SafetyRisk"].str.lower().eq("extreme").all()
    )
    wechat_ok = bool(
        df[df["PropositionClean_v2"].map(lambda x: contains_keywords(x, WECHAT_KEYWORDS))]["PublicExposure"]
        .str.lower()
        .eq("already_public")
        .all()
    )

    checklist = {
        "OGC facts present (>=8)": len(ogc_facts) >= 8,
        "OGC facts TruthStatus=True": truth_ok,
        "OGC facts have EventDate": event_ok,
        "Monkey/Resume labeled high/extreme": monkey_ok,
        "EsuWiki facts labeled extreme": esuwiki_ok,
        "WeChat/Zhihu facts are already_public": wechat_ok,
    }

    safety_counts = df["SafetyRisk"].value_counts(dropna=False).to_dict()
    exposure_counts = df["PublicExposure"].value_counts(dropna=False).to_dict()

    sample_lines = [
        f"- {row['FactID']} — {row['PropositionClean_v2'][:200].strip()}"
        for _, row in df.head(20).iterrows()
    ]

    lines = [
        "# Final Production CSV Report",
        "",
        f"- Final fact count: {len(df)}",
        f"- OGC fact count: {len(ogc_facts)}",
        "",
        "## SafetyRisk distribution",
    ]
    for label, count in safety_counts.items():
        lines.append(f"- {label or 'missing'}: {count}")

    lines.append("")
    lines.append("## PublicExposure distribution")
    for label, count in exposure_counts.items():
        lines.append(f"- {label or 'missing'}: {count}")

    lines.append("")
    lines.append("## Checklist")
    for item, status in checklist.items():
        lines.append(f"- [{'x' if status else ' '}] {item}")

    lines.append("")
    lines.append("## Sample of Top 20 Facts")
    lines.extend(sample_lines if sample_lines else ["- (no facts)"])

    lines.append("")
    lines.append("## Do This Next")
    lines.append("```")
    lines.append("python scripts/create_final_production_csv.py")
    lines.append("```")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_base_dataframe()
    df = ensure_proposition_clean(df)
    df = merge_ogc_rows(df)
    df = ensure_ogc_coverage(df)
    df = apply_category_rules(df)
    df = sort_and_trim(df)
    final_df = select_final_columns(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    create_report(final_df)
    print(f"[done] Final production CSV written to {OUTPUT_PATH} (rows={len(final_df)})")


if __name__ == "__main__":
    main()
