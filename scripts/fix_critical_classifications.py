#!/usr/bin/env python3
"""Fix SafetyRisk/PublicExposure on top-salience facts and merge KG extracts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

SOURCE_PATH = Path("case_law_data/top_1000_facts_with_clean_propositions.csv")
FALLBACK_PATH = Path("case_law_data/top_1000_facts_for_chatgpt_v2.csv")
KG_PATH = Path("case_law_data/extracted_kg_facts.csv")
OUTPUT_PATH = Path("case_law_data/top_1000_facts_for_chatgpt_v3.csv")
REPORT_PATH = Path("reports/analysis_outputs/critical_classifications_fix_report.md")


WRAPPER_PREFIX = "The document refers to "


def ensure_proposition_clean(df: pd.DataFrame) -> pd.DataFrame:
    if "PropositionClean" in df.columns:
        return df

    def strip_wrapper(text: str) -> str:
        if isinstance(text, str) and text.startswith(WRAPPER_PREFIX):
            return text[len(WRAPPER_PREFIX) :].lstrip()
        return text or ""

    df = df.copy()
    df.insert(0, "PropositionClean", df["Proposition"].map(strip_wrapper))
    return df


def load_input() -> pd.DataFrame:
    if SOURCE_PATH.exists():
        df = pd.read_csv(SOURCE_PATH)
        return ensure_proposition_clean(df)
    if not FALLBACK_PATH.exists():
        raise FileNotFoundError("No input CSV found.")
    df = pd.read_csv(FALLBACK_PATH)
    df = ensure_proposition_clean(df)
    SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SOURCE_PATH, index=False)
    return df


def contains(text: str, keywords: Iterable[str]) -> bool:
    if not isinstance(text, str):
        return False
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def apply_classification_fixes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    if "ClassificationFixed_v2" not in df.columns:
        df["ClassificationFixed_v2"] = ""

    df_sorted = df.sort_values(by="causal_salience_score", ascending=False)
    top_indices = set(df_sorted.index[:200])
    high_salience_mask = (df["causal_salience_score"] >= 0.7) | df.index.isin(top_indices)

    change_counts = {
        "safety": 0,
        "exposure": 0,
        "OGC": 0,
        "Wechat": 0,
        "EsuWiki": 0,
        "Monkey": 0,
    }

    def add_note(idx: int, note: str) -> None:
        current = df.at[idx, "ClassificationFixed_v2"]
        df.at[idx, "ClassificationFixed_v2"] = f"{current}; {note}".strip("; ")

    def set_safety(idx: int, value: str, note: str) -> bool:
        current = (df.at[idx, "SafetyRisk"] or "").strip().lower()
        if current in {"", "none", "low", "medium"} and current != value.lower():
            df.at[idx, "SafetyRisk"] = value
            change_counts["safety"] += 1
            add_note(idx, note)
            return True
        return False

    def set_exposure(idx: int, value: str, note: str) -> bool:
        current = (df.at[idx, "PublicExposure"] or "").strip().lower()
        if current != value.lower():
            df.at[idx, "PublicExposure"] = value
            change_counts["exposure"] += 1
            add_note(idx, note)
            return True
        return False

    harm_keywords = ["harm", "threat", "danger", "persecution", "torture", "risk", "retaliation", "exposure"]

    for idx, row in df[high_salience_mask].iterrows():
        prop = str(row.get("PropositionClean") or row.get("Proposition") or "")
        lower = prop.lower()

        # OGC non-response
        if contains(lower, ["ogc", "office of general counsel", "general counsel"]) and contains(
            lower,
            ["did not respond", "no response", "no reply", "no acknowledgement", "failed to acknowledge", "ignored"],
        ):
            if set_safety(idx, "high", "OGC non-response safety"):
                change_counts["OGC"] += 1
            continue

        # EsuWiki / torture
        if contains(lower, ["esuwiki", "niu tengyu", "torture", "crackdown", "14 years"]):
            changed = set_safety(idx, "extreme", "EsuWiki torture safety")
            if (row.get("PublicExposure") or "").strip().lower() == "not_public":
                changed = set_exposure(idx, "partially_public", "EsuWiki visibility") or changed
            if changed:
                change_counts["EsuWiki"] += 1
            continue

        # WeChat / Zhihu / Sohu / Baidu articles
        if contains(lower, ["wechat", "zhihu", "sohu", "baidu", "published", "article"]):
            changed = set_exposure(idx, "already_public", "Publication exposure")
            if contains(lower, harm_keywords):
                changed = set_safety(idx, "high", "Publication harm risk") or changed
            elif (row.get("SafetyRisk") or "").strip().lower() in {"", "none"}:
                changed = set_safety(idx, "medium", "Publication medium risk") or changed
            if changed:
                change_counts["Wechat"] += 1
            continue

        # Monkey / Resume articles
        if contains(lower, ["monkey", "résumé", "resume", "screenshot"]):
            changed = set_exposure(idx, "already_public", "Monkey/resume exposure")
            changed = set_safety(idx, "high", "Monkey/resume safety") or changed
            if changed:
                change_counts["Monkey"] += 1
            continue

    return df, change_counts


def merge_kg_facts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if not KG_PATH.exists():
        return df, 0

    kg_df = pd.read_csv(KG_PATH)
    kg_df = ensure_proposition_clean(kg_df)
    missing_cols = [col for col in df.columns if col not in kg_df.columns]
    for col in missing_cols:
        kg_df[col] = ""
    kg_df = kg_df[df.columns]
    combined = pd.concat([df, kg_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["PropositionClean"], keep="first")
    return combined, len(kg_df)


def write_report(stats: dict[str, int], total_rows: int, final_df: pd.DataFrame, kg_count: int) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Critical Classification Fix Report",
        "",
        f"- SafetyRisk fixes applied: {stats['safety']}",
        f"- PublicExposure fixes applied: {stats['exposure']}",
        f"- OGC fixes: {stats['OGC']}",
        f"- WeChat/Zhihu fixes: {stats['Wechat']}",
        f"- EsuWiki fixes: {stats['EsuWiki']}",
        f"- Monkey/Resume fixes: {stats['Monkey']}",
        f"- KG facts merged: {kg_count}",
        f"- Final fact count: {total_rows}",
        "",
        "## SafetyRisk distribution",
    ]

    safety_counts = final_df["SafetyRisk"].value_counts(dropna=False).to_dict()
    for value, count in safety_counts.items():
        lines.append(f"- {value or 'missing'}: {count}")

    lines.append("")
    lines.append("## PublicExposure distribution")
    exposure_counts = final_df["PublicExposure"].value_counts(dropna=False).to_dict()
    for value, count in exposure_counts.items():
        lines.append(f"- {value or 'missing'}: {count}")

    lines.append("")
    lines.append("## Sample Fixes")

    samples = final_df[final_df["ClassificationFixed_v2"].astype(str) != ""].head(10)
    for _, row in samples.iterrows():
        lines.append(f"- {row['ClassificationFixed_v2']}: {row['PropositionClean'][:200]}")

    lines.append("")
    lines.extend(
        [
            "## Do This Next",
            "```",
            "python scripts/fix_critical_classifications.py",
            "```",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_input()
    df_fixed, stats = apply_classification_fixes(df)
    combined_df, kg_count = merge_kg_facts(df_fixed)

    combined_df = combined_df.sort_values(by="causal_salience_score", ascending=False)
    if len(combined_df) > 1000:
        combined_df = combined_df.head(1000)

    columns = ["PropositionClean"] + [col for col in combined_df.columns if col != "PropositionClean"]
    combined_df = combined_df[columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_PATH, index=False)

    write_report(stats, len(combined_df), combined_df, kg_count)
    print(
        f"[done] Safety fixes: {stats['safety']}, exposure fixes: {stats['exposure']}, "
        f"KG facts merged: {kg_count}, final rows: {len(combined_df)}"
    )


if __name__ == "__main__":
    main()
