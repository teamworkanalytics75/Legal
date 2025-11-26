#!/usr/bin/env python3
"""Extract missing critical facts (OGC non-response, Zhihu, Statement 2, EsuWiki)."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from document_ingestion.ExtractText import extract_text

SALIENT_PATH = Path("case_law_data/facts_truth_table_salient.csv")
OUTPUT_FACTS_PATH = Path("case_law_data/missing_critical_facts.csv")
COMBINED_SALIENT_PATH = Path("case_law_data/facts_truth_table_salient_complete.csv")
REPORT_PATH = Path("reports/analysis_outputs/missing_facts_coverage_report.md")
SEARCH_ROOTS = [
    Path("case_law_data/lawsuit_source_documents"),
    Path("Harvard - The Art of War"),
    Path("EsuWiki"),
]
ALLOWED_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".json",
    ".csv",
    ".xml",
    ".xdsl",
    ".yml",
    ".yaml",
    ".docx",
    ".pdf",
}


@dataclass
class PatternConfig:
    name: str
    regex: Optional[re.Pattern[str]]
    event_type: str
    notes: str


OGC_TERMS = [
    "ogc",
    "office of general counsel",
    "general counsel",
    "your office",
]
NONRESPONSE_TERMS = [
    "did not respond",
    "no response",
    "no reply",
    "within 14 days",
    "without acknowledgement",
    "no acknowledgment",
    "failed to acknowledge",
    "ignored",
    "silence",
    "has not acknowledged",
    "not acknowledged",
    "has not replied",
    "hasn't replied",
    "never responded",
]


PATTERNS: list[PatternConfig] = [
    PatternConfig(
        name="OGC non-response",
        regex=None,
        event_type="Communication",
        notes="Extract explicit mentions that Harvard OGC failed to respond within deadlines.",
    ),
    PatternConfig(
        name="Zhihu amplification",
        regex=re.compile(r"Zhihu[^.]{0,220}", re.IGNORECASE),
        event_type="Publication",
        notes="Capture republication/visibility events on Zhihu.",
    ),
    PatternConfig(
        name="Statement 2 dating",
        regex=re.compile(
            r"(Statement\s*2[^.]{0,220}(back[- ]?dated|false date|30 April|11 May|29 April))",
            re.IGNORECASE,
        ),
        event_type="Publication",
        notes="Capture false/back-dating of Statement 2.",
    ),
    PatternConfig(
        name="EsuWiki / Niu Tengyu chain",
        regex=re.compile(r"(EsuWiki|Niu\s+Tengyu|CCP crackdown|torture severity)[^.]{0,220}", re.IGNORECASE),
        event_type="Harm",
        notes="Capture CCP crackdown / Niu Tengyu torture causation links.",
    ),
]

PATTERN_LIMITS = {
    "OGC non-response": 100,
    "Zhihu amplification": 60,
    "Statement 2 dating": 60,
    "EsuWiki / Niu Tengyu chain": 60,
}


def iter_text_files() -> Iterable[Path]:
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in ALLOWED_SUFFIXES:
                yield path


def read_text(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in {".docx", ".pdf"}:
        try:
            return extract_text(path)  # type: ignore[arg-type]
        except Exception as exc:
            print(f"[warn] Failed to extract text from {path}: {exc}")
            return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return None


def extract_sentences(text: str) -> list[str]:
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for sentence in raw_sentences:
        snippet = sentence.strip()
        if len(snippet) < 40:
            continue
        cleaned.append(snippet)
    return cleaned


def scan_file(path: Path) -> list[dict[str, str]]:
    text = read_text(path)
    if not text:
        return []
    sentences = extract_sentences(text)
    matches: list[dict[str, str]] = []
    source_format = path.suffix.lower().lstrip(".")
    for sentence in sentences:
        for pattern in PATTERNS:
            if pattern.name == "OGC non-response":
                lower = sentence.lower()
                if not any(term in lower for term in OGC_TERMS):
                    continue
                if not any(term in lower for term in NONRESPONSE_TERMS):
                    continue
                matches.append(
                    {
                        "Pattern": pattern.name,
                        "Proposition": sentence.strip(),
                        "SourceDocument": str(path),
                        "EventType": pattern.event_type,
                        "EvidenceType": "Document",
                        "Notes": pattern.notes,
                        "SourceFormat": source_format or "unknown",
                    }
                )
                break
            elif pattern.regex and pattern.regex.search(sentence):
                matches.append(
                    {
                        "Pattern": pattern.name,
                        "Proposition": sentence.strip(),
                        "SourceDocument": str(path),
                        "EventType": pattern.event_type,
                        "EvidenceType": "Document",
                        "Notes": pattern.notes,
                        "SourceFormat": source_format or "unknown",
                    }
                )
                break
    return matches


def build_missing_facts() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for idx, path in enumerate(iter_text_files(), start=1):
        file_matches = scan_file(path)
        rows.extend(file_matches)
        if idx % 200 == 0:
            print(f"[scan] processed {idx} files, facts so far: {len(rows)}")
    df = pd.DataFrame(rows).drop_duplicates(subset=["Proposition"])
    if df.empty:
        return df
    if PATTERN_LIMITS:
        limited_frames = []
        for pattern, group in df.groupby("Pattern"):
            limit = PATTERN_LIMITS.get(pattern)
            if limit:
                limited_frames.append(group.head(limit))
            else:
                limited_frames.append(group)
        df = pd.concat(limited_frames, ignore_index=True)
    df.insert(0, "FactID", [f"MISSING_{i+1:04d}" for i in range(len(df))])
    df["Subject"] = ""
    df["Verb"] = ""
    df["Object"] = ""
    df["EventDate"] = ""
    df["EventLocation"] = ""
    df["ActorRole"] = ""
    df["Speaker"] = ""
    df["TruthStatus"] = "True"
    df["SourceExcerpt"] = ""
    df["SafetyRisk"] = ""
    df["PublicExposure"] = ""
    df["RiskRationale"] = ""
    df["confidence_tier"] = 2
    df["causal_salience_score"] = 0.6
    df["confidence_reason"] = "pattern_match"
    df["causal_salience_reason"] = df["Pattern"]
    return df


def merge_with_salient(new_facts: pd.DataFrame) -> pd.DataFrame:
    if not SALIENT_PATH.exists():
        raise FileNotFoundError(f"Missing {SALIENT_PATH}")
    salient = pd.read_csv(SALIENT_PATH)
    combined = pd.concat([salient, new_facts], ignore_index=True) if not new_facts.empty else salient
    combined = combined.drop_duplicates(subset=["Proposition"], keep="first")
    return combined


def write_report(facts_df: pd.DataFrame, combined_df: pd.DataFrame) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Missing Facts Coverage Report",
        "",
        f"- Missing facts extracted: {len(facts_df)}",
        f"- Combined salient facts: {len(combined_df)}",
        "",
    ]
    if not facts_df.empty:
        pattern_counts = facts_df["Pattern"].value_counts().to_dict()
        lines.append("## Pattern Counts")
        for pattern, count in pattern_counts.items():
            lines.append(f"- {pattern}: {count}")
        lines.append("")
        format_counts = facts_df["SourceFormat"].fillna("unknown").value_counts().to_dict()
        lines.append("## Source Formats")
        for fmt, count in format_counts.items():
            lines.append(f"- {fmt}: {count}")
        lines.append("")
        sample = facts_df.head(5)
        lines.append("## Sample Facts")
        for _, row in sample.iterrows():
            lines.append(f"- **{row['Pattern']}** — {row['Proposition'][:200]} ({row['SourceDocument']})")
        lines.append("")
    lines.extend(
        [
            "## Do This Next",
            "```",
            "python scripts/extract_missing_critical_facts.py",
            "```",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_facts = build_missing_facts()
    if not new_facts.empty:
        new_facts.to_csv(OUTPUT_FACTS_PATH, index=False)
    else:
        OUTPUT_FACTS_PATH.write_text("", encoding="utf-8")
    combined = merge_with_salient(new_facts)
    COMBINED_SALIENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_SALIENT_PATH, index=False)
    write_report(new_facts, combined)
    print(f"[done] new facts: {len(new_facts)}, combined salient size: {len(combined)}")


if __name__ == "__main__":
    main()
