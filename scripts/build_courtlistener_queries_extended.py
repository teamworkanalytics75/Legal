#!/usr/bin/env python3
"""Construct expanded CourtListener query packs across lawsuit themes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

SUFFIX = "legal-bert-base-uncased"


@dataclass
class QueryPack:
    label: str
    docs: Sequence[str]
    include_core: Sequence[str] = field(default_factory=list)
    include_context: Sequence[str] = field(default_factory=list)
    filters: Sequence[str] = field(default_factory=list)
    notes: str = ""


PACKS: List[QueryPack] = [
    QueryPack(
        label="HK Statement of Claim",
        docs=["Exhibit2CertifiedStatementOfClaimHongKong2Jun202515.docx"],
        filters=["party:\"Harvard\" OR \"President and Fellows of Harvard\"", "date:2018-01-01 TO 2025-12-31"],
        notes="Focus on Harvard as party, PRC sovereign disputes, and cross-border defamation.",
    ),
    QueryPack(
        label="OGC Intake (April 2025)",
        docs=[
            "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-L – Email from Malcolm Grayson to Harvard OGC (7 Apr 2025).docx",
            "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-M – Email from Malcolm Grayson to Harvard OGC (18 Apr 2025).docx",
        ],
        filters=["docket:massachusetts OR docket:\"First Circuit\""],
        notes="Settlement privilege, without prejudice communications, OGC handling of student club disputes.",
    ),
    QueryPack(
        label="OGC Escalation (Aug 2025)",
        docs=[
            "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-M2– Email from Malcolm Grayson to Harvard OGC (August 11 2025).docx",
            "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/1. Deep Research/ChatGPT Chats/Email to Harvard OGC - Spoliation Awareness + §1001 Notice - Aug 2025.docx",
        ],
        filters=["docket:\"District of Massachusetts\""],
        notes="Meet-and-confer breakdowns, identification of recipients, threats of §1001 / spoliation notices.",
    ),
    QueryPack(
        label="Harvard Club & Alumni Admissions",
        docs=[
            "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-F — Email Chain_ Malcolm Grayson ↔ HAA Clubs & SIGs (30 Apr–10 May 2019) – “Greetings”.docx",
            "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-G — Email from Malcolm Grayson to HAA Clubs & SIGs (20 May 2019) – Request for Clarification & Apology.docx",
            "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-I — Email from Marlyn McGrath to Malcolm Grayson (24 May 2019) – Reply to Exhibit 6-H.docx",
        ],
        filters=["topic:\"admissions\" OR topic:\"alumni\""],
        notes="Admissions office communications, alumni club governance, reputational harm claims.",
    ),
    QueryPack(
        label="PRC Coordination & Surveillance",
        docs=[
            "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-D -  Email Chain_ Malcolm Grayson ↔ Yi Wang (23–30 Apr 2019).docx",
            "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-D-2 – WeChat Messages & Transcript_ MJ Tang ↔ Malcolm Grayson (Late Apr 2019).docx",
            "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/1. Deep Research/1. Today/Admissibility of Online Electronic Evidence in Chinese Courts.docx",
        ],
        filters=["topic:\"foreign agent\" OR topic:\"surveillance\""],
        notes="WeChat surveillance, PRC liaison efforts, admissibility of digital evidence, diaspora coordination.",
    ),
    QueryPack(
        label="Direct Xi / Propaganda References",
        docs=["Exhibit7TheTwoXiSlides.docx"],
        filters=["topic:\"propaganda\" OR topic:\"foreign influence\""],
        notes="Target cases involving Xi Jinping, PRC propaganda, foreign sovereign messaging.",
    ),
]


def load_keywords(suffix: str) -> pd.DataFrame:
    path = Path("reports") / "analysis_outputs" / f"lawsuit_keywords__{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Keyword file not found: {path}")
    return pd.read_csv(path)


def parse_tokens(keyword_string: str, limit: int = 10) -> List[str]:
    tokens = []
    for idx, item in enumerate(keyword_string.split(",")):
        text = item.strip().split(" (")[0].strip()
        if text:
            tokens.append(text)
        if len(tokens) >= limit:
            break
    return tokens


def aggregate_tokens(docs: Iterable[str], keyword_df: pd.DataFrame, top_core: int = 6, top_context: int = 12) -> tuple[List[str], List[str]]:
    core_tokens: List[str] = []
    context_tokens: List[str] = []
    for doc in docs:
        rows = keyword_df[keyword_df["doc"] == doc]
        if rows.empty:
            continue
        tokens = parse_tokens(rows.iloc[0]["keywords"], limit=top_context)
        core_tokens.extend(tokens[:top_core])
        context_tokens.extend(tokens[top_core:])
    # Deduplicate while preserving order
    def dedupe(seq: Iterable[str]) -> List[str]:
        seen = set()
        result = []
        for token in seq:
            key = token.lower()
            if key not in seen:
                seen.add(key)
                result.append(token)
        return result

    return dedupe(core_tokens), dedupe(context_tokens)


def format_clause(tokens: Iterable[str]) -> str:
    formatted = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.lower().startswith("§") or token.lower().startswith("section"):
            formatted.append(f'"{token}"')
        elif any(c.isdigit() for c in token) or " " in token or token.isalpha():
            formatted.append(f'"{token}"')
        else:
            formatted.append(token)
    seen = set()
    unique = []
    for item in formatted:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return " OR ".join(unique) if unique else '""'


def build_query(core_tokens: List[str], context_tokens: List[str], filters: Sequence[str]) -> str:
    clauses: List[str] = []
    core_clause = format_clause(core_tokens)
    context_clause = format_clause(context_tokens)

    if core_clause.strip('" '):
        clauses.append(f"({core_clause})")
    if context_clause.strip('" ') and context_clause != core_clause:
        clauses.append(f"({context_clause})")

    # Always anchor on Harvard unless already explicit
    if "harvard" not in core_clause.lower() and "harvard" not in context_clause.lower():
        clauses.append('("Harvard" OR "Harvard University" OR "President and Fellows of Harvard College")')

    for flt in filters:
        clauses.append(f"({flt})")

    return " AND ".join(clauses) if clauses else '""'


def main() -> None:
    keyword_df = load_keywords(SUFFIX)
    available_docs = set(keyword_df["doc"])

    rows: List[Dict[str, str]] = []
    for pack in PACKS:
        docs = [doc for doc in pack.docs if doc in available_docs]
        if not docs:
            continue

        core, context = aggregate_tokens(docs, keyword_df)
        core = list(dict.fromkeys((*pack.include_core, *core)))
        context = list(dict.fromkeys((*pack.include_context, *context)))

        query = build_query(core, context, pack.filters)

        rows.append(
            {
                "category": pack.label,
                "documents": "; ".join(docs),
                "core_keywords": ", ".join(core[:12]),
                "context_keywords": ", ".join(context[:12]),
                "filters": "; ".join(pack.filters) if pack.filters else "",
                "notes": pack.notes,
                "query": query,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = Path("case_law_data") / "exports" / "COURTLISTENER_QUERY_PACKS.csv"
    out_md = Path("case_law_data") / "exports" / "COURTLISTENER_QUERY_PACKS.md"

    df.to_csv(out_csv, index=False)

    md_lines = [
        "# Expanded CourtListener Query Packs",
        "",
        f"Model suffix: `{SUFFIX}`",
        "",
        "Each pack bundles top TF-IDF keywords by allegation set, adds recommended filters, and supplies a ready-to-run query.",
        "",
    ]
    for row in rows:
        md_lines.append(f"## {row['category']}")
        md_lines.append(f"**Documents:** {row['documents']}")
        if row["notes"]:
            md_lines.append(f"**Notes:** {row['notes']}")
        if row["filters"]:
            md_lines.append(f"**Suggested filters:** {row['filters']}")
        md_lines.append("**Core keywords:**")
        for token in row["core_keywords"].split(", "):
            if token:
                md_lines.append(f"- {token}")
        if row["context_keywords"]:
            md_lines.append("")
            md_lines.append("**Context keywords:**")
            for token in row["context_keywords"].split(", "):
                if token:
                    md_lines.append(f"- {token}")
        md_lines.append("")
        md_lines.append("**Query:**")
        md_lines.append(f"`{row['query']}`")
        md_lines.append("")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
