#!/usr/bin/env python3
"""Build CourtListener query suggestions from lawsuit keyword analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

FOCUS_DOCS = {
    "Exhibit2CertifiedStatementOfClaimHongKong2Jun202515.docx": {
        "label": "HK Statement of Claim",
        "notes": "Consider filtering for Harvard as party, foreign sovereign disputes, and 2018–present.",
    },
    "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-L – Email from Malcolm Grayson to Harvard OGC (7 Apr 2025).docx": {
        "label": "OGC Email, 7 Apr 2025",
        "notes": "Useful for settlement privilege / OGC correspondence patterns; filter for D. Mass. or Harvard OGC references.",
    },
    "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-M – Email from Malcolm Grayson to Harvard OGC (18 Apr 2025).docx": {
        "label": "OGC Email, 18 Apr 2025",
        "notes": "Focus on formal service, cross-border defendants, and Harvard legal counsel disputes.",
    },
    "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-M2– Email from Malcolm Grayson to Harvard OGC (August 11 2025).docx": {
        "label": "OGC Email, 11 Aug 2025",
        "notes": "Target meet-and-confer breakdowns, identification of recipients, discovery motion practice.",
    },
}

SYNONYMS: Dict[str, List[str]] = {
    "harvard": ['"Harvard University"', '"President and Fellows of Harvard College"'],
    "club": ['"Harvard Club"', '"Hong Kong Club"'],
    "plaintiff": ['"claimant"', '"applicant"'],
    "defendants": ['"respondents"', '"defendant"'],
    "statement": ['"statement of claim"', '"formal pleadings"'],
    "schedule": ['"timetable"', '"schedule of particulars"'],
    "particulars": ['"statement of particulars"'],
    "china": ['"PRC"', '"People\'s Republic of China"', '"Chinese Communist Party"'],
    "settlement discussions": ['"settlement negotiations"', '"without prejudice"'],
    "formal service": ['"service of process"', '"formal service"', '"service abroad"'],
    "meet confer": ['"meet and confer"', '"Rule 26(f)"'],
    "recipients": ['"email recipients"'],
    "identify": ['"identify recipients"', '"identify parties"'],
    "confidential": ['"confidential information"', '"protective order"'],
    "discuss": ['"discussion"', '"discuss settlement"'],
    "prejudice settlement": ['"without prejudice"', '"settlement privilege"'],
    "settlement": ['"settlement negotiations"', '"settlement communications"'],
}


def load_keywords(suffix: str) -> pd.DataFrame:
    path = Path("reports") / "analysis_outputs" / f"lawsuit_keywords__{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Keyword file not found: {path}")
    return pd.read_csv(path)


def parse_tokens(keyword_string: str) -> List[str]:
    tokens = []
    for item in keyword_string.split(","):
        parts = item.strip().split(" (")
        token = parts[0].strip()
        if token:
            tokens.append(token)
    seen = set()
    deduped = []
    for token in tokens:
        key = token.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(token)
    return deduped


def expand_token(token: str) -> List[str]:
    normalized = token.lower()
    expansions = [token]
    if normalized in SYNONYMS:
        expansions.extend(SYNONYMS[normalized])
    return expansions


def format_or_clause(tokens: Iterable[str]) -> str:
    formatted = []
    for token in tokens:
        phrase = token.strip().strip('"')
        if not phrase:
            continue
        if phrase.lower() == "harvard":
            formatted.append('"Harvard"')
            formatted.extend(SYNONYMS.get("harvard", []))
            continue
        if " " in phrase or "-" in phrase or phrase.isalpha():
            formatted.append(f'"{phrase}"')
        else:
            formatted.append(phrase)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in formatted:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return " OR ".join(unique) if unique else '""'


def build_query(core_tokens: List[str], context_tokens: List[str]) -> str:
    if core_tokens:
        core_expanded = []
        for token in core_tokens:
            core_expanded.extend(expand_token(token))
        core_clause = format_or_clause(core_expanded)
    else:
        core_clause = '""'

    if context_tokens:
        context_expanded = []
        for token in context_tokens:
            context_expanded.extend(expand_token(token))
        context_clause = format_or_clause(context_expanded)
    else:
        context_clause = '""'

    query_parts = []
    if core_clause.strip('" '):
        query_parts.append(f"({core_clause})")
    if context_clause.strip('" '):
        query_parts.append(f"({context_clause})")
    if "harvard" not in core_clause.lower() and "harvard" not in context_clause.lower():
        harvard_clause = format_or_clause(expand_token("harvard"))
        query_parts.append(f"({harvard_clause})")

    return " AND ".join(query_parts)


def main() -> None:
    suffix = "legal-bert-base-uncased"
    keywords_df = load_keywords(suffix)
    doc_set = set(keywords_df["doc"])

    lines = [
        "# CourtListener Query Suggestions",
        "",
        f"Model suffix: `{suffix}`",
        "",
        "Each query is built from the top TF-IDF tokens in the corresponding lawsuit document, "
        "expanded with common synonyms where useful. Use these as starting points and "
        "adjust filters (jurisdiction, date range, party) inside CourtListener.",
        "",
    ]

    for doc, metadata in FOCUS_DOCS.items():
        if doc not in doc_set:
            continue
        label = metadata["label"]
        notes = metadata.get("notes", "")
        token_str = keywords_df.loc[keywords_df["doc"] == doc, "keywords"].iloc[0]
        tokens = parse_tokens(token_str)
        core_tokens = tokens[:6]
        context_tokens = tokens[6:12]
        query = build_query(core_tokens, context_tokens)

        lines.append(f"## {label}")
        lines.append(f"**Source doc:** `{doc}`")
        if notes:
            lines.append(f"**Notes:** {notes}")

        lines.append("**Core keywords:**")
        for token in core_tokens:
            lines.append(f"- {token}")

        if context_tokens:
            lines.append("")
            lines.append("**Context keywords:**")
            for token in context_tokens:
                lines.append(f"- {token}")

        lines.append("")
        lines.append("**Suggested CourtListener query:**")
        lines.append(f"`{query}`")
        lines.append("")

    out_path = Path("case_law_data") / "exports" / "COURTLISTENER_QUERIES.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
