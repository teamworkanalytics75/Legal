#!/usr/bin/env python3
"""Generate Markdown summary of lawsuit keywords and similar case-law matches."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


FOCUS_DOCS = {
    "Exhibit2CertifiedStatementOfClaimHongKong2Jun202515.docx": "HK Statement of Claim",
    "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-L – Email from Malcolm Grayson to Harvard OGC (7 Apr 2025).docx": "OGC Email, 7 Apr 2025",
    "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-M – Email from Malcolm Grayson to Harvard OGC (18 Apr 2025).docx": "OGC Email, 18 Apr 2025",
    "Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Exhibit 6-M2– Email from Malcolm Grayson to Harvard OGC (August 11 2025).docx": "OGC Email, 11 Aug 2025",
}


def load_tables(suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path("reports") / "analysis_outputs"
    keywords = pd.read_csv(base / f"lawsuit_keywords__{suffix}.csv")
    matches = pd.read_csv(base / f"lawsuit_similarity__{suffix}.csv")
    return keywords, matches


def create_markdown(suffix: str) -> str:
    keywords, matches = load_tables(suffix)
    case_meta = pd.read_csv(
        Path("case_law_data") / "features" / "unified_features.csv",
        usecols=["cluster_id", "case_name", "court", "corpus_type", "outcome"],
    )

    lines: list[str] = [
        "# Lawsuit Keyword & Case Similarity Summary",
        "",
        f"Model suffix: `{suffix}`",
        "",
    ]

    doc_set = set(keywords["doc"])
    for doc, label in FOCUS_DOCS.items():
        if doc not in doc_set:
            continue

        keyword_str = keywords.loc[keywords["doc"] == doc, "keywords"].iloc[0]
        top_tokens = [token.strip() for token in keyword_str.split(",")[:12]]

        lines.append(f"## {label}")
        lines.append("**Top keywords (TF-IDF):**")
        for token in top_tokens:
            lines.append(f"- {token}")

        doc_matches = (
            matches[matches["doc"] == doc]
            .merge(case_meta, on="cluster_id", how="left")
            .sort_values("rank")
            .head(10)
        )
        lines.append("")
        lines.append("**Closest case-law matches:**")
        for _, row in doc_matches.iterrows():
            case = row.get("case_name", "Unknown case")
            court = row.get("court", "Unknown court")
            corpus = row.get("corpus_type", "unknown")
            outcome = row.get("outcome", "unknown")
            similarity = row["similarity"]
            rank = int(row["rank"])
            lines.append(
                f"- #{rank}: {case} ({court}) — corpus: {corpus}, outcome: {outcome}, similarity {similarity:.3f}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    suffix = "legal-bert-base-uncased"
    markdown = create_markdown(suffix)
    out_path = Path("case_law_data") / "exports" / "LAWSUIT_KEYWORDS_AND_MATCHES.md"
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
