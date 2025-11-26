"""
Post-processing helpers for STORM-inspired research outputs.
Adds methodology sections, case tables, and enforces citation anchors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple


CITATION_PATTERN = re.compile(r"\[\d+\]")


@dataclass
class CitationValidationResult:
    ok: bool
    missing_sections: List[str]
    missing_paragraphs: int


def ensure_methodology_section(report: str, metadata: Dict[str, Any], search_results: List[Dict[str, Any]]) -> str:
    """
    Guarantee a methodology block near the top of the report.
    """
    if "## Methodology" in report:
        return report

    llm_model = metadata.get("model", "unknown-model")
    primary_count = sum(1 for r in search_results if r.get("is_primary"))
    filtered_count = len(search_results)

    method_section = f"""
## Methodology

- **LLM:** {llm_model} via Ollama (local inference)
- **Search Engine:** DuckDuckGo + LegalBERT re-ranking (bert-base-uncased fallback)
- **Candidates Evaluated:** {filtered_count}
- **Primary PRC Sources Captured:** {primary_count}
- **Local Corpus:** 1782 PDF database queried via LlamaIndex (BAAI/bge-small-en-v1.5 embeddings)
- **Filtering Criteria:** Approved domain whitelist, snippet length ≥ 80 chars, dictionary/speed-test domains excluded
- **Limitations:** English-language bias; PRC primary coverage dependent on public releases; no paywalled sources retrieved

"""

    insertion_point = report.find("## Executive Summary")
    if insertion_point != -1:
        return report[:insertion_point] + method_section + report[insertion_point:]

    return report + "\n" + method_section


def build_case_catalog(search_results: List[Dict[str, Any]]) -> List[Tuple[str, str, str, str, str]]:
    """
    Build a catalog of candidate cases from search results.
    """
    catalog = []
    seen_domains = set()

    for result in search_results:
        url = result.get("href", "")
        domain = result.get("domain", "")
        if not url or domain in seen_domains:
            continue

        title = result.get("title", "Unnamed Case")
        snippet = result.get("body", "")
        mapped_domain = domain.replace("www.", "")

        catalog.append(
            (
                title,
                mapped_domain,
                result.get("query", ""),
                snippet[:200].replace("\n", " "),
                url,
            )
        )
        seen_domains.add(domain)

        if len(catalog) >= 6:
            break

    return catalog


def insert_case_table(report: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Insert a case incidents table near the end of the report.
    """
    catalog = build_case_catalog(search_results)
    if not catalog:
        return report

    table_lines = [
        "## Documented Incidents & Cross-Border Signals",
        "",
        "| Incident | Source Domain | Search Query | Summary Signal | URL |",
        "|----------|---------------|--------------|----------------|-----|",
    ]
    for title, domain, query, snippet, url in catalog:
        safe_title = title.replace("|", "\\|")
        safe_query = query.replace("|", "\\|")
        safe_snippet = snippet.replace("|", "\\|")
        safe_url = url.replace("|", "%7C")
        table_lines.append(
            f"| {safe_title} | {domain} | {safe_query} | {safe_snippet} | {safe_url} |"
        )

    table_text = "\n".join(table_lines) + "\n"

    if "## Documented Incidents & Cross-Border Signals" in report:
        return report

    if "## Research Methodology" in report:
        return report.replace("## Research Methodology", table_text + "\n## Research Methodology", 1)

    return report + "\n" + table_text


def enforce_citation_anchors(report: str) -> CitationValidationResult:
    """
    Ensure each paragraph has at least one citation anchor.
    """
    paragraphs = [p for p in report.split("\n\n") if p.strip()]
    missing_count = 0
    missing_sections: List[str] = []

    current_section = "Introduction"

    for paragraph in paragraphs:
        if paragraph.startswith("#"):
            current_section = paragraph.strip("# ").strip()
            continue

        if current_section.lower() in {"references", "sources", "documented incidents & cross-border signals"}:
            continue

        if not CITATION_PATTERN.search(paragraph):
            missing_count += 1
            missing_sections.append(current_section)

    return CitationValidationResult(ok=missing_count == 0, missing_sections=missing_sections, missing_paragraphs=missing_count)


def add_citation_notes(report: str, search_results: List[Dict[str, Any]]) -> str:
    """
    Append a note if citations are missing.
    """
    validation = enforce_citation_anchors(report)
    if validation.ok:
        return report

    note = f"""
> ⚠️ Citation Notice: {validation.missing_paragraphs} paragraphs require inline citations.
> Sections needing attention: {', '.join(sorted(set(validation.missing_sections)))}.
"""
    return note + "\n\n" + report


def format_fact_table(facts: List[Any]) -> str:
    if not facts:
        return ""

    lines = [
        "## Fact Extraction Summary",
        "",
        "| Date | Actor / Platform | Action / Enforcement | Narrative Highlight | Case Reference | Source |",
        "|------|------------------|----------------------|---------------------|----------------|--------|",
    ]
    for fact in facts:
        date = fact.date or "-"
        actor = fact.actor or fact.domain or "-"
        action = fact.action or "-"
        narrative = (fact.narrative or fact.source_snippet or "-").replace("|", "\\|")
        case_ref = fact.case_reference or "-"
        source = fact.url.replace("|", "%7C")
        lines.append(f"| {date} | {actor} | {action} | {narrative} | {case_ref} | {source} |")

    return "\n".join(lines) + "\n"


def append_quality_summary(report: str, quality_score: Dict[str, Any]) -> str:
    if not quality_score:
        return report

    summary_lines = [
        "## Quality Assessment (Automated)",
        "",
        f"- **Overall Score:** {quality_score['overall']}/100",
        f"- **Citations:** {quality_score['details']['citations']} (score {quality_score['details']['citation_score']})",
        f"- **Primary Sources:** {quality_score['details']['primary_sources']} (score {quality_score['details']['source_score']})",
        f"- **Facts Extracted:** {quality_score['details']['facts']} (score {quality_score['details']['fact_score']})",
        f"- **Word Count:** {quality_score['details']['word_count']} (score {quality_score['details']['length_score']})",
        f"- **Structure Bonus:** {quality_score['details']['structure_bonus']}",
    ]
    if quality_score.get("notes"):
        summary_lines.append("- **Notes:** " + "; ".join(quality_score["notes"]))

    return report + "\n" + "\n".join(summary_lines) + "\n"


def postprocess_report(
    report: str,
    metadata: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    facts: List[Any],
    quality_score: Dict[str, Any],
) -> str:
    """
    Full post-processing pipeline: methodology, case catalog, citation notice.
    """
    report = ensure_methodology_section(report, metadata, search_results)
    report = insert_case_table(report, search_results)
    fact_table = format_fact_table(facts)
    if fact_table:
        report += "\n" + fact_table
    report = add_citation_notes(report, search_results)
    report = append_quality_summary(report, quality_score)
    return report
