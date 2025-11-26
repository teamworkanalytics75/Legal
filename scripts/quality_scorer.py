"""
Quality scoring for STORM-inspired research reports.
Evaluates source mix, citation coverage, facts extracted, and length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, List

CITATION_PATTERN = re.compile(r"\[\d+\]")


@dataclass
class QualityScore:
    overall: float
    details: Dict[str, float]
    notes: List[str]


def score_report(report: str, search_results: List[Dict[str, Any]], facts: List[Any], metadata: Dict[str, Any]) -> QualityScore:
    """
    Produce a basic quality score (0-100) based on heuristics.
    """
    notes: List[str] = []

    word_count = len(report.split())
    citations = len(CITATION_PATTERN.findall(report))
    primary_sources = sum(1 for r in search_results if r.get("is_primary"))
    total_sources = len(search_results)
    fact_count = len(facts)

    citation_score = min(citations / max(1, word_count / 400), 1.0) * 25  # expect ~1 citation per 400 words
    source_score = 15 + min(primary_sources, 4) * 5
    if total_sources < 10:
        source_score -= 10
        notes.append("Fewer than 10 unique sources captured.")
    if primary_sources == 0:
        notes.append("No PRC primary sources in this run.")

    fact_score = min(fact_count / 4, 1.0) * 20
    if fact_count < 3:
        notes.append("Fewer than 3 concrete facts extracted.")

    length_score = min(word_count / 4000, 1.0) * 20
    structure_bonus = 10 if "Documented Incidents & Cross-Border Signals" in report else 0

    overall = citation_score + source_score + fact_score + length_score + structure_bonus
    overall = max(0.0, min(100.0, overall))

    details = {
        "citation_score": round(citation_score, 2),
        "source_score": round(source_score, 2),
        "fact_score": round(fact_score, 2),
        "length_score": round(length_score, 2),
        "structure_bonus": round(structure_bonus, 2),
        "word_count": word_count,
        "citations": citations,
        "primary_sources": primary_sources,
        "facts": fact_count,
    }

    return QualityScore(overall=round(overall, 2), details=details, notes=notes)
