"""Temporal evidence filtering for date-aware Bayesian inference.

This module adds temporal awareness to prevent anachronistic reasoning
(e.g., using 2025 emails as evidence for 2019 knowledge).

Key principle: Evidence can only be used if it occurred BEFORE the query time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple


# Temporal metadata for key evidence
EVIDENCE_TIMELINE = {
    "OGC": {"type": "Email", "date": "2025", "year": 2025},
    "Email": {"type": "Communication", "date": "2025", "year": 2025},
    "Statements": {"type": "Document", "date": "2025", "year": 2025},
}


def get_evidence_date(node_id: str) -> Optional[int]:
    """Get the year when evidence became available.
    
    Args:
        node_id: Node identifier
        
    Returns:
        Year (int) or None if no temporal information
    """
    if node_id in EVIDENCE_TIMELINE:
        return EVIDENCE_TIMELINE[node_id]["year"]
    
    # Try to extract year from node name
    for year in range(2015, 2026):
        if str(year) in node_id:
            return year
    
    return None


def filter_evidence_by_date(
    evidence: Dict[str, str],
    query_date: int,
    strict: bool = True
) -> Tuple[Dict[str, str], List[str]]:
    """Filter evidence to only include facts that existed at query time.
    
    Args:
        evidence: Evidence dictionary
        query_date: Year of the query (e.g., 2019)
        strict: If True, exclude evidence from after query_date
        
    Returns:
        Tuple of (filtered_evidence, excluded_items)
        
    Example:
        >>> evidence = {'OGC': 'Involved', 'Harvard': 'True', 'April 2019': 'Past'}
        >>> filtered, excluded = filter_evidence_by_date(evidence, 2019)
        >>> # OGC excluded (2025 email), April 2019 included, Harvard included (timeless)
    """
    filtered = {}
    excluded = []
    
    for node_id, state in evidence.items():
        evidence_date = get_evidence_date(node_id)
        
        if evidence_date is None:
            # No temporal info - assume timeless (like Harvard existence)
            filtered[node_id] = state
        elif evidence_date <= query_date or not strict:
            # Evidence existed at query time
            filtered[node_id] = state
        else:
            # Evidence is from the future relative to query
            excluded.append(f"{node_id} (from {evidence_date})")
    
    return filtered, excluded


def get_temporal_summary(evidence: Dict[str, str], query_date: int) -> str:
    """Get human-readable summary of temporal filtering.
    
    Args:
        evidence: Original evidence
        query_date: Query date
        
    Returns:
        Formatted string explaining filtering
    """
    filtered, excluded = filter_evidence_by_date(evidence, query_date)
    
    lines = [
        f"Temporal Evidence Filter (Query Date: {query_date})",
        "-" * 60,
        "",
        f"Evidence that existed in {query_date}:",
    ]
    
    if filtered:
        for node, state in filtered.items():
            ev_date = get_evidence_date(node)
            if ev_date:
                lines.append(f" - {node} = {state} (from {ev_date})")
            else:
                lines.append(f" - {node} = {state} (timeless)")
    else:
        lines.append(" (None)")
    
    lines.append("")
    lines.append(f"Evidence EXCLUDED (occurred after {query_date}):")
    
    if excluded:
        for item in excluded:
            lines.append(f" - {item}")
    else:
        lines.append(" (None)")
    
    return "\n".join(lines)


def apply_temporal_always_on_evidence(
    user_evidence: Dict[str, str],
    query_date: int
) -> Tuple[Dict[str, str], str]:
    """Apply always-on evidence with temporal filtering.
    
    This combines the always-on evidence system with temporal awareness:
    - Always-on facts are only applied if they existed at query time
    - Prevents anachronistic reasoning (2025 emails -> 2019 knowledge)
    
    Args:
        user_evidence: User-provided evidence
        query_date: Year of the query (e.g., 2019)
        
    Returns:
        Tuple of (final_evidence, explanation_text)
    """
    from .always_on_evidence import get_always_on_evidence
    
    # Get always-on evidence
    always_on = get_always_on_evidence()
    
    # Filter always-on by date
    filtered_always_on, excluded_always_on = filter_evidence_by_date(always_on, query_date)
    
    # Merge filtered always-on with user evidence
    final_evidence = dict(user_evidence)
    final_evidence.update(filtered_always_on)
    
    # Create explanation
    explanation_lines = [
        f"Temporal filtering for query date: {query_date}",
        "",
        "Always-on evidence INCLUDED (existed at query time):"
    ]
    
    if filtered_always_on:
        for node, state in filtered_always_on.items():
            explanation_lines.append(f" - {node} = {state}")
    else:
        explanation_lines.append(" (None - all always-on evidence is from after query date)")
    
    if excluded_always_on:
        explanation_lines.append("")
        explanation_lines.append("Always-on evidence EXCLUDED (occurred after query date):")
        for item in excluded_always_on:
            explanation_lines.append(f" - {item}")
    
    explanation = "\n".join(explanation_lines)
    
    return final_evidence, explanation


def create_temporal_node_mapping() -> Dict[str, Dict[str, str]]:
    """Create mapping of general nodes to year-specific versions.
    
    Returns:
        Dictionary mapping general nodes to year-specific versions
    """
    mapping = {}
    
    # Key events by year
    mapping["OGC_Email"] = {
        "2019": None, # No OGC emails in 2019
        "2020": None,
        "2021": None,
        "2022": None,
        "2023": None,
        "2024": None,
        "2025": "OGC", # OGC emails were in 2025
    }
    
    mapping["Statements"] = {
        "2019": None, # Statements not made in 2019
        "2020": None,
        "2021": None,
        "2022": None,
        "2023": None,
        "2024": None,
        "2025": "Statements", # Statements made in 2025
    }
    
    return mapping


def get_evidence_for_year(evidence_type: str, year: int) -> Optional[str]:
    """Check if evidence of given type existed in given year.
    
    Args:
        evidence_type: Type of evidence (e.g., "OGC_Email")
        year: Year to check
        
    Returns:
        Node ID if evidence existed that year, None otherwise
    """
    mapping = create_temporal_node_mapping()
    
    if evidence_type in mapping:
        year_map = mapping[evidence_type]
        return year_map.get(str(year))
    
    return None

