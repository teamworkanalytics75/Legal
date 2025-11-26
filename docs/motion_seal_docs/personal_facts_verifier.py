"""
Utilities for verifying that generated motions reference personal corpus facts.

This module exposes a single entry point,
`verify_motion_uses_personal_facts`, which scans a motion for mandatory
lawsuit-specific references (HK Statement, OGC emails, critical dates, and
allegations) and reports which facts were found or missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple

try:
    from writer_agents.code.validation.fact_graph_query import FactGraphQuery
except Exception:  # pragma: no cover - optional dependency
    FactGraphQuery = None  # type: ignore[misc]

logger = logging.getLogger(__name__)


@dataclass
class FactRule:
    """Represents a single fact requirement with compiled regex patterns."""

    name: str
    description: str
    patterns: Iterable[str]
    optional: bool = False
    is_negative: bool = False
    compiled_patterns: List[Pattern[str]] = field(init=False)

    def __post_init__(self) -> None:
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.patterns
        ]

    def match(
        self,
        text: str,
        original_text: str,
        aliases: Optional[Iterable[str]] = None,
    ) -> Optional[Tuple[str, bool]]:
        """Return snippet if rule satisfied.

        Returns (snippet, is_violation) where is_violation indicates a negative rule hit.
        """
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                start, end = match.span()
                snippet = original_text[start:end]
                return snippet, self.is_negative

        if aliases:
            for alias in aliases:
                if not alias:
                    continue
                alias_pattern = re.compile(re.escape(alias), re.IGNORECASE)
                match = alias_pattern.search(original_text)
                if match:
                    start, end = match.span()
                    snippet = original_text[start:end]
                    return snippet, self.is_negative
        return None


DEFAULT_FACT_RULES: Tuple[FactRule, ...] = (
    FactRule(
        name="hk_statement",
        description="References the Hong Kong Statement of Claim",
        patterns=(r"hong\s+kong\s+statement\s+of\s+claim", r"\bhk\s+statement\b"),
    ),
    FactRule(
        name="ogc_emails",
        description="References the Harvard OGC emails / Office of General Counsel",
        patterns=(r"\bogc\b", r"office\s+of\s+general\s+counsel"),
    ),
    FactRule(
        name="date_april_7_2025",
        description="Mentions April 7, 2025 OGC notice",
        patterns=(r"april\s+7,\s*2025", r"7\s+april\s+2025"),
    ),
    FactRule(
        name="date_april_18_2025",
        description="Mentions April 18, 2025 OGC follow-up",
        patterns=(r"april\s+18,\s*2025", r"18\s+april\s+2025"),
    ),
    FactRule(
        name="date_june_2_2025",
        description="Mentions June 2, 2025 HK filing",
        patterns=(r"june\s+2,\s*2025", r"2\s+june\s+2025"),
    ),
    FactRule(
        name="date_june_4_2025",
        description="Mentions June 4, 2025 arrests / threats",
        patterns=(r"june\s+4,\s*2025", r"4\s+june\s+2025"),
    ),
    FactRule(
        name="allegation_defamation",
        description="Explains defamation allegation",
        patterns=(r"\bdefamation\b", r"\bdefamatory\b"),
    ),
    FactRule(
        name="allegation_privacy_breach",
        description="Describes privacy breach",
        patterns=(r"privacy\s+breach", r"\bprivacy\b"),
    ),
    FactRule(
        name="allegation_retaliation",
        description="Describes retaliation",
        patterns=(r"\bretaliation\b", r"\bretaliatory\b"),
    ),
    FactRule(
        name="allegation_harassment",
        description="Describes harassment",
        patterns=(r"\bharassment\b", r"\bharassing\b"),
    ),
    FactRule(
        name="timeline_april_ogc_emails",
        description="Timeline reference to April 2025 OGC emails",
        patterns=(
            r"april\s+2025[^.]{0,120}ogc",
            r"ogc[^.]{0,120}april\s+2025",
            r"april\s+2025[^.]{0,120}office\s+of\s+general\s+counsel",
        ),
    ),
    FactRule(
        name="timeline_june_2025_arrests",
        description="Timeline reference to June 2025 arrests or threats",
        patterns=(
            r"june\s+2025[^.]{0,120}arrest",
            r"arrest[^.]{0,120}june\s+2025",
            r"june\s+2025[^.]{0,120}detention",
        ),
    ),
)

RULE_TO_GRAPH_TYPE: Dict[str, str] = {
    "hk_statement": "document_reference",
    "ogc_emails": "organization",
    "date_april_7_2025": "date",
    "date_april_18_2025": "date",
    "date_june_2_2025": "date",
    "date_june_4_2025": "date",
    "allegation_defamation": "allegation",
    "allegation_privacy_breach": "allegation",
    "allegation_retaliation": "allegation",
    "allegation_harassment": "allegation",
    "timeline_april_ogc_emails": "timeline_event",
    "timeline_june_2025_arrests": "timeline_event",
}

RULE_TO_GRAPH_TYPE: Dict[str, str] = {
    "hk_statement": "document_reference",
    "ogc_emails": "organization",
    "date_april_7_2025": "date",
    "date_april_18_2025": "date",
    "date_june_2_2025": "date",
    "date_june_4_2025": "date",
    "allegation_defamation": "allegation",
    "allegation_privacy_breach": "allegation",
    "allegation_retaliation": "allegation",
    "allegation_harassment": "allegation",
    "timeline_april_ogc_emails": "timeline_event",
    "timeline_june_2025_arrests": "timeline_event",
}


def _gather_aliases(personal_corpus_facts: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return aliases per rule name derived from user-provided data."""
    aliases: Dict[str, List[str]] = {}
    provided_aliases = personal_corpus_facts.get("aliases", {})
    if isinstance(provided_aliases, dict):
        for key, values in provided_aliases.items():
            if isinstance(values, (list, tuple, set)):
                aliases[key] = [str(value) for value in values if value]
            elif isinstance(values, str):
                aliases[key] = [values]

    # If case_insights style dict passed, use fact_blocks text to enhance
    fact_blocks = personal_corpus_facts.get("fact_blocks")
    if isinstance(fact_blocks, dict):
        hk_text = fact_blocks.get("hk_retaliation_events") or fact_blocks.get("hk_allegation_defamation")
        if hk_text:
            aliases.setdefault("hk_statement", []).append(hk_text[:120])
        ogc_text = fact_blocks.get("ogc_email_1_threat")
        if ogc_text:
            aliases.setdefault("ogc_emails", []).append(ogc_text[:120])

    return aliases


def _normalize_motion_text(motion_text: str) -> Tuple[str, str]:
    if not isinstance(motion_text, str):
        raise TypeError("motion_text must be a string")
    stripped = motion_text.strip()
    return stripped, stripped.lower()


def _extract_aliases_from_case_insights(case_insights: Dict[str, Any]) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    fact_blocks = case_insights.get("fact_blocks") or {}
    case_summary = case_insights.get("case_summary", "")

    action_pattern = re.compile(r"action\s+no\.?\s*\d+", re.IGNORECASE)
    hk_phrase_pattern = re.compile(r"(hk|hong\s+kong)\s+statement", re.IGNORECASE)

    def _append_alias(key: str, value: str) -> None:
        normalized = value.strip()
        if not normalized:
            return
        aliases.setdefault(key, [])
        if normalized not in aliases[key]:
            aliases[key].append(normalized[:240])

    def add_alias(key: str, text: Optional[str]) -> None:
        if not text:
            return
        snippet = text.strip()
        if not snippet:
            return
        _append_alias(key, snippet)
        for candidate in re.split(r"[.;]", snippet):
            candidate = candidate.strip()
            if 5 <= len(candidate) <= 160:
                _append_alias(key, candidate)
        match = action_pattern.search(snippet)
        if match:
            _append_alias(key, match.group(0))
        match = hk_phrase_pattern.search(snippet)
        if match:
            _append_alias(key, match.group(0))

    add_alias("hk_statement", fact_blocks.get("hk_allegation_defamation"))
    add_alias("hk_statement", fact_blocks.get("hk_retaliation_events"))
    add_alias("ogc_emails", fact_blocks.get("ogc_email_1_threat"))
    add_alias("ogc_emails", fact_blocks.get("ogc_email_2_non_response"))
    add_alias("ogc_emails", fact_blocks.get("ogc_email_3_meet_confer"))
    add_alias("timeline_april_ogc_emails", fact_blocks.get("ogc_email_1_threat"))
    add_alias("timeline_april_ogc_emails", fact_blocks.get("ogc_email_2_non_response"))
    add_alias("timeline_june_2025_arrests", fact_blocks.get("safety_concerns"))
    add_alias("timeline_june_2025_arrests", fact_blocks.get("harvard_retaliation_events"))

    if "Hong Kong Statement of Claim" in case_summary:
        add_alias("hk_statement", "Hong Kong Statement of Claim")
    if "Action No. 771" in case_summary:
        add_alias("hk_statement", "Action No. 771")
    if "Office of General Counsel" in case_summary:
        add_alias("ogc_emails", "Office of General Counsel")

    return aliases


def _aliases_from_corpus_dir(corpus_dir: Optional[Path]) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    if not corpus_dir:
        return aliases

    try:
        if not corpus_dir.exists():
            return aliases
    except OSError:
        return aliases

    for path in sorted(corpus_dir.glob("*.txt")):
        stem = path.stem.strip()
        if not stem:
            continue
        lowered = stem.lower()
        if any(keyword in lowered for keyword in ("statement", "claim")):
            aliases.setdefault("hk_statement", []).append(stem)
        if "ogc" in lowered or "email" in lowered:
            aliases.setdefault("ogc_emails", []).append(stem)
            aliases.setdefault("timeline_april_ogc_emails", []).append(stem)
        if "june" in lowered and ("arrest" in lowered or "detention" in lowered):
            aliases.setdefault("timeline_june_2025_arrests", []).append(stem)
    return aliases


def verify_motion_uses_personal_facts(
    motion_text: str,
    personal_corpus_facts: Optional[Dict[str, Any]] = None,
    required_rules: Optional[Iterable[FactRule]] = None,
    negative_rules: Optional[Iterable[FactRule]] = None,
    fact_graph_query: Optional["FactGraphQuery"] = None,
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Verify that the motion references lawsuit-specific personal facts.

    Args:
        motion_text: Generated motion text to inspect.
        personal_corpus_facts: Optional dict with metadata (case_insights,
            alias overrides, etc.).
        required_rules: Optional iterable of FactRule overrides for advanced use.

    Returns:
        Tuple of (is_valid, missing_fact_names, verification_details)
    """
    personal_corpus_facts = personal_corpus_facts or {}
    rules = tuple(required_rules) if required_rules else DEFAULT_FACT_RULES
    original_text, normalized_text = _normalize_motion_text(motion_text)
    aliases_map = _gather_aliases(personal_corpus_facts)

    matches: Dict[str, str] = {}
    evaluated_rules: List[Dict[str, Any]] = []
    missing: List[str] = []
    violations: List[str] = []
    negative_rules = tuple(negative_rules) if negative_rules else NEGATIVE_FACT_RULES

    motion_lower = original_text.lower()
    graph_fact_coverage: Dict[str, Any] = {}
    graph_missing_pairs: List[Dict[str, str]] = []
    graph_suggestions: List[str] = []

    graph_summary: Optional[Dict[str, Any]] = None
    if fact_graph_query and FactGraphQuery:
        graph_fact_coverage = _graph_fact_coverage(motion_lower, fact_graph_query)
        if graph_fact_coverage:
            graph_summary = _summarize_graph_coverage(graph_fact_coverage)
            graph_missing_pairs = graph_summary.get("missing_facts", [])
            graph_suggestions = graph_summary.get("suggestions", [])

    for rule in rules:
        alias_values = aliases_map.get(rule.name, [])
        match_result = rule.match(normalized_text, original_text, alias_values)
        matched = match_result is not None and not (match_result and match_result[1])
        graph_snippet: Optional[str] = None
        if not matched and fact_graph_query and FactGraphQuery:
            graph_snippet = _match_fact_via_graph(rule.name, motion_lower, fact_graph_query)
            matched = bool(graph_snippet)
        evaluated_rules.append(
            {
                "name": rule.name,
                "description": rule.description,
                "matched": matched,
                "optional": rule.optional,
                "alias_count": len(alias_values),
            }
        )
        if match_result:
            snippet, _ = match_result
            matches[rule.name] = snippet
        elif graph_snippet:
            matches[rule.name] = graph_snippet
        elif not rule.optional:
            missing.append(rule.name)

    # Evaluate negative rules (violations)
    evaluated_negative: List[Dict[str, Any]] = []
    for rule in negative_rules:
        match_result = rule.match(normalized_text, original_text, aliases_map.get(rule.name))
        violation = bool(match_result)
        evaluated_negative.append(
            {
                "name": rule.name,
                "description": rule.description,
                "violated": violation,
            }
        )
        if violation:
            snippet, _ = match_result  # Negative rules always treat match as violation
            violations.append(rule.name)
            matches[rule.name] = snippet

    total_required = len([rule for rule in rules if not rule.optional])
    coverage = (total_required - len(missing)) / total_required if total_required else 1.0

    details: Dict[str, Any] = {
        "matches": matches,
        "evaluated_rules": evaluated_rules,
        "negative_rules": evaluated_negative,
        "coverage": round(coverage, 4),
        "total_required": total_required,
        "violations": violations,
    }
    if graph_fact_coverage:
        details["graph_fact_coverage"] = graph_fact_coverage
        details["graph_missing_facts"] = graph_missing_pairs
        if graph_summary is None:
            graph_summary = _summarize_graph_coverage(graph_fact_coverage)
        details["graph_coverage_summary"] = graph_summary

    is_valid = len(missing) == 0 and not violations
    return is_valid, missing, violations, details


def _match_fact_via_graph(rule_name: str, motion_lower: str, fact_query: "FactGraphQuery") -> Optional[str]:
    graph_type = RULE_TO_GRAPH_TYPE.get(rule_name)
    if not graph_type:
        return None
    try:
        candidates = fact_query.get_all_facts_by_type(graph_type)
    except Exception as exc:  # pragma: no cover - defensive
        logger = logging.getLogger(__name__)
        logger.debug("KnowledgeGraph lookup failed for %s: %s", rule_name, exc)
        candidates = []

    for candidate in candidates:
        value = (candidate.get("value") or candidate.get("fact_value") or "").strip()
        if value and value.lower() in motion_lower:
            return value

    semantic_match = _match_fact_semantically(motion_lower, graph_type, fact_query)
    return semantic_match


def _match_fact_semantically(
    motion_text: str,
    fact_type: str,
    fact_query: "FactGraphQuery",
) -> Optional[str]:
    try:
        matches = fact_query.find_similar_facts(
            motion_text,
            fact_type=fact_type,
            top_k=1,
            similarity_threshold=0.4,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger = logging.getLogger(__name__)
        logger.debug("KnowledgeGraph semantic match failed for %s: %s", fact_type, exc)
        return None
    if matches:
        return matches[0].fact_value
    return None


def _graph_fact_coverage(
    motion_lower: str,
    fact_query: "FactGraphQuery",
) -> Dict[str, Dict[str, Any]]:
    coverage: Dict[str, Dict[str, Any]] = {}
    try:
        hierarchy = fact_query.get_fact_hierarchy()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("KnowledgeGraph hierarchy lookup failed: %s", exc)
        return coverage

    for fact_type, values in hierarchy.items():
        normalized_values = [
            str(value).strip()
            for value in values
            if isinstance(value, str) and str(value).strip()
        ]
        if not normalized_values:
            continue
        present: List[str] = []
        missing: List[str] = []
        for value in normalized_values:
            token = value.lower()
            if token and token in motion_lower:
                present.append(value)
            else:
                missing.append(value)
        coverage[fact_type] = {
            "total": len(normalized_values),
            "present": present,
            "missing": missing,
            "coverage": round(len(present) / len(normalized_values), 4),
        }
    return coverage


def _summarize_graph_coverage(
    graph_coverage: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_types": len(graph_coverage),
        "missing_facts": [],
        "suggestions": [],
    }
    if not graph_coverage:
        summary["average_coverage"] = 0.0
        return summary

    coverage_values = [section.get("coverage", 0.0) for section in graph_coverage.values()]
    summary["average_coverage"] = round(sum(coverage_values) / len(coverage_values), 4)

    missing_entries: List[Dict[str, str]] = []
    suggestions: List[str] = []
    for fact_type, section in graph_coverage.items():
        for value in section.get("missing", [])[:3]:
            entry = {"fact_type": fact_type, "value": value}
            missing_entries.append(entry)
            if len(suggestions) < 5:
                suggestions.append(f"Add detail for {value} ({fact_type})")
    summary["missing_facts"] = missing_entries
    summary["suggestions"] = suggestions
    return summary


def verify_motion_with_case_insights(
    motion_text: str,
    case_insights_path: Optional[Path] = None,
    corpus_dir: Optional[Path] = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Convenience wrapper that loads lawsuit_facts_extracted.json and extracts aliases automatically.

    Args:
        motion_text: Generated motion text to verify.
        case_insights_path: Optional path to lawsuit_facts_extracted.json
            (defaults to writer_agents/outputs/lawsuit_facts_extracted.json, also supports case_insights.json for backward compatibility).
        corpus_dir: Personal corpus directory (defaults to case_law_data/lawsuit_source_documents).

    Returns:
        Tuple of (is_valid, missing_fact_names, verification_details).
    """
    # Try new name first, then old name for backward compatibility
    default_path = Path("writer_agents/outputs/lawsuit_facts_extracted.json")
    if not case_insights_path:
        case_insights_path = default_path if default_path.exists() else Path("writer_agents/outputs/case_insights.json")
    elif not case_insights_path.exists() and case_insights_path.name == "case_insights.json":
        # If old name provided but doesn't exist, try new name
        new_path = case_insights_path.parent / "lawsuit_facts_extracted.json"
        if new_path.exists():
            case_insights_path = new_path
    if not case_insights_path.exists():
        raise FileNotFoundError(f"Case insights file not found: {case_insights_path}")

    try:
        case_insights = json.loads(case_insights_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse case insights JSON: {case_insights_path}") from exc

    personal_data: Dict[str, Any] = {
        "fact_blocks": case_insights.get("fact_blocks") or {},
    }

    alias_map = _extract_aliases_from_case_insights(case_insights)
    default_corpus = Path("case_law_data/lawsuit_source_documents")
    old_corpus = Path("case_law_data/tmp_corpus")
    corpus_path = corpus_dir or (default_corpus if default_corpus.exists() else old_corpus)
    corpus_aliases = _aliases_from_corpus_dir(corpus_path)
    for key, values in corpus_aliases.items():
        alias_map.setdefault(key, [])
        for value in values:
            if value not in alias_map[key]:
                alias_map[key].append(value)
    if alias_map:
        personal_data["aliases"] = alias_map

    return verify_motion_uses_personal_facts(motion_text, personal_data)


__all__ = [
    "verify_motion_uses_personal_facts",
    "verify_motion_with_case_insights",
    "FactRule",
    "DEFAULT_FACT_RULES",
]
NEGATIVE_FACT_RULES: Tuple[FactRule, ...] = (
    FactRule(
        name="not_prc_citizen",
        description="Must NOT claim PRC citizenship when source documents state US citizenship",
        patterns=(
            r"home\s+country\s+of\s+prc",
            r"prc\s+citizen",
            r"citizen\s+of\s+prc",
            r"prc\s+national",
            r"national\s+of\s+prc",
        ),
        is_negative=True,
    ),
    # Example extensibility: forbid fabricated courthouse locations
    FactRule(
        name="not_wrong_court_location",
        description="Must NOT relocate the courthouse/case venue to an unsupported city",
        patterns=(
            r"district\s+of\s+hong\s+kong",
            r"beijing\s+district\s+court",
        ),
        is_negative=True,
    ),
)
