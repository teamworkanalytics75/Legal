#!/usr/bin/env python3
"""
Motion Document Analysis Module for CatBoost Feature Extraction.

This module provides functions to analyze motion documents and extract
features for CatBoost model training and prediction.
"""

import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import perfect outline features if available
try:
    outline_features_path = Path(__file__).parents[3] / "case_law_data" / "scripts" / "extract_perfect_outline_features.py"
    if outline_features_path.exists():
        sys.path.insert(0, str(outline_features_path.parent))
        from extract_perfect_outline_features import extract_perfect_outline_features
        OUTLINE_FEATURES_AVAILABLE = True
    else:
        OUTLINE_FEATURES_AVAILABLE = False
        logger.warning("Perfect outline features module not found - outline features will not be included")
except ImportError:
    OUTLINE_FEATURES_AVAILABLE = False
    logger.warning("Could not import perfect outline features - outline features will not be included")


PERSONAL_FACT_BLOCK_PATHS = [
    Path("writer_agents/outputs/case_insights.json"),
    Path("case_law_data/results/case_insights.json"),
]

PERSONAL_FACT_KEYWORDS: Dict[str, List[str]] = {
    "hk_allegation_defamation": [
        "statement 1",
        "statement 2",
        "defamatory",
        "harvard club",
    ],
    "hk_allegation_ccp_family": [
        "xi mingze",
        "xi jinping",
        "ccp",
        "photograph",
    ],
    "hk_allegation_competitor": [
        "weiqi zhang",
        "blue oak",
        "competitor",
    ],
    "hk_retaliation_events": [
        "april 19, 2019",
        "april 29, 2019",
        "april 2025",
        "republished",
    ],
    "ogc_email_1_threat": [
        "april 7, 2025",
        "reckless endangerment",
        "ogc",
    ],
    "ogc_email_2_non_response": [
        "april 18, 2025",
        "no response",
        "follow-up email",
    ],
    "ogc_email_3_meet_confer": [
        "august 11, 2025",
        "meet-and-confer",
        "local rule 7.1",
    ],
    "harvard_retaliation_events": [
        "republication",
        "harvard club",
        "ogc",
    ],
    "privacy_leak_events": [
        "personal information",
        "privacy leak",
    ],
    "safety_concerns": [
        "political persecution",
        "torture",
        "bodily harm",
    ],
}

PERSONAL_KEYWORD_GROUPS: Dict[str, List[str]] = {
    "personal_mentions_harvard": ["harvard", "ogc", "office of general counsel"],
    "personal_mentions_xi_family": ["xi mingze", "xi jinping", "xi family"],
    "personal_mentions_ccp": [
        "ccp",
        "communist party",
        "people's republic of china",
        "prc",
    ],
    "personal_mentions_timeline": [
        "april 7, 2025",
        "april 18, 2025",
        "august 11, 2025",
        "june 2, 2025",
    ],
}


def compute_draft_features(text: str) -> Dict[str, float]:
    """
    Compute draft features for CatBoost model.

    Args:
        text: The motion document text to analyze

    Returns:
        Dictionary of feature names and their computed values
    """
    if not text:
        return _get_default_features()

    text_lower = text.lower()

    features = {
        # Privacy-related features
        "mentions_privacy": float(text_lower.count("privacy")),
        "mentions_harassment": float(text_lower.count("harass")),
        "mentions_safety": float(text_lower.count("safety")),
        "mentions_retaliation": float(text_lower.count("retaliat")),

        # Citation features
        "citation_count": float(len(re.findall(r'\d+\s+[A-Z]\.\s*\d+', text))),

        # Content quality features
        "public_interest_mentions": float(text_lower.count("public interest")),
        "transparency_mentions": float(text_lower.count("transparency")),
        "privacy_harm_count": float(text_lower.count("harm")),

        # Document structure features
        "section_count": float(len(re.findall(r'^[IVX]+\.', text, re.MULTILINE))),
        "paragraph_count": float(len(text.split('\n\n'))),
        "word_count": float(len(text.split())),

        # Legal terminology features
        "legal_terms": float(sum([
            text_lower.count("motion"),
            text_lower.count("petition"),
            text_lower.count("complaint"),
            text_lower.count("affidavit"),
            text_lower.count("declaration")
        ])),

        # Argument strength indicators
        "supporting_evidence": float(text_lower.count("evidence")),
        "case_citations": float(text_lower.count(" v. ") + text_lower.count(" f. ")),
        "statutory_citations": float(len(re.findall(r'\d+\s+[A-Z]\.\s*C\.', text))),
    }

    # Normalize features by document length
    word_count = features["word_count"]
    if word_count > 0:
        for key in ["mentions_privacy", "mentions_harassment", "mentions_safety",
                   "mentions_retaliation", "public_interest_mentions",
                   "transparency_mentions", "privacy_harm_count", "legal_terms",
                   "supporting_evidence"]:
            features[key] = features[key] / word_count * 1000  # Per 1000 words

    # Add perfect outline features if available
    if OUTLINE_FEATURES_AVAILABLE:
        try:
            outline_features = extract_perfect_outline_features(text)
            # Convert all outline features to float
            for key, value in outline_features.items():
                features[key] = float(value)
            logger.debug("Added perfect outline features to draft features")
        except Exception as e:
            logger.warning(f"Could not extract perfect outline features: {e}")

    # Inject personal corpus coverage metrics
    personal_metrics = _compute_personal_corpus_features(text_lower, word_count)
    features.update(personal_metrics)

    return features


def _get_default_features() -> Dict[str, float]:
    """Return default feature values for empty or invalid text."""
    features = {
        "mentions_privacy": 0.0,
        "mentions_harassment": 0.0,
        "mentions_safety": 0.0,
        "mentions_retaliation": 0.0,
        "citation_count": 0.0,
        "public_interest_mentions": 0.0,
        "transparency_mentions": 0.0,
        "privacy_harm_count": 0.0,
        "section_count": 0.0,
        "paragraph_count": 0.0,
        "word_count": 0.0,
        "legal_terms": 0.0,
        "supporting_evidence": 0.0,
        "case_citations": 0.0,
        "statutory_citations": 0.0,
        "personal_fact_blocks_total": 0.0,
        "personal_fact_blocks_hit": 0.0,
        "personal_fact_block_hit_ratio": 0.0,
        "personal_mentions_harvard": 0.0,
        "personal_mentions_xi_family": 0.0,
        "personal_mentions_ccp": 0.0,
        "personal_mentions_timeline": 0.0,
    }

    # Add default outline features if available
    if OUTLINE_FEATURES_AVAILABLE:
        outline_defaults = {
            "transition_legal_standard_to_factual_background": 0.0,
            "has_bullet_points": 0.0,
            "enumeration_count": 0.0,
            "enumeration_density": 0.0,
            "max_enumeration_depth": 0.0,
            "enumeration_in_danger_safety": 0.0,
            "enumeration_in_privacy_harm": 0.0,
            "enumeration_in_protective_measures": 0.0,
            "danger_safety_position": 999.0,
            "balancing_test_position": 999.0,
            "has_legal_standard": 0.0,
            "has_factual_background": 0.0,
            "has_privacy_harm": 0.0,
            "has_danger_safety": 0.0,
            "has_balancing_test": 0.0,
            "has_protective_measures": 0.0,
            "total_sections": 0.0,
        }
        features.update(outline_defaults)

    for fact_key in PERSONAL_FACT_KEYWORDS:
        features[f"personal_fact_hit_{fact_key}"] = 0.0

    return features


@lru_cache(maxsize=1)
def _load_personal_fact_blocks() -> Dict[str, str]:
    """Load fact blocks generated from the personal corpus pipeline."""
    for path in PERSONAL_FACT_BLOCK_PATHS:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                fact_blocks = payload.get("fact_blocks", {})
                if fact_blocks:
                    logger.debug("Loaded %d fact blocks from %s", len(fact_blocks), path)
                    return {k: str(v) for k, v in fact_blocks.items()}
            except Exception as exc:
                logger.debug("Could not parse %s: %s", path, exc)
    return {}


def _compute_personal_corpus_features(text_lower: str, word_count: float) -> Dict[str, float]:
    """
    Compute coverage metrics that tie the draft back to the curated personal corpus.
    """
    features: Dict[str, float] = {
        "personal_fact_blocks_total": 0.0,
        "personal_fact_blocks_hit": 0.0,
        "personal_fact_block_hit_ratio": 0.0,
        "personal_mentions_harvard": 0.0,
        "personal_mentions_xi_family": 0.0,
        "personal_mentions_ccp": 0.0,
        "personal_mentions_timeline": 0.0,
    }

    # Keyword groups (normalized per 1k words)
    for feature_name, keywords in PERSONAL_KEYWORD_GROUPS.items():
        raw_hits = sum(text_lower.count(keyword) for keyword in keywords)
        features[feature_name] = (
            (raw_hits / word_count) * 1000 if word_count else float(raw_hits)
        )

    fact_blocks = _load_personal_fact_blocks()
    total_blocks = len(PERSONAL_FACT_KEYWORDS)
    features["personal_fact_blocks_total"] = float(total_blocks)

    if total_blocks == 0 or not fact_blocks:
        # Provide per-block defaults so CatBoost schema remains stable
        for fact_key in PERSONAL_FACT_KEYWORDS:
            features[f"personal_fact_hit_{fact_key}"] = 0.0
        return features

    hits = 0
    for fact_key, keywords in PERSONAL_FACT_KEYWORDS.items():
        hit = any(keyword in text_lower for keyword in keywords)
        if not hit and fact_key in fact_blocks:
            # Fall back to checking trimmed snippets if keywords not defined
            snippet = " ".join(fact_blocks[fact_key].lower().split()[:6])
            hit = snippet and snippet in text_lower
        if hit:
            hits += 1
        features[f"personal_fact_hit_{fact_key}"] = 1.0 if hit else 0.0

    features["personal_fact_blocks_hit"] = float(hits)
    features["personal_fact_block_hit_ratio"] = (
        hits / total_blocks if total_blocks else 0.0
    )
    return features


def analyze_motion_structure(text: str) -> Dict[str, Any]:
    """
    Analyze the structure of a motion document.

    Args:
        text: The motion document text

    Returns:
        Dictionary containing structure analysis results
    """
    if not text:
        return {"error": "Empty text provided"}

    analysis = {
        "has_introduction": bool(re.search(r'^I\.?\s*INTRODUCTION', text, re.MULTILINE | re.IGNORECASE)),
        "has_facts": bool(re.search(r'^II\.?\s*FACTS', text, re.MULTILINE | re.IGNORECASE)),
        "has_argument": bool(re.search(r'^III\.?\s*ARGUMENT', text, re.MULTILINE | re.IGNORECASE)),
        "has_conclusion": bool(re.search(r'^IV\.?\s*CONCLUSION', text, re.MULTILINE | re.IGNORECASE)),
        "section_count": len(re.findall(r'^[IVX]+\.', text, re.MULTILINE)),
        "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
        "citation_count": len(re.findall(r'\d+\s+[A-Z]\.\s*\d+', text)),
    }

    return analysis


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from motion document.

    Args:
        text: The motion document text
        max_phrases: Maximum number of phrases to return

    Returns:
        List of key phrases
    """
    if not text:
        return []

    # Simple phrase extraction based on common legal patterns
    phrases = []

    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', text)
    phrases.extend(quoted)

    # Extract phrases with legal significance
    legal_patterns = [
        r'privacy\s+harm[s]?',
        r'personal\s+information',
        r'public\s+interest',
        r'compelling\s+interest',
        r'strict\s+scrutiny',
        r'reasonable\s+expectation',
    ]

    for pattern in legal_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        phrases.extend(matches)

    # Remove duplicates and limit length
    unique_phrases = list(dict.fromkeys(phrases))
    return unique_phrases[:max_phrases]


def validate_motion_format(text: str) -> Dict[str, Any]:
    """
    Validate motion document format and completeness.

    Args:
        text: The motion document text

    Returns:
        Dictionary containing validation results
    """
    if not text:
        return {"valid": False, "errors": ["Empty document"]}

    errors = []
    warnings = []

    # Check for required sections
    required_sections = ["INTRODUCTION", "ARGUMENT", "CONCLUSION"]
    for section in required_sections:
        if not re.search(f'^[IVX]+\.?\s*{section}', text, re.MULTILINE | re.IGNORECASE):
            errors.append(f"Missing required section: {section}")

    # Check for minimum length
    word_count = len(text.split())
    if word_count < 100:
        warnings.append(f"Document very short: {word_count} words")
    elif word_count < 500:
        warnings.append(f"Document short: {word_count} words")

    # Check for citations
    citation_count = len(re.findall(r'\d+\s+[A-Z]\.\s*\d+', text))
    if citation_count == 0:
        warnings.append("No case citations found")
    elif citation_count < 3:
        warnings.append(f"Few citations: {citation_count}")

    # Check for legal terminology
    legal_terms = ["motion", "petition", "relief", "court", "plaintiff", "defendant"]
    found_terms = sum(1 for term in legal_terms if term in text.lower())
    if found_terms < 3:
        warnings.append("Limited legal terminology")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "word_count": word_count,
        "citation_count": citation_count,
        "legal_term_count": found_terms
    }
