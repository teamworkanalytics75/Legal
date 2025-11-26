#!/usr/bin/env python3
"""
Feature-to-Rule Translation Engine.

Consumes CatBoost top features and generates rule specifications
for atomic SK plugins.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATTERNS_FILE = Path(__file__).parent / "granted_patterns.jsonl"
SUMMARY_FILE = Path(__file__).parent / "pattern_summary.json"
RULES_DIR = Path(__file__).parents[1] / "sk_plugins" / "rules"
WINNING_CITATIONS_FILE = PROJECT_ROOT / "case_law_data" / "analysis" / "winning_citations_federal.json"


def load_patterns() -> List[Dict[str, Any]]:
    """Load extracted patterns from granted cases."""
    patterns = []
    if PATTERNS_FILE.exists():
        with open(PATTERNS_FILE, 'r') as f:
            for line in f:
                patterns.append(json.loads(line.strip()))
    return patterns


def load_summary() -> Dict[str, Any]:
    """Load pattern summary statistics."""
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, 'r') as f:
            return json.load(f)
    return {}


def load_winning_citations() -> List[Dict[str, Any]]:
    """Load winning citations database."""
    if WINNING_CITATIONS_FILE.exists():
        with open(WINNING_CITATIONS_FILE, 'r') as f:
            return json.load(f)
    return []


def calculate_feature_thresholds(patterns: List[Dict], shap_importance: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Calculate thresholds and averages for each feature."""
    feature_stats = {}

    # Group patterns by feature
    for feature_name, importance in shap_importance.items():
        if importance < 0.01:  # Skip low-importance features
            continue

        values = []
        for pattern in patterns:
            if feature_name in pattern.get("shap_scores", {}):
                values.append(pattern["shap_scores"][feature_name])

        if values:
            feature_stats[feature_name] = {
                "shap_importance": importance,
                "minimum_threshold": max(1, int(np.percentile(values, 25))),  # 25th percentile
                "successful_case_average": float(np.mean(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
                "count": len(values)
            }

    return feature_stats


def generate_privacy_rules(feature_stats: Dict, patterns: List[Dict]) -> Dict[str, Any]:
    """Generate rules for privacy mentions feature."""
    privacy_stats = feature_stats.get("mentions_privacy", {})

    # Extract privacy-related phrases from patterns
    privacy_phrases = []
    for pattern in patterns:
        if pattern.get("harm_mentions", {}).get("privacy", 0) > 0:
            # In practice, would extract actual phrases from text
            privacy_phrases.extend([
                "privacy harm",
                "personal information disclosure",
                "expectation of privacy",
                "privacy interest",
                "confidential information"
            ])

    return {
        "feature_name": "mentions_privacy",
        "shap_importance": privacy_stats.get("shap_importance", 0.0),
        "minimum_threshold": privacy_stats.get("minimum_threshold", 3),
        "successful_case_average": privacy_stats.get("successful_case_average", 5.0),
        "recommended_phrases": list(set(privacy_phrases))[:10],
        "chroma_query_template": "privacy harm personal information disclosure {case_context}",
        "validation_criteria": {
            "min_mentions": privacy_stats.get("minimum_threshold", 3),
            "required_context": ["harm", "disclosure", "expectation"],
            "target_average": privacy_stats.get("successful_case_average", 5.0)
        }
    }


def generate_harassment_rules(feature_stats: Dict, patterns: List[Dict]) -> Dict[str, Any]:
    """Generate rules for harassment mentions feature."""
    harassment_stats = feature_stats.get("mentions_harassment", {})

    return {
        "feature_name": "mentions_harassment",
        "shap_importance": harassment_stats.get("shap_importance", 0.0),
        "minimum_threshold": harassment_stats.get("minimum_threshold", 1),
        "successful_case_average": harassment_stats.get("successful_case_average", 2.0),
        "recommended_phrases": [
            "harassment",
            "retaliation",
            "intimidation",
            "adverse action",
            "reprisal"
        ],
        "chroma_query_template": "harassment retaliation intimidation {case_context}",
        "validation_criteria": {
            "min_mentions": harassment_stats.get("minimum_threshold", 1),
            "required_context": ["harassment", "retaliation", "causation"],
            "target_average": harassment_stats.get("successful_case_average", 2.0)
        }
    }


def generate_safety_rules(feature_stats: Dict, patterns: List[Dict]) -> Dict[str, Any]:
    """Generate rules for safety mentions feature."""
    safety_stats = feature_stats.get("mentions_safety", {})

    return {
        "feature_name": "mentions_safety",
        "shap_importance": safety_stats.get("shap_importance", 0.0),
        "minimum_threshold": safety_stats.get("minimum_threshold", 1),
        "successful_case_average": safety_stats.get("successful_case_average", 1.5),
        "recommended_phrases": [
            "safety concerns",
            "danger",
            "threats",
            "physical harm",
            "security risk"
        ],
        "chroma_query_template": "safety concerns danger threats {case_context}",
        "validation_criteria": {
            "min_mentions": safety_stats.get("minimum_threshold", 1),
            "required_context": ["safety", "danger", "harm"],
            "target_average": safety_stats.get("successful_case_average", 1.5)
        }
    }


def generate_retaliation_rules(feature_stats: Dict, patterns: List[Dict]) -> Dict[str, Any]:
    """Generate rules for retaliation mentions feature."""
    retaliation_stats = feature_stats.get("mentions_retaliation", {})

    return {
        "feature_name": "mentions_retaliation",
        "shap_importance": retaliation_stats.get("shap_importance", 0.0),
        "minimum_threshold": retaliation_stats.get("minimum_threshold", 1),
        "successful_case_average": retaliation_stats.get("successful_case_average", 1.2),
        "recommended_phrases": [
            "retaliation",
            "adverse action",
            "reprisal",
            "retaliatory conduct",
            "punitive action"
        ],
        "chroma_query_template": "retaliation adverse action reprisal {case_context}",
        "validation_criteria": {
            "min_mentions": retaliation_stats.get("minimum_threshold", 1),
            "required_context": ["retaliation", "adverse", "causal"],
            "target_average": retaliation_stats.get("successful_case_average", 1.2)
        }
    }


def generate_citation_rules(feature_stats: Dict, patterns: List[Dict], winning_citations: List[Dict]) -> Dict[str, Any]:
    """Generate rules for citation requirements."""
    citation_stats = feature_stats.get("citation_count", {})

    # Extract section-specific citation patterns
    section_patterns = {}
    for pattern in patterns:
        for section, density in pattern.get("citation_density", {}).items():
            if section not in section_patterns:
                section_patterns[section] = []
            section_patterns[section].append(density)

    # Calculate averages per section
    section_requirements = {}
    for section, densities in section_patterns.items():
        if densities:
            section_requirements[section] = {
                "min_citations": max(1, int(np.percentile(densities, 25))),
                "density_per_100_words": float(np.mean(densities)),
                "required_authorities": [cite["citation"] for cite in winning_citations[:5]]
            }

    return {
        "feature_name": "citation_count",
        "shap_importance": citation_stats.get("shap_importance", 0.0),
        "minimum_threshold": citation_stats.get("minimum_threshold", 3),
        "successful_case_average": citation_stats.get("successful_case_average", 5.0),
        "sections": section_requirements,
        "winning_citations": winning_citations[:20],
        "validation_criteria": {
            "min_total_citations": citation_stats.get("minimum_threshold", 3),
            "required_winning_cites": 2,
            "target_average": citation_stats.get("successful_case_average", 5.0)
        }
    }


def generate_harm_count_rules(feature_stats: Dict, patterns: List[Dict]) -> Dict[str, Any]:
    """Generate rules for privacy harm count feature."""
    harm_count_stats = feature_stats.get("privacy_harm_count", {})

    return {
        "feature_name": "privacy_harm_count",
        "shap_importance": harm_count_stats.get("shap_importance", 0.0),
        "minimum_threshold": harm_count_stats.get("minimum_threshold", 2),
        "successful_case_average": harm_count_stats.get("successful_case_average", 3.0),
        "harm_types": ["privacy", "harassment", "safety", "retaliation"],
        "validation_criteria": {
            "min_harm_types": harm_count_stats.get("minimum_threshold", 2),
            "required_types": ["privacy"],
            "target_average": harm_count_stats.get("successful_case_average", 3.0)
        }
    }


def generate_public_interest_rules(feature_stats: Dict, patterns: List[Dict]) -> Dict[str, Any]:
    """Generate rules for public interest feature."""
    public_interest_stats = feature_stats.get("mentions_public_interest", {})

    return {
        "feature_name": "mentions_public_interest",
        "shap_importance": public_interest_stats.get("shap_importance", 0.0),
        "minimum_threshold": public_interest_stats.get("minimum_threshold", 1),
        "successful_case_average": public_interest_stats.get("successful_case_average", 2.0),
        "recommended_phrases": [
            "limited public interest",
            "minimal public interest",
            "no public interest",
            "private matter",
            "personal dispute"
        ],
        "chroma_query_template": "limited public interest {case_context}",
        "validation_criteria": {
            "min_mentions": public_interest_stats.get("minimum_threshold", 1),
            "required_context": ["limited", "minimal", "private"],
            "target_average": public_interest_stats.get("successful_case_average", 2.0)
        }
    }


def generate_transparency_rules(feature_stats: Dict, patterns: List[Dict]) -> Dict[str, Any]:
    """Generate rules for transparency feature."""
    transparency_stats = feature_stats.get("mentions_transparency", {})

    return {
        "feature_name": "mentions_transparency",
        "shap_importance": transparency_stats.get("shap_importance", 0.0),
        "minimum_threshold": transparency_stats.get("minimum_threshold", 1),
        "successful_case_average": transparency_stats.get("successful_case_average", 1.5),
        "recommended_phrases": [
            "court transparency",
            "public access",
            "open court",
            "public interest",
            "First Amendment"
        ],
        "chroma_query_template": "court transparency public access {case_context}",
        "validation_criteria": {
            "min_mentions": transparency_stats.get("minimum_threshold", 1),
            "required_context": ["transparency", "public", "access"],
            "target_average": transparency_stats.get("successful_case_average", 1.5)
        }
    }


def generate_section_structure_rules(patterns: List[Dict]) -> Dict[str, Any]:
    """Generate rules for section structure."""
    section_frequency = Counter()

    for pattern in patterns:
        for section in pattern.get("section_structure", []):
            section_frequency[section] += 1

    # Define required section order
    required_sections = [
        "INTRODUCTION",
        "PRIVACY HARM ANALYSIS",
        "LEGAL STANDARD",
        "ARGUMENT",
        "CONCLUSION"
    ]

    return {
        "feature_name": "section_structure",
        "required_sections": required_sections,
        "section_frequency": dict(section_frequency.most_common(10)),
        "min_sections": 3,
        "validation_criteria": {
            "required_order": required_sections,
            "min_sections": 3,
            "must_include": ["INTRODUCTION", "ARGUMENT"]
        }
    }


def translate_features_to_rules() -> None:
    """Main function to translate CatBoost features to rule configs."""
    logger.info("Starting feature-to-rule translation...")

    # Load data
    patterns = load_patterns()
    summary = load_summary()
    winning_citations = load_winning_citations()

    if not patterns:
        logger.error("No patterns found. Run audit_catboost_patterns.py first.")
        return

    logger.info(f"Loaded {len(patterns)} patterns and {len(winning_citations)} winning citations")

    # Get SHAP importance from summary
    shap_importance = summary.get("top_features", {})

    # Calculate feature thresholds
    feature_stats = calculate_feature_thresholds(patterns, shap_importance)

    # Create rules directory
    RULES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate individual feature rules
    rule_generators = {
        "mentions_privacy_rules.json": generate_privacy_rules,
        "mentions_harassment_rules.json": generate_harassment_rules,
        "mentions_safety_rules.json": generate_safety_rules,
        "mentions_retaliation_rules.json": generate_retaliation_rules,
        "citation_requirements.json": generate_citation_rules,
        "harm_thresholds.json": generate_harm_count_rules,
        "public_interest_rules.json": generate_public_interest_rules,
        "transparency_rules.json": generate_transparency_rules,
        "section_structure.json": generate_section_structure_rules
    }

    for filename, generator in rule_generators.items():
        if filename == "citation_requirements.json":
            rules = generator(feature_stats, patterns, winning_citations)
        elif filename == "section_structure.json":
            rules = generator(patterns)
        else:
            rules = generator(feature_stats, patterns)

        # Save rules
        rules_path = RULES_DIR / filename
        with open(rules_path, 'w') as f:
            json.dump(rules, f, indent=2)

        logger.info(f"Generated rules: {rules_path}")

    logger.info(f"Generated {len(rule_generators)} rule files in {RULES_DIR}")


if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    translate_features_to_rules()
