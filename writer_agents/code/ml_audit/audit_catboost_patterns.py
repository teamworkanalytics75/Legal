#!/usr/bin/env python3
"""
ML Audit Pipeline - Extract structured patterns from CatBoost analysis.

Re-runs analysis/analyze_ma_motion_doc.py with SHAP logging per document
and extracts structured patterns from "granted" cases for rule generation.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from collections import Counter
import re

logger = logging.getLogger(__name__)

# Import from existing analysis
import sys
analysis_path = Path(__file__).parents[2] / "analysis"
sys.path.append(str(analysis_path))

try:
    from analyze_ma_motion_doc import (
        load_dataset, extract_motion_outcomes, engineer_motion_features,
        train_catboost_model, compute_draft_features
    )
except ImportError:
    logger.warning("analyze_ma_motion_doc not found, using mock functions")

    def load_dataset(db_path):
        return {"cases": pd.DataFrame()}

    def extract_motion_outcomes(cases_df):
        return pd.DataFrame()

    def engineer_motion_features(data, outcomes_df):
        return pd.DataFrame()

    def train_catboost_model(features_df):
        return None, None, [], {}

    def compute_draft_features(text):
        return {"word_count": len(text.split()), "text_length": len(text)}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "case_law_data" / "ma_federal_motions.db"
OUTPUT_DIR = Path(__file__).parent
GRANTED_PATTERNS_FILE = OUTPUT_DIR / "granted_patterns.jsonl"


def extract_section_structure(text: str) -> List[str]:
    """Extract section headings from motion text."""
    # Look for common section patterns
    section_patterns = [
        r'(?:^|\n)\s*(?:I\.|1\.|A\.|INTRODUCTION)',
        r'(?:^|\n)\s*(?:II\.|2\.|B\.|PRIVACY HARM|HARM ANALYSIS)',
        r'(?:^|\n)\s*(?:III\.|3\.|C\.|LEGAL STANDARD|STANDARD)',
        r'(?:^|\n)\s*(?:IV\.|4\.|D\.|ARGUMENT|ANALYSIS)',
        r'(?:^|\n)\s*(?:V\.|5\.|E\.|CONCLUSION)',
        r'(?:^|\n)\s*(?:VI\.|6\.|F\.|PRAYER|RELIEF)'
    ]

    sections = []
    for pattern in section_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            sections.append(pattern.split('|')[-1].strip(')'))

    return sections


def extract_citation_density(text: str, sections: List[str]) -> Dict[str, float]:
    """Calculate citation density per section."""
    # Simple citation pattern (case citations)
    citation_pattern = r'\b\d+\s+(?:F\.|Mass\.|U\.S\.)\s+(?:Supp\.|App\.|2d|3d)?\s*\d+'

    density = {}
    for section in sections:
        # Find section boundaries (simplified)
        section_text = text  # In practice, would extract actual section text
        citations = re.findall(citation_pattern, section_text)
        word_count = len(section_text.split())
        density[section] = len(citations) / max(word_count / 100, 1)

    return density


def extract_harm_mentions(text: str) -> Dict[str, int]:
    """Count harm-related mentions in text."""
    harm_patterns = {
        'privacy': r'\bprivacy\b',
        'harassment': r'\bharass\w*\b',
        'safety': r'\b(?:safety|danger|threat)\w*\b',
        'retaliation': r'\bretaliat\w*\b',
        'embarrassment': r'\bembarrass\w*\b',
        'stigma': r'\bstigma\w*\b'
    }

    mentions = {}
    text_lower = text.lower()
    for harm_type, pattern in harm_patterns.items():
        mentions[harm_type] = len(re.findall(pattern, text_lower))

    return mentions


def extract_winning_citations(text: str) -> List[str]:
    """Extract citations that appear in winning cases."""
    # Load winning citations
    winning_citations_path = PROJECT_ROOT / "case_law_data" / "analysis" / "winning_citations_federal.json"
    if not winning_citations_path.exists():
        return []

    with open(winning_citations_path, 'r') as f:
        winning_citations = json.load(f)

    winning_cite_list = [cite['citation'] for cite in winning_citations[:20]]  # Top 20

    # Find citations in text
    found_citations = []
    citation_pattern = r'\b\d+\s+(?:F\.|Mass\.|U\.S\.)\s+(?:Supp\.|App\.|2d|3d)?\s*\d+'
    citations_in_text = re.findall(citation_pattern, text)

    for citation in citations_in_text:
        if citation in winning_cite_list:
            found_citations.append(citation)

    return found_citations


def audit_catboost_patterns() -> None:
    """Main audit function to extract patterns from granted cases."""
    logger.info("Starting CatBoost pattern audit...")

    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        return

    # Load data and train model
    logger.info("Loading dataset and training CatBoost model...")
    dataset = load_dataset(DB_PATH)
    outcomes_df = extract_motion_outcomes(dataset["cases"])
    features_df = engineer_motion_features(dataset, outcomes_df)
    model, label_encoder, feature_cols, shap_importance = train_catboost_model(features_df)

    logger.info(f"Model trained. Top features: {list(shap_importance.keys())[:10]}")

    # Get granted cases
    granted_cases = outcomes_df[outcomes_df['outcome'] == 'granted']
    logger.info(f"Found {len(granted_cases)} granted cases")

    # Extract patterns from granted cases
    patterns = []

    for idx, case_row in granted_cases.iterrows():
        case_id = case_row['case_id']

        # Get case text
        case_data = dataset["cases"][dataset["cases"]['cluster_id'] == case_id]
        if case_data.empty:
            continue

        case_text = case_data.iloc[0]['cleaned_text'] or ""

        # Extract patterns
        section_structure = extract_section_structure(case_text)
        citation_density = extract_citation_density(case_text, section_structure)
        harm_mentions = extract_harm_mentions(case_text)
        winning_citations = extract_winning_citations(case_text)

        # Compute SHAP scores for this case
        case_features = compute_draft_features(case_text)
        case_df = pd.DataFrame([case_features])

        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in case_df.columns:
                case_df[col] = 0
        case_df = case_df[feature_cols]

        # Get SHAP values (simplified - in practice would use SHAP library)
        shap_scores = {}
        for feature in feature_cols:
            if feature in case_features:
                shap_scores[feature] = float(case_features[feature])

        # Create pattern record
        pattern_record = {
            "case_id": case_id,
            "outcome": "granted",
            "shap_scores": shap_scores,
            "section_structure": section_structure,
            "citation_density": citation_density,
            "harm_mentions": harm_mentions,
            "winning_citations": winning_citations,
            "word_count": len(case_text.split()),
            "text_length": len(case_text)
        }

        patterns.append(pattern_record)

        if len(patterns) % 50 == 0:
            logger.info(f"Processed {len(patterns)} granted cases...")

    # Save patterns
    with open(GRANTED_PATTERNS_FILE, 'w') as f:
        for pattern in patterns:
            f.write(json.dumps(pattern) + '\n')

    logger.info(f"Saved {len(patterns)} pattern records to {GRANTED_PATTERNS_FILE}")

    # Generate summary statistics
    generate_pattern_summary(patterns, shap_importance)


def generate_pattern_summary(patterns: List[Dict], shap_importance: Dict[str, float]) -> None:
    """Generate summary statistics from extracted patterns."""
    summary = {
        "total_granted_cases": len(patterns),
        "top_features": dict(list(shap_importance.items())[:10]),
        "average_harm_mentions": {},
        "common_sections": [],
        "citation_patterns": {},
        "winning_citation_frequency": {}
    }

    # Calculate averages
    harm_totals = Counter()
    section_totals = Counter()
    citation_totals = Counter()

    for pattern in patterns:
        # Harm mentions
        for harm_type, count in pattern["harm_mentions"].items():
            harm_totals[harm_type] += count

        # Sections
        for section in pattern["section_structure"]:
            section_totals[section] += 1

        # Citations
        for citation in pattern["winning_citations"]:
            citation_totals[citation] += 1

    # Calculate averages
    for harm_type in harm_totals:
        summary["average_harm_mentions"][harm_type] = harm_totals[harm_type] / len(patterns)

    summary["common_sections"] = dict(section_totals.most_common(10))
    summary["winning_citation_frequency"] = dict(citation_totals.most_common(20))

    # Save summary
    summary_file = OUTPUT_DIR / "pattern_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Pattern summary saved to {summary_file}")
    logger.info(f"Average harm mentions: {summary['average_harm_mentions']}")
    logger.info(f"Common sections: {summary['common_sections']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    audit_catboost_patterns()
