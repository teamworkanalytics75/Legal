#!/usr/bin/env python3
"""
Maximum Case Adjudication System
================================

This script runs the advanced adjudicator on ALL remaining unclear cases
from the entire corpus to maximize our dataset and model performance.

Strategy:
1. Load ALL 747 cases from the corpus
2. Identify cases that haven't been adjudicated yet
3. Run advanced NLP analysis on all unclear cases
4. Retrain the model with the expanded dataset
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any, Tuple, Optional
import pickle

# Advanced NLP libraries
import spacy
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaximumAdjudicationSystem:
    """Maximum case adjudication system for comprehensive coverage."""

    def __init__(self):
        """Initialize the maximum adjudication system."""
        logger.info("Initializing Maximum Adjudication System...")

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp.max_length = max(self.nlp.max_length, 2_000_000)
            logger.info("✓ spaCy model loaded")
        except OSError:
            logger.error("✗ spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise

        # Initialize sentence transformer
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Sentence transformer loaded")
        except Exception as e:
            logger.warning(f"✗ Sentence transformer failed: {e}")
            self.sentence_model = None

        # Initialize NLTK components
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            logger.info("✓ NLTK components loaded")
        except Exception as e:
            logger.warning(f"✗ NLTK setup failed: {e}")
            self.stop_words = set()
            self.lemmatizer = None

        # Define advanced legal patterns
        self._define_legal_patterns()

        # Reference texts for similarity analysis
        self._load_reference_texts()

        logger.info("✓ Maximum Adjudication System initialized successfully")

    def _define_legal_patterns(self):
        """Define comprehensive legal patterns for outcome detection."""

        # Strong success patterns (high confidence)
        self.strong_success_patterns = [
            # Direct grants
            re.compile(r'\bmotion\s+(?:is\s+)?granted\b', re.IGNORECASE),
            re.compile(r'\bapplication\s+(?:is\s+)?granted\b', re.IGNORECASE),
            re.compile(r'\bpetition\s+(?:is\s+)?granted\b', re.IGNORECASE),
            re.compile(r'\bcourt\s+grants?\b', re.IGNORECASE),
            re.compile(r'\bgranted\s+(?:in\s+)?(?:full|part)\b', re.IGNORECASE),

            # Approval language
            re.compile(r'\bapprov(?:ed|ing|al)\b', re.IGNORECASE),
            re.compile(r'\ballow(?:ed|ing)\b', re.IGNORECASE),
            re.compile(r'\bpermit(?:ted|ting)\b', re.IGNORECASE),
            re.compile(r'\bauthoriz(?:ed|ing)\b', re.IGNORECASE),

            # Positive outcomes
            re.compile(r'\bfavor(?:able|ed)\b', re.IGNORECASE),
            re.compile(r'\bsuccess(?:ful|fully)\b', re.IGNORECASE),
            re.compile(r'\baccept(?:ed|ing|able)\b', re.IGNORECASE),
        ]

        # Strong failure patterns (high confidence)
        self.strong_failure_patterns = [
            # Direct denials
            re.compile(r'\bmotion\s+(?:is\s+)?denied\b', re.IGNORECASE),
            re.compile(r'\bapplication\s+(?:is\s+)?denied\b', re.IGNORECASE),
            re.compile(r'\bpetition\s+(?:is\s+)?denied\b', re.IGNORECASE),
            re.compile(r'\bcourt\s+denies?\b', re.IGNORECASE),
            re.compile(r'\bdenied\s+(?:in\s+)?(?:full|part)\b', re.IGNORECASE),

            # Rejection language
            re.compile(r'\breject(?:ed|ing)\b', re.IGNORECASE),
            re.compile(r'\bdisallow(?:ed|ing)\b', re.IGNORECASE),
            re.compile(r'\bprohibit(?:ed|ing)\b', re.IGNORECASE),
            re.compile(r'\bdeclin(?:ed|ing)\b', re.IGNORECASE),

            # Negative outcomes
            re.compile(r'\bunfavor(?:able|ed)\b', re.IGNORECASE),
            re.compile(r'\bunsuccess(?:ful|fully)\b', re.IGNORECASE),
            re.compile(r'\bunaccept(?:able|ed)\b', re.IGNORECASE),
        ]

        # Contextual patterns (medium confidence)
        self.contextual_patterns = {
            'success': [
                re.compile(r'\bdiscovery\s+(?:is\s+)?(?:permitted|allowed|authorized)\b', re.IGNORECASE),
                re.compile(r'\bdiscovery\s+(?:shall|will)\s+(?:proceed|continue)\b', re.IGNORECASE),
                re.compile(r'\bgood\s+cause\s+(?:shown|demonstrated|established)\b', re.IGNORECASE),
                re.compile(r'\binterests\s+of\s+justice\s+(?:require|support|favor)\b', re.IGNORECASE),
            ],
            'failure': [
                re.compile(r'\bdiscovery\s+(?:is\s+)?(?:denied|prohibited|restricted)\b', re.IGNORECASE),
                re.compile(r'\bdiscovery\s+(?:shall|will)\s+(?:not|cease)\s+(?:proceed|continue)\b', re.IGNORECASE),
                re.compile(r'\binsufficient\s+(?:showing|evidence|cause)\b', re.IGNORECASE),
                re.compile(r'\bnot\s+(?:in\s+)?the\s+interests?\s+of\s+justice\b', re.IGNORECASE),
            ]
        }

        # Legal terminology patterns
        self.legal_terminology = {
            'procedural': [
                re.compile(r'\bjurisdiction\b', re.IGNORECASE),
                re.compile(r'\bvenue\b', re.IGNORECASE),
                re.compile(r'\bstanding\b', re.IGNORECASE),
                re.compile(r'\bripeness\b', re.IGNORECASE),
            ],
            'substantive': [
                re.compile(r'\bintel\s+(?:factors?|test|analysis)\b', re.IGNORECASE),
                re.compile(r'\bdiscretion\b', re.IGNORECASE),
                re.compile(r'\bgood\s+cause\b', re.IGNORECASE),
                re.compile(r'\binterests?\s+of\s+justice\b', re.IGNORECASE),
            ]
        }

    def _load_reference_texts(self):
        """Load reference texts for similarity analysis."""
        self.reference_texts = {
            'success': [
                "The motion is granted. The court finds good cause for the discovery request.",
                "The application is approved. Discovery shall proceed as requested.",
                "The petition is granted in full. The interests of justice support this discovery.",
            ],
            'failure': [
                "The motion is denied. The court finds insufficient cause for the discovery request.",
                "The application is rejected. Discovery is not warranted under these circumstances.",
                "The petition is denied. The interests of justice do not support this discovery.",
            ]
        }

    def load_all_corpus_cases(self) -> List[Dict[str, Any]]:
        """Load all cases from the corpus."""
        logger.info("Loading all corpus cases...")

        cases_dir = Path("data/case_law/1782_discovery")
        all_cases = []

        for json_file in cases_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                    case_data['file_name'] = json_file.stem
                    all_cases.append(case_data)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")

        logger.info(f"✓ Loaded {len(all_cases)} total corpus cases")
        return all_cases

    def identify_unclear_cases(self, all_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify cases that need adjudication."""
        logger.info("Identifying unclear cases...")

        # Load previously adjudicated cases
        previously_adjudicated = set()

        # Check advanced adjudicated cases
        advanced_file = Path("data/case_law/advanced_adjudicated_cases.json")
        if advanced_file.exists():
            with open(advanced_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for case in data.get('adjudicated_cases', []):
                    previously_adjudicated.add(case.get('file_name', ''))

        # Check previous adjudicated cases
        previous_file = Path("data/case_law/adjudicated_cases.json")
        if previous_file.exists():
            with open(previous_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for case in data.get('adjudicated_cases', []):
                    previously_adjudicated.add(case.get('file_name', ''))

        logger.info(f"✓ Found {len(previously_adjudicated)} previously adjudicated cases")

        # Identify unclear cases
        unclear_cases = []
        for case in all_cases:
            file_name = case.get('file_name', '')

            # Skip if already adjudicated
            if file_name in previously_adjudicated:
                continue

            # Combine all text sources
            all_text = " ".join([
                case.get("opinion_text", ""),
                case.get("caseNameFull_text", ""),
                case.get("attorney_text", ""),
                case.get("extracted_text", "")
            ]).lower()

            if not all_text.strip():
                continue

            # Check for clear outcome indicators
            clear_success_patterns = [
                r"motion\s+granted", r"application\s+granted", r"petition\s+granted",
                r"granted\s+in\s+part", r"granted\s+in\s+full", r"court\s+grants"
            ]

            clear_failure_patterns = [
                r"motion\s+denied", r"application\s+denied", r"petition\s+denied",
                r"denied\s+in\s+part", r"denied\s+in\s+full", r"court\s+denies"
            ]

            success_count = sum(len(re.findall(pattern, all_text)) for pattern in clear_success_patterns)
            failure_count = sum(len(re.findall(pattern, all_text)) for pattern in clear_failure_patterns)

            # If unclear (no clear patterns or conflicting patterns), add to unclear cases
            if success_count == 0 and failure_count == 0:
                unclear_cases.append(case)
            elif success_count > 0 and failure_count > 0 and abs(success_count - failure_count) <= 1:
                unclear_cases.append(case)

        logger.info(f"✓ Identified {len(unclear_cases)} unclear cases for adjudication")
        return unclear_cases

    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive features from text."""
        features = {}

        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(sent_tokenize(text))

        # spaCy analysis
        doc = self.nlp(text)
        features['entities'] = [ent.text for ent in doc.ents]
        features['legal_entities'] = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'LAW', 'GPE']]
        features['noun_phrases'] = [chunk.text for chunk in doc.noun_chunks]

        # Sentiment analysis
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity

        # Pattern matching
        features['strong_success_matches'] = sum(1 for pattern in self.strong_success_patterns if pattern.search(text))
        features['strong_failure_matches'] = sum(1 for pattern in self.strong_failure_patterns if pattern.search(text))

        # Contextual pattern matching
        features['contextual_success_matches'] = sum(1 for pattern in self.contextual_patterns['success'] if pattern.search(text))
        features['contextual_failure_matches'] = sum(1 for pattern in self.contextual_patterns['failure'] if pattern.search(text))

        # Legal terminology
        features['procedural_terms'] = sum(1 for pattern in self.legal_terminology['procedural'] if pattern.search(text))
        features['substantive_terms'] = sum(1 for pattern in self.legal_terminology['substantive'] if pattern.search(text))

        return features

    def calculate_semantic_similarity(self, text: str) -> Dict[str, float]:
        """Calculate semantic similarity to reference texts."""
        if not self.sentence_model:
            return {'success_similarity': 0.0, 'failure_similarity': 0.0}

        try:
            # Encode the input text
            text_embedding = self.sentence_model.encode([text])

            similarities = {}
            for outcome_type, ref_texts in self.reference_texts.items():
                # Encode reference texts
                ref_embeddings = self.sentence_model.encode(ref_texts)

                # Calculate cosine similarity
                similarities[f'{outcome_type}_similarity'] = float(
                    cosine_similarity(text_embedding, ref_embeddings).max()
                )

            return similarities
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return {'success_similarity': 0.0, 'failure_similarity': 0.0}

    def calculate_confidence_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for the analysis."""
        confidence_factors = []

        # Pattern strength factor
        total_patterns = (analysis['strong_success_matches'] +
                         analysis['strong_failure_matches'] +
                         analysis['contextual_success_matches'] +
                         analysis['contextual_failure_matches'])

        if total_patterns > 0:
            pattern_strength = min(total_patterns / 5.0, 1.0)  # Normalize to 0-1
            confidence_factors.append(pattern_strength)

        # Semantic similarity factor
        semantic_diff = abs(analysis['success_similarity'] - analysis['failure_similarity'])
        confidence_factors.append(semantic_diff)

        # Legal terminology factor
        legal_terms = analysis['procedural_terms'] + analysis['substantive_terms']
        legal_factor = min(legal_terms / 3.0, 1.0)  # Normalize to 0-1
        confidence_factors.append(legal_factor)

        # Text quality factor
        text_quality = min(analysis['text_length'] / 1000.0, 1.0)  # Normalize to 0-1
        confidence_factors.append(text_quality)

        # Calculate weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Pattern, semantic, legal, quality
        confidence = sum(w * f for w, f in zip(weights, confidence_factors))

        return min(max(confidence, 0.0), 1.0)  # Clamp to 0-1

    def adjudicate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Adjudicate a single case using advanced analysis."""

        # Combine all text sources
        all_text = " ".join([
            case.get("opinion_text", ""),
            case.get("caseNameFull_text", ""),
            case.get("attorney_text", ""),
            case.get("extracted_text", "")
        ]).strip()

        if not all_text:
            return {
                "outcome": "UNCLEAR",
                "confidence": 0.0,
                "reasoning": "No text available for analysis",
                "features": {},
                "pattern_matches": {},
                "semantic_similarity": {}
            }

        # Extract features
        features = self.extract_features(all_text)

        # Calculate semantic similarity
        semantic_similarity = self.calculate_semantic_similarity(all_text)
        features.update(semantic_similarity)

        # Determine outcome based on multiple factors
        outcome_scores = {
            'SUCCESS': 0.0,
            'FAILURE': 0.0,
            'MIXED': 0.0
        }

        # Pattern-based scoring
        pattern_weight = 0.4
        outcome_scores['SUCCESS'] += (features['strong_success_matches'] + features['contextual_success_matches']) * pattern_weight
        outcome_scores['FAILURE'] += (features['strong_failure_matches'] + features['contextual_failure_matches']) * pattern_weight

        # Semantic similarity scoring
        semantic_weight = 0.3
        outcome_scores['SUCCESS'] += features['success_similarity'] * semantic_weight
        outcome_scores['FAILURE'] += features['failure_similarity'] * semantic_weight

        # Sentiment scoring
        sentiment_weight = 0.2
        if features['sentiment_polarity'] > 0.1:
            outcome_scores['SUCCESS'] += features['sentiment_polarity'] * sentiment_weight
        elif features['sentiment_polarity'] < -0.1:
            outcome_scores['FAILURE'] += abs(features['sentiment_polarity']) * sentiment_weight

        # Legal terminology bonus
        legal_weight = 0.1
        if features['substantive_terms'] > 0:
            # More substantive legal terms suggest more detailed analysis
            outcome_scores['SUCCESS'] += min(features['substantive_terms'] / 5.0, 1.0) * legal_weight
            outcome_scores['FAILURE'] += min(features['substantive_terms'] / 5.0, 1.0) * legal_weight

        # Determine final outcome
        max_score = max(outcome_scores.values())
        if max_score < 0.3:  # Low confidence threshold
            outcome = "UNCLEAR"
        elif abs(outcome_scores['SUCCESS'] - outcome_scores['FAILURE']) < 0.1:
            outcome = "MIXED"
        elif outcome_scores['SUCCESS'] > outcome_scores['FAILURE']:
            outcome = "SUCCESS"
        else:
            outcome = "FAILURE"

        # Calculate confidence
        confidence = self.calculate_confidence_score(features)

        # Generate reasoning
        reasoning_parts = []
        if features['strong_success_matches'] > 0:
            reasoning_parts.append(f"Found {features['strong_success_matches']} strong success patterns")
        if features['strong_failure_matches'] > 0:
            reasoning_parts.append(f"Found {features['strong_failure_matches']} strong failure patterns")
        if features['contextual_success_matches'] > 0:
            reasoning_parts.append(f"Found {features['contextual_success_matches']} contextual success patterns")
        if features['contextual_failure_matches'] > 0:
            reasoning_parts.append(f"Found {features['contextual_failure_matches']} contextual failure patterns")

        if semantic_similarity['success_similarity'] > 0.7:
            reasoning_parts.append("High semantic similarity to successful cases")
        if semantic_similarity['failure_similarity'] > 0.7:
            reasoning_parts.append("High semantic similarity to failed cases")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Limited pattern matches found"

        return {
            "outcome": outcome,
            "confidence": confidence,
            "reasoning": reasoning,
            "features": features,
            "pattern_matches": {
                "strong_success": features['strong_success_matches'],
                "strong_failure": features['strong_failure_matches'],
                "contextual_success": features['contextual_success_matches'],
                "contextual_failure": features['contextual_failure_matches']
            },
            "semantic_similarity": semantic_similarity,
            "outcome_scores": outcome_scores
        }

    def adjudicate_all_unclear_cases(self) -> List[Dict[str, Any]]:
        """Adjudicate all unclear cases from the corpus."""
        logger.info("Starting maximum case adjudication...")

        # Load all cases
        all_cases = self.load_all_corpus_cases()

        # Identify unclear cases
        unclear_cases = self.identify_unclear_cases(all_cases)

        # Adjudicate unclear cases
        adjudicated_cases = []

        for i, case in enumerate(unclear_cases):
            logger.info(f"Adjudicating case {i+1}/{len(unclear_cases)}: {case.get('file_name', 'unknown')}")

            try:
                analysis = self.adjudicate_case(case)

                adjudicated_case = {
                    "file_name": case.get("file_name", ""),
                    "case_name": case.get("case_name", ""),
                    "cluster_id": case.get("cluster_id"),
                    "court_id": case.get("court_id", ""),
                    "date_filed": case.get("date_filed", ""),
                    "original_text_length": case.get("total_text_length", 0),
                    "adjudicated_outcome": analysis["outcome"],
                    "confidence_score": analysis["confidence"],
                    "reasoning": analysis["reasoning"],
                    "pattern_matches": analysis["pattern_matches"],
                    "semantic_similarity": analysis["semantic_similarity"],
                    "outcome_scores": analysis["outcome_scores"],
                    "features": analysis["features"]
                }

                adjudicated_cases.append(adjudicated_case)

                logger.info(f"✓ Adjudicated as {analysis['outcome']} (confidence: {analysis['confidence']:.3f})")

            except Exception as e:
                logger.error(f"✗ Error adjudicating case {case.get('file_name', 'unknown')}: {e}")
                continue

        logger.info(f"Completed adjudication of {len(adjudicated_cases)} cases")
        return adjudicated_cases

    def generate_report(self, adjudicated_cases: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive report."""

        # Calculate statistics
        total_cases = len(adjudicated_cases)
        outcome_counts = {}
        confidence_scores = []

        for case in adjudicated_cases:
            outcome = case['adjudicated_outcome']
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            confidence_scores.append(case['confidence_score'])

        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        # Generate report
        report = f"""# Maximum Case Adjudication Report

## 📊 **Summary Statistics**

- **Total Cases Adjudicated**: {total_cases:,}
- **Average Confidence Score**: {avg_confidence:.3f}
- **Adjudication Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 **Outcome Distribution**

"""

        for outcome, count in sorted(outcome_counts.items()):
            percentage = (count / total_cases) * 100 if total_cases > 0 else 0
            report += f"- **{outcome}**: {count:,} cases ({percentage:.1f}%)\n"

        report += f"""
## 🔍 **Confidence Analysis**

- **High Confidence (>0.7)**: {sum(1 for c in confidence_scores if c > 0.7):,} cases
- **Medium Confidence (0.4-0.7)**: {sum(1 for c in confidence_scores if 0.4 <= c <= 0.7):,} cases
- **Low Confidence (<0.4)**: {sum(1 for c in confidence_scores if c < 0.4):,} cases

## 🏆 **Top High-Confidence Cases**

"""

        # Sort by confidence and show top cases
        sorted_cases = sorted(adjudicated_cases, key=lambda x: x['confidence_score'], reverse=True)
        for i, case in enumerate(sorted_cases[:10]):
            report += f"{i+1}. **{case['file_name']}** - {case['adjudicated_outcome']} (confidence: {case['confidence_score']:.3f})\n"
            report += f"   - Reasoning: {case['reasoning']}\n\n"

        report += f"""
## 📈 **Pattern Analysis**

### Success Patterns Found
- **Strong Success Patterns**: {sum(case['pattern_matches']['strong_success'] for case in adjudicated_cases):,} total matches
- **Contextual Success Patterns**: {sum(case['pattern_matches']['contextual_success'] for case in adjudicated_cases):,} total matches

### Failure Patterns Found
- **Strong Failure Patterns**: {sum(case['pattern_matches']['strong_failure'] for case in adjudicated_cases):,} total matches
- **Contextual Failure Patterns**: {sum(case['pattern_matches']['contextual_failure'] for case in adjudicated_cases):,} total matches

## 🧠 **Semantic Analysis**

### Average Semantic Similarity
- **Success Similarity**: {np.mean([case['semantic_similarity']['success_similarity'] for case in adjudicated_cases]):.3f}
- **Failure Similarity**: {np.mean([case['semantic_similarity']['failure_similarity'] for case in adjudicated_cases]):.3f}

## 🎯 **Next Steps**

1. **High-Confidence Cases**: {sum(1 for c in confidence_scores if c > 0.7):,} cases ready for model training
2. **Medium-Confidence Cases**: {sum(1 for c in confidence_scores if 0.4 <= c <= 0.7):,} cases may need manual review
3. **Low-Confidence Cases**: {sum(1 for c in confidence_scores if c < 0.4):,} cases require additional analysis

## 📋 **Detailed Results**

All adjudicated cases have been saved to `data/case_law/maximum_adjudicated_cases.json` with full feature analysis and confidence scores.

---
*Report generated by Maximum Adjudication System using spaCy, NLTK, TextBlob, and Sentence Transformers*
"""

        return report

def main():
    """Main execution function."""
    logger.info("Starting Maximum Case Adjudication Process...")

    try:
        # Initialize adjudication system
        system = MaximumAdjudicationSystem()

        # Adjudicate all unclear cases
        adjudicated_cases = system.adjudicate_all_unclear_cases()

        # Save results
        output_file = Path("data/case_law/maximum_adjudicated_cases.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'adjudication_date': datetime.now().isoformat(),
                'total_cases': len(adjudicated_cases),
                'adjudicated_cases': adjudicated_cases
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved {len(adjudicated_cases)} adjudicated cases to {output_file}")

        # Generate and save report
        report = system.generate_report(adjudicated_cases)
        report_file = Path("data/case_law/maximum_adjudication_report.md")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"✓ Generated comprehensive report: {report_file}")

        # Print summary
        print("\n" + "="*70)
        print("🎉 MAXIMUM CASE ADJUDICATION COMPLETE!")
        print("="*70)
        print(f"📊 Total Cases Adjudicated: {len(adjudicated_cases):,}")

        outcome_counts = {}
        for case in adjudicated_cases:
            outcome = case['adjudicated_outcome']
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        for outcome, count in sorted(outcome_counts.items()):
            print(f"🎯 {outcome}: {count:,} cases")

        avg_confidence = np.mean([case['confidence_score'] for case in adjudicated_cases])
        print(f"🎯 Average Confidence: {avg_confidence:.3f}")
        print(f"📄 Report: {report_file}")
        print(f"💾 Data: {output_file}")
        print("="*70)

    except Exception as e:
        logger.error(f"Maximum adjudication process failed: {e}")
        raise

if __name__ == "__main__":
    main()
