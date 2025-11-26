#!/usr/bin/env python3
"""
Petition Feature Extractor

Extracts features from petition text using NLP analysis.
Creates a feature table for machine learning model training.

Usage: python scripts/create_petition_features.py
"""

import json
import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PetitionFeatureExtractor:
    def __init__(self):
        self.sample_petitions_file = "data/case_law/sample_petitions.json"
        self.petitions_text_dir = "data/petitions_text"
        self.output_file = "data/case_law/petition_features.json"

        # Feature patterns
        self.citation_patterns = {
            'intel_corp': re.compile(r'intel\s+corp', re.IGNORECASE),
            'chevron': re.compile(r'chevron', re.IGNORECASE),
            'euromepa': re.compile(r'euromepa', re.IGNORECASE),
            'hegna': re.compile(r'hegna', re.IGNORECASE),
            'advanced_micro': re.compile(r'advanced\s+micro', re.IGNORECASE),
            'mees': re.compile(r'mees', re.IGNORECASE),
            'esmerian': re.compile(r'esmerian', re.IGNORECASE),
            'naranjo': re.compile(r'naranjo', re.IGNORECASE),
            'buiter': re.compile(r'buiter', re.IGNORECASE),
            'hourani': re.compile(r'hourani', re.IGNORECASE),
            'delano_farms': re.compile(r'delano\s+farms', re.IGNORECASE),
            'posco': re.compile(r'posco', re.IGNORECASE)
        }

        self.legal_patterns = {
            'protective_order': re.compile(r'protective\s+order', re.IGNORECASE),
            'confidentiality': re.compile(r'confidentiality', re.IGNORECASE),
            'privilege': re.compile(r'privilege', re.IGNORECASE),
            'burden': re.compile(r'burden', re.IGNORECASE),
            'relevance': re.compile(r'relevance', re.IGNORECASE),
            'proportionality': re.compile(r'proportionality', re.IGNORECASE),
            'cost': re.compile(r'cost', re.IGNORECASE),
            'time': re.compile(r'time', re.IGNORECASE),
            'scope': re.compile(r'scope', re.IGNORECASE),
            'foreign_tribunal': re.compile(r'foreign\s+tribunal', re.IGNORECASE),
            'international_arbitration': re.compile(r'international\s+arbitration', re.IGNORECASE),
            'discovery': re.compile(r'discovery', re.IGNORECASE),
            'subpoena': re.compile(r'subpoena', re.IGNORECASE),
            'deposition': re.compile(r'deposition', re.IGNORECASE),
            'document_production': re.compile(r'document\s+production', re.IGNORECASE)
        }

        self.intel_factors = {
            'factor_1': re.compile(r'factor\s+1|first\s+factor', re.IGNORECASE),
            'factor_2': re.compile(r'factor\s+2|second\s+factor', re.IGNORECASE),
            'factor_3': re.compile(r'factor\s+3|third\s+factor', re.IGNORECASE),
            'factor_4': re.compile(r'factor\s+4|fourth\s+factor', re.IGNORECASE),
            'intel_factors': re.compile(r'intel\s+factors', re.IGNORECASE)
        }

        self.procedural_patterns = {
            'ex_parte': re.compile(r'ex\s+parte', re.IGNORECASE),
            'intervenor': re.compile(r'intervenor', re.IGNORECASE),
            'amicus': re.compile(r'amicus', re.IGNORECASE),
            'motion_to_quash': re.compile(r'motion\s+to\s+quash', re.IGNORECASE),
            'motion_to_compel': re.compile(r'motion\s+to\s+compel', re.IGNORECASE),
            'seal': re.compile(r'seal', re.IGNORECASE),
            'redact': re.compile(r'redact', re.IGNORECASE)
        }

        self.outcome_patterns = {
            'granted': re.compile(r'granted|grant', re.IGNORECASE),
            'denied': re.compile(r'denied|deny', re.IGNORECASE),
            'affirmed': re.compile(r'affirmed|affirm', re.IGNORECASE),
            'reversed': re.compile(r'reversed|reverse', re.IGNORECASE),
            'vacated': re.compile(r'vacated|vacate', re.IGNORECASE),
            'mixed': re.compile(r'granted\s+in\s+part|denied\s+in\s+part', re.IGNORECASE)
        }

    def load_sample_petitions(self) -> List[Dict[str, Any]]:
        """Load sample petitions from JSON file."""
        logger.info("Loading sample petitions...")

        if not os.path.exists(self.sample_petitions_file):
            logger.error(f"Sample petitions file not found: {self.sample_petitions_file}")
            return []

        with open(self.sample_petitions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        petitions = data.get('petitions', [])
        logger.info(f"Loaded {len(petitions)} sample petitions")
        return petitions

    def extract_basic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text features."""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'paragraph_count': len(text.split('\n\n')),
            'avg_sentence_length': 0,
            'avg_word_length': 0
        }

        # Calculate averages
        if features['sentence_count'] > 0:
            features['avg_sentence_length'] = features['word_count'] / features['sentence_count']

        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)

        return features

    def extract_citation_features(self, text: str) -> Dict[str, int]:
        """Extract citation count features."""
        features = {}

        for citation, pattern in self.citation_patterns.items():
            matches = pattern.findall(text)
            features[f'citation_{citation}'] = len(matches)

        # Total citations
        features['total_citations'] = sum(features.values())

        return features

    def extract_legal_features(self, text: str) -> Dict[str, int]:
        """Extract legal terminology features."""
        features = {}

        for term, pattern in self.legal_patterns.items():
            matches = pattern.findall(text)
            features[f'legal_{term}'] = len(matches)

        return features

    def extract_intel_factor_features(self, text: str) -> Dict[str, int]:
        """Extract Intel factor features."""
        features = {}

        for factor, pattern in self.intel_factors.items():
            matches = pattern.findall(text)
            features[f'intel_{factor}'] = len(matches)

        return features

    def extract_procedural_features(self, text: str) -> Dict[str, int]:
        """Extract procedural features."""
        features = {}

        for proc, pattern in self.procedural_patterns.items():
            matches = pattern.findall(text)
            features[f'procedural_{proc}'] = len(matches)

        return features

    def extract_outcome_features(self, text: str) -> Dict[str, int]:
        """Extract outcome-related features."""
        features = {}

        for outcome, pattern in self.outcome_patterns.items():
            matches = pattern.findall(text)
            features[f'outcome_{outcome}'] = len(matches)

        return features

    def extract_jurisdiction_features(self, text: str) -> Dict[str, int]:
        """Extract jurisdiction-related features."""
        jurisdiction_patterns = {
            'washington': re.compile(r'washington', re.IGNORECASE),
            'california': re.compile(r'california', re.IGNORECASE),
            'new_york': re.compile(r'new\s+york', re.IGNORECASE),
            'massachusetts': re.compile(r'massachusetts', re.IGNORECASE),
            'texas': re.compile(r'texas', re.IGNORECASE),
            'florida': re.compile(r'florida', re.IGNORECASE),
            'nebraska': re.compile(r'nebraska', re.IGNORECASE),
            'maryland': re.compile(r'maryland', re.IGNORECASE),
            'wisconsin': re.compile(r'wisconsin', re.IGNORECASE)
        }

        features = {}
        for jurisdiction, pattern in jurisdiction_patterns.items():
            matches = pattern.findall(text)
            features[f'jurisdiction_{jurisdiction}'] = len(matches)

        return features

    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features (simplified)."""
        # Simple sentiment analysis based on positive/negative words
        positive_words = ['grant', 'approve', 'allow', 'permit', 'favor', 'support', 'benefit', 'help', 'assist']
        negative_words = ['deny', 'reject', 'refuse', 'oppose', 'burden', 'cost', 'harm', 'prejudice', 'unfair']

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_words = len(text.split())

        features = {
            'sentiment_positive': positive_count / max(total_words, 1),
            'sentiment_negative': negative_count / max(total_words, 1),
            'sentiment_polarity': (positive_count - negative_count) / max(total_words, 1)
        }

        return features

    def extract_complexity_features(self, text: str) -> Dict[str, float]:
        """Extract complexity features."""
        sentences = re.split(r'[.!?]+', text)
        words = text.split()

        if not sentences or not words:
            return {
                'complexity_avg_sentence_length': 0,
                'complexity_legal_density': 0,
                'complexity_citation_density': 0
            }

        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)

        # Legal term density
        legal_terms = sum(len(pattern.findall(text)) for pattern in self.legal_patterns.values())
        legal_density = legal_terms / max(len(words), 1)

        # Citation density
        citations = sum(len(pattern.findall(text)) for pattern in self.citation_patterns.values())
        citation_density = citations / max(len(words), 1)

        return {
            'complexity_avg_sentence_length': avg_sentence_length,
            'complexity_legal_density': legal_density,
            'complexity_citation_density': citation_density
        }

    def extract_features_from_petition(self, petition: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all features from a single petition."""
        text = petition.get('text_content', '')

        if not text:
            logger.warning(f"No text content for petition: {petition.get('case_name', 'Unknown')}")
            return {}

        # Extract all feature categories
        features = {
            'petition_id': petition.get('filename', ''),
            'case_name': petition.get('case_name', ''),
            'docket_id': petition.get('docket_id'),
            'docket_number': petition.get('docket_number', ''),
            'court': petition.get('court', ''),
            'outcome': petition.get('outcome', ''),
            'petition_type': petition.get('petition_type', ''),
            'text_length': petition.get('text_length', 0)
        }

        # Add extracted features
        features.update(self.extract_basic_features(text))
        features.update(self.extract_citation_features(text))
        features.update(self.extract_legal_features(text))
        features.update(self.extract_intel_factor_features(text))
        features.update(self.extract_procedural_features(text))
        features.update(self.extract_outcome_features(text))
        features.update(self.extract_jurisdiction_features(text))
        features.update(self.extract_sentiment_features(text))
        features.update(self.extract_complexity_features(text))

        return features

    def create_feature_table(self, petitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create feature table from all petitions."""
        logger.info("Creating petition feature table...")

        feature_table = []

        for petition in petitions:
            features = self.extract_features_from_petition(petition)
            if features:
                feature_table.append(features)

        logger.info(f"Created feature table with {len(feature_table)} petitions")
        return feature_table

    def analyze_feature_distribution(self, feature_table: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of features."""
        logger.info("Analyzing feature distribution...")

        analysis = {
            'total_petitions': len(feature_table),
            'outcome_distribution': {},
            'court_distribution': {},
            'feature_statistics': {},
            'top_citations': {},
            'top_legal_terms': {}
        }

        if not feature_table:
            return analysis

        # Count outcomes
        for petition in feature_table:
            outcome = petition.get('outcome', 'UNKNOWN')
            analysis['outcome_distribution'][outcome] = analysis['outcome_distribution'].get(outcome, 0) + 1

        # Count courts
        for petition in feature_table:
            court = petition.get('court', 'UNKNOWN')
            analysis['court_distribution'][court] = analysis['court_distribution'].get(court, 0) + 1

        # Calculate feature statistics
        numeric_features = ['text_length', 'word_count', 'total_citations', 'sentiment_polarity']
        for feature in numeric_features:
            values = [p.get(feature, 0) for p in feature_table if isinstance(p.get(feature), (int, float))]
            if values:
                analysis['feature_statistics'][feature] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

        # Top citations
        citation_features = [f for f in feature_table[0].keys() if f.startswith('citation_')]
        for citation in citation_features:
            total_count = sum(p.get(citation, 0) for p in feature_table)
            if total_count > 0:
                analysis['top_citations'][citation] = total_count

        # Top legal terms
        legal_features = [f for f in feature_table[0].keys() if f.startswith('legal_')]
        for legal in legal_features:
            total_count = sum(p.get(legal, 0) for p in feature_table)
            if total_count > 0:
                analysis['top_legal_terms'][legal] = total_count

        return analysis

    def save_feature_table(self, feature_table: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Save the feature table and analysis."""
        logger.info("Saving petition feature table...")

        # Save feature table
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'creation_date': datetime.now().isoformat(),
                'total_petitions': len(feature_table),
                'analysis': analysis,
                'features': feature_table
            }, f, indent=2, ensure_ascii=False)

        # Save analysis report
        report_file = "data/case_law/petition_features_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 📊 Petition Features Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Petitions**: {analysis['total_petitions']}\n\n")

            f.write("## 🎯 Outcome Distribution\n\n")
            for outcome, count in analysis['outcome_distribution'].items():
                f.write(f"- **{outcome}**: {count} petitions\n")

            f.write("\n## 🏛️ Court Distribution\n\n")
            for court, count in sorted(analysis['court_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"- **{court}**: {count} petitions\n")

            f.write("\n## 📈 Feature Statistics\n\n")
            for feature, stats in analysis['feature_statistics'].items():
                f.write(f"- **{feature}**: mean={stats['mean']:.2f}, min={stats['min']}, max={stats['max']}\n")

            f.write("\n## 🔗 Top Citations\n\n")
            for citation, count in sorted(analysis['top_citations'].items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"- **{citation}**: {count} mentions\n")

            f.write("\n## ⚖️ Top Legal Terms\n\n")
            for term, count in sorted(analysis['top_legal_terms'].items(), key=lambda x: x[1], reverse=True)[:10]:
                f.write(f"- **{term}**: {count} mentions\n")

        logger.info(f"✓ Feature table saved to: {self.output_file}")
        logger.info(f"✓ Analysis report saved to: {report_file}")

    def run_extraction(self):
        """Run the complete feature extraction process."""
        logger.info("🚀 Starting Petition Feature Extraction")
        logger.info("="*80)

        # Load sample petitions
        petitions = self.load_sample_petitions()
        if not petitions:
            logger.error("No petitions to process")
            return

        # Create feature table
        feature_table = self.create_feature_table(petitions)

        # Analyze feature distribution
        analysis = self.analyze_feature_distribution(feature_table)

        # Save results
        self.save_feature_table(feature_table, analysis)

        logger.info("🎉 Petition feature extraction complete!")

        # Print summary
        print(f"\n📊 Summary:")
        print(f"  Total petitions processed: {analysis['total_petitions']}")
        print(f"  Outcome distribution: {analysis['outcome_distribution']}")
        print(f"  Feature statistics: {len(analysis['feature_statistics'])} features analyzed")
        print(f"  Top citations: {len(analysis['top_citations'])} citations found")
        print(f"  Top legal terms: {len(analysis['top_legal_terms'])} legal terms found")

def main():
    """Main function."""
    print("📊 Petition Feature Extractor")
    print("="*80)

    extractor = PetitionFeatureExtractor()
    extractor.run_extraction()

    print("\n✅ Petition feature extraction complete!")
    print("Check petition_features.json and petition_features_report.md for results.")

if __name__ == "__main__":
    main()
