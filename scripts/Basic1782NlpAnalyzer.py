#!/usr/bin/env python3
"""
Basic NLP Analysis for 1782 Discovery Cases
==========================================

Step 2: Analyze extracted text to find patterns
- Count legal keywords and phrases
- Identify case outcomes (granted/denied)
- Extract key entities (courts, parties, dates)
- Find common patterns in 1782 cases
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Counter
from collections import defaultdict, Counter
import spacy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Basic1782Analyzer:
    """Basic NLP analyzer for 1782 discovery cases."""

    def __init__(self, text_directory: str = "data/case_law/extracted_text"):
        """Initialize the analyzer."""
        self.text_directory = Path(text_directory)
        self.output_dir = Path("data/case_law/analysis_results")
        self.output_dir.mkdir(exist_ok=True)

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded")
        except OSError:
            logger.error("❌ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Legal patterns for 1782 analysis
        self.legal_patterns = {
            'outcome_keywords': [
                'granted', 'denied', 'approved', 'rejected', 'sustained', 'overruled',
                'motion granted', 'motion denied', 'petition granted', 'petition denied',
                'discovery granted', 'discovery denied', 'application granted', 'application denied',
                'order granting', 'order denying', 'memorandum granting', 'memorandum denying'
            ],
            '1782_keywords': [
                '28 u.s.c. 1782', '28 usc 1782', 'section 1782', 'statute 1782',
                'international discovery', 'foreign proceeding', 'cross-border discovery',
                'assistance to foreign tribunal', 'foreign litigation'
            ],
            'court_keywords': [
                'district court', 'federal court', 'circuit court', 'bankruptcy court',
                'magistrate judge', 'district judge', 'circuit judge', 'bankruptcy judge'
            ],
            'discovery_keywords': [
                'discovery', 'deposition', 'interrogatory', 'request for production',
                'subpoena', 'document production', 'witness testimony', 'document request'
            ],
            'procedure_keywords': [
                'motion', 'petition', 'application', 'order', 'memorandum', 'brief',
                'hearing', 'oral argument', 'written submission', 'ex parte'
            ]
        }

        # Analysis results
        self.analysis_results = {}
        self.pattern_counts = defaultdict(Counter)
        self.case_outcomes = Counter()

    def analyze_text_file(self, text_file: Path) -> Dict[str, Any]:
        """Analyze a single text file."""
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()

            analysis = {
                'filename': text_file.name,
                'text_length': len(text),
                'word_count': len(text.split()),
                'pattern_matches': {},
                'outcome_prediction': None,
                'entities': [],
                'key_phrases': []
            }

            text_lower = text.lower()

            # Count pattern matches
            for pattern_type, keywords in self.legal_patterns.items():
                matches = []
                for keyword in keywords:
                    count = text_lower.count(keyword.lower())
                    if count > 0:
                        matches.append((keyword, count))
                analysis['pattern_matches'][pattern_type] = matches

            # Predict outcome
            analysis['outcome_prediction'] = self._predict_outcome(text_lower)

            # Extract entities if spaCy is available
            if self.nlp:
                doc = self.nlp(text[:10000])  # Process first 10k chars for speed

                # Extract named entities
                entities = []
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
                analysis['entities'] = entities[:50]  # Limit to 50 entities

                # Extract key phrases (noun phrases)
                key_phrases = []
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) >= 2:  # Multi-word phrases
                        key_phrases.append(chunk.text.strip())
                analysis['key_phrases'] = key_phrases[:20]  # Top 20 phrases

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {text_file.name}: {e}")
            return {'filename': text_file.name, 'error': str(e)}

    def _predict_outcome(self, text_lower: str) -> Dict[str, Any]:
        """Predict case outcome based on text patterns."""
        # Count positive and negative outcome indicators
        positive_indicators = ['granted', 'approved', 'sustained', 'motion granted', 'petition granted', 'order granting']
        negative_indicators = ['denied', 'rejected', 'overruled', 'motion denied', 'petition denied', 'order denying']

        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)

        total_indicators = positive_count + negative_count

        if total_indicators == 0:
            return {'prediction': 'unclear', 'confidence': 0.0, 'reasoning': 'No clear outcome indicators'}

        if positive_count > negative_count:
            prediction = 'granted'
            confidence = positive_count / total_indicators
        elif negative_count > positive_count:
            prediction = 'denied'
            confidence = negative_count / total_indicators
        else:
            prediction = 'mixed'
            confidence = 0.5

        return {
            'prediction': prediction,
            'confidence': confidence,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'reasoning': f'Found {positive_count} positive and {negative_count} negative indicators'
        }

    def analyze_all_texts(self) -> Dict[str, Any]:
        """Analyze all extracted text files."""
        text_files = list(self.text_directory.glob("*.txt"))
        logger.info(f"📁 Found {len(text_files)} text files to analyze")

        results = {
            'total_files': len(text_files),
            'analyzed_files': 0,
            'analysis_results': {},
            'summary_stats': {},
            'pattern_summary': {},
            'outcome_summary': {}
        }

        for i, text_file in enumerate(text_files, 1):
            logger.info(f"🔍 Analyzing {i}/{len(text_files)}: {text_file.name}")

            analysis = self.analyze_text_file(text_file)
            results['analysis_results'][text_file.name] = analysis
            results['analyzed_files'] += 1

            # Aggregate pattern counts
            if 'pattern_matches' in analysis:
                for pattern_type, matches in analysis['pattern_matches'].items():
                    for keyword, count in matches:
                        self.pattern_counts[pattern_type][keyword] += count

            # Aggregate outcomes
            if 'outcome_prediction' in analysis:
                outcome = analysis['outcome_prediction'].get('prediction', 'unclear')
                self.case_outcomes[outcome] += 1

            # Progress update every 25 files
            if i % 25 == 0:
                logger.info(f"📊 Progress: {i}/{len(text_files)} files analyzed")

        # Generate summary statistics
        results['summary_stats'] = self._generate_summary_stats(results['analysis_results'])
        results['pattern_summary'] = dict(self.pattern_counts)
        results['outcome_summary'] = dict(self.case_outcomes)

        logger.info(f"✅ Analysis complete: {results['analyzed_files']}/{results['total_files']} files analyzed")
        return results

    def _generate_summary_stats(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from analysis results."""
        if not analysis_results:
            return {}

        stats = {
            'total_files_analyzed': len(analysis_results),
            'total_text_length': sum(result.get('text_length', 0) for result in analysis_results.values()),
            'average_text_length': 0,
            'total_word_count': sum(result.get('word_count', 0) for result in analysis_results.values()),
            'average_word_count': 0,
            'entity_counts': Counter(),
            'court_distributions': Counter()
        }

        # Calculate averages
        text_lengths = [result.get('text_length', 0) for result in analysis_results.values()]
        word_counts = [result.get('word_count', 0) for result in analysis_results.values()]

        stats['average_text_length'] = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        stats['average_word_count'] = sum(word_counts) / len(word_counts) if word_counts else 0

        # Count entities
        for result in analysis_results.values():
            for entity in result.get('entities', []):
                stats['entity_counts'][entity.get('label', 'UNKNOWN')] += 1

        return stats

    def save_results(self, results: Dict[str, Any]):
        """Save analysis results to files."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full results
        results_file = self.output_dir / f"1782_basic_analysis_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save summary report
        summary_file = self.output_dir / f"1782_basic_summary_{timestamp}.md"
        self._generate_summary_report(results, summary_file)

        logger.info(f"💾 Results saved to: {results_file}")
        logger.info(f"📋 Summary report saved to: {summary_file}")

    def _generate_summary_report(self, results: Dict[str, Any], output_file: Path):
        """Generate a markdown summary report."""
        from datetime import datetime
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 1782 Discovery Cases - Basic NLP Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overview
            f.write("## 📊 Overview\n\n")
            f.write(f"- **Total Files Analyzed:** {results['analyzed_files']}\n")
            f.write(f"- **Analysis Success Rate:** {(results['analyzed_files']/results['total_files'])*100:.1f}%\n\n")

            # Summary Statistics
            stats = results.get('summary_stats', {})
            f.write("## 📈 Summary Statistics\n\n")
            f.write(f"- **Average Text Length:** {stats.get('average_text_length', 0):.0f} characters\n")
            f.write(f"- **Average Word Count:** {stats.get('average_word_count', 0):.0f} words\n")
            f.write(f"- **Total Text Analyzed:** {stats.get('total_text_length', 0):,} characters\n")
            f.write(f"- **Total Words Analyzed:** {stats.get('total_word_count', 0):,} words\n\n")

            # Outcome Predictions
            f.write("## ⚖️ Case Outcome Predictions\n\n")
            outcome_counts = results.get('outcome_summary', {})
            total_outcomes = sum(outcome_counts.values())
            if total_outcomes > 0:
                for outcome, count in outcome_counts.most_common():
                    percentage = (count / total_outcomes) * 100
                    f.write(f"- **{outcome.title()}:** {count} cases ({percentage:.1f}%)\n")
            f.write("\n")

            # Legal Pattern Analysis
            f.write("## ⚖️ Legal Pattern Analysis\n\n")
            patterns = results.get('pattern_summary', {})
            for pattern_type, keyword_counts in patterns.items():
                f.write(f"### {pattern_type.replace('_', ' ').title()}\n\n")
                for keyword, count in keyword_counts.most_common(10):
                    f.write(f"- **{keyword}:** {count} occurrences\n")
                f.write("\n")

            # Entity Analysis
            f.write("## 🏷️ Entity Analysis\n\n")
            entity_counts = stats.get('entity_counts', {})
            for entity_type, count in entity_counts.most_common(10):
                f.write(f"- **{entity_type}:** {count} occurrences\n")
            f.write("\n")

            f.write("---\n")
            f.write("*Report generated by Basic 1782 Analyzer*\n")

def main():
    """Main execution function."""
    print("🚀 Starting Basic NLP Analysis for 1782 Discovery Cases")
    print("=" * 60)

    # Initialize analyzer
    analyzer = Basic1782Analyzer()

    # Analyze all texts
    results = analyzer.analyze_all_texts()

    # Save results
    analyzer.save_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 BASIC NLP ANALYSIS COMPLETE!")
    print(f"✅ Analyzed: {results['analyzed_files']}/{results['total_files']} files")

    # Outcome summary
    outcome_counts = results.get('outcome_summary', {})
    if outcome_counts:
        print("\n⚖️ OUTCOME PREDICTIONS:")
        total_outcomes = sum(outcome_counts.values())
        for outcome, count in outcome_counts.most_common():
            percentage = (count / total_outcomes) * 100
            print(f"  {outcome.title()}: {count} cases ({percentage:.1f}%)")

    # Pattern summary
    patterns = results.get('pattern_summary', {})
    if patterns:
        print("\n🔍 TOP LEGAL PATTERNS:")
        for pattern_type, keyword_counts in patterns.items():
            top_keyword = keyword_counts.most_common(1)[0] if keyword_counts else ("none", 0)
            print(f"  {pattern_type.replace('_', ' ').title()}: {top_keyword[0]} ({top_keyword[1]} times)")

    print(f"\n💾 Results saved to: data/case_law/analysis_results/")
    print("🎯 Ready for advanced pattern analysis!")

if __name__ == "__main__":
    main()
