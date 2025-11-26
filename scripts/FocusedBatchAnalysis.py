#!/usr/bin/env python3
"""
Focused Batch NLP Analysis

This script performs focused NLP analysis on the newly extracted text,
building on existing patterns discovered in the PDF analysis.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FocusedNLPAnalyzer:
    """Focused NLP analysis building on existing patterns."""

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.corpus_dir = Path("data/case_law/1782_discovery")

        # Load existing patterns from PDF analysis
        self.existing_patterns = self._load_existing_patterns()

        # Legal patterns (enhanced based on existing analysis)
        self.legal_patterns = {
            'section_1782': [
                r'28\s*u\.s\.c\.?\s*(?:\u00a7)?\s*1782',
                r'section\s*1782',
                r'(?:\u00a7)\s*1782',
            ],
            'intel_corp': [
                r'intel\s+corp',
                r'intel\s+corporation',
            ],
            'discovery_terms': [
                r'discovery',
                r'subpoena',
                r'deposition',
                r'evidence',
                r'document\s+production',
            ],
            'foreign_proceedings': [
                r'foreign\s+tribunal',
                r'international\s+proceeding',
                r'foreign\s+court',
                r'arbitration',
                r'foreign\s+litigation',
            ],
            'outcome_indicators': [
                r'granted',
                r'denied',
                r'denying',
                r'granting',
                r'approve',
                r'approval',
                r'reject',
                r'rejection',
            ]
        }

        # Compile patterns
        for category, patterns in self.legal_patterns.items():
            self.legal_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def _load_existing_patterns(self) -> Dict[str, Any]:
        """Load patterns from existing PDF analysis."""
        try:
            with open("data/case_law/simple_pdf_analysis.json", 'r', encoding='utf-8') as f:
                pdf_data = json.load(f)

            patterns = {
                'total_cases': len(pdf_data),
                'outcomes': Counter(),
                'intel_factors': defaultdict(int),
                'avg_text_length': 0,
                'avg_section_1782_mentions': 0
            }

            text_lengths = []
            section_1782_counts = []

            for case in pdf_data:
                patterns['outcomes'][case['outcome']['outcome']] += 1

                for factor, value in case['intel_factors'].items():
                    if value:
                        patterns['intel_factors'][factor] += 1

                text_lengths.append(case['text_length'])
                section_1782_counts.append(case['section_1782_mentions'])

            patterns['avg_text_length'] = sum(text_lengths) / len(text_lengths)
            patterns['avg_section_1782_mentions'] = sum(section_1782_counts) / len(section_1782_counts)

            logger.info(f"Loaded existing patterns from {patterns['total_cases']} PDF cases")
            return patterns

        except Exception as e:
            logger.warning(f"Could not load existing patterns: {e}")
            return {}

    def get_cases_with_text(self) -> List[Path]:
        """Get all case files that have extracted text."""
        case_files = []

        for case_file in self.corpus_dir.glob("*.json"):
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                if case_data.get('extracted_text') and len(case_data['extracted_text'].strip()) > 100:
                    case_files.append(case_file)

            except Exception as e:
                logger.error(f"Error reading {case_file.name}: {e}")

        logger.info(f"Found {len(case_files)} cases with extractable text")
        return case_files

    def analyze_text_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text for legal patterns."""
        analysis = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'pattern_matches': {},
            'outcome_prediction': 'unclear',
            'confidence': 0.0,
            'notable_findings': []
        }

        # Count pattern matches
        for category, patterns in self.legal_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            analysis['pattern_matches'][category] = len(matches)

        # Predict outcome based on patterns
        granted_indicators = analysis['pattern_matches']['outcome_indicators']
        section_1782_count = analysis['pattern_matches']['section_1782']
        intel_count = analysis['pattern_matches']['intel_corp']

        # Simple outcome prediction logic
        if section_1782_count > 10:  # High §1782 mentions suggest detailed analysis
            if intel_count > 0:  # Intel Corp citation suggests precedent
                analysis['outcome_prediction'] = 'likely_granted'
                analysis['confidence'] = 0.7
            else:
                analysis['outcome_prediction'] = 'likely_denied'
                analysis['confidence'] = 0.6
        else:
            analysis['outcome_prediction'] = 'unclear'
            analysis['confidence'] = 0.3

        # Notable findings
        if section_1782_count > 20:
            analysis['notable_findings'].append(f"High §1782 references ({section_1782_count})")

        if intel_count > 0:
            analysis['notable_findings'].append(f"Intel Corp citation ({intel_count})")

        if analysis['pattern_matches']['foreign_proceedings'] > 5:
            analysis['notable_findings'].append("Strong foreign proceeding focus")

        return analysis

    def analyze_batch(self, case_files: List[Path], batch_num: int) -> Dict[str, Any]:
        """Analyze a batch of cases."""
        logger.info(f"Analyzing batch {batch_num} ({len(case_files)} cases)...")

        batch_results = {
            'batch_number': batch_num,
            'cases_analyzed': len(case_files),
            'case_analyses': [],
            'batch_patterns': {},
            'comparison_to_existing': {}
        }

        total_text_length = 0
        all_pattern_matches = defaultdict(int)
        outcome_predictions = Counter()
        notable_cases = []

        for case_file in case_files:
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                text = case_data.get('extracted_text', '')
                case_name = case_data.get('caseName', case_file.stem)

                # Analyze the text
                analysis = self.analyze_text_patterns(text)
                analysis['case_name'] = case_name
                analysis['file_name'] = case_file.name

                batch_results['case_analyses'].append(analysis)

                # Aggregate data
                total_text_length += analysis['text_length']
                outcome_predictions[analysis['outcome_prediction']] += 1

                # Aggregate pattern matches
                for category, count in analysis['pattern_matches'].items():
                    all_pattern_matches[category] += count

                # Identify notable cases
                if analysis['notable_findings']:
                    notable_cases.append({
                        'case_name': case_name,
                        'findings': analysis['notable_findings'],
                        'confidence': analysis['confidence']
                    })

            except Exception as e:
                logger.error(f"Error analyzing {case_file.name}: {e}")

        # Calculate batch statistics
        batch_results['batch_patterns'] = {
            'total_text_length': total_text_length,
            'pattern_totals': dict(all_pattern_matches),
            'outcome_predictions': dict(outcome_predictions),
            'notable_cases': notable_cases,
            'avg_text_length': total_text_length / len(case_files) if case_files else 0
        }

        # Compare to existing patterns
        if self.existing_patterns:
            batch_results['comparison_to_existing'] = {
                'text_length_comparison': {
                    'batch_avg': batch_results['batch_patterns']['avg_text_length'],
                    'existing_avg': self.existing_patterns['avg_text_length'],
                    'difference': batch_results['batch_patterns']['avg_text_length'] - self.existing_patterns['avg_text_length']
                },
                'section_1782_comparison': {
                    'batch_total': all_pattern_matches['section_1782'],
                    'batch_avg': all_pattern_matches['section_1782'] / len(case_files) if case_files else 0,
                    'existing_avg': self.existing_patterns['avg_section_1782_mentions']
                }
            }

        return batch_results

    def run_batch_analysis(self) -> None:
        """Run focused batch analysis."""
        logger.info("="*80)
        logger.info("STARTING FOCUSED BATCH NLP ANALYSIS")
        logger.info("="*80)

        case_files = self.get_cases_with_text()
        total_cases = len(case_files)

        logger.info(f"Found {total_cases} cases with text content")
        logger.info(f"Processing in batches of {self.batch_size}")

        if self.existing_patterns:
            logger.info(f"Building on existing analysis of {self.existing_patterns['total_cases']} PDF cases")
            logger.info(f"Existing patterns: {dict(self.existing_patterns['outcomes'])}")

        # Process in batches
        for i in range(0, total_cases, self.batch_size):
            batch_files = case_files[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESSING BATCH {batch_num}")
            logger.info(f"{'='*60}")

            # Analyze batch
            batch_results = self.analyze_batch(batch_files, batch_num)

            # Report batch findings
            self._report_batch_findings(batch_results, batch_num)

        logger.info("\n🎉 Focused batch analysis completed!")

    def _report_batch_findings(self, batch_results: Dict, batch_num: int) -> None:
        """Report findings from a batch."""
        logger.info(f"\n🔍 BATCH {batch_num} FINDINGS:")
        logger.info(f"Cases analyzed: {batch_results['cases_analyzed']}")
        logger.info(f"Total text length: {batch_results['batch_patterns']['total_text_length']:,} characters")
        logger.info(f"Average text length: {batch_results['batch_patterns']['avg_text_length']:,.0f} characters")

        # Pattern analysis
        patterns = batch_results['batch_patterns']['pattern_totals']
        logger.info(f"\n📊 Legal Pattern Analysis:")
        for category, count in patterns.items():
            logger.info(f"  {category.replace('_', ' ').title()}: {count} matches")

        # Outcome predictions
        outcomes = batch_results['batch_patterns']['outcome_predictions']
        logger.info(f"\n🎯 Outcome Predictions:")
        for outcome, count in outcomes.items():
            percentage = count / batch_results['cases_analyzed'] * 100
            logger.info(f"  {outcome.replace('_', ' ').title()}: {count} cases ({percentage:.1f}%)")

        # Notable cases
        notable_cases = batch_results['batch_patterns']['notable_cases']
        if notable_cases:
            logger.info(f"\n⭐ Notable Cases:")
            for case in notable_cases[:3]:  # Show top 3
                logger.info(f"  {case['case_name']}: {', '.join(case['findings'])}")

        # Comparison to existing patterns
        if batch_results['comparison_to_existing']:
            comparison = batch_results['comparison_to_existing']
            logger.info(f"\n📈 Comparison to Existing PDF Analysis:")

            text_comp = comparison['text_length_comparison']
            logger.info(f"  Text Length: Batch avg {text_comp['batch_avg']:,.0f} vs Existing {text_comp['existing_avg']:,.0f}")

            section_comp = comparison['section_1782_comparison']
            logger.info(f"  §1782 References: Batch avg {section_comp['batch_avg']:.1f} vs Existing {section_comp['existing_avg']:.1f}")


def main():
    """Main entry point."""
    logger.info("Starting focused batch NLP analysis...")

    # Create analyzer
    analyzer = FocusedNLPAnalyzer(batch_size=64)

    # Run analysis
    analyzer.run_batch_analysis()


if __name__ == "__main__":
    main()
