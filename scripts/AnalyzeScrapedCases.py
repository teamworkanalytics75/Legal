#!/usr/bin/env python3
"""
Analyze Scraped Cases for §1782 Content

Analyze the scraped cases to identify which ones are actually about §1782 discovery.
"""

import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CaseAnalyzer:
    """Analyze scraped cases for §1782 content."""

    def __init__(self):
        """Initialize the analyzer."""
        self.data_dir = Path(__file__).parent.parent / "data" / "case_law"
        self.scraped_cases_file = self.data_dir / "courtlistener_scraped_1782_cases.json"

    def load_scraped_cases(self):
        """Load scraped cases."""
        try:
            with open(self.scraped_cases_file, 'r', encoding='utf-8') as f:
                cases = json.load(f)

            logger.info(f"Loaded {len(cases)} scraped cases")
            return cases

        except Exception as e:
            logger.error(f"Error loading scraped cases: {e}")
            return []

    def analyze_case_content(self, case):
        """Analyze a single case for §1782 content."""
        case_name = case.get('case_name', '').lower()
        case_text = case.get('case_text', '').lower()

        # §1782 indicators
        indicators = {
            'section_1782': ['1782', 'section 1782', '28 usc 1782', '28 u.s.c. 1782'],
            'discovery_terms': ['international discovery', 'foreign discovery', 'judicial assistance', 'letter rogatory'],
            'application_terms': ['application', 'pursuant to', 'in re', 'matter of'],
            'foreign_proceedings': ['foreign proceeding', 'foreign tribunal', 'international tribunal', 'arbitration']
        }

        scores = {}
        for category, terms in indicators.items():
            score = 0
            for term in terms:
                if term in case_name:
                    score += 2  # Higher weight for case name matches
                if term in case_text:
                    score += 1
            scores[category] = score

        # Calculate total score
        total_score = sum(scores.values())

        # Determine if it's likely a §1782 case
        is_1782 = (
            scores['section_1782'] > 0 or  # Direct §1782 reference
            (scores['discovery_terms'] > 0 and scores['application_terms'] > 0) or  # Discovery + application
            (scores['foreign_proceedings'] > 0 and scores['application_terms'] > 0)  # Foreign proceeding + application
        )

        return {
            'scores': scores,
            'total_score': total_score,
            'is_1782': is_1782,
            'confidence': 'high' if total_score >= 3 else 'medium' if total_score >= 1 else 'low'
        }

    def analyze_all_cases(self):
        """Analyze all scraped cases."""
        cases = self.load_scraped_cases()
        if not cases:
            return

        logger.info("Analyzing all cases for §1782 content...")

        analyzed_cases = []
        verified_1782 = []
        likely_1782 = []
        unclear = []

        for i, case in enumerate(cases, 1):
            logger.info(f"Analyzing case {i}/{len(cases)}: {case.get('case_name', 'Unknown')[:80]}...")

            analysis = self.analyze_case_content(case)
            case['analysis'] = analysis

            analyzed_cases.append(case)

            # Categorize based on analysis
            if analysis['is_1782'] and analysis['confidence'] == 'high':
                verified_1782.append(case)
            elif analysis['is_1782'] and analysis['confidence'] == 'medium':
                likely_1782.append(case)
            else:
                unclear.append(case)

        # Log results
        logger.info(f"\nAnalysis Results:")
        logger.info(f"Verified §1782 cases: {len(verified_1782)}")
        logger.info(f"Likely §1782 cases: {len(likely_1782)}")
        logger.info(f"Unclear cases: {len(unclear)}")

        # Show sample verified cases
        logger.info(f"\nSample verified §1782 cases:")
        for i, case in enumerate(verified_1782[:5], 1):
            logger.info(f"  {i}. {case.get('case_name', 'Unknown')}")
            logger.info(f"     Score: {case['analysis']['total_score']}")
            logger.info(f"     Confidence: {case['analysis']['confidence']}")

        # Save results
        self.save_analysis_results(analyzed_cases, verified_1782, likely_1782, unclear)

        return analyzed_cases, verified_1782, likely_1782, unclear

    def save_analysis_results(self, all_cases, verified_1782, likely_1782, unclear):
        """Save analysis results."""
        try:
            # Save all analyzed cases
            all_cases_path = self.data_dir / "analyzed_1782_cases.json"
            with open(all_cases_path, 'w', encoding='utf-8') as f:
                json.dump(all_cases, f, indent=2, ensure_ascii=False)

            # Save verified §1782 cases
            verified_path = self.data_dir / "verified_1782_cases_analyzed.json"
            with open(verified_path, 'w', encoding='utf-8') as f:
                json.dump(verified_1782, f, indent=2, ensure_ascii=False)

            # Save likely §1782 cases
            likely_path = self.data_dir / "likely_1782_cases.json"
            with open(likely_path, 'w', encoding='utf-8') as f:
                json.dump(likely_1782, f, indent=2, ensure_ascii=False)

            # Save summary
            summary = {
                'total_cases_analyzed': len(all_cases),
                'verified_1782_cases': len(verified_1782),
                'likely_1782_cases': len(likely_1782),
                'unclear_cases': len(unclear),
                'success_rate': f"{len(verified_1782 + likely_1782)}/{len(all_cases)} ({((len(verified_1782) + len(likely_1782)) / len(all_cases) * 100):.1f}%)"
            }

            summary_path = self.data_dir / "case_analysis_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"Analysis results saved:")
            logger.info(f"  - All cases: {all_cases_path}")
            logger.info(f"  - Verified §1782: {verified_path}")
            logger.info(f"  - Likely §1782: {likely_path}")
            logger.info(f"  - Summary: {summary_path}")

        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")


def main():
    """Main entry point."""
    analyzer = CaseAnalyzer()
    all_cases, verified_1782, likely_1782, unclear = analyzer.analyze_all_cases()

    if all_cases:
        print(f"\nANALYSIS COMPLETE!")
        print(f"Total cases analyzed: {len(all_cases)}")
        print(f"Verified §1782 cases: {len(verified_1782)}")
        print(f"Likely §1782 cases: {len(likely_1782)}")
        print(f"Unclear cases: {len(unclear)}")
        print(f"Success rate: {len(verified_1782 + likely_1782)}/{len(all_cases)} ({((len(verified_1782) + len(likely_1782)) / len(all_cases) * 100):.1f}%)")


if __name__ == "__main__":
    main()
