#!/usr/bin/env python3
"""
Court Outcome Extractor

This script extracts the actual court decisions (granted/denied) from the case text,
replacing our pattern-based predictions with real outcomes.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CourtOutcomeExtractor:
    """Extract actual court outcomes from case text."""

    def __init__(self):
        self.corpus_dir = Path("data/case_law/1782_discovery")

        # Patterns to find court decisions
        self.decision_patterns = {
            'grant_patterns': [
                r'granted\s+(?:the\s+)?(?:application|petition|motion|request)',
                r'(?:application|petition|motion|request)\s+(?:is\s+)?granted',
                r'we\s+grant',
                r'the\s+court\s+grants',
                r'order\s+granting',
                r'granting\s+(?:the\s+)?(?:application|petition|motion)',
                r'it\s+is\s+(?:hereby\s+)?ordered\s+that\s+(?:the\s+)?(?:application|petition|motion)\s+is\s+granted',
                r'(?:application|petition|motion)\s+is\s+hereby\s+granted',
            ],
            'deny_patterns': [
                r'denied\s+(?:the\s+)?(?:application|petition|motion|request)',
                r'(?:application|petition|motion|request)\s+(?:is\s+)?denied',
                r'we\s+deny',
                r'the\s+court\s+denies',
                r'order\s+denying',
                r'denying\s+(?:the\s+)?(?:application|petition|motion)',
                r'it\s+is\s+(?:hereby\s+)?ordered\s+that\s+(?:the\s+)?(?:application|petition|motion)\s+is\s+denied',
                r'(?:application|petition|motion)\s+is\s+hereby\s+denied',
            ],
            'affirm_patterns': [
                r'affirmed',
                r'affirm\s+(?:the\s+)?(?:order|judgment|decision)',
                r'the\s+(?:order|judgment|decision)\s+is\s+affirmed',
            ],
            'reverse_patterns': [
                r'reversed',
                r'reverse\s+(?:the\s+)?(?:order|judgment|decision)',
                r'the\s+(?:order|judgment|decision)\s+is\s+reversed',
            ],
            'vacate_patterns': [
                r'vacated',
                r'vacate\s+(?:the\s+)?(?:order|judgment|decision)',
                r'the\s+(?:order|judgment|decision)\s+is\s+vacated',
            ]
        }

        # Compile patterns
        for category, patterns in self.decision_patterns.items():
            self.decision_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def extract_outcome_from_text(self, text: str, case_name: str) -> Dict[str, Any]:
        """Extract the actual court outcome from case text."""
        outcome_analysis = {
            'outcome': 'unclear',
            'confidence': 0.0,
            'evidence': [],
            'decision_context': '',
            'method': 'pattern_matching'
        }

        if not text or len(text.strip()) < 100:
            return outcome_analysis

        # Find all decision patterns
        decision_matches = {}
        for category, patterns in self.decision_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(text)
                matches.extend(found)
            decision_matches[category] = matches

        # Analyze the matches
        grant_matches = decision_matches['grant_patterns']
        deny_matches = decision_matches['deny_patterns']
        affirm_matches = decision_matches['affirm_patterns']
        reverse_matches = decision_matches['reverse_patterns']
        vacate_matches = decision_matches['vacate_patterns']

        # Determine outcome based on matches
        if grant_matches and not deny_matches:
            outcome_analysis['outcome'] = 'granted'
            outcome_analysis['confidence'] = 0.9
            outcome_analysis['evidence'] = grant_matches[:3]  # First 3 matches
        elif deny_matches and not grant_matches:
            outcome_analysis['outcome'] = 'denied'
            outcome_analysis['confidence'] = 0.9
            outcome_analysis['evidence'] = deny_matches[:3]
        elif grant_matches and deny_matches:
            # Both present - need to analyze context
            if len(grant_matches) > len(deny_matches):
                outcome_analysis['outcome'] = 'granted'
                outcome_analysis['confidence'] = 0.7
                outcome_analysis['evidence'] = grant_matches[:2]
            elif len(deny_matches) > len(grant_matches):
                outcome_analysis['outcome'] = 'denied'
                outcome_analysis['confidence'] = 0.7
                outcome_analysis['evidence'] = deny_matches[:2]
            else:
                outcome_analysis['outcome'] = 'mixed'
                outcome_analysis['confidence'] = 0.5
                outcome_analysis['evidence'] = grant_matches[:1] + deny_matches[:1]
        elif affirm_matches:
            outcome_analysis['outcome'] = 'affirmed'
            outcome_analysis['confidence'] = 0.8
            outcome_analysis['evidence'] = affirm_matches[:2]
        elif reverse_matches:
            outcome_analysis['outcome'] = 'reversed'
            outcome_analysis['confidence'] = 0.8
            outcome_analysis['evidence'] = reverse_matches[:2]
        elif vacate_matches:
            outcome_analysis['outcome'] = 'vacated'
            outcome_analysis['confidence'] = 0.8
            outcome_analysis['evidence'] = vacate_matches[:2]
        else:
            # Try to find decision context manually
            decision_context = self._find_decision_context(text)
            if decision_context:
                outcome_analysis['decision_context'] = decision_context
                outcome_analysis['method'] = 'context_analysis'
                outcome_analysis['confidence'] = 0.3

        return outcome_analysis

    def _find_decision_context(self, text: str) -> Optional[str]:
        """Find decision context when patterns don't match."""
        # Look for common decision phrases
        decision_phrases = [
            r'for the foregoing reasons.*?(?:granted|denied|affirmed|reversed)',
            r'accordingly.*?(?:granted|denied|affirmed|reversed)',
            r'therefore.*?(?:granted|denied|affirmed|reversed)',
            r'it is ordered.*?(?:granted|denied|affirmed|reversed)',
            r'the court.*?(?:grants|denies|affirms|reverses)',
        ]

        for phrase in decision_phrases:
            match = re.search(phrase, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0)[:200]  # First 200 chars

        return None

    def analyze_case_file(self, case_file: Path) -> Dict[str, Any]:
        """Analyze a single case file for outcome."""
        try:
            with open(case_file, 'r', encoding='utf-8') as f:
                case_data = json.load(f)

            case_name = case_data.get('caseName', case_file.stem)
            text = case_data.get('extracted_text', '')

            if not text or len(text.strip()) < 100:
                return {
                    'file_name': case_file.name,
                    'case_name': case_name,
                    'outcome': 'no_text',
                    'confidence': 0.0,
                    'evidence': [],
                    'error': 'No extractable text'
                }

            # Extract outcome
            outcome_analysis = self.extract_outcome_from_text(text, case_name)

            return {
                'file_name': case_file.name,
                'case_name': case_name,
                'text_length': len(text),
                'outcome': outcome_analysis['outcome'],
                'confidence': outcome_analysis['confidence'],
                'evidence': outcome_analysis['evidence'],
                'decision_context': outcome_analysis.get('decision_context', ''),
                'method': outcome_analysis['method']
            }

        except Exception as e:
            logger.error(f"Error analyzing {case_file.name}: {e}")
            return {
                'file_name': case_file.name,
                'case_name': case_file.stem,
                'outcome': 'error',
                'confidence': 0.0,
                'evidence': [],
                'error': str(e)
            }

    def run_outcome_extraction(self) -> None:
        """Extract outcomes from all cases with text."""
        logger.info("="*80)
        logger.info("EXTRACTING ACTUAL COURT OUTCOMES")
        logger.info("="*80)

        # Get all case files with text
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

        # Analyze each case
        results = []
        outcome_counts = {}

        for i, case_file in enumerate(case_files, 1):
            logger.info(f"Analyzing case {i}/{len(case_files)}: {case_file.name}")

            result = self.analyze_case_file(case_file)
            results.append(result)

            # Count outcomes
            outcome = result['outcome']
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

            # Log interesting cases
            if result['confidence'] > 0.7:
                logger.info(f"  ✓ {outcome.upper()} (confidence: {result['confidence']:.1f})")
                if result['evidence']:
                    logger.info(f"    Evidence: {result['evidence'][0]}")
            elif result['outcome'] == 'unclear':
                logger.info(f"  ? UNCLEAR - needs manual review")

        # Generate summary
        logger.info("\n" + "="*80)
        logger.info("OUTCOME EXTRACTION SUMMARY")
        logger.info("="*80)

        total_cases = len(results)
        logger.info(f"Total cases analyzed: {total_cases}")

        for outcome, count in sorted(outcome_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_cases * 100
            logger.info(f"{outcome.upper()}: {count} cases ({percentage:.1f}%)")

        # High confidence cases
        high_confidence = [r for r in results if r['confidence'] > 0.7]
        logger.info(f"\nHigh confidence outcomes: {len(high_confidence)} cases")

        # Cases needing manual review
        unclear_cases = [r for r in results if r['outcome'] in ['unclear', 'mixed', 'no_text']]
        logger.info(f"Cases needing manual review: {len(unclear_cases)} cases")

        # Save results
        self._save_results(results, outcome_counts)

        logger.info("\n🎉 Outcome extraction completed!")

    def _save_results(self, results: List[Dict], outcome_counts: Dict) -> None:
        """Save extraction results."""
        # Save detailed results
        results_path = Path("data/case_law/court_outcomes_extracted.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'extraction_date': datetime.now().isoformat(),
                'total_cases': len(results),
                'outcome_counts': outcome_counts,
                'results': results
            }, f, indent=2, ensure_ascii=False)

        # Save summary report
        report_path = Path("data/case_law/court_outcomes_report.md")
        report_content = f"""# 🏛️ Court Outcomes Extraction Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary

| Metric | Value |
|--------|-------|
| **Total Cases Analyzed** | {len(results)} |
| **High Confidence Outcomes** | {len([r for r in results if r['confidence'] > 0.7])} |
| **Cases Needing Manual Review** | {len([r for r in results if r['outcome'] in ['unclear', 'mixed', 'no_text']])} |

## 🎯 Outcome Distribution

"""

        for outcome, count in sorted(outcome_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(results) * 100
            report_content += f"- **{outcome.title()}**: {count} cases ({percentage:.1f}%)\n"

        report_content += f"""
## 🔍 High Confidence Cases

"""

        high_confidence = [r for r in results if r['confidence'] > 0.7]
        for result in high_confidence[:10]:  # Show first 10
            report_content += f"- **{result['case_name']}**: {result['outcome'].upper()} (confidence: {result['confidence']:.1f})\n"

        report_content += f"""
## ❓ Cases Needing Manual Review

"""

        unclear_cases = [r for r in results if r['outcome'] in ['unclear', 'mixed', 'no_text']]
        for result in unclear_cases[:10]:  # Show first 10
            report_content += f"- **{result['case_name']}**: {result['outcome']} - {result.get('error', 'No error message')}\n"

        report_content += f"""
## 📁 Data Files

- **Detailed Results**: `data/case_law/court_outcomes_extracted.json`
- **This Report**: `data/case_law/court_outcomes_report.md`

---

**This report provides actual court outcomes extracted from case text, replacing pattern-based predictions with real decisions.**
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"✓ Detailed results saved to: {results_path}")
        logger.info(f"✓ Summary report saved to: {report_path}")


def main():
    """Main entry point."""
    logger.info("Starting court outcome extraction...")

    extractor = CourtOutcomeExtractor()
    extractor.run_outcome_extraction()


if __name__ == "__main__":
    main()
