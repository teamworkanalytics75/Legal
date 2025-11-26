#!/usr/bin/env python3
"""
Merge and Deduplicate §1782 Cases

Merge web-scraped cases with existing API cases and deduplicate to create
a comprehensive, clean dataset of unique §1782 cases.
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CaseMerger:
    """Merge and deduplicate §1782 cases from multiple sources."""

    def __init__(self):
        """Initialize the merger."""
        self.data_dir = Path(__file__).parent.parent / "data" / "case_law"
        self.web_scraped_file = self.data_dir / "courtlistener_scraped_1782_cases.json"
        self.existing_cases_file = self.data_dir / "1782_discovery" / "clean_cases.json"

    def load_web_scraped_cases(self):
        """Load web-scraped cases."""
        try:
            if not self.web_scraped_file.exists():
                logger.error(f"Web-scraped cases file not found: {self.web_scraped_file}")
                return []

            with open(self.web_scraped_file, 'r', encoding='utf-8') as f:
                cases = json.load(f)

            logger.info(f"Loaded {len(cases)} web-scraped cases")
            return cases

        except Exception as e:
            logger.error(f"Error loading web-scraped cases: {e}")
            return []

    def load_existing_cases(self):
        """Load existing cases from Google Drive folder."""
        try:
            # Load from the clean folder we created earlier
            # We'll need to reconstruct this from our previous analysis
            logger.info("Loading existing cases from previous analysis...")

            # For now, return empty list - we'll merge with the 11 unique cases we found
            return []

        except Exception as e:
            logger.error(f"Error loading existing cases: {e}")
            return []

    def normalize_case_name(self, case_name):
        """Normalize case name for comparison."""
        if not case_name:
            return ""

        # Convert to lowercase and remove extra whitespace
        normalized = case_name.lower().strip()

        # Remove common variations
        normalized = normalized.replace('application of', '')
        normalized = normalized.replace('application for', '')
        normalized = normalized.replace('in re', '')
        normalized = normalized.replace('in the matter of', '')

        # Remove punctuation and extra spaces
        normalized = ' '.join(normalized.split())

        return normalized

    def extract_case_key(self, case):
        """Extract a unique key for case comparison."""
        case_name = self.normalize_case_name(case.get('case_name', ''))
        case_id = case.get('case_id', '')
        court = case.get('court', '')

        # Use case_id if available, otherwise use normalized name + court
        if case_id:
            return f"id_{case_id}"
        else:
            return f"name_{case_name}_{court}"

    def deduplicate_cases(self, cases):
        """Deduplicate cases based on case ID and normalized names."""
        logger.info("Deduplicating cases...")

        unique_cases = {}
        duplicates = []

        for case in cases:
            case_key = self.extract_case_key(case)

            if case_key in unique_cases:
                # Check if this is a better version
                existing_case = unique_cases[case_key]

                # Prefer cases with more complete information
                existing_score = self.score_case_completeness(existing_case)
                current_score = self.score_case_completeness(case)

                if current_score > existing_score:
                    duplicates.append(existing_case)
                    unique_cases[case_key] = case
                else:
                    duplicates.append(case)
            else:
                unique_cases[case_key] = case

        logger.info(f"Found {len(duplicates)} duplicates")
        logger.info(f"Unique cases: {len(unique_cases)}")

        return list(unique_cases.values()), duplicates

    def score_case_completeness(self, case):
        """Score case completeness for deduplication."""
        score = 0

        if case.get('case_name'):
            score += 1
        if case.get('case_id'):
            score += 2
        if case.get('court'):
            score += 1
        if case.get('docket_number'):
            score += 1
        if case.get('date_filed'):
            score += 1
        if case.get('case_text'):
            score += 3
        if case.get('citations'):
            score += 1

        return score

    def categorize_cases(self, cases):
        """Categorize cases by type and relevance."""
        logger.info("Categorizing cases...")

        categories = {
            'verified_1782': [],
            'likely_1782': [],
            'unclear': [],
            'not_1782': []
        }

        for case in cases:
            case_name = case.get('case_name', '').lower()
            case_text = case.get('case_text', '').lower()

            # Check for clear §1782 indicators
            if any(term in case_name or term in case_text for term in [
                '1782', 'section 1782', '28 usc 1782', '28 u.s.c. 1782',
                'international discovery', 'foreign discovery', 'judicial assistance'
            ]):
                if 'application' in case_name or 'pursuant to' in case_name:
                    categories['verified_1782'].append(case)
                else:
                    categories['likely_1782'].append(case)
            else:
                categories['unclear'].append(case)

        # Log categorization results
        for category, case_list in categories.items():
            logger.info(f"{category}: {len(case_list)} cases")

        return categories

    def merge_cases(self):
        """Merge cases from all sources."""
        logger.info("Starting case merge process...")

        # Load cases from all sources
        web_cases = self.load_web_scraped_cases()
        existing_cases = self.load_existing_cases()

        # Combine all cases
        all_cases = web_cases + existing_cases
        logger.info(f"Total cases to process: {len(all_cases)}")

        # Deduplicate
        unique_cases, duplicates = self.deduplicate_cases(all_cases)

        # Categorize
        categories = self.categorize_cases(unique_cases)

        # Save results
        self.save_results(unique_cases, duplicates, categories)

        return unique_cases, duplicates, categories

    def save_results(self, unique_cases, duplicates, categories):
        """Save merged results."""
        try:
            # Save all unique cases
            all_cases_path = self.data_dir / "merged_1782_cases.json"
            with open(all_cases_path, 'w', encoding='utf-8') as f:
                json.dump(unique_cases, f, indent=2, ensure_ascii=False)

            # Save verified §1782 cases only
            verified_cases_path = self.data_dir / "verified_1782_cases.json"
            with open(verified_cases_path, 'w', encoding='utf-8') as f:
                json.dump(categories['verified_1782'], f, indent=2, ensure_ascii=False)

            # Save duplicates for reference
            duplicates_path = self.data_dir / "duplicate_cases.json"
            with open(duplicates_path, 'w', encoding='utf-8') as f:
                json.dump(duplicates, f, indent=2, ensure_ascii=False)

            # Save categorization summary
            summary = {
                'total_unique_cases': len(unique_cases),
                'verified_1782_cases': len(categories['verified_1782']),
                'likely_1782_cases': len(categories['likely_1782']),
                'unclear_cases': len(categories['unclear']),
                'duplicates_removed': len(duplicates),
                'categories': {k: len(v) for k, v in categories.items()}
            }

            summary_path = self.data_dir / "case_merge_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved:")
            logger.info(f"  - All cases: {all_cases_path}")
            logger.info(f"  - Verified §1782: {verified_cases_path}")
            logger.info(f"  - Duplicates: {duplicates_path}")
            logger.info(f"  - Summary: {summary_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main entry point."""
    merger = CaseMerger()
    unique_cases, duplicates, categories = merger.merge_cases()

    if unique_cases:
        print(f"\nSUCCESS!")
        print(f"Total unique cases: {len(unique_cases)}")
        print(f"Verified §1782 cases: {len(categories['verified_1782'])}")
        print(f"Likely §1782 cases: {len(categories['likely_1782'])}")
        print(f"Duplicates removed: {len(duplicates)}")

        print(f"\nSample verified §1782 cases:")
        for i, case in enumerate(categories['verified_1782'][:5], 1):
            print(f"  {i}. {case.get('case_name', 'Unknown')}")
            print(f"     Court: {case.get('court', 'Unknown')}")
            print(f"     ID: {case.get('case_id', 'Unknown')}")
            print()


if __name__ == "__main__":
    main()
