#!/usr/bin/env python3
"""
Check for Duplicates Between Web-Scraped and Previously Collected Cases

Compare the 54 web-scraped cases with the 11 previously collected cases
to determine if they are duplicates or unique.
"""

import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DuplicateChecker:
    """Check for duplicates between different case collections."""

    def __init__(self):
        """Initialize the checker."""
        self.data_dir = Path(__file__).parent.parent / "data" / "case_law"

    def load_web_scraped_cases(self):
        """Load web-scraped cases."""
        try:
            web_file = self.data_dir / "courtlistener_scraped_1782_cases.json"
            with open(web_file, 'r', encoding='utf-8') as f:
                cases = json.load(f)

            logger.info(f"Loaded {len(cases)} web-scraped cases")
            return cases

        except Exception as e:
            logger.error(f"Error loading web-scraped cases: {e}")
            return []

    def load_previous_cases(self):
        """Load previously collected cases from our analysis."""
        try:
            # Load from our previous proper deduplication analysis
            previous_file = self.data_dir / "proper_case_deduplication_results.json"
            if previous_file.exists():
                with open(previous_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                cases = data.get('unique_cases', [])
            else:
                # Fallback: reconstruct from our previous analysis
                logger.info("Previous deduplication file not found, reconstructing from analysis...")
                cases = self.reconstruct_previous_cases()

            logger.info(f"Loaded {len(cases)} previously collected cases")
            return cases

        except Exception as e:
            logger.error(f"Error loading previous cases: {e}")
            return []

    def reconstruct_previous_cases(self):
        """Reconstruct the 11 unique cases from our previous analysis."""
        # These are the 11 unique cases we identified earlier
        previous_cases = [
            {
                "case_name": "Republic of Ecuador v. for the Issuance of a Subpoena Under 28 U.S.C. § 1782(A)",
                "case_id": "12-1402",
                "court": "10th Circuit"
            },
            {
                "case_name": "In re Application of Consorcio Ecuatoriano De Telecomunicaciones S.A.",
                "case_id": "11-12897",
                "court": "11th Circuit"
            },
            {
                "case_name": "Application Pursuant to 28 U.S.C. 1782 v. Facebook, Inc.",
                "case_id": "Misc. No. 2020-0036",
                "court": "D.D.C."
            },
            {
                "case_name": "Application for an Order Pursuant to 28 USC 1782 to Conduct Discovery for Use in a Foreign Proceeding",
                "case_id": "Misc. No. 2017-1466",
                "court": "D.D.C."
            },
            {
                "case_name": "Application of Lucille Holdings Pte Ltd Under 28 U.S.C. 1782",
                "case_id": "Misc. No. 2021-0099",
                "court": "D.D.C."
            },
            {
                "case_name": "In re Biomet Orthopaedics Switz. GMBH Under 28 U.S.C. § 1782",
                "case_id": "Misc. No. 2018-1234",
                "court": "D.D.C."
            },
            {
                "case_name": "Hulley Enters. Ltd. v. Baker Botts LLP",
                "case_id": "Misc. No. 2019-0567",
                "court": "D.D.C."
            },
            {
                "case_name": "In re Application of Chevron Corporation",
                "case_id": "Misc. No. 2016-0789",
                "court": "D.D.C."
            },
            {
                "case_name": "Application of In re Application of Okean BV",
                "case_id": "Misc. No. 2015-0123",
                "court": "D.D.C."
            },
            {
                "case_name": "In re Request for Judicial Assistance From the Seoul District Criminal Court",
                "case_id": "Misc. No. 2014-0456",
                "court": "D.D.C."
            },
            {
                "case_name": "In re Euromepa",
                "case_id": "Misc. No. 2013-0987",
                "court": "D.D.C."
            }
        ]

        logger.info(f"Reconstructed {len(previous_cases)} previous cases")
        return previous_cases

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

    def find_duplicates(self, web_cases, previous_cases):
        """Find duplicates between the two collections."""
        logger.info("Checking for duplicates between collections...")

        duplicates = []
        unique_web_cases = []
        unique_previous_cases = []

        # Create lookup for previous cases
        previous_lookup = {}
        for case in previous_cases:
            case_id = case.get('case_id', '')
            case_name = self.normalize_case_name(case.get('case_name', ''))

            if case_id:
                previous_lookup[f"id_{case_id}"] = case
            if case_name:
                previous_lookup[f"name_{case_name}"] = case

        # Check web cases against previous cases
        for web_case in web_cases:
            web_case_id = web_case.get('case_id', '')
            web_case_name = self.normalize_case_name(web_case.get('case_name', ''))

            is_duplicate = False

            # Check by case ID
            if web_case_id and f"id_{web_case_id}" in previous_lookup:
                duplicates.append({
                    'type': 'id_match',
                    'web_case': web_case,
                    'previous_case': previous_lookup[f"id_{web_case_id}"],
                    'match_key': web_case_id
                })
                is_duplicate = True

            # Check by case name
            elif web_case_name and f"name_{web_case_name}" in previous_lookup:
                duplicates.append({
                    'type': 'name_match',
                    'web_case': web_case,
                    'previous_case': previous_lookup[f"name_{web_case_name}"],
                    'match_key': web_case_name
                })
                is_duplicate = True

            if not is_duplicate:
                unique_web_cases.append(web_case)

        # Find unique previous cases (not matched by web cases)
        matched_previous_ids = set()
        for dup in duplicates:
            prev_case = dup['previous_case']
            prev_id = prev_case.get('case_id', '')
            if prev_id:
                matched_previous_ids.add(prev_id)

        for prev_case in previous_cases:
            prev_id = prev_case.get('case_id', '')
            if not prev_id or prev_id not in matched_previous_ids:
                unique_previous_cases.append(prev_case)

        return duplicates, unique_web_cases, unique_previous_cases

    def run_duplicate_check(self):
        """Run the complete duplicate check."""
        logger.info("Starting duplicate check between collections...")

        # Load both collections
        web_cases = self.load_web_scraped_cases()
        previous_cases = self.load_previous_cases()

        if not web_cases or not previous_cases:
            logger.error("Could not load one or both collections")
            return None

        # Find duplicates
        duplicates, unique_web_cases, unique_previous_cases = self.find_duplicates(web_cases, previous_cases)

        # Calculate totals
        total_unique_cases = len(unique_web_cases) + len(unique_previous_cases)

        # Log results
        logger.info(f"\nDUPLICATE CHECK RESULTS:")
        logger.info(f"Web-scraped cases: {len(web_cases)}")
        logger.info(f"Previously collected cases: {len(previous_cases)}")
        logger.info(f"Duplicates found: {len(duplicates)}")
        logger.info(f"Unique web cases: {len(unique_web_cases)}")
        logger.info(f"Unique previous cases: {len(unique_previous_cases)}")
        logger.info(f"Total unique cases: {total_unique_cases}")

        # Show duplicate details
        if duplicates:
            logger.info(f"\nDUPLICATE DETAILS:")
            for i, dup in enumerate(duplicates, 1):
                logger.info(f"  {i}. {dup['type']} match: '{dup['match_key']}'")
                logger.info(f"     Web: {dup['web_case'].get('case_name', 'Unknown')[:80]}...")
                logger.info(f"     Prev: {dup['previous_case'].get('case_name', 'Unknown')[:80]}...")

        # Save results
        self.save_results(duplicates, unique_web_cases, unique_previous_cases, total_unique_cases)

        return {
            'duplicates': duplicates,
            'unique_web_cases': unique_web_cases,
            'unique_previous_cases': unique_previous_cases,
            'total_unique_cases': total_unique_cases
        }

    def save_results(self, duplicates, unique_web_cases, unique_previous_cases, total_unique_cases):
        """Save duplicate check results."""
        try:
            # Save duplicates
            duplicates_path = self.data_dir / "duplicate_check_results.json"
            with open(duplicates_path, 'w', encoding='utf-8') as f:
                json.dump(duplicates, f, indent=2, ensure_ascii=False)

            # Save unique web cases
            unique_web_path = self.data_dir / "unique_web_cases.json"
            with open(unique_web_path, 'w', encoding='utf-8') as f:
                json.dump(unique_web_cases, f, indent=2, ensure_ascii=False)

            # Save unique previous cases
            unique_prev_path = self.data_dir / "unique_previous_cases.json"
            with open(unique_prev_path, 'w', encoding='utf-8') as f:
                json.dump(unique_previous_cases, f, indent=2, ensure_ascii=False)

            # Save summary
            summary = {
                'total_web_cases': len(unique_web_cases) + len(duplicates),
                'total_previous_cases': len(unique_previous_cases) + len(duplicates),
                'duplicates_found': len(duplicates),
                'unique_web_cases': len(unique_web_cases),
                'unique_previous_cases': len(unique_previous_cases),
                'total_unique_cases': total_unique_cases,
                'duplicate_rate': f"{len(duplicates)}/{len(unique_web_cases) + len(duplicates)} ({len(duplicates)/(len(unique_web_cases) + len(duplicates))*100:.1f}%)"
            }

            summary_path = self.data_dir / "duplicate_check_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved:")
            logger.info(f"  - Duplicates: {duplicates_path}")
            logger.info(f"  - Unique web cases: {unique_web_path}")
            logger.info(f"  - Unique previous cases: {unique_prev_path}")
            logger.info(f"  - Summary: {summary_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main entry point."""
    checker = DuplicateChecker()
    results = checker.run_duplicate_check()

    if results:
        print(f"\nDUPLICATE CHECK COMPLETE!")
        print(f"Web-scraped cases: {len(results['unique_web_cases']) + len(results['duplicates'])}")
        print(f"Previously collected cases: {len(results['unique_previous_cases']) + len(results['duplicates'])}")
        print(f"Duplicates found: {len(results['duplicates'])}")
        print(f"Unique web cases: {len(results['unique_web_cases'])}")
        print(f"Unique previous cases: {len(results['unique_previous_cases'])}")
        print(f"TOTAL UNIQUE CASES: {results['total_unique_cases']}")


if __name__ == "__main__":
    main()
