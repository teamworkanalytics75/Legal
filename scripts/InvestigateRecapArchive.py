#!/usr/bin/env python3
"""
RECAP Archive Investigation for §1782 Cases

Investigate RECAP Archive (CourtListener's PACER data) to find additional
§1782 cases that might not be in CourtListener's opinion database.
"""

import sys
import json
import logging
import requests
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CourtListener client
import importlib.util
spec = importlib.util.spec_from_file_location(
    "download_case_law",
    Path(__file__).parent.parent / "document_ingestion" / "download_case_law.py"
)
download_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_module)
CourtListenerClient = download_module.CourtListenerClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RECAPInvestigator:
    """Investigate RECAP Archive for §1782 cases."""

    def __init__(self):
        """Initialize the investigator."""
        self.client = CourtListenerClient()
        self.base_url = "https://www.courtlistener.com/api/rest/v3/"
        self.session = requests.Session()

    def search_recap_dockets(self, search_terms, limit=100):
        """Search RECAP dockets for §1782 cases."""
        try:
            logger.info(f"Searching RECAP dockets for: {search_terms}")

            # RECAP docket search endpoint
            url = f"{self.base_url}search/"

            params = {
                'q': search_terms,
                'type': 'd',  # Docket search
                'order_by': 'score desc',
                'stat_Precedential': 'on',  # Only precedential
                'format': 'json',
                'page_size': min(limit, 100)
            }

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            logger.info(f"Found {data.get('count', 0)} RECAP dockets")

            return data

        except Exception as e:
            logger.error(f"Error searching RECAP dockets: {e}")
            return None

    def search_recap_documents(self, search_terms, limit=100):
        """Search RECAP documents for §1782 cases."""
        try:
            logger.info(f"Searching RECAP documents for: {search_terms}")

            # RECAP document search endpoint
            url = f"{self.base_url}search/"

            params = {
                'q': search_terms,
                'type': 'r',  # RECAP document search
                'order_by': 'score desc',
                'stat_Precedential': 'on',  # Only precedential
                'format': 'json',
                'page_size': min(limit, 100)
            }

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            logger.info(f"Found {data.get('count', 0)} RECAP documents")

            return data

        except Exception as e:
            logger.error(f"Error searching RECAP documents: {e}")
            return None

    def analyze_recap_results(self, results, result_type):
        """Analyze RECAP search results."""
        if not results:
            return None

        logger.info(f"\nAnalyzing {result_type} results...")

        count = results.get('count', 0)
        results_list = results.get('results', [])

        logger.info(f"Total {result_type}: {count}")
        logger.info(f"Returned {result_type}: {len(results_list)}")

        # Analyze results
        courts = {}
        years = {}
        case_types = {}

        for item in results_list:
            # Court analysis
            court = item.get('court', 'Unknown')
            courts[court] = courts.get(court, 0) + 1

            # Year analysis
            date_filed = item.get('date_filed')
            if date_filed:
                year = date_filed[:4]
                years[year] = years.get(year, 0) + 1

            # Case type analysis
            case_name = item.get('caseName', '')
            if '1782' in case_name.lower():
                case_types['1782_mention'] = case_types.get('1782_mention', 0) + 1
            elif 'discovery' in case_name.lower():
                case_types['discovery_mention'] = case_types.get('discovery_mention', 0) + 1
            elif 'international' in case_name.lower():
                case_types['international_mention'] = case_types.get('international_mention', 0) + 1

        # Show top results
        logger.info(f"\nTop courts for {result_type}:")
        for court, count in sorted(courts.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {court}: {count}")

        logger.info(f"\nTop years for {result_type}:")
        for year, count in sorted(years.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {year}: {count}")

        logger.info(f"\nCase type mentions:")
        for case_type, count in case_types.items():
            logger.info(f"  {case_type}: {count}")

        return {
            'total_count': count,
            'returned_count': len(results_list),
            'courts': courts,
            'years': years,
            'case_types': case_types,
            'results': results_list
        }

    def test_multiple_search_terms(self):
        """Test multiple search terms for §1782 cases."""
        search_terms = [
            "1782",
            "28 USC 1782",
            "28 U.S.C. 1782",
            "section 1782",
            "international discovery",
            "foreign discovery",
            "judicial assistance",
            "letter rogatory",
            "Intel Corp",
            "ZF Automotive"
        ]

        logger.info("Testing multiple search terms...")

        all_results = {}

        for term in search_terms:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing search term: '{term}'")
            logger.info(f"{'='*60}")

            # Search dockets
            docket_results = self.search_recap_dockets(term, limit=50)
            docket_analysis = self.analyze_recap_results(docket_results, "dockets")

            # Search documents
            doc_results = self.search_recap_documents(term, limit=50)
            doc_analysis = self.analyze_recap_results(doc_results, "documents")

            all_results[term] = {
                'dockets': docket_analysis,
                'documents': doc_analysis
            }

        return all_results

    def run_investigation(self):
        """Run the complete RECAP investigation."""
        logger.info("Starting RECAP Archive investigation...")

        # Test multiple search terms
        results = self.test_multiple_search_terms()

        # Summary
        logger.info("\n" + "="*80)
        logger.info("RECAP INVESTIGATION SUMMARY")
        logger.info("="*80)

        total_dockets = 0
        total_documents = 0

        for term, data in results.items():
            docket_count = data['dockets']['total_count'] if data['dockets'] else 0
            doc_count = data['documents']['total_count'] if data['documents'] else 0

            total_dockets += docket_count
            total_documents += doc_count

            logger.info(f"{term}: {docket_count} dockets, {doc_count} documents")

        logger.info(f"\nTotal unique dockets found: {total_dockets}")
        logger.info(f"Total unique documents found: {total_documents}")

        return results


def main():
    """Main entry point."""
    investigator = RECAPInvestigator()
    results = investigator.run_investigation()

    if results:
        print(f"\nRECAP Investigation Complete!")
        print(f"Check the logs above for detailed results.")


if __name__ == "__main__":
    main()
