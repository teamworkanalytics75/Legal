#!/usr/bin/env python3
"""
Comprehensive §1782 Case Download Strategy

Since RECAP Archive is blocked, this script implements a multi-pronged approach:
1. Google Scholar web scraping for additional cases
2. Targeted CourtListener searches by specific districts
3. Broader keyword searches with better filtering
"""

import sys
import json
import logging
import requests
import time
from pathlib import Path
from urllib.parse import quote_plus
import re

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


class Comprehensive1782Downloader:
    """Comprehensive §1782 case downloader using multiple strategies."""

    def __init__(self):
        """Initialize the downloader."""
        self.client = CourtListenerClient()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # High-volume districts for §1782 cases
        self.target_districts = [
            'mad',  # D. Mass
            'nys',  # SDNY
            'nye',  # EDNY
            'cand', # ND Cal
            'cacd', # CD Cal
            'flsd', # SD Fla
            'txsd', # SD Tex
            'ilnd', # ND Ill
            'wawd', # WD Wash
            'nynd', # ND NY
        ]

        # Circuit courts
        self.target_circuits = [
            'ca1',  # 1st Circuit
            'ca2',  # 2nd Circuit
            'ca9',  # 9th Circuit
            'ca11', # 11th Circuit
            'ca5',  # 5th Circuit
        ]

    def search_google_scholar(self, query, max_results=50):
        """Search Google Scholar for §1782 cases."""
        try:
            logger.info(f"Searching Google Scholar for: {query}")

            # Google Scholar search URL
            search_url = f"https://scholar.google.com/scholar?q={quote_plus(query)}&as_sdt=2006&as_vis=1"

            response = self.session.get(search_url)
            response.raise_for_status()

            # Parse results (basic parsing - would need more sophisticated parsing for production)
            content = response.text

            # Look for case citations and titles
            case_patterns = [
                r'(\d{4} WL \d+)',  # Westlaw citations
                r'(\d{3} F\.3d \d+)',  # Federal Reporter
                r'(\d{3} F\. Supp\. \d+)',  # Federal Supplement
                r'(\d{3} F\. Supp\. 2d \d+)',  # Federal Supplement 2d
                r'(\d{3} F\. Supp\. 3d \d+)',  # Federal Supplement 3d
            ]

            found_cases = []
            for pattern in case_patterns:
                matches = re.findall(pattern, content)
                found_cases.extend(matches)

            # Look for case titles mentioning 1782
            title_pattern = r'<h3[^>]*>.*?1782.*?</h3>'
            title_matches = re.findall(title_pattern, content, re.IGNORECASE | re.DOTALL)

            logger.info(f"Found {len(found_cases)} case citations")
            logger.info(f"Found {len(title_matches)} case titles mentioning 1782")

            return {
                'citations': found_cases,
                'titles': title_matches,
                'total_found': len(found_cases) + len(title_matches)
            }

        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}")
            return None

    def search_courtlistener_by_district(self, district, keywords, limit=100):
        """Search CourtListener for §1782 cases in specific districts."""
        try:
            logger.info(f"Searching {district} for: {keywords}")

            results = []

            for keyword in keywords:
                response = self.client.search_opinions(
                    courts=[district],
                    keywords=[keyword],
                    limit=limit,
                    include_non_precedential=True
                )

                if response and 'results' in response:
                    results.extend(response['results'])
                    logger.info(f"Found {len(response['results'])} cases for '{keyword}' in {district}")

                # Rate limiting
                time.sleep(1)

            return results

        except Exception as e:
            logger.error(f"Error searching {district}: {e}")
            return []

    def search_courtlistener_by_circuit(self, circuit, keywords, limit=100):
        """Search CourtListener for §1782 cases in specific circuits."""
        try:
            logger.info(f"Searching {circuit} for: {keywords}")

            results = []

            for keyword in keywords:
                response = self.client.search_opinions(
                    courts=[circuit],
                    keywords=[keyword],
                    limit=limit,
                    include_non_precedential=True
                )

                if response and 'results' in response:
                    results.extend(response['results'])
                    logger.info(f"Found {len(response['results'])} cases for '{keyword}' in {circuit}")

                # Rate limiting
                time.sleep(1)

            return results

        except Exception as e:
            logger.error(f"Error searching {circuit}: {e}")
            return []

    def filter_1782_cases(self, cases):
        """Filter cases to only include actual §1782 cases."""
        filtered_cases = []

        for case in cases:
            case_name = case.get('caseName', '').lower()
            case_text = case.get('text', '').lower()

            # Check if it's actually about §1782
            if any(term in case_name or term in case_text for term in [
                '1782', 'section 1782', '28 usc 1782', '28 u.s.c. 1782',
                'international discovery', 'foreign discovery', 'judicial assistance'
            ]):
                filtered_cases.append(case)

        return filtered_cases

    def run_comprehensive_search(self):
        """Run comprehensive search using all strategies."""
        logger.info("Starting comprehensive §1782 case search...")

        all_results = {
            'google_scholar': {},
            'district_searches': {},
            'circuit_searches': {},
            'total_cases': 0,
            'unique_cases': set()
        }

        # Search terms
        search_terms = [
            '1782',
            '28 USC 1782',
            'international discovery',
            'foreign discovery',
            'judicial assistance',
            'letter rogatory'
        ]

        # 1. Google Scholar searches
        logger.info("\n" + "="*60)
        logger.info("GOOGLE SCHOLAR SEARCHES")
        logger.info("="*60)

        for term in search_terms:
            scholar_results = self.search_google_scholar(term, max_results=50)
            if scholar_results:
                all_results['google_scholar'][term] = scholar_results
                logger.info(f"Google Scholar '{term}': {scholar_results['total_found']} results")

            time.sleep(2)  # Rate limiting

        # 2. District court searches
        logger.info("\n" + "="*60)
        logger.info("DISTRICT COURT SEARCHES")
        logger.info("="*60)

        for district in self.target_districts:
            district_results = self.search_courtlistener_by_district(district, search_terms, limit=50)
            filtered_results = self.filter_1782_cases(district_results)

            all_results['district_searches'][district] = {
                'total_found': len(district_results),
                'filtered_1782': len(filtered_results),
                'cases': filtered_results
            }

            logger.info(f"{district}: {len(district_results)} total, {len(filtered_results)} §1782 cases")

            # Add to unique cases
            for case in filtered_results:
                case_id = case.get('id')
                if case_id:
                    all_results['unique_cases'].add(case_id)

        # 3. Circuit court searches
        logger.info("\n" + "="*60)
        logger.info("CIRCUIT COURT SEARCHES")
        logger.info("="*60)

        for circuit in self.target_circuits:
            circuit_results = self.search_courtlistener_by_circuit(circuit, search_terms, limit=50)
            filtered_results = self.filter_1782_cases(circuit_results)

            all_results['circuit_searches'][circuit] = {
                'total_found': len(circuit_results),
                'filtered_1782': len(filtered_results),
                'cases': filtered_results
            }

            logger.info(f"{circuit}: {len(circuit_results)} total, {len(filtered_results)} §1782 cases")

            # Add to unique cases
            for case in filtered_results:
                case_id = case.get('id')
                if case_id:
                    all_results['unique_cases'].add(case_id)

        # Calculate totals
        all_results['total_cases'] = len(all_results['unique_cases'])

        # Summary
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE SEARCH SUMMARY")
        logger.info("="*80)

        total_district_cases = sum(data['filtered_1782'] for data in all_results['district_searches'].values())
        total_circuit_cases = sum(data['filtered_1782'] for data in all_results['circuit_searches'].values())

        logger.info(f"Google Scholar results: {len(all_results['google_scholar'])} search terms")
        logger.info(f"District court cases: {total_district_cases}")
        logger.info(f"Circuit court cases: {total_circuit_cases}")
        logger.info(f"Total unique cases found: {all_results['total_cases']}")

        return all_results

    def save_results(self, results, filename="comprehensive_1782_search_results.json"):
        """Save search results to file."""
        try:
            output_path = Path(__file__).parent.parent / "data" / "case_law" / filename

            # Convert set to list for JSON serialization
            results_copy = results.copy()
            results_copy['unique_cases'] = list(results_copy['unique_cases'])

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main entry point."""
    downloader = Comprehensive1782Downloader()
    results = downloader.run_comprehensive_search()

    if results:
        downloader.save_results(results)
        print(f"\nSUCCESS!")
        print(f"Total unique §1782 cases found: {results['total_cases']}")
        print(f"Results saved to comprehensive_1782_search_results.json")


if __name__ == "__main__":
    main()
