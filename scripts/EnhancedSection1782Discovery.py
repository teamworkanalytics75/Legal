#!/usr/bin/env python3
"""
Enhanced Section 1782 Case Discovery - Using improved techniques from motion-to-seal work.

This script applies the advanced search strategies we learned from finding 1,400+ motion-to-seal
cases to discover more §1782 cases. We originally only found ~136 cases with simple queries.

New techniques applied:
1. Multiple query variations (phrases, citations, case names)
2. Date range filtering (2010+ for modern cases)
3. Jurisdiction targeting (SDNY, DDC, ND Cal, etc. - high-volume §1782 courts)
4. Both precedential and non-precedential
5. Better duplicate detection
6. Integration with corpus_slice classification
"""

import os
import sys
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# CourtListener API configuration
COURTLISTENER_API_TOKEN = os.environ.get("COURTLISTENER_API_TOKEN")
if not COURTLISTENER_API_TOKEN:
    raise EnvironmentError(
        "COURTLISTENER_API_TOKEN is not set. Export the token via environment variable or .env file."
    )
COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
HEADERS = {
    'Authorization': f'Token {COURTLISTENER_API_TOKEN}',
    'User-Agent': 'The MatrixLegalAI/2.0 (Research/Education)'
}

# High-volume §1782 districts (based on our corpus)
HIGH_VOLUME_DISTRICTS = [
    'caed',  # E.D. Cal
    'cand',  # N.D. Cal
    'cacd',  # C.D. Cal
    'nysd',  # S.D.N.Y.
    'nyed',  # E.D.N.Y.
    'dcd',   # D.D.C.
    'flsd',  # S.D. NJ
    'njd',   # D.N.J.
    'flsd',  # S.D. Fla
    'txnd',  # N.D. Tex
    'ilsd',  # S.D. Ill
    'mad',   # D. Mass
]


class Enhanced1782Discovery:
    """Enhanced Section 1782 case discovery using improved techniques."""

    def __init__(self, output_dir: Path = Path("case_law_data/enhanced_1782_discovery")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seen_cluster_ids: Set[int] = set()
        self.seen_case_hashes: Set[str] = set()
        self.all_cases: List[Dict] = []

    def get_case_hash(self, case: Dict) -> str:
        """Create a unique hash for duplicate detection."""
        cluster_id = case.get('id') or case.get('cluster_id', '')
        case_name = case.get('caseName') or case.get('case_name', '')
        date_filed = case.get('dateFiled') or case.get('date_filed', '')
        hash_string = f"{cluster_id}|{case_name}|{date_filed}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def is_1782_relevant(self, case: Dict) -> bool:
        """Check if case is relevant to §1782 discovery."""
        text_fields = [
            case.get('caseName', ''),
            case.get('snippet', ''),
            case.get('description', ''),
        ]

        text = ' '.join(text_fields).lower()

        # Strong indicators
        strong_indicators = [
            '28 u.s.c. 1782',
            '28 usc 1782',
            'section 1782',
            '§ 1782',
            '§1782',
            '1782(a)',
            '1782 application',
            'discovery for use in foreign proceeding',
            'judicial assistance',
        ]

        # Check for strong indicators
        for indicator in strong_indicators:
            if indicator in text:
                return True

        # Weak indicators (require multiple)
        weak_indicators = [
            'foreign tribunal',
            'foreign proceeding',
            'intel corp',
            'euromepa',
            'brandi-dohrn',
        ]

        weak_count = sum(1 for ind in weak_indicators if ind in text)
        return weak_count >= 2

    def search_with_query(self, query: str, date_after: str = "2010-01-01",
                         max_results: int = 500, court_filter: Optional[List[str]] = None) -> List[Dict]:
        """Search with a specific query."""
        logger.info(f"🔍 Searching: '{query}' (after {date_after})")

        cases = []
        page = 1
        per_page = 100

        while len(cases) < max_results:
            params = {
                'q': query,
                'format': 'json',
                'order_by': 'dateFiled desc',
                'stat_Precedential': 'on',
                'stat_Non-Precedential': 'on',
                'dateFiled__gte': date_after,
                'page': page,
                'per_page': per_page,
            }

            # Add court filter if specified
            if court_filter:
                params['court'] = ','.join(court_filter)

            # Retry logic for API issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    url = f"{COURTLISTENER_BASE_URL}/search/"
                    response = requests.get(url, headers=HEADERS, params=params, timeout=30)

                    # Handle rate limiting / server errors
                    if response.status_code == 502 or response.status_code == 503:
                        wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                        logger.warning(f"  ⚠️ Server error {response.status_code}, waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    break  # Success, exit retry loop
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        logger.warning(f"  ⚠️ Request error, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise  # Final attempt failed

            # Process successful response
            try:

                results = data.get('results', [])
                if not results:
                    break

                for case in results:
                    # Deduplicate
                    cluster_id = case.get('id') or case.get('cluster_id')
                    if cluster_id and cluster_id in self.seen_cluster_ids:
                        continue

                    case_hash = self.get_case_hash(case)
                    if case_hash in self.seen_case_hashes:
                        continue

                    # Filter for §1782 relevance
                    if self.is_1782_relevant(case):
                        self.seen_cluster_ids.add(cluster_id)
                        self.seen_case_hashes.add(case_hash)
                        cases.append(case)
                        logger.info(f"  ✓ Found: {case.get('caseName', 'Unknown')[:60]} (ID: {cluster_id})")

                page += 1
                time.sleep(0.2)  # Rate limiting

            except Exception as e:
                logger.error(f"  ❌ Error on page {page}: {e}")
                break

        logger.info(f"  Found {len(cases)} unique §1782 cases for query '{query}'")
        return cases

    def comprehensive_search(self, date_after: str = "2010-01-01",
                           max_per_query: int = 300) -> List[Dict]:
        """Run comprehensive search with multiple query strategies."""
        logger.info("="*80)
        logger.info("ENHANCED SECTION 1782 CASE DISCOVERY")
        logger.info("="*80)
        logger.info(f"Date range: {date_after} to present")
        logger.info(f"Max results per query: {max_per_query}")
        logger.info("")

        # Comprehensive query list (learned from motion-to-seal work)
        queries = [
            # Direct citations
            "28 U.S.C. 1782",
            "28 USC 1782",
            "section 1782",
            "§ 1782",
            "§1782",

            # Common phrases
            "discovery for use in foreign proceeding",
            "judicial assistance foreign tribunal",
            "1782 application",
            "1782 discovery",

            # Landmark cases (likely to cite §1782)
            "Intel Corp Advanced Micro Devices",
            "Euromepa Esmerian",
            "Brandi-Dohrn",
            "ZF Automotive Luxshare",
            "Servotronics Rolls-Royce",

            # Common patterns
            "foreign tribunal discovery",
            "international judicial assistance",
            "assistance before foreign tribunal",
            "discovery pursuant to 28 USC 1782",

            # Alternative phrasings
            "applicant foreign proceeding",
            "subpoena foreign proceeding",
        ]

        all_discovered = []

        # Strategy 1: Broad search (no court filter)
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 1: BROAD SEARCH (All Courts)")
        logger.info("="*80)

        for query in queries:
            cases = self.search_with_query(query, date_after, max_per_query)
            all_discovered.extend(cases)
            if len(all_discovered) >= 1000:  # Safety limit
                logger.warning("Hit 1000 case limit, stopping")
                break

        # Strategy 2: Targeted high-volume districts
        logger.info("\n" + "="*80)
        logger.info("STRATEGY 2: TARGETED HIGH-VOLUME DISTRICTS")
        logger.info("="*80)

        key_queries = ["28 U.S.C. 1782", "section 1782", "discovery for use in foreign proceeding"]
        for query in key_queries[:3]:  # Top 3 queries only for efficiency
            cases = self.search_with_query(
                query, date_after, max_per_query // 3,
                court_filter=HIGH_VOLUME_DISTRICTS
            )
            all_discovered.extend(cases)

        # Deduplicate final list
        final_cases = []
        seen = set()
        for case in all_discovered:
            cluster_id = case.get('id') or case.get('cluster_id')
            if cluster_id and cluster_id not in seen:
                seen.add(cluster_id)
                final_cases.append(case)

        logger.info("\n" + "="*80)
        logger.info(f"TOTAL DISCOVERED: {len(final_cases)} unique §1782 cases")
        logger.info("="*80)

        return final_cases

    def save_results(self, cases: List[Dict], filename: str = "enhanced_1782_discovery.json"):
        """Save discovered cases to JSON."""
        output_path = self.output_dir / filename
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved {len(cases)} cases to {output_path}")

        # Also save summary CSV
        import csv
        csv_path = self.output_dir / filename.replace('.json', '.csv')
        if cases:
            with csv_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['cluster_id', 'case_name', 'court', 'date_filed', 'url'])
                writer.writeheader()
                for case in cases:
                    writer.writerow({
                        'cluster_id': case.get('id') or case.get('cluster_id', ''),
                        'case_name': case.get('caseName') or case.get('case_name', ''),
                        'court': case.get('court', ''),
                        'date_filed': case.get('dateFiled') or case.get('date_filed', ''),
                        'url': f"https://www.courtlistener.com/opinion/{case.get('id', '')}/",
                    })
        logger.info(f"💾 Saved summary CSV to {csv_path}")


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced §1782 case discovery")
    parser.add_argument("--date-after", default="2010-01-01", help="Search cases after this date")
    parser.add_argument("--max-per-query", type=int, default=300, help="Max results per query")
    parser.add_argument("--output-dir", type=Path, default=Path("case_law_data/enhanced_1782_discovery"))
    args = parser.parse_args()

    discoverer = Enhanced1782Discovery(args.output_dir)
    cases = discoverer.comprehensive_search(args.date_after, args.max_per_query)
    discoverer.save_results(cases)

    logger.info(f"\n✅ Discovery complete! Found {len(cases)} unique §1782 cases")
    logger.info(f"Compare to original ~136 cases - potential {len(cases) - 136} new cases found!")


if __name__ == "__main__":
    main()
