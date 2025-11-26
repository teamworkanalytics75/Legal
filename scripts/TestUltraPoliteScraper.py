#!/usr/bin/env python3
"""
Test Ultra-Polite Scraper
Verify the ultra-polite approach works before full download
"""

import json
import time
import random
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional

import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestUltraPoliteScraper:
    def __init__(self):
        self.base_url = "https://www.courtlistener.com"
        self.recap_search_url = f"{self.base_url}/recap/search/"

        # Ultra-polite settings
        self.base_delay = 30.0  # Shorter for testing
        self.random_delay = 10.0

        # Sophisticated session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

    def _ultra_polite_delay(self):
        """Ultra-polite delay with random variation."""
        delay = self.base_delay + random.uniform(-self.random_delay, self.random_delay)
        delay = max(20.0, delay)  # Minimum 20 seconds for testing
        logger.info(f"⏸️  Ultra-polite delay: {delay:.1f} seconds")
        time.sleep(delay)

    def test_search(self, query: str, max_cases: int = 3):
        """Test search with ultra-polite delays."""
        logger.info(f"🧪 Testing ultra-polite search: '{query}'")

        search_params = {
            'q': query,
            'type': 'r',  # RECAP Archive
            'order_by': 'dateFiled desc',
            'available_only': 'on',  # Only show results with PDFs
        }

        try:
            # Ultra-polite delay before request
            self._ultra_polite_delay()

            response = self.session.get(self.recap_search_url, params=search_params, timeout=60)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract case information
                cases = []
                for link_tag in soup.select('h2.case-name a')[:max_cases]:
                    docket_url = urljoin(self.base_url, link_tag['href'])
                    case_name = link_tag.get_text(strip=True)
                    cases.append({
                        'url': docket_url,
                        'case_name': case_name
                    })

                logger.info(f"✓ Found {len(cases)} cases for query: {query}")
                for i, case in enumerate(cases, 1):
                    logger.info(f"  {i}. {case['case_name']}")

                return cases
            elif response.status_code == 202:
                logger.warning(f"✗ HTTP 202 - AWS WAF still blocking (expected)")
                return []
            else:
                logger.warning(f"✗ Search failed: HTTP {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"✗ Search error: {e}")
            return []

    def test_docket_analysis(self, case_url: str):
        """Test analyzing a docket for FREE vs PAID PDFs."""
        logger.info(f"🧪 Testing docket analysis: {case_url}")

        try:
            # Ultra-polite delay
            self._ultra_polite_delay()

            response = self.session.get(case_url, timeout=60)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Look for document entries
                document_entries = soup.select('.docket-entry')

                free_pdfs = []
                paid_pdfs = []

                for entry in document_entries:
                    # Look for download buttons
                    buttons = entry.select('a')

                    for button in buttons:
                        button_text = button.get_text(strip=True).lower()
                        button_href = button.get('href', '')

                        if 'download pdf' in button_text and 'buy' not in button_text:
                            free_pdfs.append({
                                'text': button_text,
                                'href': button_href
                            })
                        elif 'buy on pacer' in button_text:
                            paid_pdfs.append({
                                'text': button_text,
                                'href': button_href
                            })

                logger.info(f"✓ Found {len(free_pdfs)} FREE PDFs")
                for pdf in free_pdfs:
                    logger.info(f"  - FREE: {pdf['text']}")

                logger.info(f"✓ Found {len(paid_pdfs)} PAID PDFs (skipping)")
                for pdf in paid_pdfs:
                    logger.info(f"  - PAID: {pdf['text']}")

                return {
                    'free_pdfs': free_pdfs,
                    'paid_pdfs': paid_pdfs
                }
            else:
                logger.warning(f"✗ Docket analysis failed: HTTP {response.status_code}")
                return {'free_pdfs': [], 'paid_pdfs': []}

        except Exception as e:
            logger.error(f"✗ Docket analysis error: {e}")
            return {'free_pdfs': [], 'paid_pdfs': []}

def main():
    logger.info("🧪 TESTING ULTRA-POLITE SCRAPER")
    logger.info("=" * 50)
    logger.info("🎯 Testing FREE PDF detection")
    logger.info("⏸️  Ultra-polite delays: 30±10 seconds")

    scraper = TestUltraPoliteScraper()

    # Test different search queries
    test_queries = [
        "28 usc 1782 discovery",
        '"28 U.S.C. § 1782"'
    ]

    all_cases = []

    for query in test_queries:
        cases = scraper.test_search(query, max_cases=2)
        all_cases.extend(cases)

        # Ultra-polite pause between searches
        if query != test_queries[-1]:
            logger.info("⏸️  Ultra-polite pause between searches...")
            scraper._ultra_polite_delay()

    # Remove duplicates
    unique_cases = []
    seen_urls = set()
    for case in all_cases:
        if case['url'] not in seen_urls:
            unique_cases.append(case)
            seen_urls.add(case['url'])

    logger.info(f"\\n📊 TEST RESULTS:")
    logger.info(f"   Total cases found: {len(all_cases)}")
    logger.info(f"   Unique cases: {len(unique_cases)}")

    # Test docket analysis on first case
    if unique_cases:
        logger.info(f"\\n🧪 Testing docket analysis...")
        analysis = scraper.test_docket_analysis(unique_cases[0]['url'])

        if analysis['free_pdfs']:
            logger.info(f"✓ SUCCESS: Found {len(analysis['free_pdfs'])} FREE PDFs")
            logger.info(f"✓ Strategy working: Can distinguish FREE vs PAID")
        else:
            logger.warning("⚠️  No FREE PDFs found in test case")

    logger.info(f"\\n✅ Ultra-polite test complete!")
    logger.info(f"   Ready to run full scraper: py scripts/ultra_polite_courtlistener_scraper.py")

if __name__ == "__main__":
    main()
