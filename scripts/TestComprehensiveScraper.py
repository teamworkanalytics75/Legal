#!/usr/bin/env python3
"""
Test Comprehensive Scraper with Small Sample
Verify the scraper works before running full download
"""

import json
import time
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional

import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestScraper:
    def __init__(self):
        self.base_url = "https://www.courtlistener.com"
        self.recap_search_url = f"{self.base_url}/recap/search/"

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MalcolmGrayson §1782 research (malcolmgrayson00@gmail.com) - test scraper',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        self.pdf_dir = Path("data/recap_petitions_test")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    def test_search(self, query: str, max_cases: int = 5):
        """Test search functionality with a small sample."""
        logger.info(f"🧪 Testing search: '{query}'")

        search_params = {
            'q': query,
            'type': 'r',  # RECAP Archive
            'order_by': 'dateFiled desc',
            'available_only': 'on',  # Only show results with PDFs
        }

        try:
            response = self.session.get(self.recap_search_url, params=search_params, timeout=30)
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
            else:
                logger.warning(f"✗ Search failed: HTTP {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"✗ Search error: {e}")
            return []

    def test_download(self, case_url: str):
        """Test downloading PDFs from a single case."""
        logger.info(f"🧪 Testing download from: {case_url}")

        try:
            response = self.session.get(case_url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find PDF links
                pdf_links = []
                for link_tag in soup.select('a[href$=".pdf"]'):
                    pdf_url = urljoin(self.base_url, link_tag['href'])
                    file_name = Path(urlparse(pdf_url).path).name
                    pdf_links.append({
                        'url': pdf_url,
                        'filename': file_name
                    })

                logger.info(f"✓ Found {len(pdf_links)} PDF links")
                for pdf in pdf_links:
                    logger.info(f"  - {pdf['filename']}")

                return pdf_links
            else:
                logger.warning(f"✗ Case page failed: HTTP {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"✗ Download test error: {e}")
            return []

def main():
    logger.info("🧪 TESTING COMPREHENSIVE SCRAPER")
    logger.info("=" * 50)

    scraper = TestScraper()

    # Test different search queries
    test_queries = [
        "28 usc 1782 discovery",
        '"28 U.S.C. § 1782"',
        '"Application for Judicial Assistance"'
    ]

    all_cases = []

    for query in test_queries:
        cases = scraper.test_search(query, max_cases=3)
        all_cases.extend(cases)
        time.sleep(5)  # Be polite

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

    # Test downloading from first case
    if unique_cases:
        logger.info(f"\\n🧪 Testing download from first case...")
        pdf_links = scraper.test_download(unique_cases[0]['url'])

        if pdf_links:
            logger.info(f"✓ Download test successful - found {len(pdf_links)} PDFs")
        else:
            logger.warning("✗ No PDFs found in test case")

    logger.info(f"\\n✅ Test complete!")
    logger.info(f"   Ready to run full scraper: py scripts/comprehensive_courtlistener_scraper.py")

if __name__ == "__main__":
    main()
