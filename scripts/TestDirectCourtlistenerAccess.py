#!/usr/bin/env python3
"""
Direct CourtListener Scraper
Target specific docket URLs based on the screenshots
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

class DirectCourtListenerScraper:
    def __init__(self):
        self.base_url = "https://www.courtlistener.com"

        # Try different URL patterns
        self.search_urls = [
            f"{self.base_url}/recap/search/",
            f"{self.base_url}/search/",
            f"{self.base_url}/dockets/",
        ]

        # Ultra-polite settings
        self.base_delay = 30.0
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
        delay = max(20.0, delay)
        logger.info(f"⏸️  Ultra-polite delay: {delay:.1f} seconds")
        time.sleep(delay)

    def test_direct_search(self):
        """Test direct search using the exact interface you found."""
        logger.info("🧪 Testing direct search interface")

        # Try the exact search interface you used
        search_url = "https://www.courtlistener.com/recap/search/"

        # Test with the exact query you used
        test_queries = [
            "28 usc 1782 discovery",
            '"28 U.S.C. § 1782"',
            '"Application for Judicial Assistance"'
        ]

        for query in test_queries:
            logger.info(f"🧪 Testing query: '{query}'")

            # Ultra-polite delay
            self._ultra_polite_delay()

            try:
                # Try GET request first
                response = self.session.get(search_url, timeout=60)
                logger.info(f"✓ GET request successful: HTTP {response.status_code}")

                # Try POST request with form data
                form_data = {
                    'q': query,
                    'type': 'r',  # RECAP Archive
                    'available_only': 'on',  # Only show results with PDFs
                }

                self._ultra_polite_delay()

                response = self.session.post(search_url, data=form_data, timeout=60)
                logger.info(f"✓ POST request successful: HTTP {response.status_code}")

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Look for case links
                    case_links = soup.select('h2.case-name a')
                    logger.info(f"✓ Found {len(case_links)} case links")

                    for i, link in enumerate(case_links[:3], 1):
                        case_name = link.get_text(strip=True)
                        case_url = urljoin(self.base_url, link['href'])
                        logger.info(f"  {i}. {case_name}")
                        logger.info(f"     URL: {case_url}")

                    if case_links:
                        return case_links[:3]  # Return first 3 cases for testing

            except Exception as e:
                logger.error(f"✗ Search error: {e}")

        return []

    def test_direct_docket_access(self):
        """Test accessing specific docket URLs from your screenshots."""
        logger.info("🧪 Testing direct docket access")

        # URLs from your screenshots
        test_dockets = [
            "https://www.courtlistener.com/docket/3:25-mc-00010/efg-bank-ag/",
            "https://www.courtlistener.com/docket/1:24-mc-00575/navios-south-american-logistics-inc/",
            "https://www.courtlistener.com/docket/5:25-mc-80042/hmd-global-oy/",
        ]

        for docket_url in test_dockets:
            logger.info(f"🧪 Testing docket: {docket_url}")

            # Ultra-polite delay
            self._ultra_polite_delay()

            try:
                response = self.session.get(docket_url, timeout=60)
                logger.info(f"✓ Docket access: HTTP {response.status_code}")

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Look for PDF download buttons
                    download_buttons = soup.select('a[href*=".pdf"]')
                    logger.info(f"✓ Found {len(download_buttons)} PDF links")

                    free_pdfs = []
                    paid_pdfs = []

                    for button in download_buttons:
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

                    logger.info(f"  FREE PDFs: {len(free_pdfs)}")
                    for pdf in free_pdfs:
                        logger.info(f"    - {pdf['text']}")

                    logger.info(f"  PAID PDFs: {len(paid_pdfs)}")
                    for pdf in paid_pdfs:
                        logger.info(f"    - {pdf['text']}")

                    if free_pdfs:
                        return free_pdfs[0]  # Return first FREE PDF for testing

            except Exception as e:
                logger.error(f"✗ Docket access error: {e}")

        return None

    def test_pdf_download(self, pdf_info):
        """Test downloading a FREE PDF."""
        if not pdf_info:
            logger.warning("No PDF info provided for download test")
            return False

        logger.info(f"🧪 Testing PDF download: {pdf_info['text']}")

        # Ultra-polite delay
        self._ultra_polite_delay()

        try:
            pdf_url = urljoin(self.base_url, pdf_info['href'])
            logger.info(f"Downloading from: {pdf_url}")

            response = self.session.get(pdf_url, timeout=60)
            logger.info(f"✓ PDF download: HTTP {response.status_code}")

            if response.status_code == 200:
                # Check if it's actually a PDF
                content_type = response.headers.get('content-type', '')
                if 'pdf' in content_type.lower():
                    logger.info(f"✓ SUCCESS: Downloaded PDF ({len(response.content)} bytes)")
                    return True
                else:
                    logger.warning(f"⚠️  Response not a PDF: {content_type}")
            else:
                logger.warning(f"✗ PDF download failed: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"✗ PDF download error: {e}")

        return False

def main():
    logger.info("🧪 TESTING DIRECT COURTLISTENER ACCESS")
    logger.info("=" * 60)
    logger.info("🎯 Testing direct docket access")
    logger.info("⏸️  Ultra-polite delays: 30±10 seconds")

    scraper = DirectCourtListenerScraper()

    # Test 1: Direct search
    logger.info("\\n🧪 Test 1: Direct Search Interface")
    case_links = scraper.test_direct_search()

    # Test 2: Direct docket access
    logger.info("\\n🧪 Test 2: Direct Docket Access")
    pdf_info = scraper.test_direct_docket_access()

    # Test 3: PDF download
    if pdf_info:
        logger.info("\\n🧪 Test 3: PDF Download")
        success = scraper.test_pdf_download(pdf_info)

        if success:
            logger.info("\\n✅ SUCCESS: Direct access working!")
            logger.info("   Ready to run full scraper with direct URLs")
        else:
            logger.warning("\\n⚠️  PDF download failed")
    else:
        logger.warning("\\n⚠️  No FREE PDFs found for testing")

    logger.info("\\n✅ Direct access test complete!")

if __name__ == "__main__":
    main()
