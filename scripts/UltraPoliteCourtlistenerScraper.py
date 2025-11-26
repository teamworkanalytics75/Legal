#!/usr/bin/env python3
"""
Ultra-Polite CourtListener PDF Scraper
Targets only FREE PDFs with sophisticated anti-detection
"""

import json
import os
import time
import random
import logging
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote_plus
from typing import Dict, List, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraPoliteCourtListenerScraper:
    def __init__(self):
        self.base_url = "https://www.courtlistener.com"
        self.recap_search_url = f"{self.base_url}/recap/search/"

        # Ultra-polite settings
        self.base_delay = 90.0  # 90 seconds base delay
        self.random_delay = 30.0  # ±30 seconds variation
        self.max_retries = 5

        # Sophisticated session with realistic browser behavior
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })

        self.pdf_dir = Path("data/recap_petitions_ultra_polite")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        self.scraping_log = []
        self.downloaded_hashes = set()
        self.all_case_urls = set()

        self.stats = {
            'total_searches': 0,
            'total_cases_found': 0,
            'total_pdfs_downloaded': 0,
            'total_errors': 0,
            'duplicates_skipped': 0,
            'free_pdfs_found': 0,
            'paid_pdfs_skipped': 0,
            'last_run': None
        }

        # Load existing files
        self.load_existing_files()

    def load_existing_files(self):
        """Load existing downloaded files to avoid duplicates."""
        for pdf_file in self.pdf_dir.glob("*.pdf"):
            try:
                with open(pdf_file, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    self.downloaded_hashes.add(file_hash)
            except Exception as e:
                logger.warning(f"Could not hash existing file {pdf_file}: {e}")

        logger.info(f"✓ Loaded {len(self.downloaded_hashes)} existing file hashes")

    def _ultra_polite_delay(self):
        """Ultra-polite delay with random variation."""
        delay = self.base_delay + random.uniform(-self.random_delay, self.random_delay)
        delay = max(60.0, delay)  # Minimum 60 seconds
        logger.info(f"⏸️  Ultra-polite delay: {delay:.1f} seconds")
        time.sleep(delay)

    def _make_request(self, url: str, params: Optional[Dict] = None, method: str = 'GET') -> Optional[requests.Response]:
        """Make HTTP request with ultra-polite delays and retry logic."""

        for attempt in range(self.max_retries):
            # Ultra-polite delay before each request
            if attempt > 0:
                self._ultra_polite_delay()
            else:
                # Shorter delay for first attempt
                time.sleep(random.uniform(10, 20))

            try:
                if method == 'GET':
                    response = self.session.get(url, params=params, timeout=60)
                else:
                    response = self.session.post(url, params=params, timeout=60)

                if response.status_code == 200:
                    return response
                elif response.status_code == 202:
                    logger.warning(f"HTTP 202 (Accepted) - AWS WAF blocking. Attempt {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        wait_time = 300 + (attempt * 120)  # 5, 7, 9, 11 minutes
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                elif response.status_code == 429:
                    wait_time = 600 + (attempt * 300)  # 10, 15, 20, 25 minutes
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 180 + (attempt * 60)  # 3, 4, 5, 6 minutes
                    time.sleep(wait_time)

        self.stats['total_errors'] += 1
        return None

    def search_recap_archive(self, query: str, max_pages: int = 5) -> List[Dict[str, Any]]:
        """Search RECAP Archive with ultra-polite delays."""
        logger.info(f"🔍 Ultra-polite search: '{query}'")
        self.stats['total_searches'] += 1

        all_dockets = []

        for page in range(1, max_pages + 1):
            logger.info(f"  📄 Processing page {page}/{max_pages}")

            search_params = {
                'q': query,
                'type': 'r',  # RECAP Archive
                'order_by': 'dateFiled desc',
                'available_only': 'on',  # Only show results with PDFs
                'page': page
            }

            response = self._make_request(self.recap_search_url, params=search_params)
            if not response:
                logger.warning(f"Search failed for query: {query}, page: {page}")
                break

            soup = BeautifulSoup(response.text, 'html.parser')

            # Check if we've reached the end
            if "No results found" in soup.get_text() or "Page" not in soup.get_text():
                logger.info(f"  ✓ Reached end of results for query: {query}")
                break

            # Extract docket links
            page_dockets = []
            for link_tag in soup.select('h2.case-name a'):
                docket_url = urljoin(self.base_url, link_tag['href'])
                case_name = link_tag.get_text(strip=True)

                if docket_url not in self.all_case_urls:
                    self.all_case_urls.add(docket_url)
                    page_dockets.append({
                        'url': docket_url,
                        'case_name': case_name,
                        'query': query,
                        'page': page
                    })

            if not page_dockets:
                logger.info(f"  ✓ No new cases found on page {page}")
                break

            all_dockets.extend(page_dockets)
            logger.info(f"  ✓ Found {len(page_dockets)} new cases on page {page}")

            # Ultra-polite pause between pages
            if page < max_pages:
                time.sleep(random.uniform(30, 60))

        logger.info(f"✓ Total unique cases found for '{query}': {len(all_dockets)}")
        return all_dockets

    def download_docket_pdfs(self, docket_info: Dict[str, Any]) -> List[Path]:
        """Download only FREE PDFs from docket page."""
        docket_url = docket_info['url']
        case_name = docket_info['case_name']

        logger.info(f"📄 Analyzing docket: {case_name}")

        response = self._make_request(docket_url)
        if not response:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        downloaded_pdfs = []

        # Find all document entries
        document_entries = soup.select('.docket-entry')

        for entry in document_entries:
            # Look for "Download PDF" buttons (FREE)
            download_buttons = entry.select('a[href*=".pdf"]')

            for button in download_buttons:
                button_text = button.get_text(strip=True).lower()

                # Only download FREE PDFs
                if 'download pdf' in button_text and 'buy' not in button_text:
                    pdf_url = urljoin(self.base_url, button['href'])
                    file_name = Path(urlparse(pdf_url).path).name

                    # Skip if already downloaded
                    if (self.pdf_dir / file_name).exists():
                        logger.info(f"  - PDF already exists: {file_name}")
                        downloaded_pdfs.append(self.pdf_dir / file_name)
                        continue

                    # Download FREE PDF
                    logger.info(f"  - Downloading FREE PDF: {file_name}")
                    pdf_response = self._make_request(pdf_url)

                    if pdf_response and pdf_response.status_code == 200:
                        save_path = self.pdf_dir / file_name

                        try:
                            with open(save_path, 'wb') as f:
                                f.write(pdf_response.content)

                            # Check for duplicates
                            with open(save_path, 'rb') as f:
                                file_hash = hashlib.md5(f.read()).hexdigest()

                            if file_hash in self.downloaded_hashes:
                                logger.info(f"  ✓ Duplicate detected, removing: {file_name}")
                                save_path.unlink()
                                self.stats['duplicates_skipped'] += 1
                            else:
                                self.downloaded_hashes.add(file_hash)
                                downloaded_pdfs.append(save_path)
                                self.stats['total_pdfs_downloaded'] += 1
                                self.stats['free_pdfs_found'] += 1
                                logger.info(f"  ✓ Saved FREE PDF: {file_name}")

                        except Exception as e:
                            logger.error(f"  ✗ Error saving {file_name}: {e}")
                    else:
                        logger.warning(f"  ✗ Failed to download {file_name}")

                # Count PAID PDFs we're skipping
                elif 'buy on pacer' in button_text:
                    self.stats['paid_pdfs_skipped'] += 1

        return downloaded_pdfs

    def run_ultra_polite_search(self):
        """Run ultra-polite comprehensive search."""
        logger.info("🚀 Ultra-Polite CourtListener PDF Scraper")
        logger.info("=" * 70)
        logger.info("🎯 Targeting ONLY FREE PDFs (Download PDF buttons)")
        logger.info("⏸️  Ultra-polite delays: 90±30 seconds between requests")
        logger.info("🛡️  Sophisticated anti-detection measures")

        # Comprehensive search queries
        search_queries = [
            "28 usc 1782 discovery",
            '"28 U.S.C. § 1782"',
            '"28 USC 1782"',
            '"Application for Judicial Assistance"',
            '"foreign proceeding pursuant to 28"',
            '"ex parte application" AND "1782"',
            '"discovery for use in foreign proceeding"',
            '"letter rogatory" AND "1782"',
        ]

        all_dockets = []

        # Search all queries with ultra-polite delays
        for i, query in enumerate(search_queries, 1):
            logger.info(f"\\n🔍 Search {i}/{len(search_queries)}: {query}")
            dockets = self.search_recap_archive(query, max_pages=3)  # Start with fewer pages
            all_dockets.extend(dockets)

            # Ultra-polite pause between different search queries
            if i < len(search_queries):
                logger.info(f"⏸️  Ultra-polite pause between searches...")
                self._ultra_polite_delay()

        # Remove duplicates
        unique_dockets = []
        seen_urls = set()
        for docket in all_dockets:
            if docket['url'] not in seen_urls:
                unique_dockets.append(docket)
                seen_urls.add(docket['url'])

        logger.info(f"\\n📊 ULTRA-POLITE SEARCH RESULTS:")
        logger.info(f"   Total searches: {len(search_queries)}")
        logger.info(f"   Total cases found: {len(all_dockets)}")
        logger.info(f"   Unique cases: {len(unique_dockets)}")
        logger.info(f"   PDFs will be saved to: {self.pdf_dir.absolute()}")

        # Process all unique dockets with ultra-polite delays
        batch_size = 2  # Very small batches
        for i in range(0, len(unique_dockets), batch_size):
            batch_dockets = unique_dockets[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"\\n🔄 Processing Batch {batch_num}: {len(batch_dockets)} dockets")
            logger.info(f"📁 Check folder: {self.pdf_dir.absolute()}")

            for j, docket in enumerate(batch_dockets):
                case_num = i + j + 1
                logger.info(f"Processing {case_num}/{len(unique_dockets)}: {docket['case_name']}")

                downloaded_pdfs = self.download_docket_pdfs(docket)

                self.scraping_log.append({
                    'docket_url': docket['url'],
                    'case_name': docket['case_name'],
                    'query': docket['query'],
                    'page': docket['page'],
                    'pdfs_downloaded': [str(p) for p in downloaded_pdfs],
                    'status': 'success' if downloaded_pdfs else 'no_free_pdfs'
                })

                logger.info(f"📄 Total FREE PDFs downloaded: {self.stats['total_pdfs_downloaded']}")
                logger.info(f"💰 PAID PDFs skipped: {self.stats['paid_pdfs_skipped']}")

            # Save progress after each batch
            self.save_progress()

            # Ultra-polite pause between batches
            if i + batch_size < len(unique_dockets):
                logger.info(f"⏸️  Ultra-polite pause between batches...")
                self._ultra_polite_delay()

        self.generate_summary_report()
        logger.info(f"\\n✅ Ultra-polite scraping complete!")
        logger.info(f"📁 Final FREE PDF count: {self.stats['total_pdfs_downloaded']}")
        logger.info(f"💰 PAID PDFs skipped: {self.stats['paid_pdfs_skipped']}")

    def save_progress(self):
        """Save current progress to log file."""
        log_file = Path("data/case_law/ultra_polite_scraping_log.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'scraping_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stats': self.stats,
                'results': self.scraping_log
            }, f, indent=2, ensure_ascii=False)

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_path = Path("data/case_law/ultra_polite_scraping_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Ultra-Polite CourtListener PDF Scraping Report\\n\\n")
            f.write(f"**Date**: {self.stats['last_run']}\\n")
            f.write(f"**Total Searches**: {self.stats['total_searches']}\\n")
            f.write(f"**Total Cases Found**: {self.stats['total_cases_found']}\\n")
            f.write(f"**FREE PDFs Downloaded**: {self.stats['total_pdfs_downloaded']}\\n")
            f.write(f"**PAID PDFs Skipped**: {self.stats['paid_pdfs_skipped']}\\n")
            f.write(f"**Duplicates Skipped**: {self.stats['duplicates_skipped']}\\n")
            f.write(f"**Total Errors**: {self.stats['total_errors']}\\n\\n")

            f.write("## Strategy\\n")
            f.write("- **Ultra-polite delays**: 90±30 seconds between requests\\n")
            f.write("- **Target**: Only FREE PDFs (Download PDF buttons)\\n")
            f.write("- **Skip**: PAID PDFs (Buy on PACER buttons)\\n")
            f.write("- **Anti-detection**: Sophisticated browser simulation\\n\\n")

            f.write("## Downloaded FREE PDFs by Case\\n")
            for entry in self.scraping_log:
                f.write(f"- **Case**: {entry['case_name']}\\n")
                f.write(f"  - **Query**: {entry['query']}\\n")
                f.write(f"  - **Status**: {entry['status']}\\n")
                if entry['pdfs_downloaded']:
                    f.write(f"  - **FREE PDFs**: {', '.join([Path(p).name for p in entry['pdfs_downloaded']])}\\n")
                f.write("\\n")

if __name__ == "__main__":
    scraper = UltraPoliteCourtListenerScraper()
    scraper.stats['last_run'] = time.strftime('%Y-%m-%d %H:%M:%S')
    scraper.run_ultra_polite_search()
