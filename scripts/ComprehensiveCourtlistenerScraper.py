#!/usr/bin/env python3
"""
Comprehensive CourtListener PDF Scraper
Downloads ALL §1782 PDFs using multiple search strategies
"""

import json
import os
import time
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

class ComprehensiveCourtListenerScraper:
    def __init__(self):
        self.base_url = "https://www.courtlistener.com"
        self.recap_search_url = f"{self.base_url}/recap/search/"

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MalcolmGrayson §1782 research (malcolmgrayson00@gmail.com) - comprehensive scraper',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        self.pdf_dir = Path("data/recap_petitions_comprehensive")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        self.delay = 25.0  # seconds between requests (be very polite)
        self.max_retries = 3

        self.scraping_log = []
        self.downloaded_hashes = set()
        self.all_case_urls = set()  # Track all unique case URLs

        self.stats = {
            'total_searches': 0,
            'total_cases_found': 0,
            'total_pdfs_downloaded': 0,
            'total_errors': 0,
            'duplicates_skipped': 0,
            'last_run': None
        }

        # Load existing downloaded files to avoid duplicates
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

    def _make_request(self, url: str, params: Optional[Dict] = None, method: str = 'GET') -> Optional[requests.Response]:
        """Helper to make HTTP requests with delay and error handling."""
        time.sleep(self.delay)

        for attempt in range(self.max_retries):
            try:
                if method == 'GET':
                    response = self.session.get(url, params=params, timeout=30)
                else:
                    response = self.session.post(url, params=params, timeout=30)

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    wait_time = self.delay * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))

        self.stats['total_errors'] += 1
        return None

    def search_recap_archive(self, query: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        """Searches the RECAP Archive for dockets matching the query."""
        logger.info(f"🔍 Searching RECAP Archive for: '{query}'")
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

            # Check if we've reached the end of results
            if "No results found" in soup.get_text() or "Page" not in soup.get_text():
                logger.info(f"  ✓ Reached end of results for query: {query}")
                break

            # Extract docket links from this page
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

            # Brief pause between pages
            time.sleep(5)

        logger.info(f"✓ Total unique cases found for '{query}': {len(all_dockets)}")
        return all_dockets

    def download_docket_pdfs(self, docket_info: Dict[str, Any]) -> List[Path]:
        """Downloads all PDFs from a docket page."""
        docket_url = docket_info['url']
        case_name = docket_info['case_name']

        logger.info(f"📄 Downloading PDFs from: {case_name}")

        response = self._make_request(docket_url)
        if not response:
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        downloaded_pdfs = []

        # Find all PDF download links
        for link_tag in soup.select('a[href$=".pdf"]'):
            pdf_url = urljoin(self.base_url, link_tag['href'])
            file_name = Path(urlparse(pdf_url).path).name

            # Skip if already downloaded
            if (self.pdf_dir / file_name).exists():
                logger.info(f"  - PDF already exists: {file_name}")
                downloaded_pdfs.append(self.pdf_dir / file_name)
                continue

            # Download PDF
            logger.info(f"  - Downloading: {file_name}")
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
                        logger.info(f"  ✓ Saved: {file_name}")

                except Exception as e:
                    logger.error(f"  ✗ Error saving {file_name}: {e}")
            else:
                logger.warning(f"  ✗ Failed to download {file_name}")

        return downloaded_pdfs

    def run_comprehensive_search(self):
        """Runs comprehensive search using all identified search terms."""
        logger.info("🚀 Comprehensive CourtListener PDF Scraper")
        logger.info("=" * 70)

        # All search terms identified from the CourtListener interface
        search_queries = [
            # Primary searches (from your interface)
            "28 usc 1782 discovery",
            '"28 U.S.C. § 1782"',
            '"28 USC 1782"',
            '"Application for Judicial Assistance"',

            # Additional comprehensive searches
            '"foreign proceeding pursuant to 28"',
            '"ex parte application" AND "1782"',
            '"discovery for use in foreign proceeding"',
            '"28 U.S.C. 1782"',
            '"miscellaneous case" AND "1782"',
            '"letter rogatory" AND "1782"',

            # Broader searches to catch everything
            '"1782" AND "discovery"',
            '"1782" AND "foreign"',
            '"1782" AND "tribunal"',
            '"1782" AND "arbitration"',

            # Specific case types
            '"in re application" AND "1782"',
            '"in re ex parte" AND "1782"',
            '"motion for discovery" AND "1782"',
        ]

        all_dockets = []

        # Search all queries
        for i, query in enumerate(search_queries, 1):
            logger.info(f"\\n🔍 Search {i}/{len(search_queries)}: {query}")
            dockets = self.search_recap_archive(query, max_pages=15)  # More pages per query
            all_dockets.extend(dockets)

            # Pause between different search queries
            if i < len(search_queries):
                logger.info(f"⏸️  Pausing 30 seconds before next search...")
                time.sleep(30)

        # Remove duplicates based on URL
        unique_dockets = []
        seen_urls = set()
        for docket in all_dockets:
            if docket['url'] not in seen_urls:
                unique_dockets.append(docket)
                seen_urls.add(docket['url'])

        logger.info(f"\\n📊 COMPREHENSIVE SEARCH RESULTS:")
        logger.info(f"   Total searches: {len(search_queries)}")
        logger.info(f"   Total cases found: {len(all_dockets)}")
        logger.info(f"   Unique cases: {len(unique_dockets)}")
        logger.info(f"   PDFs will be saved to: {self.pdf_dir.absolute()}")

        # Process all unique dockets
        batch_size = 5  # Smaller batches to be very polite
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
                    'status': 'success' if downloaded_pdfs else 'no_pdfs_found'
                })

                logger.info(f"📄 Total PDFs downloaded: {self.stats['total_pdfs_downloaded']}")

            # Save progress after each batch
            self.save_progress()

            # Longer pause between batches
            if i + batch_size < len(unique_dockets):
                logger.info(f"⏸️  Pausing {self.delay} seconds before next batch...")
                time.sleep(self.delay)

        self.generate_summary_report()
        logger.info(f"\\n✅ Comprehensive scraping complete!")
        logger.info(f"📁 Final PDF count: {self.stats['total_pdfs_downloaded']}")

    def save_progress(self):
        """Save current progress to log file."""
        log_file = Path("data/case_law/comprehensive_scraping_log.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'scraping_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stats': self.stats,
                'results': self.scraping_log
            }, f, indent=2, ensure_ascii=False)

    def generate_summary_report(self):
        """Generates a comprehensive markdown summary report."""
        report_path = Path("data/case_law/comprehensive_scraping_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive CourtListener PDF Scraping Report\\n\\n")
            f.write(f"**Date**: {self.stats['last_run']}\\n")
            f.write(f"**Total Searches**: {self.stats['total_searches']}\\n")
            f.write(f"**Total Cases Found**: {self.stats['total_cases_found']}\\n")
            f.write(f"**Total PDFs Downloaded**: {self.stats['total_pdfs_downloaded']}\\n")
            f.write(f"**Duplicates Skipped**: {self.stats['duplicates_skipped']}\\n")
            f.write(f"**Total Errors**: {self.stats['total_errors']}\\n\\n")

            f.write("## Search Queries Used\\n")
            queries = list(set([entry['query'] for entry in self.scraping_log]))
            for query in queries:
                f.write(f"- {query}\\n")
            f.write("\\n")

            f.write("## Downloaded PDFs by Case\\n")
            for entry in self.scraping_log:
                f.write(f"- **Case**: {entry['case_name']}\\n")
                f.write(f"  - **Query**: {entry['query']}\\n")
                f.write(f"  - **Status**: {entry['status']}\\n")
                if entry['pdfs_downloaded']:
                    f.write(f"  - **PDFs**: {', '.join([Path(p).name for p in entry['pdfs_downloaded']])}\\n")
                f.write("\\n")

if __name__ == "__main__":
    scraper = ComprehensiveCourtListenerScraper()
    scraper.stats['last_run'] = time.strftime('%Y-%m-%d %H:%M:%S')
    scraper.run_comprehensive_search()
