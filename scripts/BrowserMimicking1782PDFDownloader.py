#!/usr/bin/env python3
"""
Browser-Mimicking 1782 PDF Downloader - Uses realistic browser headers and behavior
"""

import json
import time
import logging
import hashlib
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import re
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrowserMimicking1782PDFDownloader:
    def __init__(self, config_path: str = "document_ingestion/courtlistener_config.json"):
        """Initialize the browser-mimicking 1782 PDF downloader."""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.web_base_url = "https://www.courtlistener.com"

        # PDF storage directory
        self.pdf_dir = Path("data/case_law/1782_browser_pdfs")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.base_delay = 3.0
        self.last_request_time = 0

        # Statistics tracking
        self.stats = {
            'total_cases_found': 0,
            'total_pdfs_downloaded': 0,
            'total_errors': 0,
            'duplicates_skipped': 0,
            'last_run': None
        }

        # Load existing PDF hashes
        self.downloaded_hashes = set()
        self.load_existing_pdfs()

        # Session for maintaining cookies
        self.session = requests.Session()

        logger.info("Browser-Mimicking 1782 PDF Downloader initialized")
        logger.info(f"PDF storage: {self.pdf_dir.absolute()}")

    def load_existing_pdfs(self):
        """Load existing PDF files to avoid duplicates."""
        for pdf_file in self.pdf_dir.rglob("*.pdf"):
            try:
                with open(pdf_file, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    self.downloaded_hashes.add(file_hash)
            except Exception as e:
                logger.warning(f"Could not hash existing file {pdf_file}: {e}")

        logger.info(f"✓ Loaded {len(self.downloaded_hashes)} existing PDF hashes")

    def get_realistic_headers(self) -> Dict[str, str]:
        """Get realistic browser headers."""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]

        return {
            'User-Agent': random.choice(user_agents),
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
            'Cache-Control': 'max-age=0'
        }

    def rate_limit(self):
        """Enforce realistic rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.base_delay:
            sleep_time = self.base_delay - elapsed + random.uniform(0.5, 2.0)
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def make_request(self, url: str, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """Make a realistic request to CourtListener."""
        self.rate_limit()

        headers = self.get_realistic_headers()

        for attempt in range(max_retries):
            try:
                logger.info(f"Making request to: {url} (attempt {attempt + 1})")

                response = self.session.get(url, headers=headers, timeout=30)

                if response.status_code == 200:
                    logger.info(f"✓ Successfully loaded: {url}")
                    return BeautifulSoup(response.content, 'html.parser')
                elif response.status_code == 403:
                    logger.warning(f"403 Forbidden - waiting longer before retry")
                    time.sleep(random.uniform(10, 20))
                    continue
                elif response.status_code == 429:
                    logger.warning(f"Rate limited - waiting 60 seconds")
                    time.sleep(60)
                    continue
                else:
                    logger.error(f"Request failed: {response.status_code}")
                    return None

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(5, 10))
                    continue
                return None

        return None

    def search_recap_archive_realistic(self, max_results: int = 50) -> List[str]:
        """Search RECAP archive using realistic browser behavior."""
        logger.info(f"🔍 Searching RECAP archive realistically for 1782 cases (max {max_results})")

        case_urls = []

        # First, visit the main RECAP page to establish session
        main_url = f"{self.web_base_url}/recap/"
        logger.info(f"Visiting main RECAP page: {main_url}")

        soup = self.make_request(main_url)
        if not soup:
            logger.error("Failed to load main RECAP page")
            return case_urls

        # Now search for 1782 cases
        search_url = f"{self.web_base_url}/recap/search/"

        # Build search parameters exactly like in your screenshot
        search_params = {
            'q': '28 usc 1782',
            'available_only': 'on',
            'order_by': 'dateFiled desc'
        }

        # Build the full URL with parameters
        param_string = '&'.join([f"{k}={v}" for k, v in search_params.items()])
        full_url = f"{search_url}?{param_string}"

        logger.info(f"Searching RECAP archive: {full_url}")

        soup = self.make_request(full_url)
        if not soup:
            logger.error("Failed to load RECAP search page")
            return case_urls

        # Look for case links in the search results
        # Based on your screenshot, cases have URLs like /docket/69863999/1/case-name/
        case_links = soup.find_all('a', href=re.compile(r'/docket/\d+/'))

        for link in case_links:
            href = link.get('href')
            if href:
                full_url = urljoin(self.web_base_url, href)
                case_urls.append(full_url)
                logger.info(f"✓ Found case URL: {full_url}")

        logger.info(f"✓ Found {len(case_urls)} case URLs from RECAP search")
        return case_urls

    def get_all_pdfs_from_case(self, case_url: str) -> List[Dict[str, Any]]:
        """Get all PDFs from a case page using realistic browser behavior."""
        logger.info(f"📄 Getting all PDFs from case: {case_url}")

        soup = self.make_request(case_url)
        if not soup:
            logger.error(f"Failed to load case page: {case_url}")
            return []

        pdfs = []

        # Look for all PDF links on the page
        pdf_links = soup.find_all('a', href=re.compile(r'\.pdf'))

        for link in pdf_links:
            href = link.get('href')
            if href:
                # Make sure it's a full URL
                if href.startswith('/'):
                    pdf_url = urljoin(self.web_base_url, href)
                else:
                    pdf_url = href

                # Get link text for description
                link_text = link.get_text(strip=True)

                pdf_info = {
                    'url': pdf_url,
                    'type': 'recap_pdf',
                    'description': link_text,
                    'case_url': case_url
                }

                pdfs.append(pdf_info)
                logger.info(f"  ✓ Found PDF: {link_text} -> {pdf_url}")

        # Look for RECAP document links
        recap_links = soup.find_all('a', href=re.compile(r'/recap/'))

        for link in recap_links:
            href = link.get('href')
            if href:
                if href.startswith('/'):
                    full_url = urljoin(self.web_base_url, href)
                else:
                    full_url = href

                link_text = link.get_text(strip=True)

                # Check if this looks like a document link
                if any(keyword in link_text.lower() for keyword in ['pdf', 'document', 'filing', 'motion', 'order', 'brief', 'memorandum']):
                    pdf_info = {
                        'url': full_url,
                        'type': 'recap_document',
                        'description': link_text,
                        'case_url': case_url
                    }

                    pdfs.append(pdf_info)
                    logger.info(f"  ✓ Found RECAP document: {link_text} -> {full_url}")

        # Look for docket entry links that might contain PDFs
        docket_links = soup.find_all('a', href=re.compile(r'/docket/\d+/'))

        for link in docket_links:
            href = link.get('href')
            if href and href != case_url:  # Don't include the current case URL
                if href.startswith('/'):
                    full_url = urljoin(self.web_base_url, href)
                else:
                    full_url = href

                link_text = link.get_text(strip=True)

                pdf_info = {
                    'url': full_url,
                    'type': 'docket_entry',
                    'description': link_text,
                    'case_url': case_url
                }

                pdfs.append(pdf_info)
                logger.info(f"  ✓ Found docket entry: {link_text} -> {full_url}")

        logger.info(f"✓ Found {len(pdfs)} total PDFs/documents from case")
        return pdfs

    def download_pdf(self, pdf_info: Dict[str, Any], case_name: str) -> Optional[Path]:
        """Download a PDF file using realistic browser behavior."""
        pdf_url = pdf_info['url']
        pdf_type = pdf_info['type']

        try:
            # Create filename
            safe_case_name = "".join(c for c in case_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_case_name = safe_case_name.replace(" ", "_")[:50]

            filename = f"{safe_case_name}_{pdf_type}.pdf"
            save_path = self.pdf_dir / filename

            # Skip if already downloaded
            if save_path.exists():
                logger.info(f"  - PDF already exists: {filename}")
                return save_path

            logger.info(f"📥 Downloading {pdf_type}: {filename}")
            logger.info(f"   URL: {pdf_url}")

            headers = self.get_realistic_headers()

            response = self.session.get(pdf_url, headers=headers, timeout=60)

            if response.status_code == 200:
                # Check if it's actually a PDF
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and not pdf_url.lower().endswith('.pdf'):
                    logger.info(f"  - Skipping non-PDF content: {content_type}")
                    return None

                # Save the file
                with open(save_path, 'wb') as f:
                    f.write(response.content)

                # Check for duplicates
                with open(save_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in self.downloaded_hashes:
                    logger.info(f"  ✓ Duplicate detected, removing: {filename}")
                    save_path.unlink()
                    self.stats['duplicates_skipped'] += 1
                    return None
                else:
                    self.downloaded_hashes.add(file_hash)
                    self.stats['total_pdfs_downloaded'] += 1
                    logger.info(f"  ✓ Downloaded PDF: {filename}")
                    return save_path
            else:
                logger.warning(f"  ✗ Failed to download {filename}: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"  ✗ Error downloading {filename}: {e}")
            return None

    def process_case(self, case_url: str):
        """Process a single case."""
        logger.info(f"\n📋 Processing case: {case_url}")

        try:
            # Extract case name from URL
            case_name = case_url.split('/')[-2] if '/' in case_url else 'Unknown'

            # Get all PDFs from this case
            case_pdfs = self.get_all_pdfs_from_case(case_url)

            if not case_pdfs:
                logger.info(f"✗ No PDFs found for: {case_name}")
                return

            all_downloaded_pdfs = []

            # Download each PDF
            for pdf_info in case_pdfs:
                pdf_path = self.download_pdf(pdf_info, case_name)
                if pdf_path:
                    all_downloaded_pdfs.append(pdf_path)

            if all_downloaded_pdfs:
                logger.info(f"✓ Downloaded {len(all_downloaded_pdfs)} PDFs for {case_name}")
            else:
                logger.info(f"✗ No PDFs downloaded for {case_name}")

            self.stats['total_cases_found'] += 1

        except Exception as e:
            logger.error(f"Error processing case {case_url}: {e}")
            self.stats['total_errors'] += 1

    def run_browser_mimicking_download(self, max_cases: int = 10):
        """Run browser-mimicking 1782 PDF download session."""
        logger.info("🚀 Browser-Mimicking 1782 Discovery PDF Downloader")
        logger.info("=" * 70)
        logger.info("🎯 Downloading ALL available PDFs using realistic browser behavior")
        logger.info("📄 Focus: 1782 discovery cases with PDFs from RECAP Archive")
        logger.info(f"📁 PDFs will be saved to: {self.pdf_dir.absolute()}")

        # Search for cases in RECAP archive
        case_urls = self.search_recap_archive_realistic(max_results=max_cases)

        if not case_urls:
            logger.info("No 1782 discovery cases found in RECAP archive")
            return

        logger.info(f"Found {len(case_urls)} 1782 discovery cases to process")

        # Process each case
        for i, case_url in enumerate(case_urls):
            case_num = i + 1
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing case {case_num}/{len(case_urls)}")
            logger.info(f"{'='*50}")

            self.process_case(case_url)

            # Pause between cases
            if case_num < len(case_urls):
                pause_time = random.uniform(5, 10)
                logger.info(f"⏸️  Pause between cases: {pause_time:.1f} seconds")
                time.sleep(pause_time)

        # Generate final report
        self.generate_summary_report()

        logger.info(f"\n✅ Browser-mimicking 1782 PDF download complete!")
        logger.info(f"📊 Final statistics:")
        logger.info(f"   Cases found: {self.stats['total_cases_found']}")
        logger.info(f"   Total PDFs downloaded: {self.stats['total_pdfs_downloaded']}")
        logger.info(f"   Duplicates skipped: {self.stats['duplicates_skipped']}")
        logger.info(f"   Errors: {self.stats['total_errors']}")

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_path = Path("data/case_law/1782_browser_download_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Browser-Mimicking 1782 Discovery PDF Download Report\n\n")
            f.write(f"**Date**: {self.stats['last_run']}\n")
            f.write(f"**Cases Found**: {self.stats['total_cases_found']}\n")
            f.write(f"**Total PDFs Downloaded**: {self.stats['total_pdfs_downloaded']}\n")
            f.write(f"**Duplicates Skipped**: {self.stats['duplicates_skipped']}\n")
            f.write(f"**Total Errors**: {self.stats['total_errors']}\n\n")

            f.write("## Strategy\n")
            f.write("- **Browser Mimicking**: Uses realistic browser headers and behavior\n")
            f.write("- **Session Management**: Maintains cookies and session state\n")
            f.write("- **Rate Limiting**: Implements realistic delays between requests\n")
            f.write("- **RECAP Focus**: Targets CourtListener RECAP archive specifically\n")
            f.write("- **Anti-Detection**: Uses random user agents and realistic timing\n\n")

            f.write("## PDF Storage\n")
            f.write(f"PDFs are stored in: `{self.pdf_dir.absolute()}`\n\n")

def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Browser-Mimicking 1782 Discovery PDF Downloader")
    parser.add_argument("--max-cases", type=int, default=10, help="Maximum cases to process")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Browser-Mimicking 1782 Discovery PDF Downloader")
    print("="*60 + "\n")

    downloader = BrowserMimicking1782PDFDownloader()
    downloader.stats['last_run'] = datetime.now().isoformat()

    downloader.run_browser_mimicking_download(max_cases=args.max_cases)

if __name__ == "__main__":
    main()
