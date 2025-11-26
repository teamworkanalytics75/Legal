#!/usr/bin/env python3
"""
Hybrid 1782 PDF Downloader - Uses both API and web scraping to get ALL PDFs
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Hybrid1782PDFDownloader:
    def __init__(self, config_path: str = "document_ingestion/courtlistener_config.json"):
        """Initialize the hybrid 1782 PDF downloader."""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.base_url = self.config['api']['base_url']
        self.web_base_url = "https://www.courtlistener.com"

        # PDF storage directory
        self.pdf_dir = Path("data/case_law/1782_hybrid_pdfs")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.base_delay = 1.0
        self.last_request_time = 0

        # Statistics tracking
        self.stats = {
            'total_cases_found': 0,
            'total_pdfs_downloaded': 0,
            'total_errors': 0,
            'duplicates_skipped': 0,
            'api_pdfs_downloaded': 0,
            'web_scraped_pdfs_downloaded': 0,
            'last_run': None
        }

        # Load existing PDF hashes
        self.downloaded_hashes = set()
        self.load_existing_pdfs()

        logger.info("Hybrid 1782 PDF Downloader initialized")
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

    def rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.base_delay:
            time.sleep(self.base_delay - elapsed)
        self.last_request_time = time.time()

    def make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Make a request to the CourtListener API."""
        self.rate_limit()

        url = f"{self.base_url}{endpoint}"

        # Add API token to headers
        headers = {
            'Authorization': f'Token {self.config["api"]["api_token"]}',
            'User-Agent': self.config["api"]["user_agent"]
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limited, waiting 60 seconds...")
                time.sleep(60)
                return self.make_api_request(endpoint, params)
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text[:200]}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    def make_web_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make a request to the CourtListener website."""
        self.rate_limit()

        headers = {
            'User-Agent': self.config["api"]["user_agent"]
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
            else:
                logger.error(f"Web request failed: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Web request error: {e}")
            return None

    def search_1782_cases_api(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search CourtListener API for 1782 discovery cases."""
        logger.info(f"🔍 API search for 1782 discovery cases (max {max_results})")

        all_cases = []

        # Search with different keyword combinations
        search_queries = [
            "28 USC 1782",
            "28 U.S.C. 1782",
            "discovery foreign proceeding",
            "judicial assistance"
        ]

        for query in search_queries:
            logger.info(f"Searching with query: {query}")

            params = {
                'q': query,
                'format': 'json',
                'order_by': 'dateFiled desc',
                'stat_Precedential': 'on',
                'page_size': min(50, max_results)
            }

            response = self.make_api_request('/search/', params)

            if response and 'results' in response:
                results = response['results']
                logger.info(f"Found {len(results)} results for '{query}'")

                # Add unique cases
                for result in results:
                    if result not in all_cases:
                        all_cases.append(result)
                        logger.info(f"✓ Added case: {result.get('caseName', 'Unknown')}")

            time.sleep(1)  # Small delay between queries

        logger.info(f"✓ Found {len(all_cases)} total unique 1782-related cases")
        return all_cases[:max_results]

    def search_1782_cases_web(self, max_results: int = 50) -> List[str]:
        """Search CourtListener website for 1782 discovery cases."""
        logger.info(f"🌐 Web search for 1782 discovery cases (max {max_results})")

        case_urls = []

        # Search the RECAP archive for 1782 cases
        search_url = f"{self.web_base_url}/recap/search/"

        search_params = {
            'q': '28 usc 1782',
            'available_only': 'on',  # Only show results with PDFs
            'order_by': 'dateFiled desc'
        }

        logger.info(f"Searching RECAP archive: {search_url}")

        soup = self.make_web_request(search_url)
        if not soup:
            logger.error("Failed to load RECAP search page")
            return case_urls

        # Look for case links
        case_links = soup.find_all('a', href=re.compile(r'/docket/\d+/'))

        for link in case_links[:max_results]:
            href = link.get('href')
            if href:
                full_url = urljoin(self.web_base_url, href)
                case_urls.append(full_url)
                logger.info(f"✓ Found case URL: {full_url}")

        logger.info(f"✓ Found {len(case_urls)} case URLs from web search")
        return case_urls

    def get_pdfs_from_case_url(self, case_url: str) -> List[Dict[str, Any]]:
        """Get all PDFs from a case URL by scraping the web page."""
        logger.info(f"📄 Scraping PDFs from case URL: {case_url}")

        soup = self.make_web_request(case_url)
        if not soup:
            logger.error(f"Failed to load case page: {case_url}")
            return []

        pdfs = []

        # Look for PDF links in the page
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
                    'type': 'web_scraped',
                    'description': link_text,
                    'source': 'web_scraping'
                }

                pdfs.append(pdf_info)
                logger.info(f"  ✓ Found PDF: {link_text} -> {pdf_url}")

        # Also look for RECAP document links
        recap_links = soup.find_all('a', href=re.compile(r'/recap/'))

        for link in recap_links:
            href = link.get('href')
            if href and 'pdf' in href.lower():
                if href.startswith('/'):
                    pdf_url = urljoin(self.web_base_url, href)
                else:
                    pdf_url = href

                link_text = link.get_text(strip=True)

                pdf_info = {
                    'url': pdf_url,
                    'type': 'recap_document',
                    'description': link_text,
                    'source': 'web_scraping'
                }

                pdfs.append(pdf_info)
                logger.info(f"  ✓ Found RECAP PDF: {link_text} -> {pdf_url}")

        logger.info(f"✓ Found {len(pdfs)} PDFs from web scraping")
        return pdfs

    def get_pdfs_from_api_case(self, case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get PDFs from an API case result."""
        case_name = case_data.get('caseName', 'Unknown')
        logger.info(f"📄 Getting PDFs from API case: {case_name}")

        pdfs = []

        # Get opinions from cluster
        opinions = case_data.get('opinions', [])
        logger.info(f"Found {len(opinions)} opinions in cluster")

        for i, opinion in enumerate(opinions):
            opinion_id = opinion.get('id')
            if opinion_id:
                logger.info(f"Processing opinion {i+1}: ID {opinion_id}")

                # Get opinion details
                opinion_response = self.make_api_request(f'/opinions/{opinion_id}/')
                if opinion_response:
                    pdf_url = opinion_response.get('download_url') or opinion_response.get('pdf_url')

                    if pdf_url and pdf_url.lower().endswith('.pdf'):
                        pdf_info = {
                            'url': pdf_url,
                            'type': f'api_opinion_{i+1}',
                            'description': f"Opinion {i+1}",
                            'source': 'api'
                        }
                        pdfs.append(pdf_info)
                        logger.info(f"  ✓ Found API PDF: {pdf_url}")

        logger.info(f"✓ Found {len(pdfs)} PDFs from API")
        return pdfs

    def download_pdf(self, pdf_info: Dict[str, Any], case_name: str) -> Optional[Path]:
        """Download a PDF file."""
        pdf_url = pdf_info['url']
        pdf_type = pdf_info['type']
        source = pdf_info.get('source', 'unknown')

        try:
            # Create filename
            safe_case_name = "".join(c for c in case_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_case_name = safe_case_name.replace(" ", "_")[:50]

            filename = f"{safe_case_name}_{pdf_type}_{source}.pdf"
            save_path = self.pdf_dir / filename

            # Skip if already downloaded
            if save_path.exists():
                logger.info(f"  - PDF already exists: {filename}")
                return save_path

            logger.info(f"📥 Downloading {pdf_type} ({source}): {filename}")
            logger.info(f"   URL: {pdf_url}")

            response = requests.get(pdf_url, timeout=60)

            if response.status_code == 200:
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

                    if source == 'api':
                        self.stats['api_pdfs_downloaded'] += 1
                    else:
                        self.stats['web_scraped_pdfs_downloaded'] += 1

                    logger.info(f"  ✓ Downloaded PDF: {filename}")
                    return save_path
            else:
                logger.warning(f"  ✗ Failed to download {filename}: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"  ✗ Error downloading {filename}: {e}")
            return None

    def process_case_hybrid(self, case_data: Dict[str, Any] = None, case_url: str = None):
        """Process a single case using both API and web scraping."""
        if case_data:
            case_name = case_data.get('caseName', 'Unknown')
            logger.info(f"\n📋 Processing API case: {case_name}")

            # Get PDFs from API
            api_pdfs = self.get_pdfs_from_api_case(case_data)

            # Download API PDFs
            for pdf_info in api_pdfs:
                self.download_pdf(pdf_info, case_name)

            self.stats['total_cases_found'] += 1

        elif case_url:
            logger.info(f"\n📋 Processing web case: {case_url}")

            # Get PDFs from web scraping
            web_pdfs = self.get_pdfs_from_case_url(case_url)

            # Extract case name from URL or page
            case_name = case_url.split('/')[-2] if '/' in case_url else 'Unknown'

            # Download web PDFs
            for pdf_info in web_pdfs:
                self.download_pdf(pdf_info, case_name)

            self.stats['total_cases_found'] += 1

    def run_hybrid_download(self, max_cases: int = 20):
        """Run hybrid 1782 PDF download session."""
        logger.info("🚀 Hybrid 1782 Discovery PDF Downloader")
        logger.info("=" * 70)
        logger.info("🎯 Downloading ALL available PDFs using API + Web Scraping")
        logger.info("📄 Sources: CourtListener API + RECAP Archive Web Scraping")
        logger.info(f"📁 PDFs will be saved to: {self.pdf_dir.absolute()}")

        # 1. Search for cases using API
        api_cases = self.search_1782_cases_api(max_results=max_cases//2)

        # 2. Search for cases using web scraping
        web_case_urls = self.search_1782_cases_web(max_results=max_cases//2)

        total_cases = len(api_cases) + len(web_case_urls)
        logger.info(f"Found {total_cases} total cases to process ({len(api_cases)} API + {len(web_case_urls)} web)")

        # 3. Process API cases
        for i, case_data in enumerate(api_cases):
            case_num = i + 1
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing API case {case_num}/{len(api_cases)}")
            logger.info(f"{'='*50}")

            self.process_case_hybrid(case_data=case_data)

            # Pause between cases
            if case_num < len(api_cases):
                time.sleep(2)

        # 4. Process web cases
        for i, case_url in enumerate(web_case_urls):
            case_num = i + 1
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing web case {case_num}/{len(web_case_urls)}")
            logger.info(f"{'='*50}")

            self.process_case_hybrid(case_url=case_url)

            # Pause between cases
            if case_num < len(web_case_urls):
                time.sleep(2)

        # Generate final report
        self.generate_summary_report()

        logger.info(f"\n✅ Hybrid 1782 PDF download complete!")
        logger.info(f"📊 Final statistics:")
        logger.info(f"   Cases found: {self.stats['total_cases_found']}")
        logger.info(f"   Total PDFs downloaded: {self.stats['total_pdfs_downloaded']}")
        logger.info(f"   API PDFs downloaded: {self.stats['api_pdfs_downloaded']}")
        logger.info(f"   Web scraped PDFs downloaded: {self.stats['web_scraped_pdfs_downloaded']}")
        logger.info(f"   Duplicates skipped: {self.stats['duplicates_skipped']}")
        logger.info(f"   Errors: {self.stats['total_errors']}")

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_path = Path("data/case_law/1782_hybrid_download_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Hybrid 1782 Discovery PDF Download Report\n\n")
            f.write(f"**Date**: {self.stats['last_run']}\n")
            f.write(f"**Cases Found**: {self.stats['total_cases_found']}\n")
            f.write(f"**Total PDFs Downloaded**: {self.stats['total_pdfs_downloaded']}\n")
            f.write(f"**API PDFs Downloaded**: {self.stats['api_pdfs_downloaded']}\n")
            f.write(f"**Web Scraped PDFs Downloaded**: {self.stats['web_scraped_pdfs_downloaded']}\n")
            f.write(f"**Duplicates Skipped**: {self.stats['duplicates_skipped']}\n")
            f.write(f"**Total Errors**: {self.stats['total_errors']}\n\n")

            f.write("## Strategy\n")
            f.write("- **Hybrid Approach**: Uses both CourtListener API and web scraping\n")
            f.write("- **API Search**: Finds cases through CourtListener API\n")
            f.write("- **Web Scraping**: Searches RECAP archive directly for additional cases\n")
            f.write("- **Comprehensive Coverage**: Gets PDFs from both sources\n\n")

            f.write("## PDF Storage\n")
            f.write(f"PDFs are stored in: `{self.pdf_dir.absolute()}`\n\n")

def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid 1782 Discovery PDF Downloader")
    parser.add_argument("--max-cases", type=int, default=20, help="Maximum cases to process")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Hybrid 1782 Discovery PDF Downloader")
    print("="*60 + "\n")

    downloader = Hybrid1782PDFDownloader()
    downloader.stats['last_run'] = datetime.now().isoformat()

    downloader.run_hybrid_download(max_cases=args.max_cases)

if __name__ == "__main__":
    main()
