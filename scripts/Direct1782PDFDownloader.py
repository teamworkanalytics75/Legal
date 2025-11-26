#!/usr/bin/env python3
"""
Direct 1782 Discovery PDF Downloader
Searches CourtListener directly for 1782 cases and downloads ALL available PDFs
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Direct1782PDFDownloader:
    def __init__(self, config_path: str = "document_ingestion/courtlistener_config.json"):
        """Initialize the direct 1782 PDF downloader."""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.base_url = self.config['api']['base_url']

        # PDF storage directory
        self.pdf_dir = Path("data/case_law/1782_discovery_pdfs")
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
            'opinions_downloaded': 0,
            'docket_entries_downloaded': 0,
            'exhibits_downloaded': 0,
            'initial_filings_downloaded': 0,
            'last_run': None
        }

        # Load existing PDF hashes
        self.downloaded_hashes = set()
        self.load_existing_pdfs()

        logger.info("Direct 1782 Discovery PDF Downloader initialized")
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
        """Enforce rate limiting between API requests."""
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
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    def search_1782_cases_direct(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search CourtListener directly for 1782 discovery cases."""
        logger.info(f"🔍 Searching CourtListener for 1782 discovery cases (max {max_results})")

        # Get 1782 keywords from config
        keywords = self.config['filters']['topics']['1782_discovery']

        all_cases = []

        # Search with different keyword combinations
        search_queries = [
            "28 USC 1782",
            "28 U.S.C. 1782",
            "discovery foreign proceeding",
            "judicial assistance",
            "Intel Corp",
            "Brandi-Dohrn",
            "Euromepa"
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

    def get_all_pdfs_from_cluster(self, cluster_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all PDFs from a cluster including opinions, docket entries, and exhibits."""
        cluster_id = cluster_data.get('cluster_id')
        case_name = cluster_data.get('caseName', 'Unknown')

        logger.info(f"📄 Getting ALL PDFs from cluster {cluster_id}: {case_name}")

        all_pdfs = []

        # Get opinions from cluster
        opinions = cluster_data.get('opinions', [])
        logger.info(f"Found {len(opinions)} opinions in cluster")

        for i, opinion in enumerate(opinions):
            opinion_id = opinion.get('id')
            if opinion_id:
                logger.info(f"Processing opinion {i+1}: ID {opinion_id}")

                # Get opinion details
                opinion_response = self.make_api_request(f'/opinions/{opinion_id}/')
                if opinion_response:
                    pdf_info = self.extract_pdf_info(opinion_response, f"opinion_{i+1}")
                    if pdf_info:
                        all_pdfs.append(pdf_info)
                        self.stats['opinions_downloaded'] += 1

        # Get docket information if available
        docket_id = cluster_data.get('docket_id')
        if docket_id:
            logger.info(f"Getting docket entries for docket {docket_id}")
            docket_pdfs = self.get_docket_pdfs(docket_id)
            all_pdfs.extend(docket_pdfs)

        logger.info(f"✓ Found {len(all_pdfs)} total PDFs for cluster {cluster_id}")
        return all_pdfs

    def extract_pdf_info(self, opinion_data: Dict[str, Any], pdf_type: str) -> Optional[Dict[str, Any]]:
        """Extract PDF information from opinion data."""
        pdf_url = opinion_data.get('download_url') or opinion_data.get('pdf_url')

        if not pdf_url:
            return None

        # Only process actual PDF URLs, not HTML pages
        if not pdf_url.lower().endswith('.pdf'):
            logger.info(f"  - Skipping non-PDF URL: {pdf_url}")
            return None

        # Determine if this is an initial filing or opinion
        opinion_type = opinion_data.get('type', '').lower()
        is_initial_filing = any(term in opinion_type for term in ['motion', 'petition', 'application', 'complaint'])

        if is_initial_filing:
            self.stats['initial_filings_downloaded'] += 1

        return {
            'url': pdf_url,
            'type': pdf_type,
            'is_initial_filing': is_initial_filing,
            'opinion_id': opinion_data.get('id'),
            'page_count': opinion_data.get('page_count', 0),
            'sha1': opinion_data.get('sha1', '')
        }

    def get_docket_pdfs(self, docket_id: str) -> List[Dict[str, Any]]:
        """Get PDFs from docket entries."""
        logger.info(f"📋 Getting docket PDFs for docket {docket_id}")

        # Get docket details
        docket_response = self.make_api_request(f'/dockets/{docket_id}/')
        if not docket_response:
            return []

        docket_pdfs = []

        # Get docket entries
        entries_response = self.make_api_request(f'/dockets/{docket_id}/docket-entries/')
        if entries_response and 'results' in entries_response:
            entries = entries_response['results']
            logger.info(f"Found {len(entries)} docket entries")

            for i, entry in enumerate(entries):
                # Check for PDF attachments
                if 'recap_documents' in entry:
                    for doc in entry['recap_documents']:
                        if doc.get('filepath', '').endswith('.pdf'):
                            pdf_info = {
                                'url': doc.get('filepath'),
                                'type': f"docket_entry_{i+1}",
                                'is_initial_filing': False,
                                'docket_entry_id': entry.get('id'),
                                'page_count': 0,
                                'sha1': ''
                            }
                            docket_pdfs.append(pdf_info)
                            self.stats['docket_entries_downloaded'] += 1

        logger.info(f"✓ Found {len(docket_pdfs)} docket PDFs")
        return docket_pdfs

    def download_pdf(self, pdf_info: Dict[str, Any], case_name: str) -> Optional[Path]:
        """Download a PDF file."""
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
                    logger.info(f"  ✓ Downloaded PDF: {filename}")
                    return save_path
            else:
                logger.warning(f"  ✗ Failed to download {filename}: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"  ✗ Error downloading {filename}: {e}")
            return None

    def process_1782_case_comprehensive(self, cluster_data: Dict[str, Any]):
        """Process a single 1782 case comprehensively."""
        case_name = cluster_data.get('caseName', 'Unknown')
        logger.info(f"\n📋 Processing 1782 case: {case_name}")

        try:
            # Get all PDFs from this cluster
            cluster_pdfs = self.get_all_pdfs_from_cluster(cluster_data)

            if not cluster_pdfs:
                logger.info(f"✗ No PDFs found for: {case_name}")
                return

            all_downloaded_pdfs = []

            # Download each PDF
            for pdf_info in cluster_pdfs:
                pdf_path = self.download_pdf(pdf_info, case_name)
                if pdf_path:
                    all_downloaded_pdfs.append(pdf_path)

            if all_downloaded_pdfs:
                logger.info(f"✓ Downloaded {len(all_downloaded_pdfs)} PDFs for {case_name}")
            else:
                logger.info(f"✗ No PDFs downloaded for {case_name}")

            self.stats['total_cases_found'] += 1

        except Exception as e:
            logger.error(f"Error processing case {case_name}: {e}")
            self.stats['total_errors'] += 1

    def run_direct_1782_download(self, max_cases: int = 50):
        """Run direct 1782 PDF download session."""
        logger.info("🚀 Direct 1782 Discovery PDF Downloader")
        logger.info("=" * 70)
        logger.info("🎯 Searching CourtListener directly for 1782 discovery cases")
        logger.info("📄 Downloading ALL available PDFs: Opinions, Docket Entries, Exhibits, Initial Filings")
        logger.info(f"📁 PDFs will be saved to: {self.pdf_dir.absolute()}")

        # Search for 1782 cases directly
        cases = self.search_1782_cases_direct(max_results=max_cases)

        if not cases:
            logger.info("No 1782 discovery cases found on CourtListener")
            return

        logger.info(f"Found {len(cases)} 1782 discovery cases to process")

        # Process each case comprehensively
        for i, case_data in enumerate(cases):
            case_num = i + 1
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing case {case_num}/{len(cases)}")
            logger.info(f"{'='*50}")

            self.process_1782_case_comprehensive(case_data)

            # Save progress after each case
            self.save_progress()

            # Pause between cases
            if case_num < len(cases):
                pause_time = 5
                logger.info(f"⏸️  Pause between cases: {pause_time} seconds")
                time.sleep(pause_time)

        # Generate final report
        self.generate_summary_report()

        logger.info(f"\n✅ Direct 1782 PDF download complete!")
        logger.info(f"📊 Final statistics:")
        logger.info(f"   Cases found: {self.stats['total_cases_found']}")
        logger.info(f"   Total PDFs downloaded: {self.stats['total_pdfs_downloaded']}")
        logger.info(f"   Opinions downloaded: {self.stats['opinions_downloaded']}")
        logger.info(f"   Docket entries downloaded: {self.stats['docket_entries_downloaded']}")
        logger.info(f"   Exhibits downloaded: {self.stats['exhibits_downloaded']}")
        logger.info(f"   Initial filings downloaded: {self.stats['initial_filings_downloaded']}")
        logger.info(f"   Duplicates skipped: {self.stats['duplicates_skipped']}")
        logger.info(f"   Errors: {self.stats['total_errors']}")

    def save_progress(self):
        """Save current progress to log file."""
        log_file = Path("data/case_law/1782_direct_download_log.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'download_date': datetime.now().isoformat(),
                'stats': self.stats,
                'pdf_directory': str(self.pdf_dir.absolute())
            }, f, indent=2, ensure_ascii=False)

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_path = Path("data/case_law/1782_direct_download_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Direct 1782 Discovery PDF Download Report\n\n")
            f.write(f"**Date**: {self.stats['last_run']}\n")
            f.write(f"**Cases Found**: {self.stats['total_cases_found']}\n")
            f.write(f"**Total PDFs Downloaded**: {self.stats['total_pdfs_downloaded']}\n")
            f.write(f"**Opinions Downloaded**: {self.stats['opinions_downloaded']}\n")
            f.write(f"**Docket Entries Downloaded**: {self.stats['docket_entries_downloaded']}\n")
            f.write(f"**Exhibits Downloaded**: {self.stats['exhibits_downloaded']}\n")
            f.write(f"**Initial Filings Downloaded**: {self.stats['initial_filings_downloaded']}\n")
            f.write(f"**Duplicates Skipped**: {self.stats['duplicates_skipped']}\n")
            f.write(f"**Total Errors**: {self.stats['total_errors']}\n\n")

            f.write("## Strategy\n")
            f.write("- **Direct Search**: Searches CourtListener directly for 1782 cases\n")
            f.write("- **Multiple Queries**: Uses various 1782-related search terms\n")
            f.write("- **All PDF Types**: Downloads opinions, docket entries, exhibits, initial filings\n")
            f.write("- **Comprehensive Coverage**: Gets everything available for each case\n\n")

            f.write("## PDF Storage\n")
            f.write(f"PDFs are stored in: `{self.pdf_dir.absolute()}`\n\n")

def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Direct 1782 Discovery PDF Downloader")
    parser.add_argument("--max-cases", type=int, default=50, help="Maximum cases to process")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Direct 1782 Discovery PDF Downloader")
    print("="*60 + "\n")

    downloader = Direct1782PDFDownloader()
    downloader.stats['last_run'] = datetime.now().isoformat()

    downloader.run_direct_1782_download(max_cases=args.max_cases)

if __name__ == "__main__":
    main()
