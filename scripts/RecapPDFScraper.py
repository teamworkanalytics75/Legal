#!/usr/bin/env python3
"""
RECAP Archive PDF Scraper for §1782 Cases

Targets the RECAP Archive specifically for §1782 petitions and documents.
Much more conservative approach to avoid rate limiting.

Usage:
    python recap_pdf_scraper.py --batch-size 5 --delay 10
"""

import json
import os
import time
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlencode
from typing import Dict, List, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RECAPPDFScraper:
    def __init__(self):
        self.api_token = "1d6baa6881aa5b32acecff70866e5901a8c5bc18"
        self.recap_base = "https://www.courtlistener.com/recap/"
        self.api_base = "https://www.courtlistener.com/api/rest/v3/"

        # Very polite headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MalcolmGrayson §1782 Research (malcolmgrayson00@gmail.com) - RECAP Archive Access',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        # Output directories
        self.pdf_dir = Path("data/recap_petitions")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Very conservative rate limiting
        self.delay = 10  # 10 seconds between requests
        self.batch_delay = 60  # 1 minute between batches

        # Results tracking
        self.stats = {
            'total_searches': 0,
            'cases_found': 0,
            'pdfs_downloaded': 0,
            'errors': 0
        }
        self.scraping_log = []

    def search_recap_archive(self, query: str = "28 usc 1782", max_results: int = 100) -> List[Dict]:
        """Search RECAP Archive for §1782 cases with PDFs."""
        logger.info(f"🔍 Searching RECAP Archive for: '{query}'")

        # Build RECAP search URL
        params = {
            'q': query,
            'stat_Precedential': 'on',  # Only published cases
            'order_by': 'score desc',   # Most relevant first
            'format': 'json'
        }

        search_url = f"{self.recap_base}search/"

        try:
            logger.info(f"📡 Searching: {search_url}")
            response = self.session.get(search_url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                logger.info(f"✓ Found {len(results)} RECAP results")
                return results[:max_results]
            else:
                logger.warning(f"✗ Search failed: {response.status_code} {response.reason}")
                return []

        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching RECAP: {e}")
            return []

    def get_docket_documents(self, docket_id: str) -> List[Dict]:
        """Get all documents for a specific docket."""
        logger.info(f"📄 Getting documents for docket {docket_id}")

        # Try API first
        api_url = f"{self.api_base}dockets/{docket_id}/recap/"
        try:
            response = self.session.get(api_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                documents = data.get('results', [])
                logger.info(f"✓ Found {len(documents)} documents via API")
                return documents
        except Exception as e:
            logger.warning(f"API failed for docket {docket_id}: {e}")

        # Fallback to HTML scraping
        docket_url = f"{self.recap_base}docket/{docket_id}/"
        try:
            response = self.session.get(docket_url, timeout=30)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                documents = []

                # Look for document links
                doc_links = soup.find_all('a', href=lambda x: x and '/recap/' in x and 'document' in x)
                for link in doc_links:
                    doc_url = urljoin(docket_url, link['href'])
                    doc_text = link.get_text(strip=True)
                    documents.append({
                        'url': doc_url,
                        'description': doc_text,
                        'source': 'html_scraping'
                    })

                logger.info(f"✓ Found {len(documents)} documents via HTML scraping")
                return documents
            else:
                logger.warning(f"HTML scraping failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error getting documents for docket {docket_id}: {e}")
            return []

    def download_pdf(self, doc_url: str, filename: str) -> bool:
        """Download a PDF document."""
        try:
            logger.info(f"📥 Downloading: {filename}")
            response = self.session.get(doc_url, timeout=60)

            if response.status_code == 200:
                pdf_path = self.pdf_dir / filename
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"✓ Downloaded: {pdf_path}")
                return True
            else:
                logger.warning(f"✗ Download failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error downloading {doc_url}: {e}")
            return False

    def process_case(self, case: Dict) -> Dict[str, Any]:
        """Process a single case and download its documents."""
        case_name = case.get('case_name', 'Unknown')
        docket_id = case.get('id')

        result = {
            'case_name': case_name,
            'docket_id': docket_id,
            'status': 'processing',
            'pdfs_downloaded': 0,
            'errors': []
        }

        if not docket_id:
            result['status'] = 'no_docket_id'
            return result

        logger.info(f"🔍 Processing: {case_name} (Docket: {docket_id})")

        # Get documents for this docket
        documents = self.get_docket_documents(str(docket_id))

        if not documents:
            result['status'] = 'no_documents'
            return result

        # Download PDFs
        downloaded_count = 0
        for i, doc in enumerate(documents):
            doc_url = doc.get('url') or doc.get('download_url')
            if not doc_url:
                continue

            # Create filename
            doc_desc = doc.get('description', f'document_{i}')
            safe_desc = "".join(c for c in doc_desc if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{docket_id}_{safe_desc}.pdf"

            if self.download_pdf(doc_url, filename):
                downloaded_count += 1
                self.stats['pdfs_downloaded'] += 1

            # Be very polite between downloads
            time.sleep(self.delay)

        result['pdfs_downloaded'] = downloaded_count
        result['status'] = 'success' if downloaded_count > 0 else 'no_pdfs'

        return result

    def run(self, batch_size: int = 5, max_cases: int = 50):
        """Run the RECAP PDF scraper."""
        logger.info("🚀 RECAP Archive PDF Scraper for §1782 Cases")
        logger.info("=" * 60)

        # Search for §1782 cases
        cases = self.search_recap_archive(max_results=max_cases)

        if not cases:
            logger.error("No cases found in RECAP Archive")
            return

        self.stats['cases_found'] = len(cases)
        logger.info(f"📊 Found {len(cases)} cases to process")
        logger.info(f"📁 PDFs will be saved to: {self.pdf_dir.absolute()}")

        # Process in batches
        for batch_num in range(0, len(cases), batch_size):
            batch_cases = cases[batch_num:batch_num + batch_size]
            batch_start = batch_num + 1
            batch_end = min(batch_num + batch_size, len(cases))

            logger.info(f"\n🔄 Processing Batch {batch_num // batch_size + 1}: Cases {batch_start}-{batch_end}")
            logger.info(f"📁 Check folder: {self.pdf_dir.absolute()}")

            for i, case in enumerate(batch_cases):
                case_num = batch_num + i + 1
                logger.info(f"\nProcessing case {case_num}/{len(cases)}")

                result = self.process_case(case)
                self.scraping_log.append(result)

                # Log result
                if result['status'] == 'success':
                    logger.info(f"✓ SUCCESS: Downloaded {result['pdfs_downloaded']} PDFs")
                else:
                    logger.info(f"✗ {result['status'].upper()}")

                # Show current PDF count
                pdf_count = len(list(self.pdf_dir.glob("*.pdf")))
                logger.info(f"📄 Total PDFs downloaded so far: {pdf_count}")

            # Long pause between batches
            if batch_num + batch_size < len(cases):
                logger.info(f"⏸️  Pausing {self.batch_delay} seconds before next batch...")
                time.sleep(self.batch_delay)

        # Final report
        self.generate_summary_report()

        logger.info(f"\n✅ RECAP scraping complete!")
        logger.info(f"📁 Final PDF count: {len(list(self.pdf_dir.glob('*.pdf')))}")
        logger.info(f"📄 Check results in: {self.pdf_dir.absolute()}")

    def generate_summary_report(self):
        """Generate a summary report."""
        report_path = Path("data/case_law/recap_scraping_report.md")

        successful_cases = [r for r in self.scraping_log if r['status'] == 'success']
        total_pdfs = sum(r['pdfs_downloaded'] for r in successful_cases)

        report_content = f"""# RECAP Archive PDF Scraping Report

## Summary
- **Total Cases Processed**: {len(self.scraping_log)}
- **Successful Downloads**: {len(successful_cases)}
- **Total PDFs Downloaded**: {total_pdfs}
- **Success Rate**: {len(successful_cases)/len(self.scraping_log)*100:.1f}%

## Detailed Results
"""

        for result in self.scraping_log:
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            report_content += f"- {status_emoji} **{result['case_name']}** (Docket: {result['docket_id']}) - {result['pdfs_downloaded']} PDFs\n"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"📊 Summary report saved to: {report_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='RECAP Archive PDF Scraper for §1782 Cases')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Number of cases to process per batch (default: 5)')
    parser.add_argument('--max-cases', type=int, default=50,
                       help='Maximum number of cases to process (default: 50)')
    parser.add_argument('--delay', type=float, default=10,
                       help='Delay between requests in seconds (default: 10)')

    args = parser.parse_args()

    scraper = RECAPPDFScraper()
    scraper.delay = args.delay

    print(f"🚀 Starting RECAP scraper with:")
    print(f"   📦 Batch size: {args.batch_size}")
    print(f"   🔢 Max cases: {args.max_cases}")
    print(f"   ⏱️  Delay between requests: {args.delay}s")
    print(f"   📁 PDFs will be saved to: {scraper.pdf_dir.absolute()}")
    print()

    scraper.run(batch_size=args.batch_size, max_cases=args.max_cases)
