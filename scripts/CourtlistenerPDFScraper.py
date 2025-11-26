#!/usr/bin/env python3
"""
CourtListener PDF Scraper for §1782 Cases

Downloads PDFs from CourtListener docket pages for cases that returned 403 from RECAP API.
Uses both API fallback and HTML scraping to maximize PDF retrieval.

Usage:
    python courtlistener_pdf_scraper.py
"""

import json
import os
import time
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CourtListenerPDFScraper:
    def __init__(self):
        self.api_token = "1d6baa6881aa5b32acecff70866e5901a8c5bc18"
        self.api_base = "https://www.courtlistener.com/api/rest/v3/"
        self.web_base = "https://www.courtlistener.com"

        # Polite headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MalcolmGrayson §1782 research (malcolmgrayson00@gmail.com) - polite PDF scraper',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        # Output directories
        self.pdf_dir = Path("data/petitions_raw")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.delay = 1.5  # seconds between requests

        # Results tracking
        self.scraping_log = []
        self.stats = {
            'total_cases': 0,
            'cases_with_docket_ids': 0,
            'pdfs_found_via_api': 0,
            'pdfs_found_via_html': 0,
            'pdfs_downloaded': 0,
            'pdfs_failed': 0,
            'cases_updated': 0
        }

    def load_cases_with_docket_ids(self) -> List[Dict[str, Any]]:
        """Load cases that have docket IDs (most likely to have PDFs)."""
        logger.info("Loading cases with docket IDs...")

        cases = []
        case_dir = Path("data/case_law/1782_discovery")

        if not case_dir.exists():
            logger.error(f"Case directory not found: {case_dir}")
            return cases

        for json_file in case_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                    # Only process cases with docket IDs
                    if case_data.get('docket_id'):
                        cases.append(case_data)

            except Exception as e:
                logger.warning(f"Error loading {json_file.name}: {e}")

        logger.info(f"✓ Found {len(cases)} cases with docket IDs")
        return cases

    def get_docket_url(self, case: Dict[str, Any]) -> Optional[str]:
        """Generate CourtListener docket URL from case data."""
        docket_id = case.get('docket_id')
        case_name = case.get('case_name', '')

        if not docket_id:
            return None

        # Try to construct URL from case name
        if case_name and case_name != 'Unknown':
            # Clean case name for URL
            clean_name = case_name.lower()
            clean_name = clean_name.replace(' ', '-')
            clean_name = clean_name.replace('&', 'and')
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c in '-')
            clean_name = clean_name.strip('-')

            if clean_name:
                return f"{self.web_base}/docket/{docket_id}/{clean_name}/"

        # Fallback to generic URL
        return f"{self.web_base}/docket/{docket_id}/"

    def fetch_pdfs_via_api(self, docket_id: str) -> List[str]:
        """Try to fetch PDF URLs via RECAP API."""
        try:
            api_url = f"{self.api_base}dockets/{docket_id}/"
            headers = {'Authorization': f'Token {self.api_token}'}

            response = self.session.get(api_url, headers=headers, timeout=10)

            if response.status_code == 200:
                docket_data = response.json()
                pdf_urls = []

                # Get RECAP documents
                for recap_url in docket_data.get('recap_documents', []):
                    try:
                        doc_response = self.session.get(recap_url, headers=headers, timeout=10)
                        if doc_response.status_code == 200:
                            doc_data = doc_response.json()
                            pdf_url = doc_data.get('pdf_url')
                            if pdf_url:
                                pdf_urls.append(pdf_url)
                    except Exception as e:
                        logger.debug(f"Error fetching RECAP doc {recap_url}: {e}")

                logger.info(f"✓ API found {len(pdf_urls)} PDFs for docket {docket_id}")
                return pdf_urls
            else:
                logger.debug(f"API returned {response.status_code} for docket {docket_id}")
                return []

        except Exception as e:
            logger.debug(f"API fetch failed for docket {docket_id}: {e}")
            return []

    def scrape_pdfs_via_html(self, docket_url: str) -> List[str]:
        """Scrape PDF URLs from docket HTML page."""
        try:
            response = self.session.get(docket_url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            pdf_urls = []

            # Look for PDF download links
            pdf_links = soup.find_all('a', string=lambda text: text and 'PDF' in text.upper())

            for link in pdf_links:
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    full_url = urljoin(docket_url, href)
                    pdf_urls.append(full_url)

            # Also look for direct PDF links in docket entries
            docket_entries = soup.select('.docket-entry, .entry')
            for entry in docket_entries:
                links = entry.find_all('a', href=True)
                for link in links:
                    href = link.get('href', '')
                    if '.pdf' in href.lower() or 'pdf' in link.get_text().lower():
                        full_url = urljoin(docket_url, href)
                        pdf_urls.append(full_url)

            # Remove duplicates
            pdf_urls = list(set(pdf_urls))

            logger.info(f"✓ HTML scraping found {len(pdf_urls)} PDFs from {docket_url}")
            return pdf_urls

        except Exception as e:
            logger.warning(f"HTML scraping failed for {docket_url}: {e}")
            return []

    def download_pdf(self, pdf_url: str, case_name: str, docket_id: str) -> Optional[str]:
        """Download a PDF and return the local file path."""
        try:
            # Generate filename
            doc_id = pdf_url.rstrip('/').split('/')[-1]
            if not doc_id or doc_id == 'pdf':
                doc_id = f"doc_{int(time.time())}"

            filename = f"{docket_id}_{doc_id}.pdf"
            filepath = self.pdf_dir / filename

            # Skip if already downloaded
            if filepath.exists():
                logger.info(f"✓ PDF already exists: {filename}")
                return str(filepath)

            # Download PDF
            response = self.session.get(pdf_url, timeout=30)

            if response.status_code == 200:
                filepath.write_bytes(response.content)
                logger.info(f"✓ Downloaded PDF: {filename} ({len(response.content)} bytes)")
                return str(filepath)
            elif response.status_code == 403:
                logger.warning(f"✗ 403 Forbidden: {pdf_url}")
                return None
            else:
                logger.warning(f"✗ Download failed ({response.status_code}): {pdf_url}")
                return None

        except Exception as e:
            logger.error(f"✗ Error downloading PDF {pdf_url}: {e}")
            return None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfminer."""
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(pdf_path)
            return text.strip()
        except ImportError:
            logger.warning("pdfminer not available - install with: pip install pdfminer.six")
            return ""
        except Exception as e:
            logger.warning(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def update_case_json(self, case: Dict[str, Any], pdf_paths: List[str], docket_url: str) -> bool:
        """Update case JSON file with PDF paths and extracted text."""
        try:
            # Find the original JSON file
            case_dir = Path("data/case_law/1782_discovery")
            case_name = case.get('case_name', '')

            # Try to find the JSON file
            json_file = None
            for f in case_dir.glob("*.json"):
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        if data.get('docket_id') == case.get('docket_id'):
                            json_file = f
                            break
                except:
                    continue

            if not json_file:
                logger.warning(f"Could not find JSON file for case {case_name}")
                return False

            # Update the case data
            case['pdf_paths'] = pdf_paths
            case['docket_url'] = docket_url
            case['pdf_scraping_date'] = time.strftime('%Y-%m-%d %H:%M:%S')

            # Extract text from PDFs
            extracted_texts = []
            for pdf_path in pdf_paths:
                if os.path.exists(pdf_path):
                    text = self.extract_text_from_pdf(pdf_path)
                    if text:
                        extracted_texts.append(text)

            if extracted_texts:
                case['pdf_extracted_text'] = '\n\n--- PDF SEPARATOR ---\n\n'.join(extracted_texts)
                case['pdf_text_length'] = len(case['pdf_extracted_text'])

            # Save updated JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(case, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Updated case JSON: {json_file.name}")
            return True

        except Exception as e:
            logger.error(f"Error updating case JSON: {e}")
            return False

    def process_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single case to find and download PDFs."""
        case_name = case.get('case_name', 'Unknown')
        docket_id = case.get('docket_id')

        result = {
            'case_name': case_name,
            'docket_id': docket_id,
            'docket_url': None,
            'pdf_urls_found': [],
            'pdfs_downloaded': [],
            'pdfs_failed': [],
            'status': 'failed'
        }

        if not docket_id:
            result['status'] = 'no_docket_id'
            return result

        # Get docket URL
        docket_url = self.get_docket_url(case)
        if not docket_url:
            result['status'] = 'no_docket_url'
            return result

        result['docket_url'] = docket_url

        # Try API first
        api_pdf_urls = self.fetch_pdfs_via_api(docket_id)
        result['pdf_urls_found'].extend(api_pdf_urls)
        self.stats['pdfs_found_via_api'] += len(api_pdf_urls)

        # Fallback to HTML scraping
        html_pdf_urls = self.scrape_pdfs_via_html(docket_url)
        result['pdf_urls_found'].extend(html_pdf_urls)
        self.stats['pdfs_found_via_html'] += len(html_pdf_urls)

        # Remove duplicates
        result['pdf_urls_found'] = list(set(result['pdf_urls_found']))

        if not result['pdf_urls_found']:
            result['status'] = 'no_pdfs_found'
            return result

        # Download PDFs
        downloaded_paths = []
        for pdf_url in result['pdf_urls_found']:
            pdf_path = self.download_pdf(pdf_url, case_name, docket_id)

            if pdf_path:
                downloaded_paths.append(pdf_path)
                result['pdfs_downloaded'].append(pdf_path)
                self.stats['pdfs_downloaded'] += 1
            else:
                result['pdfs_failed'].append(pdf_url)
                self.stats['pdfs_failed'] += 1

            # Rate limiting
            time.sleep(self.delay)

        # Update case JSON if we downloaded any PDFs
        if downloaded_paths:
            if self.update_case_json(case, downloaded_paths, docket_url):
                self.stats['cases_updated'] += 1
                result['status'] = 'success'
            else:
                result['status'] = 'download_success_json_failed'
        else:
            result['status'] = 'no_pdfs_downloaded'

        return result

    def run(self, batch_size: int = 10, start_from: int = 0):
        """Run the PDF scraping process in batches."""
        logger.info("🔍 CourtListener PDF Scraper for §1782 Cases")
        logger.info("=" * 70)

        # Load cases with docket IDs
        cases = self.load_cases_with_docket_ids()
        if not cases:
            logger.error("No cases with docket IDs found")
            return

        self.stats['total_cases'] = len(cases)
        self.stats['cases_with_docket_ids'] = len(cases)

        # Process in batches
        total_cases = len(cases)
        cases_to_process = cases[start_from:]

        logger.info(f"📁 PDFs will be saved to: {self.pdf_dir.absolute()}")
        logger.info(f"📊 Processing {len(cases_to_process)} cases (starting from case {start_from + 1})")
        logger.info(f"🔄 Batch size: {batch_size} cases per batch")

        # Process in batches
        for batch_num in range(0, len(cases_to_process), batch_size):
            batch_cases = cases_to_process[batch_num:batch_num + batch_size]
            batch_start = start_from + batch_num + 1
            batch_end = min(start_from + batch_num + batch_size, total_cases)

            logger.info(f"\n🔄 Processing Batch {batch_num // batch_size + 1}: Cases {batch_start}-{batch_end}")
            logger.info(f"📁 Check folder: {self.pdf_dir.absolute()}")

            for i, case in enumerate(batch_cases):
                case_name = case.get('case_name', 'Unknown')
                case_num = start_from + batch_num + i + 1
                logger.info(f"Processing case {case_num}/{total_cases}: {case_name}")

                result = self.process_case(case)
                self.scraping_log.append(result)

                # Log result
                if result['status'] == 'success':
                    logger.info(f"✓ SUCCESS: Downloaded {len(result['pdfs_downloaded'])} PDFs")
                else:
                    logger.info(f"✗ {result['status'].upper()}: {result.get('docket_url', 'No URL')}")

                # Show current PDF count
                pdf_count = len(list(self.pdf_dir.glob("*.pdf")))
                logger.info(f"📄 Total PDFs downloaded so far: {pdf_count}")

            # Save progress after each batch
            self.save_progress()

            # Brief pause between batches
            if batch_num + batch_size < len(cases_to_process):
                logger.info(f"⏸️  Pausing 5 seconds before next batch...")
                time.sleep(5)

        # Final save and report
        self.save_progress()
        self.generate_summary_report()

        logger.info(f"\n✅ PDF scraping complete!")
        logger.info(f"📁 Final PDF count: {len(list(self.pdf_dir.glob('*.pdf')))}")
        logger.info(f"📄 Check results in: {self.pdf_dir.absolute()}")

    def save_progress(self):
        """Save current progress to log file."""
        log_file = Path("data/case_law/pdf_scraping_log.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'scraping_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stats': self.stats,
                'results': self.scraping_log
            }, f, indent=2, ensure_ascii=False)

    def generate_summary_report(self):
        """Generate a summary report of the scraping results."""
        report_file = Path("data/case_law/pdf_scraping_report.md")

        successful_cases = [r for r in self.scraping_log if r['status'] == 'success']
        failed_cases = [r for r in self.scraping_log if r['status'] != 'success']

        report_content = f"""# 📄 CourtListener PDF Scraping Report

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary Statistics

- **Total Cases Processed**: {self.stats['total_cases']}
- **Cases with Docket IDs**: {self.stats['cases_with_docket_ids']}
- **PDFs Found via API**: {self.stats['pdfs_found_via_api']}
- **PDFs Found via HTML**: {self.stats['pdfs_found_via_html']}
- **PDFs Successfully Downloaded**: {self.stats['pdfs_downloaded']}
- **PDFs Failed to Download**: {self.stats['pdfs_failed']}
- **Case JSON Files Updated**: {self.stats['cases_updated']}

## ✅ Successful Cases ({len(successful_cases)})

"""

        for result in successful_cases:
            report_content += f"- **{result['case_name']}** (Docket: {result['docket_id']})\n"
            report_content += f"  - PDFs Downloaded: {len(result['pdfs_downloaded'])}\n"
            report_content += f"  - Docket URL: {result['docket_url']}\n\n"

        report_content += f"""## ❌ Failed Cases ({len(failed_cases)})

"""

        for result in failed_cases:
            report_content += f"- **{result['case_name']}** (Docket: {result['docket_id']})\n"
            report_content += f"  - Status: {result['status']}\n"
            report_content += f"  - Docket URL: {result.get('docket_url', 'N/A')}\n\n"

        report_content += f"""## 📁 Output Files

- **PDFs Directory**: `{self.pdf_dir}`
- **Scraping Log**: `data/case_law/pdf_scraping_log.json`
- **Case JSON Files**: Updated with PDF paths and extracted text

## 🔧 Technical Details

- **Rate Limiting**: {self.delay} seconds between requests
- **User Agent**: MalcolmGrayson §1782 research (malcolmgrayson00@gmail.com)
- **Method**: API first, then HTML scraping fallback
- **Text Extraction**: pdfminer.six (if available)

## 📝 Next Steps

1. Review failed cases manually
2. Install pdfminer.six for text extraction: `pip install pdfminer.six`
3. Run text analysis on extracted PDF content
4. Update predictive models with new petition data
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"Summary report saved to: {report_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CourtListener PDF Scraper for §1782 Cases')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of cases to process per batch (default: 10)')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Case number to start from (0-based, default: 0)')
    parser.add_argument('--delay', type=float, default=1.5,
                       help='Delay between requests in seconds (default: 1.5)')

    args = parser.parse_args()

    scraper = CourtListenerPDFScraper()
    scraper.delay = args.delay  # Update delay if specified

    print(f"🚀 Starting PDF scraper with:")
    print(f"   📦 Batch size: {args.batch_size}")
    print(f"   🔢 Starting from case: {args.start_from + 1}")
    print(f"   ⏱️  Delay between requests: {args.delay}s")
    print(f"   📁 PDFs will be saved to: {scraper.pdf_dir.absolute()}")
    print()

    scraper.run(batch_size=args.batch_size, start_from=args.start_from)
