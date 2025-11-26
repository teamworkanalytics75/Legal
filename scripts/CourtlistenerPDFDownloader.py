#!/usr/bin/env python3
"""
CourtListener PDF Downloader with Database Integration
Uses Playwright for reliable browser automation and integrates with existing case law database
"""

import json
import time
import random
import logging
import hashlib
import mysql.connector
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext
import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CourtListenerPDFDownloader:
    def __init__(self, config_path: str = "document_ingestion/courtlistener_config.json"):
        """Initialize the PDF downloader with database integration."""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.base_url = "https://www.courtlistener.com"
        self.recap_search_url = f"{self.base_url}/recap/search/"

        # Database connection
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'thetimeisN0w!',
            'database': 'lawsuit_docs',
            'charset': 'utf8mb4'
        }

        # PDF storage directory
        self.pdf_dir = Path("data/case_law/pdfs")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Browser automation settings
        self.browser = None
        self.context = None
        self.page = None

        # Rate limiting and politeness
        self.base_delay = 2.0  # Base delay between requests
        self.random_delay = 1.0  # Random variation
        self.max_retries = 3

        # Statistics tracking
        self.stats = {
            'total_cases_processed': 0,
            'total_pdfs_downloaded': 0,
            'total_errors': 0,
            'duplicates_skipped': 0,
            'free_pdfs_found': 0,
            'paid_pdfs_skipped': 0,
            'cases_with_pdfs': 0,
            'cases_without_pdfs': 0,
            'last_run': None
        }

        # Load existing PDF hashes to avoid duplicates
        self.downloaded_hashes = set()
        self.load_existing_pdfs()

        logger.info("CourtListener PDF Downloader initialized")
        logger.info(f"PDF storage: {self.pdf_dir.absolute()}")
        logger.info(f"Database: {self.db_config['database']}")

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

    def setup_browser(self):
        """Setup Playwright browser with realistic settings."""
        logger.info("🔧 Setting up Playwright browser...")

        playwright = sync_playwright().start()

        # Launch browser with realistic settings
        self.browser = playwright.chromium.launch(
            headless=False,  # Set to True for production
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )

        # Create context with realistic user agent and settings
        self.context = self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/New_York'
        )

        # Set download behavior
        self.context.set_default_timeout(30000)  # 30 seconds

        # Create page
        self.page = self.context.new_page()

        # Remove automation indicators
        self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)

        logger.info("✓ Playwright browser setup complete")

    def human_like_delay(self, min_seconds: float = None, max_seconds: float = None):
        """Add human-like delays."""
        if min_seconds is None:
            min_seconds = self.base_delay
        if max_seconds is None:
            max_seconds = self.base_delay + self.random_delay

        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

    def get_cases_from_database(self, limit: int = 100, topic: str = None) -> List[Dict[str, Any]]:
        """Get cases from the database that need PDF downloads."""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)

            # Build query
            if topic:
                query = """
                    SELECT id, case_name, citation, court, date_filed, url, source_id, topic
                    FROM case_law
                    WHERE topic = %s AND download_url IS NULL
                    ORDER BY date_filed DESC
                    LIMIT %s
                """
                cursor.execute(query, (topic, limit))
            else:
                query = """
                    SELECT id, case_name, citation, court, date_filed, url, source_id, topic
                    FROM case_law
                    WHERE download_url IS NULL
                    ORDER BY date_filed DESC
                    LIMIT %s
                """
                cursor.execute(query, (limit,))

            cases = cursor.fetchall()
            cursor.close()
            conn.close()

            logger.info(f"✓ Retrieved {len(cases)} cases from database")
            return cases

        except Exception as e:
            logger.error(f"Error retrieving cases from database: {e}")
            return []

    def search_courtlistener_for_case(self, case_info: Dict[str, Any]) -> Optional[str]:
        """Search CourtListener for a specific case and return the docket URL."""
        case_name = case_info['case_name']
        citation = case_info.get('citation', '')

        logger.info(f"🔍 Searching for: {case_name}")

        try:
            # Navigate to RECAP search page
            logger.info(f"Navigating to: {self.recap_search_url}")
            self.page.goto(self.recap_search_url)
            self.human_like_delay(3, 5)

            # Check if we're on the right page
            page_title = self.page.title()
            logger.info(f"Page title: {page_title}")

            if "404" in page_title or "not found" in page_title.lower():
                logger.error("Got 404 error from CourtListener")
                return None

            # Wait for search form to load
            try:
                self.page.wait_for_selector('input[name="q"]', timeout=10000)
            except Exception as e:
                logger.error(f"Search form not found: {e}")
                return None

            # Fill search box with case name
            search_box = self.page.locator('input[name="q"]')
            search_box.clear()
            search_box.fill(case_name)
            self.human_like_delay(1, 2)

            # Check "Only show results with PDFs" if available
            try:
                pdf_only_checkbox = self.page.locator('input[name="available_only"]')
                if pdf_only_checkbox.is_visible():
                    pdf_only_checkbox.check()
                    logger.info("✓ Checked 'Only show results with PDFs'")
            except Exception as e:
                logger.info(f"PDF-only checkbox not found: {e}")

            # Submit search
            search_button = self.page.locator('input[type="submit"]')
            search_button.click()
            logger.info("✓ Search submitted")

            # Wait for results with longer timeout
            try:
                self.page.wait_for_selector('h2.case-name, .no-results, .alert', timeout=15000)
            except Exception as e:
                logger.error(f"Results not loaded: {e}")
                return None

            # Check if we got results
            if self.page.locator('.no-results').is_visible():
                logger.info(f"✗ No results found for: {case_name}")
                return None

            if self.page.locator('.alert').is_visible():
                alert_text = self.page.locator('.alert').text_content()
                logger.warning(f"Alert on page: {alert_text}")

            self.human_like_delay(2, 3)

            # Look for matching case
            case_links = self.page.locator('h2.case-name a')
            count = case_links.count()
            logger.info(f"Found {count} case links")

            for i in range(count):
                link = case_links.nth(i)
                link_text = link.text_content()
                link_href = link.get_attribute('href')

                logger.info(f"Checking case {i+1}: {link_text}")

                # Check if this looks like our case
                if self.case_matches(case_name, citation, link_text):
                    docket_url = urljoin(self.base_url, link_href)
                    logger.info(f"✓ Found matching case: {link_text}")
                    return docket_url

            logger.info(f"✗ No matching case found for: {case_name}")
            return None

        except Exception as e:
            logger.error(f"Error searching for case {case_name}: {e}")
            return None

    def case_matches(self, case_name: str, citation: str, link_text: str) -> bool:
        """Check if a search result matches our case."""
        case_name_lower = case_name.lower()
        link_text_lower = link_text.lower()

        # Extract key words from case name (remove common words)
        common_words = {'v', 'vs', 'versus', 'in', 're', 'ex', 'parte', 'the', 'of', 'and', 'or', 'for', 'to', 'a', 'an'}
        case_words = set(case_name_lower.split())
        case_words = case_words - common_words  # Remove common words

        link_words = set(link_text_lower.split())

        # Check for significant overlap (at least 2 meaningful words)
        if len(case_words) >= 2:
            common_words = case_words.intersection(link_words)
            if len(common_words) >= 2:
                logger.info(f"✓ Match found: {len(common_words)} common words: {common_words}")
                return True

        # Check for citation match if available
        if citation and citation.strip():
            citation_lower = citation.lower()
            if citation_lower in link_text_lower:
                logger.info(f"✓ Citation match found: {citation}")
                return True

        # Check for partial name match (first few words)
        case_words_list = list(case_words)
        if len(case_words_list) >= 2:
            first_two_words = ' '.join(case_words_list[:2])
            if first_two_words in link_text_lower:
                logger.info(f"✓ Partial name match found: {first_two_words}")
                return True

        logger.info(f"✗ No match: case_words={case_words}, link_text='{link_text_lower[:100]}...'")
        return False

    def download_pdfs_from_docket(self, docket_url: str, case_info: Dict[str, Any]) -> List[Path]:
        """Download all FREE PDFs from a docket page."""
        logger.info(f"📄 Processing docket: {docket_url}")

        try:
            # Navigate to docket page
            self.page.goto(docket_url)
            self.human_like_delay(2, 4)

            # Wait for page to load
            self.page.wait_for_selector('.docket-entry', timeout=10000)

            downloaded_pdfs = []

            # Find all PDF download links
            pdf_links = self.page.locator('a[href$=".pdf"]')
            count = pdf_links.count()

            logger.info(f"Found {count} PDF links on docket page")

            for i in range(count):
                link = pdf_links.nth(i)
                link_text = link.text_content().strip().lower()
                link_href = link.get_attribute('href')

                # Only download FREE PDFs
                if 'download pdf' in link_text and 'buy' not in link_text:
                    logger.info(f"  - Found FREE PDF: {link_text}")

                    # Get filename from URL
                    filename = Path(urlparse(link_href).path).name
                    save_path = self.pdf_dir / filename

                    # Skip if already downloaded
                    if save_path.exists():
                        logger.info(f"  - PDF already exists: {filename}")
                        downloaded_pdfs.append(save_path)
                        continue

                    try:
                        # Download PDF
                        with self.page.expect_download() as download_info:
                            link.click()

                        download = download_info.value

                        # Save the file
                        download.save_as(save_path)

                        # Check for duplicates
                        with open(save_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()

                        if file_hash in self.downloaded_hashes:
                            logger.info(f"  ✓ Duplicate detected, removing: {filename}")
                            save_path.unlink()
                            self.stats['duplicates_skipped'] += 1
                        else:
                            self.downloaded_hashes.add(file_hash)
                            downloaded_pdfs.append(save_path)
                            self.stats['total_pdfs_downloaded'] += 1
                            self.stats['free_pdfs_found'] += 1
                            logger.info(f"  ✓ Downloaded FREE PDF: {filename}")

                    except Exception as e:
                        logger.error(f"  ✗ Error downloading {filename}: {e}")

                # Count PAID PDFs we're skipping
                elif 'buy on pacer' in link_text:
                    self.stats['paid_pdfs_skipped'] += 1

            return downloaded_pdfs

        except Exception as e:
            logger.error(f"Error processing docket {docket_url}: {e}")
            return []

    def update_database_with_pdfs(self, case_info: Dict[str, Any], pdf_paths: List[Path]):
        """Update database with PDF information."""
        if not pdf_paths:
            return

        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()

            # Update case with PDF information
            pdf_urls = [str(pdf_path) for pdf_path in pdf_paths]
            pdf_urls_json = json.dumps(pdf_urls)

            update_query = """
                UPDATE case_law
                SET download_url = %s,
                    pdf_count = %s,
                    pdf_downloaded_date = %s
                WHERE id = %s
            """

            cursor.execute(update_query, (
                pdf_urls_json,
                len(pdf_paths),
                datetime.now(),
                case_info['id']
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"✓ Updated database for case {case_info['case_name']} with {len(pdf_paths)} PDFs")

        except Exception as e:
            logger.error(f"Error updating database for case {case_info['case_name']}: {e}")

    def process_cases_batch(self, cases: List[Dict[str, Any]], batch_size: int = 5):
        """Process a batch of cases for PDF downloads."""
        logger.info(f"🔄 Processing batch of {len(cases)} cases")

        for i, case_info in enumerate(cases):
            case_num = i + 1
            logger.info(f"\n📋 Processing case {case_num}/{len(cases)}: {case_info['case_name']}")

            try:
                # Search for the case on CourtListener
                docket_url = self.search_courtlistener_for_case(case_info)

                if docket_url:
                    # Download PDFs from docket
                    pdf_paths = self.download_pdfs_from_docket(docket_url, case_info)

                    if pdf_paths:
                        # Update database
                        self.update_database_with_pdfs(case_info, pdf_paths)
                        self.stats['cases_with_pdfs'] += 1
                        logger.info(f"✓ Downloaded {len(pdf_paths)} PDFs for {case_info['case_name']}")
                    else:
                        self.stats['cases_without_pdfs'] += 1
                        logger.info(f"✗ No FREE PDFs found for {case_info['case_name']}")
                else:
                    self.stats['cases_without_pdfs'] += 1
                    logger.info(f"✗ Case not found on CourtListener: {case_info['case_name']}")

                self.stats['total_cases_processed'] += 1

                # Human-like delay between cases
                if case_num < len(cases):
                    self.human_like_delay(3, 6)

            except Exception as e:
                logger.error(f"Error processing case {case_info['case_name']}: {e}")
                self.stats['total_errors'] += 1

    def run_pdf_download_session(self, topic: str = None, limit: int = 50):
        """Run a complete PDF download session."""
        logger.info("🚀 CourtListener PDF Downloader")
        logger.info("=" * 70)
        logger.info("🎯 Targeting FREE PDFs from CourtListener")
        logger.info("📊 Integrating with existing case law database")
        logger.info(f"📁 PDFs will be saved to: {self.pdf_dir.absolute()}")

        # Setup browser
        self.setup_browser()

        try:
            # Get cases from database
            cases = self.get_cases_from_database(limit=limit, topic=topic)

            if not cases:
                logger.info("No cases found in database that need PDF downloads")
                return

            logger.info(f"Found {len(cases)} cases to process")

            # Process cases in batches
            batch_size = 5
            for i in range(0, len(cases), batch_size):
                batch_cases = cases[i:i + batch_size]
                batch_num = i // batch_size + 1

                logger.info(f"\n🔄 Processing Batch {batch_num}: {len(batch_cases)} cases")

                self.process_cases_batch(batch_cases, batch_size)

                # Save progress after each batch
                self.save_progress()

                # Longer pause between batches
                if i + batch_size < len(cases):
                    pause_time = random.uniform(30, 60)
                    logger.info(f"⏸️  Pause between batches: {pause_time:.1f} seconds")
                    time.sleep(pause_time)

            # Generate final report
            self.generate_summary_report()

            logger.info(f"\n✅ PDF download session complete!")
            logger.info(f"📊 Final statistics:")
            logger.info(f"   Cases processed: {self.stats['total_cases_processed']}")
            logger.info(f"   Cases with PDFs: {self.stats['cases_with_pdfs']}")
            logger.info(f"   Cases without PDFs: {self.stats['cases_without_pdfs']}")
            logger.info(f"   FREE PDFs downloaded: {self.stats['total_pdfs_downloaded']}")
            logger.info(f"   PAID PDFs skipped: {self.stats['paid_pdfs_skipped']}")
            logger.info(f"   Duplicates skipped: {self.stats['duplicates_skipped']}")
            logger.info(f"   Errors: {self.stats['total_errors']}")

        finally:
            # Close browser
            if self.browser:
                self.browser.close()
                logger.info("✓ Browser closed")

    def save_progress(self):
        """Save current progress to log file."""
        log_file = Path("data/case_law/pdf_download_log.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'download_date': datetime.now().isoformat(),
                'stats': self.stats,
                'pdf_directory': str(self.pdf_dir.absolute())
            }, f, indent=2, ensure_ascii=False)

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_path = Path("data/case_law/pdf_download_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# CourtListener PDF Download Report\n\n")
            f.write(f"**Date**: {self.stats['last_run']}\n")
            f.write(f"**Cases Processed**: {self.stats['total_cases_processed']}\n")
            f.write(f"**Cases with PDFs**: {self.stats['cases_with_pdfs']}\n")
            f.write(f"**Cases without PDFs**: {self.stats['cases_without_pdfs']}\n")
            f.write(f"**FREE PDFs Downloaded**: {self.stats['total_pdfs_downloaded']}\n")
            f.write(f"**PAID PDFs Skipped**: {self.stats['paid_pdfs_skipped']}\n")
            f.write(f"**Duplicates Skipped**: {self.stats['duplicates_skipped']}\n")
            f.write(f"**Total Errors**: {self.stats['total_errors']}\n\n")

            f.write("## Strategy\n")
            f.write("- **Browser Automation**: Playwright Chromium\n")
            f.write("- **Database Integration**: MySQL case_law table\n")
            f.write("- **Target**: Only FREE PDFs (Download PDF buttons)\n")
            f.write("- **Skip**: PAID PDFs (Buy on PACER buttons)\n")
            f.write("- **Anti-Detection**: Human-like delays and realistic browser behavior\n\n")

            f.write("## PDF Storage\n")
            f.write(f"PDFs are stored in: `{self.pdf_dir.absolute()}`\n")
            f.write("Database is updated with PDF paths and metadata.\n\n")

def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Download PDFs from CourtListener")
    parser.add_argument("--topic", help="Topic to process (e.g., 1782_discovery)")
    parser.add_argument("--limit", type=int, default=50, help="Maximum cases to process")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("CourtListener PDF Downloader")
    print("="*60 + "\n")

    downloader = CourtListenerPDFDownloader()
    downloader.stats['last_run'] = datetime.now().isoformat()

    # Set headless mode if requested
    if args.headless:
        downloader.headless = True

    downloader.run_pdf_download_session(
        topic=args.topic,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
