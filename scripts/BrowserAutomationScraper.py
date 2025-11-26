#!/usr/bin/env python3
"""
Browser Automation Scraper for CourtListener
Uses Selenium to simulate real browser behavior and bypass AWS WAF
"""

import json
import time
import random
import logging
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrowserAutomationScraper:
    def __init__(self):
        self.base_url = "https://www.courtlistener.com"
        self.recap_search_url = f"{self.base_url}/recap/search/"

        self.pdf_dir = Path("data/recap_petitions_browser")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        self.driver = None
        self.downloaded_hashes = set()
        self.scraping_log = []

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

    def setup_driver(self):
        """Setup Chrome driver with realistic browser settings."""
        logger.info("🔧 Setting up Chrome driver...")

        chrome_options = Options()

        # Realistic browser settings
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_argument("--accept-lang=en-US,en;q=0.9")
        chrome_options.add_argument("--accept-encoding=gzip, deflate, br")
        chrome_options.add_argument("--dnt=1")
        chrome_options.add_argument("--connection=keep-alive")
        chrome_options.add_argument("--upgrade-insecure-requests=1")

        # Anti-detection settings
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Download settings
        prefs = {
            "download.default_directory": str(self.pdf_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # Create driver
        self.driver = webdriver.Chrome(options=chrome_options)

        # Execute script to remove automation indicators
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        logger.info("✓ Chrome driver setup complete")

    def human_like_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Add human-like delays."""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

    def search_recap_archive(self, query: str, max_pages: int = 3) -> List[Dict[str, Any]]:
        """Search RECAP Archive using browser automation."""
        logger.info(f"🔍 Browser search: '{query}'")
        self.stats['total_searches'] += 1

        try:
            # Navigate to search page
            self.driver.get(self.recap_search_url)
            self.human_like_delay(2, 4)

            # Find and fill search box
            search_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )

            # Clear and type query
            search_box.clear()
            for char in query:
                search_box.send_keys(char)
                time.sleep(random.uniform(0.05, 0.15))  # Human-like typing

            self.human_like_delay(1, 2)

            # Check "Only show results with PDFs" if available
            try:
                pdf_only_checkbox = self.driver.find_element(By.NAME, "available_only")
                if not pdf_only_checkbox.is_selected():
                    pdf_only_checkbox.click()
                    self.human_like_delay(0.5, 1)
            except NoSuchElementException:
                logger.info("PDF-only checkbox not found, continuing...")

            # Submit search
            search_button = self.driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
            search_button.click()

            # Wait for results
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h2.case-name"))
            )

            self.human_like_delay(2, 3)

            # Extract case links
            all_dockets = []

            for page in range(1, max_pages + 1):
                logger.info(f"  📄 Processing page {page}/{max_pages}")

                # Find case links
                case_links = self.driver.find_elements(By.CSS_SELECTOR, "h2.case-name a")

                if not case_links:
                    logger.info(f"  ✓ No cases found on page {page}")
                    break

                page_dockets = []
                for link in case_links:
                    case_name = link.text.strip()
                    case_url = link.get_attribute('href')

                    if case_url and case_url not in [d['url'] for d in all_dockets]:
                        page_dockets.append({
                            'url': case_url,
                            'case_name': case_name,
                            'query': query,
                            'page': page
                        })

                all_dockets.extend(page_dockets)
                logger.info(f"  ✓ Found {len(page_dockets)} new cases on page {page}")

                # Try to go to next page
                if page < max_pages:
                    try:
                        next_button = self.driver.find_element(By.CSS_SELECTOR, "a[rel='next']")
                        next_button.click()
                        self.human_like_delay(3, 5)

                        # Wait for new page to load
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "h2.case-name"))
                        )
                    except NoSuchElementException:
                        logger.info(f"  ✓ No more pages available")
                        break

            logger.info(f"✓ Total cases found for '{query}': {len(all_dockets)}")
            return all_dockets

        except TimeoutException:
            logger.error(f"✗ Search timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"✗ Search error for query '{query}': {e}")
            return []

    def download_docket_pdfs(self, docket_info: Dict[str, Any]) -> List[Path]:
        """Download FREE PDFs from docket page using browser automation."""
        docket_url = docket_info['url']
        case_name = docket_info['case_name']

        logger.info(f"📄 Analyzing docket: {case_name}")

        try:
            # Navigate to docket page
            self.driver.get(docket_url)
            self.human_like_delay(2, 4)

            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".docket-entry"))
            )

            downloaded_pdfs = []

            # Find all PDF download links
            pdf_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href$='.pdf']")

            for link in pdf_links:
                link_text = link.text.strip().lower()
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
                        # Click download link
                        self.driver.execute_script("arguments[0].click();", link)
                        self.human_like_delay(2, 4)

                        # Wait for download to complete
                        max_wait = 30
                        wait_time = 0
                        while wait_time < max_wait:
                            if save_path.exists():
                                break
                            time.sleep(1)
                            wait_time += 1

                        if save_path.exists():
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
                        else:
                            logger.warning(f"  ✗ Download failed: {filename}")

                    except Exception as e:
                        logger.error(f"  ✗ Error downloading {filename}: {e}")

                # Count PAID PDFs we're skipping
                elif 'buy on pacer' in link_text:
                    self.stats['paid_pdfs_skipped'] += 1

            return downloaded_pdfs

        except TimeoutException:
            logger.error(f"✗ Docket page timeout: {case_name}")
            return []
        except Exception as e:
            logger.error(f"✗ Error processing docket {case_name}: {e}")
            return []

    def run_browser_automation_search(self):
        """Run comprehensive search using browser automation."""
        logger.info("🚀 Browser Automation Scraper for CourtListener")
        logger.info("=" * 70)
        logger.info("🎯 Using Selenium to bypass AWS WAF")
        logger.info("📥 Targeting ONLY FREE PDFs")

        # Setup browser
        self.setup_driver()

        try:
            # Comprehensive search queries
            search_queries = [
                "28 usc 1782 discovery",
                '"28 U.S.C. § 1782"',
                '"28 USC 1782"',
                '"Application for Judicial Assistance"',
                '"foreign proceeding pursuant to 28"',
            ]

            all_dockets = []

            # Search all queries
            for i, query in enumerate(search_queries, 1):
                logger.info(f"\\n🔍 Search {i}/{len(search_queries)}: {query}")
                dockets = self.search_recap_archive(query, max_pages=2)
                all_dockets.extend(dockets)

                # Human-like pause between searches
                if i < len(search_queries):
                    pause_time = random.uniform(10, 20)
                    logger.info(f"⏸️  Human-like pause: {pause_time:.1f} seconds")
                    time.sleep(pause_time)

            # Remove duplicates
            unique_dockets = []
            seen_urls = set()
            for docket in all_dockets:
                if docket['url'] not in seen_urls:
                    unique_dockets.append(docket)
                    seen_urls.add(docket['url'])

            logger.info(f"\\n📊 BROWSER AUTOMATION RESULTS:")
            logger.info(f"   Total searches: {len(search_queries)}")
            logger.info(f"   Total cases found: {len(all_dockets)}")
            logger.info(f"   Unique cases: {len(unique_dockets)}")
            logger.info(f"   PDFs will be saved to: {self.pdf_dir.absolute()}")

            # Process all unique dockets
            batch_size = 3  # Small batches
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

                # Human-like pause between batches
                if i + batch_size < len(unique_dockets):
                    pause_time = random.uniform(15, 30)
                    logger.info(f"⏸️  Human-like pause between batches: {pause_time:.1f} seconds")
                    time.sleep(pause_time)

            self.generate_summary_report()
            logger.info(f"\\n✅ Browser automation complete!")
            logger.info(f"📁 Final FREE PDF count: {self.stats['total_pdfs_downloaded']}")
            logger.info(f"💰 PAID PDFs skipped: {self.stats['paid_pdfs_skipped']}")

        finally:
            # Close browser
            if self.driver:
                self.driver.quit()
                logger.info("✓ Browser closed")

    def save_progress(self):
        """Save current progress to log file."""
        log_file = Path("data/case_law/browser_automation_log.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'scraping_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'stats': self.stats,
                'results': self.scraping_log
            }, f, indent=2, ensure_ascii=False)

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report_path = Path("data/case_law/browser_automation_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Browser Automation Scraping Report\\n\\n")
            f.write(f"**Date**: {self.stats['last_run']}\\n")
            f.write(f"**Total Searches**: {self.stats['total_searches']}\\n")
            f.write(f"**Total Cases Found**: {self.stats['total_cases_found']}\\n")
            f.write(f"**FREE PDFs Downloaded**: {self.stats['total_pdfs_downloaded']}\\n")
            f.write(f"**PAID PDFs Skipped**: {self.stats['paid_pdfs_skipped']}\\n")
            f.write(f"**Duplicates Skipped**: {self.stats['duplicates_skipped']}\\n")
            f.write(f"**Total Errors**: {self.stats['total_errors']}\\n\\n")

            f.write("## Strategy\\n")
            f.write("- **Browser Automation**: Selenium Chrome driver\\n")
            f.write("- **Anti-Detection**: Realistic browser behavior\\n")
            f.write("- **Target**: Only FREE PDFs (Download PDF buttons)\\n")
            f.write("- **Skip**: PAID PDFs (Buy on PACER buttons)\\n\\n")

            f.write("## Downloaded FREE PDFs by Case\\n")
            for entry in self.scraping_log:
                f.write(f"- **Case**: {entry['case_name']}\\n")
                f.write(f"  - **Query**: {entry['query']}\\n")
                f.write(f"  - **Status**: {entry['status']}\\n")
                if entry['pdfs_downloaded']:
                    f.write(f"  - **FREE PDFs**: {', '.join([Path(p).name for p in entry['pdfs_downloaded']])}\\n")
                f.write("\\n")

if __name__ == "__main__":
    scraper = BrowserAutomationScraper()
    scraper.stats['last_run'] = time.strftime('%Y-%m-%d %H:%M:%S')
    scraper.run_browser_automation_search()
