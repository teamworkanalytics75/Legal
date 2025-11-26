#!/usr/bin/env python3
"""
Selenium-based RECAP Archive scraper using the refined search
Based on the user's successful search: discovery judicial assistance
"""

import time
import os
import logging
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RECAPSeleniumScraper:
    def __init__(self):
        self.driver = None
        self.pdf_dir = Path("data/recap_petitions")
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.stats = {
            'cases_processed': 0,
            'pdfs_downloaded': 0,
            'errors': 0
        }

    def setup_driver(self):
        """Set up Chrome driver with human-like settings."""
        chrome_options = Options()

        # Make it look like a real user
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
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

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            logger.info("✓ Chrome driver initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {e}")
            logger.error("Make sure ChromeDriver is installed and in PATH")
            return False

    def search_recap_archive(self):
        """Navigate to the refined search URL."""
        search_url = "https://www.courtlistener.com/?q=&type=r&order_by=dateFiled%20desc&available_only=on&description=discovery%20judicial%20assistance"

        logger.info(f"🔍 Navigating to RECAP search: {search_url}")

        try:
            self.driver.get(search_url)

            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "result"))
            )

            logger.info("✓ Search page loaded successfully")
            return True

        except TimeoutException:
            logger.error("✗ Search page failed to load")
            return False
        except Exception as e:
            logger.error(f"✗ Error loading search page: {e}")
            return False

    def get_search_results(self):
        """Get all search result links."""
        try:
            # Wait for results to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "result"))
            )

            # Find all result links
            result_links = self.driver.find_elements(By.CSS_SELECTOR, ".result a[href*='/docket/']")

            logger.info(f"✓ Found {len(result_links)} search results")
            return result_links

        except TimeoutException:
            logger.error("✗ No search results found")
            return []
        except Exception as e:
            logger.error(f"✗ Error getting search results: {e}")
            return []

    def process_case(self, case_link):
        """Process a single case and download its PDFs."""
        case_name = case_link.text.strip()
        case_url = case_link.get_attribute('href')

        logger.info(f"🔍 Processing: {case_name}")

        try:
            # Navigate to case page
            self.driver.get(case_url)

            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "docket"))
            )

            # Find PDF download links
            pdf_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='.pdf'], a[href*='/recap/']")

            downloaded_count = 0
            for pdf_link in pdf_links:
                try:
                    # Click PDF link
                    pdf_link.click()
                    time.sleep(2)  # Wait for download
                    downloaded_count += 1
                    self.stats['pdfs_downloaded'] += 1

                except Exception as e:
                    logger.warning(f"Failed to download PDF: {e}")

            logger.info(f"✓ Downloaded {downloaded_count} PDFs from {case_name}")
            self.stats['cases_processed'] += 1

            return downloaded_count

        except Exception as e:
            logger.error(f"✗ Error processing case {case_name}: {e}")
            self.stats['errors'] += 1
            return 0

    def run(self, max_cases=10):
        """Run the scraper."""
        logger.info("🚀 Starting RECAP Selenium Scraper")
        logger.info("=" * 50)

        # Setup driver
        if not self.setup_driver():
            return

        try:
            # Navigate to search
            if not self.search_recap_archive():
                return

            # Get search results
            result_links = self.get_search_results()
            if not result_links:
                logger.error("No search results found")
                return

            # Process cases
            cases_to_process = result_links[:max_cases]
            logger.info(f"📊 Processing {len(cases_to_process)} cases")
            logger.info(f"📁 PDFs will be saved to: {self.pdf_dir.absolute()}")

            for i, case_link in enumerate(cases_to_process):
                logger.info(f"\n--- Case {i+1}/{len(cases_to_process)} ---")

                downloaded = self.process_case(case_link)

                # Show progress
                total_pdfs = len(list(self.pdf_dir.glob("*.pdf")))
                logger.info(f"📄 Total PDFs downloaded so far: {total_pdfs}")

                # Be polite between cases
                time.sleep(3)

            # Final report
            logger.info(f"\n✅ Scraping complete!")
            logger.info(f"📊 Cases processed: {self.stats['cases_processed']}")
            logger.info(f"📄 PDFs downloaded: {self.stats['pdfs_downloaded']}")
            logger.info(f"❌ Errors: {self.stats['errors']}")
            logger.info(f"📁 Check results in: {self.pdf_dir.absolute()}")

        finally:
            if self.driver:
                self.driver.quit()
                logger.info("✓ Browser closed")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='RECAP Selenium Scraper')
    parser.add_argument('--max-cases', type=int, default=10,
                       help='Maximum number of cases to process (default: 10)')

    args = parser.parse_args()

    scraper = RECAPSeleniumScraper()
    scraper.run(max_cases=args.max_cases)
