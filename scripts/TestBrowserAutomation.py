#!/usr/bin/env python3
"""
Test Browser Automation Scraper
Verify Selenium approach works before full download
"""

import time
import random
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

class TestBrowserScraper:
    def __init__(self):
        self.base_url = "https://www.courtlistener.com"
        self.recap_search_url = f"{self.base_url}/recap/search/"
        self.driver = None

    def setup_driver(self):
        """Setup Chrome driver for testing."""
        logger.info("🔧 Setting up Chrome driver for testing...")

        chrome_options = Options()

        # Realistic browser settings
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        chrome_options.add_argument("--accept-lang=en-US,en;q=0.9")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Create driver
        self.driver = webdriver.Chrome(options=chrome_options)

        # Execute script to remove automation indicators
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        logger.info("✓ Chrome driver setup complete")

    def human_like_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Add human-like delays."""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

    def test_search_interface(self):
        """Test accessing the search interface."""
        logger.info("🧪 Testing search interface access")

        try:
            # Navigate to search page
            logger.info(f"Navigating to: {self.recap_search_url}")
            self.driver.get(self.recap_search_url)
            self.human_like_delay(2, 4)

            # Check if page loaded successfully
            page_title = self.driver.title
            logger.info(f"✓ Page loaded: {page_title}")

            # Look for search elements
            try:
                search_box = self.driver.find_element(By.NAME, "q")
                logger.info("✓ Search box found")

                # Test typing
                search_box.clear()
                test_query = "28 usc 1782 discovery"
                for char in test_query:
                    search_box.send_keys(char)
                    time.sleep(random.uniform(0.05, 0.15))

                logger.info(f"✓ Typed query: {test_query}")

                # Look for PDF-only checkbox
                try:
                    pdf_checkbox = self.driver.find_element(By.NAME, "available_only")
                    logger.info("✓ PDF-only checkbox found")
                    if not pdf_checkbox.is_selected():
                        pdf_checkbox.click()
                        logger.info("✓ PDF-only checkbox checked")
                except NoSuchElementException:
                    logger.info("⚠️  PDF-only checkbox not found")

                # Look for submit button
                submit_button = self.driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
                logger.info("✓ Submit button found")

                return True

            except NoSuchElementException as e:
                logger.error(f"✗ Search elements not found: {e}")
                return False

        except Exception as e:
            logger.error(f"✗ Error accessing search interface: {e}")
            return False

    def test_search_execution(self):
        """Test executing a search."""
        logger.info("🧪 Testing search execution")

        try:
            # Find search box and submit
            search_box = self.driver.find_element(By.NAME, "q")
            search_box.clear()

            test_query = "28 usc 1782 discovery"
            search_box.send_keys(test_query)
            self.human_like_delay(1, 2)

            # Submit search
            submit_button = self.driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
            submit_button.click()

            # Wait for results
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h2.case-name"))
            )

            logger.info("✓ Search executed successfully")

            # Count results
            case_links = self.driver.find_elements(By.CSS_SELECTOR, "h2.case-name a")
            logger.info(f"✓ Found {len(case_links)} case links")

            # Show first few cases
            for i, link in enumerate(case_links[:3], 1):
                case_name = link.text.strip()
                case_url = link.get_attribute('href')
                logger.info(f"  {i}. {case_name}")
                logger.info(f"     URL: {case_url}")

            return case_links[:3] if case_links else []

        except TimeoutException:
            logger.error("✗ Search timeout - no results found")
            return []
        except Exception as e:
            logger.error(f"✗ Search execution error: {e}")
            return []

    def test_docket_access(self, case_links):
        """Test accessing a docket page."""
        if not case_links:
            logger.warning("No case links provided for docket test")
            return False

        logger.info("🧪 Testing docket page access")

        try:
            # Click on first case
            first_case = case_links[0]
            case_name = first_case.text.strip()
            logger.info(f"Testing docket access: {case_name}")

            # Click the link
            first_case.click()
            self.human_like_delay(2, 4)

            # Wait for docket page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".docket-entry"))
            )

            logger.info("✓ Docket page loaded successfully")

            # Look for PDF links
            pdf_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href$='.pdf']")
            logger.info(f"✓ Found {len(pdf_links)} PDF links")

            # Analyze PDF types
            free_pdfs = []
            paid_pdfs = []

            for link in pdf_links:
                link_text = link.text.strip().lower()
                link_href = link.get_attribute('href')

                if 'download pdf' in link_text and 'buy' not in link_text:
                    free_pdfs.append({
                        'text': link_text,
                        'href': link_href
                    })
                elif 'buy on pacer' in link_text:
                    paid_pdfs.append({
                        'text': link_text,
                        'href': link_href
                    })

            logger.info(f"✓ FREE PDFs found: {len(free_pdfs)}")
            for pdf in free_pdfs:
                logger.info(f"  - {pdf['text']}")

            logger.info(f"✓ PAID PDFs found: {len(paid_pdfs)}")
            for pdf in paid_pdfs:
                logger.info(f"  - {pdf['text']}")

            return len(free_pdfs) > 0

        except TimeoutException:
            logger.error("✗ Docket page timeout")
            return False
        except Exception as e:
            logger.error(f"✗ Docket access error: {e}")
            return False

def main():
    logger.info("🧪 TESTING BROWSER AUTOMATION SCRAPER")
    logger.info("=" * 60)
    logger.info("🎯 Testing Selenium approach to bypass AWS WAF")
    logger.info("⏸️  Human-like delays and realistic browser behavior")

    scraper = TestBrowserScraper()

    try:
        # Setup browser
        scraper.setup_driver()

        # Test 1: Search interface access
        logger.info("\\n🧪 Test 1: Search Interface Access")
        interface_ok = scraper.test_search_interface()

        if not interface_ok:
            logger.error("✗ Search interface test failed")
            return

        # Test 2: Search execution
        logger.info("\\n🧪 Test 2: Search Execution")
        case_links = scraper.test_search_execution()

        if not case_links:
            logger.error("✗ Search execution test failed")
            return

        # Test 3: Docket access
        logger.info("\\n🧪 Test 3: Docket Page Access")
        docket_ok = scraper.test_docket_access(case_links)

        if docket_ok:
            logger.info("\\n✅ SUCCESS: Browser automation working!")
            logger.info("   Ready to run full scraper with Selenium")
        else:
            logger.warning("\\n⚠️  Docket access test failed")

    except Exception as e:
        logger.error(f"✗ Test error: {e}")

    finally:
        # Close browser
        if scraper.driver:
            scraper.driver.quit()
            logger.info("✓ Browser closed")

    logger.info("\\n✅ Browser automation test complete!")

if __name__ == "__main__":
    main()
