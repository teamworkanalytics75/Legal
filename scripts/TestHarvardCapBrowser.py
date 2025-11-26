#!/usr/bin/env python3
"""
Harvard CAP Browser Automation Test

Since the API doesn't work and the web interface requires JavaScript,
let's try browser automation to actually operate the site manually.
"""

import time
import json
import logging
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CAPBrowserAutomation:
    """Use browser automation to operate Harvard CAP manually."""

    def __init__(self):
        """Initialize the browser automation."""
        self.base_url = "https://case.law"
        self.results_dir = Path("data/case_law/1782_discovery/cap_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup Chrome options
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")  # Run in background
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--window-size=1920,1080")

        self.driver = None
        self.wait = None

    def setup_browser(self):
        """Setup the browser driver."""
        try:
            logger.info("Setting up Chrome browser...")
            self.driver = webdriver.Chrome(options=self.chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
            logger.info("✓ Browser setup complete")
            return True
        except Exception as e:
            logger.error(f"Failed to setup browser: {e}")
            return False

    def test_main_page(self):
        """Test if we can access the main page."""
        logger.info("Testing main page access...")

        try:
            self.driver.get(self.base_url)
            time.sleep(3)  # Wait for page to load

            # Check if page loaded
            title = self.driver.title
            logger.info(f"Page title: {title}")

            # Look for search elements
            search_elements = self.driver.find_elements(By.CSS_SELECTOR, "input[type='search'], input[name*='search'], input[placeholder*='search']")
            logger.info(f"Found {len(search_elements)} search elements")

            # Look for any input fields
            all_inputs = self.driver.find_elements(By.TAG_NAME, "input")
            logger.info(f"Found {len(all_inputs)} total input elements")

            # Save page source for analysis
            with open(self.results_dir / "main_page_source.html", 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)

            return len(search_elements) > 0

        except Exception as e:
            logger.error(f"Error accessing main page: {e}")
            return False

    def test_search_functionality(self):
        """Test the search functionality."""
        logger.info("Testing search functionality...")

        search_queries = [
            "1782",
            "28 U.S.C. 1782",
            "section 1782",
            "foreign tribunal",
            "Intel Corp",
            "ZF Automotive"
        ]

        search_results = {}

        for query in search_queries:
            logger.info(f"Testing search: '{query}'")

            try:
                # Try to find search input
                search_input = None

                # Try different selectors for search input
                search_selectors = [
                    "input[type='search']",
                    "input[name*='search']",
                    "input[placeholder*='search']",
                    "input[placeholder*='Search']",
                    "input[placeholder*='case']",
                    "input[placeholder*='Case']",
                    ".search-input",
                    "#search",
                    "[data-testid*='search']"
                ]

                for selector in search_selectors:
                    try:
                        search_input = self.driver.find_element(By.CSS_SELECTOR, selector)
                        logger.info(f"Found search input with selector: {selector}")
                        break
                    except NoSuchElementException:
                        continue

                if not search_input:
                    logger.warning(f"No search input found for query: {query}")
                    search_results[query] = {"error": "No search input found"}
                    continue

                # Clear and enter search query
                search_input.clear()
                search_input.send_keys(query)

                # Look for search button
                search_button = None
                button_selectors = [
                    "button[type='submit']",
                    "input[type='submit']",
                    "button:contains('Search')",
                    ".search-button",
                    "#search-button",
                    "[data-testid*='search']"
                ]

                for selector in button_selectors:
                    try:
                        search_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                        logger.info(f"Found search button with selector: {selector}")
                        break
                    except NoSuchElementException:
                        continue

                if search_button:
                    search_button.click()
                else:
                    # Try pressing Enter
                    search_input.send_keys("\n")

                # Wait for results
                time.sleep(3)

                # Check for results
                results = self.driver.find_elements(By.CSS_SELECTOR, ".result, .case, .opinion, .decision")
                logger.info(f"Found {len(results)} result elements")

                # Save page source
                with open(self.results_dir / f"search_{query.replace(' ', '_')}.html", 'w', encoding='utf-8') as f:
                    f.write(self.driver.page_source)

                search_results[query] = {
                    "success": True,
                    "result_count": len(results),
                    "page_source_length": len(self.driver.page_source)
                }

            except Exception as e:
                logger.error(f"Error testing search '{query}': {e}")
                search_results[query] = {"error": str(e)}

            time.sleep(2)  # Be respectful

        # Save search results
        with open(self.results_dir / "search_results.json", 'w') as f:
            json.dump(search_results, f, indent=2)

        return search_results

    def test_case_access(self):
        """Test accessing specific cases."""
        logger.info("Testing case access...")

        # Try to find any case links on the page
        case_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='case'], a[href*='opinion'], a[href*='decision']")
        logger.info(f"Found {len(case_links)} potential case links")

        case_results = {}

        for i, link in enumerate(case_links[:5]):  # Test first 5 links
            try:
                href = link.get_attribute('href')
                text = link.text.strip()

                logger.info(f"Testing case link {i+1}: {text[:50]}...")

                # Click the link
                link.click()
                time.sleep(3)

                # Check if we're on a case page
                case_indicators = self.driver.find_elements(By.CSS_SELECTOR, ".case-content, .opinion-text, .decision-text")

                case_results[f"case_{i+1}"] = {
                    "href": href,
                    "text": text,
                    "success": True,
                    "case_indicators": len(case_indicators),
                    "page_title": self.driver.title
                }

                # Go back
                self.driver.back()
                time.sleep(2)

            except Exception as e:
                logger.error(f"Error testing case link {i+1}: {e}")
                case_results[f"case_{i+1}"] = {"error": str(e)}

        # Save case results
        with open(self.results_dir / "case_access_results.json", 'w') as f:
            json.dump(case_results, f, indent=2)

        return case_results

    def run(self):
        """Run the browser automation tests."""
        logger.info("Starting Harvard CAP Browser Automation Tests")
        logger.info("The irony continues...")

        try:
            # Setup browser
            if not self.setup_browser():
                logger.error("Failed to setup browser. Cannot continue.")
                return

            # Test main page
            if not self.test_main_page():
                logger.error("Cannot access main page. CAP may be down.")
                return

            # Test search functionality
            search_results = self.test_search_functionality()

            # Test case access
            case_results = self.test_case_access()

            # Analyze results
            self.analyze_results(search_results, case_results)

            logger.info("=" * 60)
            logger.info("CAP BROWSER AUTOMATION TESTS COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {self.results_dir}")

        except Exception as e:
            logger.error(f"Error during browser automation: {e}", exc_info=True)
        finally:
            if self.driver:
                self.driver.quit()

    def analyze_results(self, search_results, case_results):
        """Analyze the browser automation results."""
        logger.info("\n" + "=" * 60)
        logger.info("CAP Browser Automation Analysis")
        logger.info("=" * 60)

        # Analyze search results
        successful_searches = 0
        total_results = 0

        for query, result in search_results.items():
            if result.get("success", False):
                successful_searches += 1
                result_count = result.get("result_count", 0)
                total_results += result_count
                logger.info(f"✓ '{query}': {result_count} results")
            else:
                logger.info(f"✗ '{query}': {result.get('error', 'Unknown error')}")

        # Analyze case access
        successful_cases = 0
        for case_id, result in case_results.items():
            if result.get("success", False):
                successful_cases += 1
                logger.info(f"✓ {case_id}: {result.get('case_indicators', 0)} case indicators")
            else:
                logger.info(f"✗ {case_id}: {result.get('error', 'Unknown error')}")

        logger.info(f"\nSUMMARY:")
        logger.info(f"Successful searches: {successful_searches}")
        logger.info(f"Total results found: {total_results}")
        logger.info(f"Successful case access: {successful_cases}")

        if successful_searches > 0 or successful_cases > 0:
            logger.info("✓ CAP browser automation appears to work")
            logger.info("✓ We can potentially automate §1782 case extraction")
        else:
            logger.info("✗ CAP browser automation may not be viable")
            logger.info("✗ May need to try different approach or give up on CAP")


def main():
    """Main entry point."""
    print("Harvard CAP Browser Automation Test")
    print("=" * 60)
    print("Testing Harvard's website with browser automation")
    print("(The irony continues...)")
    print("=" * 60)

    # Check if selenium is available
    try:
        import selenium
        logger.info(f"Selenium version: {selenium.__version__}")
    except ImportError:
        logger.error("Selenium not installed. Install with: pip install selenium")
        return

    automation = CAPBrowserAutomation()
    automation.run()


if __name__ == "__main__":
    main()
