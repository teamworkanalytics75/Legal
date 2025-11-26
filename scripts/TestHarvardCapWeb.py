#!/usr/bin/env python3
"""
Harvard CAP Web Scraping Test

Since the API doesn't work, let's try to scrape the web interface
to see if we can find §1782 cases on case.law.
"""

import requests
import json
import time
import logging
from pathlib import Path
from urllib.parse import urlencode, quote
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CAPWebScraper:
    """Test Harvard CAP web interface for §1782 cases."""

    def __init__(self):
        """Initialize the web scraper."""
        self.base_url = "https://case.law"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Test results storage
        self.results_dir = Path("data/case_law/1782_discovery/cap_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def test_main_page(self):
        """Test if we can access the main case.law page."""
        logger.info("Testing main case.law page...")

        try:
            response = self.session.get(self.base_url, timeout=10)

            if response.status_code == 200:
                logger.info(f"✓ Main page accessible: {len(response.text)} chars")

                # Look for search functionality
                soup = BeautifulSoup(response.text, 'html.parser')

                # Look for search forms
                search_forms = soup.find_all('form')
                search_inputs = soup.find_all('input', {'type': 'search'}) + soup.find_all('input', {'name': 'q'})

                logger.info(f"Found {len(search_forms)} forms")
                logger.info(f"Found {len(search_inputs)} search inputs")

                # Save the page for analysis
                with open(self.results_dir / "main_page.html", 'w', encoding='utf-8') as f:
                    f.write(response.text)

                return True
            else:
                logger.info(f"✗ Main page returned HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error accessing main page: {e}")
            return False

    def test_search_functionality(self):
        """Test the search functionality on case.law."""
        logger.info("Testing search functionality...")

        # Try different search approaches
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

            # Try different search URL patterns
            search_urls = [
                f"{self.base_url}/search/?q={quote(query)}",
                f"{self.base_url}/?q={quote(query)}",
                f"{self.base_url}/search?q={quote(query)}",
            ]

            query_results = {}

            for url in search_urls:
                try:
                    response = self.session.get(url, timeout=10)

                    query_results[url] = {
                        "status_code": response.status_code,
                        "content_length": len(response.text),
                        "success": response.status_code == 200
                    }

                    if response.status_code == 200:
                        logger.info(f"  ✓ {url}: {len(response.text)} chars")

                        # Parse for case results
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Look for case links or results
                        case_links = soup.find_all('a', href=True)
                        case_results = []

                        for link in case_links:
                            href = link.get('href', '')
                            text = link.get_text().strip()

                            # Look for case-like patterns
                            if any(pattern in href.lower() for pattern in ['case', 'opinion', 'decision']):
                                case_results.append({
                                    "href": href,
                                    "text": text[:100]  # First 100 chars
                                })

                        query_results[url]["case_results"] = case_results
                        logger.info(f"    Found {len(case_results)} potential case links")

                    else:
                        logger.info(f"  ✗ {url}: HTTP {response.status_code}")

                except Exception as e:
                    query_results[url] = {
                        "error": str(e),
                        "success": False
                    }
                    logger.info(f"  ✗ {url}: Error - {e}")

                time.sleep(1)  # Be respectful

            search_results[query] = query_results
            time.sleep(2)  # Be respectful between queries

        # Save search results
        with open(self.results_dir / "search_results.json", 'w') as f:
            json.dump(search_results, f, indent=2)

        return search_results

    def test_direct_case_access(self):
        """Test if we can access specific known cases directly."""
        logger.info("Testing direct case access...")

        # Try to access known cases directly
        known_cases = [
            "Intel Corp v Advanced Micro Devices",
            "ZF Automotive v Luxshare",
            "Brandi-Dohrn v IKB Deutsche Industriebank"
        ]

        direct_results = {}

        for case_name in known_cases:
            logger.info(f"Testing direct access: '{case_name}'")

            # Try different URL patterns for case access
            case_urls = [
                f"{self.base_url}/case/{quote(case_name)}",
                f"{self.base_url}/cases/{quote(case_name)}",
                f"{self.base_url}/opinion/{quote(case_name)}",
                f"{self.base_url}/decisions/{quote(case_name)}",
            ]

            case_results = {}

            for url in case_urls:
                try:
                    response = self.session.get(url, timeout=10)

                    case_results[url] = {
                        "status_code": response.status_code,
                        "content_length": len(response.text),
                        "success": response.status_code == 200
                    }

                    if response.status_code == 200:
                        logger.info(f"  ✓ {url}: {len(response.text)} chars")

                        # Check if it looks like a case page
                        if "opinion" in response.text.lower() or "decision" in response.text.lower():
                            logger.info(f"    Looks like a case page!")
                            case_results[url]["is_case_page"] = True
                        else:
                            case_results[url]["is_case_page"] = False

                    else:
                        logger.info(f"  ✗ {url}: HTTP {response.status_code}")

                except Exception as e:
                    case_results[url] = {
                        "error": str(e),
                        "success": False
                    }
                    logger.info(f"  ✗ {url}: Error - {e}")

                time.sleep(1)

            direct_results[case_name] = case_results
            time.sleep(2)

        # Save direct access results
        with open(self.results_dir / "direct_access_results.json", 'w') as f:
            json.dump(direct_results, f, indent=2)

        return direct_results

    def run(self):
        """Run all CAP web scraping tests."""
        logger.info("Starting Harvard CAP Web Scraping Tests")
        logger.info("The irony continues...")

        try:
            # Test main page access
            if not self.test_main_page():
                logger.error("Cannot access main case.law page. CAP may be down or blocked.")
                return

            # Test search functionality
            search_results = self.test_search_functionality()

            # Test direct case access
            direct_results = self.test_direct_case_access()

            # Analyze results
            self.analyze_web_results(search_results, direct_results)

            logger.info("=" * 60)
            logger.info("CAP WEB SCRAPING TESTS COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {self.results_dir}")

        except Exception as e:
            logger.error(f"Error during CAP web scraping: {e}", exc_info=True)

    def analyze_web_results(self, search_results, direct_results):
        """Analyze the web scraping results."""
        logger.info("\n" + "=" * 60)
        logger.info("CAP Web Scraping Analysis")
        logger.info("=" * 60)

        # Analyze search results
        successful_searches = 0
        total_case_links = 0

        for query, results in search_results.items():
            logger.info(f"\nSearch: '{query}'")

            has_success = False
            for url, data in results.items():
                if data.get("success", False):
                    has_success = True
                    case_count = len(data.get("case_results", []))
                    total_case_links += case_count
                    logger.info(f"  ✓ {url}: {case_count} case links")

            if not has_success:
                logger.info(f"  ✗ No successful searches")
            else:
                successful_searches += 1

        # Analyze direct access results
        successful_direct = 0
        for case_name, results in direct_results.items():
            has_success = any(data.get("success", False) for data in results.values())
            if has_success:
                successful_direct += 1
                logger.info(f"✓ Direct access works for: {case_name}")

        logger.info(f"\nSUMMARY:")
        logger.info(f"Successful searches: {successful_searches}")
        logger.info(f"Total case links found: {total_case_links}")
        logger.info(f"Successful direct access: {successful_direct}")

        if successful_searches > 0 or successful_direct > 0:
            logger.info("✓ CAP web interface appears accessible")
            logger.info("✓ We can potentially scrape §1782 cases from CAP")
        else:
            logger.info("✗ CAP web interface may not be accessible or may not have the cases we need")


def main():
    """Main entry point."""
    print("Harvard CAP Web Scraping Test")
    print("=" * 60)
    print("Testing Harvard's web interface for §1782 cases")
    print("(The irony continues...)")
    print("=" * 60)

    scraper = CAPWebScraper()
    scraper.run()


if __name__ == "__main__":
    main()
