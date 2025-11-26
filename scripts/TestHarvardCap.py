#!/usr/bin/env python3
"""
Harvard CAP (Caselaw Access Project) Test Script

This script tests Harvard's free legal database to see if we can find
§1782 cases that CourtListener is missing. The irony is not lost on us.

Test different search approaches on case.law to see what's available.
"""

import requests
import json
import time
import logging
from pathlib import Path
from urllib.parse import urlencode, quote

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CAPTester:
    """Test Harvard's Caselaw Access Project for §1782 cases."""

    def __init__(self):
        """Initialize the CAP tester."""
        self.base_url = "https://case.law"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Test results storage
        self.results_dir = Path("data/case_law/1782_discovery/cap_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.test_results = {}

    def test_search_endpoints(self):
        """Test different search approaches on CAP."""
        logger.info("=" * 60)
        logger.info("Testing Harvard CAP (Caselaw Access Project)")
        logger.info("=" * 60)

        # Test 1: Basic search for "1782"
        logger.info("Test 1: Searching for '1782'")
        self._test_search("1782", "basic_1782")

        # Test 2: Search for "28 U.S.C. 1782"
        logger.info("Test 2: Searching for '28 U.S.C. 1782'")
        self._test_search("28 U.S.C. 1782", "usc_1782")

        # Test 3: Search for "section 1782"
        logger.info("Test 3: Searching for 'section 1782'")
        self._test_search("section 1782", "section_1782")

        # Test 4: Search for "foreign tribunal"
        logger.info("Test 4: Searching for 'foreign tribunal'")
        self._test_search("foreign tribunal", "foreign_tribunal")

        # Test 5: Search for "Intel Corp"
        logger.info("Test 5: Searching for 'Intel Corp'")
        self._test_search("Intel Corp", "intel_corp")

        # Test 6: Search for "ZF Automotive"
        logger.info("Test 6: Searching for 'ZF Automotive'")
        self._test_search("ZF Automotive", "zf_automotive")

        time.sleep(2)  # Be respectful

    def _test_search(self, query: str, test_name: str):
        """Test a specific search query on CAP."""
        try:
            # Try different search endpoints
            search_urls = [
                f"{self.base_url}/search/?q={quote(query)}",
                f"{self.base_url}/api/v1/cases/?search={quote(query)}",
                f"{self.base_url}/api/search/?q={quote(query)}",
            ]

            results = {}

            for i, url in enumerate(search_urls):
                logger.info(f"  Trying endpoint {i+1}: {url}")

                try:
                    response = self.session.get(url, timeout=10)

                    results[f"endpoint_{i+1}"] = {
                        "url": url,
                        "status_code": response.status_code,
                        "content_length": len(response.text),
                        "content_type": response.headers.get('content-type', 'unknown'),
                        "success": response.status_code == 200
                    }

                    if response.status_code == 200:
                        # Try to parse as JSON
                        try:
                            json_data = response.json()
                            results[f"endpoint_{i+1}"]["json_data"] = json_data
                            logger.info(f"    ✓ JSON response with {len(str(json_data))} chars")
                        except:
                            # Not JSON, save HTML content
                            results[f"endpoint_{i+1}"]["html_preview"] = response.text[:500]
                            logger.info(f"    ✓ HTML response with {len(response.text)} chars")

                    else:
                        logger.info(f"    ✗ HTTP {response.status_code}")

                except Exception as e:
                    results[f"endpoint_{i+1}"] = {
                        "url": url,
                        "error": str(e),
                        "success": False
                    }
                    logger.info(f"    ✗ Error: {e}")

                time.sleep(1)  # Be respectful

            # Save results
            self.test_results[test_name] = {
                "query": query,
                "results": results
            }

            # Save to file
            with open(self.results_dir / f"{test_name}_results.json", 'w') as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            logger.error(f"Error testing search '{query}': {e}")

    def test_bulk_download_options(self):
        """Test if CAP offers bulk download options."""
        logger.info("\n" + "=" * 60)
        logger.info("Testing CAP Bulk Download Options")
        logger.info("=" * 60)

        # Test common bulk download endpoints
        bulk_urls = [
            f"{self.base_url}/api/v1/bulk/",
            f"{self.base_url}/download/",
            f"{self.base_url}/api/bulk/",
            f"{self.base_url}/datasets/",
            f"{self.base_url}/api/v1/datasets/",
        ]

        bulk_results = {}

        for url in bulk_urls:
            logger.info(f"Testing bulk endpoint: {url}")

            try:
                response = self.session.get(url, timeout=10)

                bulk_results[url] = {
                    "status_code": response.status_code,
                    "content_length": len(response.text),
                    "success": response.status_code == 200
                }

                if response.status_code == 200:
                    logger.info(f"  ✓ Available: {len(response.text)} chars")
                    try:
                        json_data = response.json()
                        bulk_results[url]["json_data"] = json_data
                    except:
                        bulk_results[url]["html_preview"] = response.text[:500]
                else:
                    logger.info(f"  ✗ HTTP {response.status_code}")

            except Exception as e:
                bulk_results[url] = {
                    "error": str(e),
                    "success": False
                }
                logger.info(f"  ✗ Error: {e}")

            time.sleep(1)

        # Save bulk results
        with open(self.results_dir / "bulk_download_test.json", 'w') as f:
            json.dump(bulk_results, f, indent=2)

        return bulk_results

    def analyze_results(self):
        """Analyze the test results to determine CAP's capabilities."""
        logger.info("\n" + "=" * 60)
        logger.info("CAP Test Results Analysis")
        logger.info("=" * 60)

        successful_searches = 0
        total_searches = len(self.test_results)

        for test_name, test_data in self.test_results.items():
            query = test_data["query"]
            results = test_data["results"]

            logger.info(f"\n{test_name.upper()}: '{query}'")

            has_results = False
            for endpoint_name, endpoint_data in results.items():
                if endpoint_data.get("success", False):
                    has_results = True
                    logger.info(f"  ✓ {endpoint_name}: {endpoint_data.get('content_length', 0)} chars")

                    # Check if it contains case data
                    if "json_data" in endpoint_data:
                        json_data = endpoint_data["json_data"]
                        if isinstance(json_data, dict) and "results" in json_data:
                            case_count = len(json_data["results"])
                            logger.info(f"    Found {case_count} cases")
                        elif isinstance(json_data, list):
                            logger.info(f"    Found {len(json_data)} items")

            if not has_results:
                logger.info(f"  ✗ No successful endpoints")
            else:
                successful_searches += 1

        logger.info(f"\nSUMMARY:")
        logger.info(f"Successful searches: {successful_searches}/{total_searches}")
        logger.info(f"CAP API availability: {'Yes' if successful_searches > 0 else 'No'}")

        if successful_searches > 0:
            logger.info("✓ CAP appears to have searchable case data")
            logger.info("✓ We can potentially extract §1782 cases from CAP")
        else:
            logger.info("✗ CAP may not have public API access")
            logger.info("✗ May need to use web scraping approach")

    def run(self):
        """Run all CAP tests."""
        logger.info("Starting Harvard CAP (Caselaw Access Project) Tests")
        logger.info("The irony of using Harvard's database to sue Harvard is noted.")

        try:
            # Test search functionality
            self.test_search_endpoints()

            # Test bulk download options
            self.test_bulk_download_options()

            # Analyze results
            self.analyze_results()

            logger.info("=" * 60)
            logger.info("CAP TESTING COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {self.results_dir}")

        except Exception as e:
            logger.error(f"Error during CAP testing: {e}", exc_info=True)


def main():
    """Main entry point."""
    print("Harvard CAP (Caselaw Access Project) Test")
    print("=" * 60)
    print("Testing Harvard's free legal database for §1782 cases")
    print("(The irony is not lost on us)")
    print("=" * 60)

    tester = CAPTester()
    tester.run()


if __name__ == "__main__":
    main()
