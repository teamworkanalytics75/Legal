#!/usr/bin/env python3
"""
Improved CourtListener Web Scraper

Inspect the actual HTML structure and scrape the 54 cases found.
"""

import sys
import json
import logging
import requests
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs
import re
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ImprovedCourtListenerScraper:
    """Improved scraper for CourtListener web interface."""

    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://www.courtlistener.com"

    def inspect_page_structure(self, url):
        """Inspect the page structure to understand how cases are displayed."""
        try:
            logger.info(f"Inspecting page structure: {url}")

            response = self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Save the HTML for inspection
            html_path = Path(__file__).parent.parent / "data" / "case_law" / "courtlistener_page.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())

            logger.info(f"HTML saved to: {html_path}")

            # Look for common patterns
            logger.info("Looking for case-related elements...")

            # Check for search results
            results_div = soup.find('div', class_='search-results')
            if results_div:
                logger.info("Found search-results div")

            # Check for case listings
            case_divs = soup.find_all('div', class_=re.compile(r'case|result|opinion'))
            logger.info(f"Found {len(case_divs)} divs with case/result/opinion classes")

            # Check for links containing case information
            case_links = soup.find_all('a', href=re.compile(r'/opinion/|/case/'))
            logger.info(f"Found {len(case_links)} case links")

            # Look for text containing "1782"
            text_1782 = soup.find_all(string=re.compile(r'1782'))
            logger.info(f"Found {len(text_1782)} text elements containing '1782'")

            # Show sample elements
            logger.info("Sample case links:")
            for i, link in enumerate(case_links[:5], 1):
                logger.info(f"  {i}. {link.get_text(strip=True)[:100]}...")
                logger.info(f"     URL: {link.get('href')}")

            return soup

        except Exception as e:
            logger.error(f"Error inspecting page: {e}")
            return None

    def extract_cases_from_inspection(self, soup):
        """Extract cases based on the actual page structure."""
        cases = []

        # Look for case links
        case_links = soup.find_all('a', href=re.compile(r'/opinion/|/case/'))

        logger.info(f"Found {len(case_links)} potential case links")

        for link in case_links:
            try:
                case_info = {}

                # Get case URL
                case_url = urljoin(self.base_url, link['href'])
                case_info['case_url'] = case_url

                # Extract case ID from URL
                url_parts = urlparse(case_url)
                path_parts = url_parts.path.split('/')
                if len(path_parts) > 1:
                    case_info['case_id'] = path_parts[-1]

                # Get case name from link text
                case_name = link.get_text(strip=True)
                if case_name:
                    case_info['case_name'] = case_name

                # Look for parent elements that might contain additional info
                parent = link.parent
                while parent and parent.name != 'body':
                    # Look for court/date info
                    court_text = parent.get_text(strip=True)
                    if 'Filed:' in court_text or 'Court:' in court_text:
                        case_info['court_date'] = court_text

                    # Look for docket number
                    if 'Docket' in court_text:
                        docket_match = re.search(r'Docket.*?(\d+)', court_text)
                        if docket_match:
                            case_info['docket_number'] = docket_match.group(1)

                    parent = parent.parent

                # Only add if we have meaningful information
                if case_info.get('case_name') and len(case_info['case_name']) > 10:
                    cases.append(case_info)

            except Exception as e:
                logger.error(f"Error processing case link: {e}")
                continue

        # Remove duplicates based on case_id
        unique_cases = []
        seen_ids = set()

        for case in cases:
            case_id = case.get('case_id')
            if case_id and case_id not in seen_ids:
                unique_cases.append(case)
                seen_ids.add(case_id)

        logger.info(f"Found {len(unique_cases)} unique cases")

        return unique_cases

    def scrape_case_details(self, case_url):
        """Scrape detailed information from a case page."""
        try:
            logger.info(f"Scraping case details: {case_url}")

            response = self.session.get(case_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            case_details = {}

            # Extract case text
            case_text_elem = soup.find('div', class_='case-text') or soup.find('div', class_='opinion')
            if case_text_elem:
                case_details['case_text'] = case_text_elem.get_text(strip=True)

            # Extract metadata
            metadata_elem = soup.find('div', class_='metadata') or soup.find('div', class_='case-info')
            if metadata_elem:
                case_details['metadata'] = metadata_elem.get_text(strip=True)

            return case_details

        except Exception as e:
            logger.error(f"Error scraping case details: {e}")
            return {}

    def run_scraping(self):
        """Run the complete scraping process."""
        logger.info("Starting improved CourtListener web scraping...")

        # The search URL you found
        search_url = "https://www.courtlistener.com/?q=28%20usc%201782%20"

        # First, inspect the page structure
        soup = self.inspect_page_structure(search_url)
        if not soup:
            logger.error("Failed to inspect page structure")
            return []

        # Extract cases based on actual structure
        cases = self.extract_cases_from_inspection(soup)

        # Add detailed information for each case
        detailed_cases = []
        for i, case in enumerate(cases, 1):
            logger.info(f"Processing case {i}/{len(cases)}: {case.get('case_name', 'Unknown')}")

            case_details = self.scrape_case_details(case['case_url'])
            case.update(case_details)

            detailed_cases.append(case)
            time.sleep(1)  # Rate limiting

        # Save results
        self.save_results(detailed_cases)

        return detailed_cases

    def save_results(self, cases):
        """Save scraped cases to file."""
        try:
            output_path = Path(__file__).parent.parent / "data" / "case_law" / "courtlistener_scraped_1782_cases.json"

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cases, f, indent=2, ensure_ascii=False)

            logger.info(f"Scraped cases saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main entry point."""
    scraper = ImprovedCourtListenerScraper()
    cases = scraper.run_scraping()

    if cases:
        print(f"\nSUCCESS!")
        print(f"Scraped {len(cases)} cases from CourtListener web interface")
        print(f"Results saved to courtlistener_scraped_1782_cases.json")

        # Show sample cases
        print(f"\nSample cases:")
        for i, case in enumerate(cases[:5], 1):
            print(f"  {i}. {case.get('case_name', 'Unknown')}")
            print(f"     URL: {case.get('case_url', 'Unknown')}")
            print()


if __name__ == "__main__":
    main()
