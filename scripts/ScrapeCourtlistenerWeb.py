#!/usr/bin/env python3
"""
CourtListener Web Scraper for §1782 Cases

Scrape CourtListener's web interface to get all 54 cases found for "28 usc 1782"
that aren't showing up in the API results.
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


class CourtListenerWebScraper:
    """Scrape CourtListener web interface for §1782 cases."""

    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://www.courtlistener.com"
        self.scraped_cases = []

    def scrape_search_results(self, search_url, max_pages=10):
        """Scrape search results from CourtListener."""
        try:
            logger.info(f"Scraping search results from: {search_url}")

            all_cases = []
            page = 1

            while page <= max_pages:
                # Add page parameter
                if '?' in search_url:
                    page_url = f"{search_url}&page={page}"
                else:
                    page_url = f"{search_url}?page={page}"

                logger.info(f"Scraping page {page}: {page_url}")

                response = self.session.get(page_url)
                response.raise_for_status()

                # Parse the page
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find case results
                case_results = self.extract_cases_from_page(soup)

                if not case_results:
                    logger.info(f"No more cases found on page {page}")
                    break

                all_cases.extend(case_results)
                logger.info(f"Found {len(case_results)} cases on page {page}")

                page += 1
                time.sleep(1)  # Rate limiting

            logger.info(f"Total cases scraped: {len(all_cases)}")
            return all_cases

        except Exception as e:
            logger.error(f"Error scraping search results: {e}")
            return []

    def extract_cases_from_page(self, soup):
        """Extract case information from a search results page."""
        cases = []

        # Look for case result containers
        # CourtListener uses specific CSS classes for case results
        case_containers = soup.find_all('div', class_='result')

        if not case_containers:
            # Try alternative selectors
            case_containers = soup.find_all('div', class_='case-result')

        if not case_containers:
            # Try looking for any div with case information
            case_containers = soup.find_all('div', string=re.compile(r'Application.*1782|In re.*1782'))

        logger.info(f"Found {len(case_containers)} case containers")

        for container in case_containers:
            case_info = self.extract_case_info(container)
            if case_info:
                cases.append(case_info)

        return cases

    def extract_case_info(self, container):
        """Extract case information from a container element."""
        try:
            case_info = {}

            # Extract case name
            case_name_elem = container.find('h3') or container.find('h4') or container.find('a', class_='case-name')
            if case_name_elem:
                case_info['case_name'] = case_name_elem.get_text(strip=True)

            # Extract court and date
            court_date_elem = container.find('div', class_='court-date') or container.find('span', class_='court')
            if court_date_elem:
                court_date_text = court_date_elem.get_text(strip=True)
                case_info['court_date'] = court_date_text

            # Extract docket number
            docket_elem = container.find('span', class_='docket') or container.find('div', class_='docket-number')
            if docket_elem:
                case_info['docket_number'] = docket_elem.get_text(strip=True)

            # Extract case URL
            link_elem = container.find('a', href=True)
            if link_elem:
                case_info['case_url'] = urljoin(self.base_url, link_elem['href'])

            # Extract case ID from URL
            if 'case_url' in case_info:
                url_parts = urlparse(case_info['case_url'])
                path_parts = url_parts.path.split('/')
                if len(path_parts) > 1:
                    case_info['case_id'] = path_parts[-1]

            # Extract status
            status_elem = container.find('span', class_='status') or container.find('div', class_='status')
            if status_elem:
                case_info['status'] = status_elem.get_text(strip=True)

            # Extract citations
            citations_elem = container.find('div', class_='citations') or container.find('span', class_='citations')
            if citations_elem:
                case_info['citations'] = citations_elem.get_text(strip=True)

            # Only return if we have at least a case name
            if case_info.get('case_name'):
                return case_info

        except Exception as e:
            logger.error(f"Error extracting case info: {e}")

        return None

    def scrape_case_details(self, case_url):
        """Scrape detailed information from a case page."""
        try:
            logger.info(f"Scraping case details: {case_url}")

            response = self.session.get(case_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract case text
            case_text_elem = soup.find('div', class_='case-text') or soup.find('div', class_='opinion')
            if case_text_elem:
                case_text = case_text_elem.get_text(strip=True)
                return case_text

        except Exception as e:
            logger.error(f"Error scraping case details: {e}")

        return None

    def run_scraping(self):
        """Run the complete scraping process."""
        logger.info("Starting CourtListener web scraping...")

        # The search URL you found
        search_url = "https://www.courtlistener.com/?q=28%20usc%201782%20"

        # Scrape all pages
        cases = self.scrape_search_results(search_url, max_pages=5)

        # Add detailed information for each case
        detailed_cases = []
        for i, case in enumerate(cases, 1):
            logger.info(f"Processing case {i}/{len(cases)}: {case.get('case_name', 'Unknown')}")

            if 'case_url' in case:
                case_text = self.scrape_case_details(case['case_url'])
                if case_text:
                    case['case_text'] = case_text

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
    scraper = CourtListenerWebScraper()
    cases = scraper.run_scraping()

    if cases:
        print(f"\nSUCCESS!")
        print(f"Scraped {len(cases)} cases from CourtListener web interface")
        print(f"Results saved to courtlistener_scraped_1782_cases.json")

        # Show sample cases
        print(f"\nSample cases:")
        for i, case in enumerate(cases[:5], 1):
            print(f"  {i}. {case.get('case_name', 'Unknown')}")
            print(f"     Court: {case.get('court_date', 'Unknown')}")
            print(f"     Docket: {case.get('docket_number', 'Unknown')}")
            print()


if __name__ == "__main__":
    main()
