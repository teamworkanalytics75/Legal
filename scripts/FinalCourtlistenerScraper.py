#!/usr/bin/env python3
"""
Final CourtListener Web Scraper

Extract all 54 cases from CourtListener's web interface using the correct HTML structure.
"""

import sys
import json
import logging
import requests
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
import re
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FinalCourtListenerScraper:
    """Final scraper for CourtListener web interface."""

    def __init__(self):
        """Initialize the scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://www.courtlistener.com"

    def scrape_all_pages(self, search_url, max_pages=5):
        """Scrape all pages of search results."""
        all_cases = []
        page = 1

        while page <= max_pages:
            # Add page parameter
            if '?' in search_url:
                page_url = f"{search_url}&page={page}"
            else:
                page_url = f"{search_url}?page={page}"

            logger.info(f"Scraping page {page}: {page_url}")

            try:
                response = self.session.get(page_url)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract cases from this page
                page_cases = self.extract_cases_from_page(soup)

                if not page_cases:
                    logger.info(f"No more cases found on page {page}")
                    break

                all_cases.extend(page_cases)
                logger.info(f"Found {len(page_cases)} cases on page {page}")

                page += 1
                time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                break

        logger.info(f"Total cases scraped: {len(all_cases)}")
        return all_cases

    def extract_cases_from_page(self, soup):
        """Extract cases from a single page."""
        cases = []

        # Find all article elements containing case information
        articles = soup.find_all('article')
        logger.info(f"Found {len(articles)} article elements")

        for article in articles:
            case_info = self.extract_case_from_article(article)
            if case_info:
                cases.append(case_info)

        return cases

    def extract_case_from_article(self, article):
        """Extract case information from an article element."""
        try:
            case_info = {}

            # Find the h3 element with the case link
            h3 = article.find('h3', class_='bottom serif')
            if not h3:
                return None

            # Extract the case link
            link = h3.find('a', class_='visitable')
            if not link:
                return None

            # Get case URL
            case_url = urljoin(self.base_url, link['href'])
            case_info['case_url'] = case_url

            # Extract case ID from URL
            url_parts = urlparse(case_url)
            path_parts = url_parts.path.split('/')
            if len(path_parts) > 1:
                case_info['case_id'] = path_parts[2]  # /opinion/ID/...

            # Extract case name (clean up the text)
            case_name = link.get_text(strip=True)
            # Remove extra whitespace and clean up
            case_name = re.sub(r'\s+', ' ', case_name)
            case_info['case_name'] = case_name

            # Extract court and date information
            court_date_elem = article.find('p', class_='bottom')
            if court_date_elem:
                court_date_text = court_date_elem.get_text(strip=True)
                case_info['court_date'] = court_date_text

                # Try to extract court
                court_match = re.search(r'\(([^)]+)\)', court_date_text)
                if court_match:
                    case_info['court'] = court_match.group(1)

                # Try to extract date
                date_match = re.search(r'Date Filed:\s*([^,]+)', court_date_text)
                if date_match:
                    case_info['date_filed'] = date_match.group(1)

                # Try to extract docket number
                docket_match = re.search(r'Docket Number:\s*([^,]+)', court_date_text)
                if docket_match:
                    case_info['docket_number'] = docket_match.group(1)

                # Try to extract status
                status_match = re.search(r'Status:\s*([^,]+)', court_date_text)
                if status_match:
                    case_info['status'] = status_match.group(1)

            # Extract citations
            citations_elem = article.find('p', class_='bottom')
            if citations_elem:
                citations_text = citations_elem.get_text(strip=True)
                if 'Citations:' in citations_text:
                    citations_match = re.search(r'Citations:\s*([^,]+)', citations_text)
                    if citations_match:
                        case_info['citations'] = citations_match.group(1)

            # Only return if we have essential information
            if case_info.get('case_name') and case_info.get('case_id'):
                return case_info

        except Exception as e:
            logger.error(f"Error extracting case from article: {e}")

        return None

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
        logger.info("Starting final CourtListener web scraping...")

        # The search URL you found
        search_url = "https://www.courtlistener.com/?q=28%20usc%201782%20"

        # Scrape all pages
        cases = self.scrape_all_pages(search_url, max_pages=5)

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
    scraper = FinalCourtListenerScraper()
    cases = scraper.run_scraping()

    if cases:
        print(f"\nSUCCESS!")
        print(f"Scraped {len(cases)} cases from CourtListener web interface")
        print(f"Results saved to courtlistener_scraped_1782_cases.json")

        # Show sample cases
        print(f"\nSample cases:")
        for i, case in enumerate(cases[:5], 1):
            print(f"  {i}. {case.get('case_name', 'Unknown')}")
            print(f"     Court: {case.get('court', 'Unknown')}")
            print(f"     Docket: {case.get('docket_number', 'Unknown')}")
            print(f"     URL: {case.get('case_url', 'Unknown')}")
            print()


if __name__ == "__main__":
    main()
