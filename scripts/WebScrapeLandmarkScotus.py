#!/usr/bin/env python3
"""
Web Scrape Landmark SCOTUS §1782 Cases

Uses web scraping to find the missing landmark SCOTUS cases since API search failed.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import time
import re

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Google Drive backup module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "google_drive_backup",
    Path(__file__).parent.parent / "document_ingestion" / "google_drive_backup.py"
)
gdrive_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdrive_module)
GoogleDriveBackup = gdrive_module.GoogleDriveBackup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class WebScrapeLandmarkSCOTUS:
    """Web scrapes landmark SCOTUS §1782 cases from various sources."""

    def __init__(self):
        """Initialize the scraper."""
        self.gdrive = GoogleDriveBackup()

        # Target folder
        self.target_folder_name = "Cleanest 1782 Database"
        self.target_folder_id = None

        # Landmark SCOTUS cases to find
        self.landmark_cases = {
            "Intel Corp. v. Advanced Micro Devices, Inc.": {
                "year": "2004",
                "citation": "542 U.S. 241",
                "docket": "03-724",
                "urls": [
                    "https://www.law.cornell.edu/supct/html/03-724.ZS.html",
                    "https://supreme.justia.com/cases/federal/us/542/241/",
                    "https://www.oyez.org/cases/2003/03-724"
                ]
            },
            "ZF Automotive US, Inc. v. Luxshare, Ltd.": {
                "year": "2022",
                "citation": "596 U.S. ___",
                "docket": "21-401",
                "urls": [
                    "https://www.law.cornell.edu/supct/html/21-401.ZS.html",
                    "https://supreme.justia.com/cases/federal/us/596/",
                    "https://www.oyez.org/cases/2021/21-401"
                ]
            },
            "Servotronics, Inc. v. Rolls-Royce PLC": {
                "year": "2021",
                "citation": "593 U.S. ___",
                "docket": "20-794",
                "urls": [
                    "https://www.law.cornell.edu/supct/html/20-794.ZS.html",
                    "https://supreme.justia.com/cases/federal/us/593/",
                    "https://www.oyez.org/cases/2020/20-794"
                ]
            }
        }

        self.found_cases = {}
        self.downloaded_cases = []

    def find_target_folder(self):
        """Find the Cleanest 1782 Database folder."""
        try:
            results = self.gdrive.service.files().list(
                q=f"name='{self.target_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields='files(id, name)'
            ).execute()

            folders = results.get('files', [])
            if folders:
                self.target_folder_id = folders[0]['id']
                logger.info(f"Found target folder: {self.target_folder_name}")
                return True
            else:
                logger.error(f"Target folder '{self.target_folder_name}' not found")
                return False

        except Exception as e:
            logger.error(f"Error finding target folder: {e}")
            return False

    def scrape_case_from_url(self, case_name: str, case_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Scrape case content from a URL."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping: {case_name}")
        logger.info(f"Citation: {case_info['citation']}")
        logger.info(f"{'='*60}")

        for url in case_info['urls']:
            try:
                logger.info(f"Trying URL: {url}")

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }

                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract case content
                case_data = self._extract_case_content(soup, case_name, case_info)

                if case_data:
                    logger.info(f"✅ Successfully scraped from: {url}")
                    return case_data
                else:
                    logger.info(f"⚠️  No usable content found at: {url}")

            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                continue

            # Add delay between requests
            time.sleep(2)

        logger.warning(f"❌ Could not scrape: {case_name}")
        return None

    def _extract_case_content(self, soup: BeautifulSoup, case_name: str, case_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract case content from BeautifulSoup object."""
        try:
            # Try to find the main content
            content_selectors = [
                'div.opinion',
                'div.case-content',
                'div.main-content',
                'div.content',
                'article',
                'main',
                'div[class*="opinion"]',
                'div[class*="case"]'
            ]

            content_text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content_text = "\n".join([elem.get_text(strip=True) for elem in elements])
                    break

            if not content_text:
                # Fallback: get all text
                content_text = soup.get_text(strip=True)

            # Check if this looks like a §1782 case
            if self._is_1782_case(content_text):
                case_data = {
                    'case_name': case_name,
                    'citation': case_info['citation'],
                    'year': case_info['year'],
                    'docket': case_info['docket'],
                    'court': 'Supreme Court',
                    'content': content_text,
                    'source': 'web_scraped',
                    'html_content': str(soup)
                }

                logger.info(f"✅ Extracted §1782 case content")
                return case_data
            else:
                logger.info(f"⚠️  Content doesn't appear to be §1782 related")
                return None

        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return None

    def _is_1782_case(self, content: str) -> bool:
        """Check if content is about §1782."""
        content_lower = content.lower()
        keywords = [
            "28 u.s.c. § 1782",
            "28 usc 1782",
            "section 1782",
            "foreign tribunal",
            "international tribunal",
            "judicial assistance",
            "foreign proceeding",
            "international discovery"
        ]

        for keyword in keywords:
            if keyword in content_lower:
                return True

        return False

    def download_case_content(self, case_data: Dict[str, Any], case_name: str) -> bool:
        """Download and save case content."""
        try:
            logger.info(f"Downloading content for: {case_name}")

            # Create clean filename
            clean_name = case_name.replace(' ', '_').replace('.', '').replace(',', '')
            filename = f"001_SCOTUS_{clean_name}.json"

            # Upload to target folder
            uploaded_id = self.gdrive.upload_file_content(
                content=json.dumps(case_data, indent=2),
                filename=filename,
                folder_id=self.target_folder_id
            )

            if uploaded_id:
                self.downloaded_cases.append({
                    'case_name': case_name,
                    'filename': filename,
                    'court': case_data.get('court', 'Supreme Court'),
                    'citation': case_data.get('citation', 'Unknown'),
                    'year': case_data.get('year', 'Unknown')
                })

                logger.info(f"✅ Successfully downloaded: {case_name}")
                return True
            else:
                logger.error(f"❌ Failed to upload: {case_name}")
                return False

        except Exception as e:
            logger.error(f"❌ Error downloading {case_name}: {e}")
            return False

    def scrape_all_landmark_cases(self):
        """Scrape all landmark SCOTUS cases."""
        logger.info("Starting web scraping of landmark SCOTUS §1782 cases...")

        # Find target folder
        if not self.find_target_folder():
            return False

        # Process each landmark case
        for case_name, case_info in self.landmark_cases.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {case_name}")
            logger.info(f"{'='*80}")

            # Scrape the case
            case_data = self.scrape_case_from_url(case_name, case_info)

            if case_data:
                self.found_cases[case_name] = {
                    'case_info': case_info,
                    'case_data': case_data
                }

                # Try to download full content
                success = self.download_case_content(case_data, case_name)
                if success:
                    logger.info(f"🎯 Successfully added {case_name} to database!")
                else:
                    logger.warning(f"📝 Case found but content not fully available")
            else:
                logger.warning(f"❌ Could not scrape: {case_name}")

            # Add delay between cases
            time.sleep(3)

        # Generate summary report
        self.generate_summary_report()
        return True

    def generate_summary_report(self):
        """Generate a summary report."""
        logger.info(f"\n{'='*80}")
        logger.info("LANDMARK SCOTUS CASE WEB SCRAPING SUMMARY")
        logger.info(f"{'='*80}")

        total_targeted = len(self.landmark_cases)
        found_count = len(self.found_cases)
        downloaded_count = len(self.downloaded_cases)

        logger.info(f"Total landmark cases targeted: {total_targeted}")
        logger.info(f"Cases found: {found_count}")
        logger.info(f"Cases downloaded: {downloaded_count}")
        logger.info(f"Success rate: {(found_count/total_targeted)*100:.1f}%")

        if self.found_cases:
            logger.info(f"\n✅ FOUND LANDMARK CASES:")
            for case_name, data in self.found_cases.items():
                case_info = data['case_info']
                case_data = data['case_data']
                logger.info(f"  - {case_name}")
                logger.info(f"    Citation: {case_info['citation']}")
                logger.info(f"    Year: {case_info['year']}")
                logger.info(f"    Source: {case_data.get('source', 'web_scraped')}")

        if self.downloaded_cases:
            logger.info(f"\n📥 DOWNLOADED CASES:")
            for case in self.downloaded_cases:
                logger.info(f"  - {case['case_name']} → {case['filename']}")

        missing_cases = []
        for case_name in self.landmark_cases.keys():
            if case_name not in self.found_cases:
                missing_cases.append(case_name)

        if missing_cases:
            logger.info(f"\n❌ MISSING LANDMARK CASES:")
            for case_name in missing_cases:
                case_info = self.landmark_cases[case_name]
                logger.info(f"  - {case_name} ({case_info['year']}, {case_info['citation']})")

        logger.info(f"\n🎯 DATABASE STATUS:")
        if downloaded_count > 0:
            logger.info(f"✅ Added {downloaded_count} landmark SCOTUS cases to Cleanest 1782 Database")
            logger.info(f"📁 Database now contains Supreme Court cases!")
        else:
            logger.info(f"⚠️  No landmark SCOTUS cases were successfully scraped")
            logger.info(f"💡 Consider manual search on other platforms")

        # Save detailed report
        report_data = {
            'summary': {
                'total_targeted': total_targeted,
                'found': found_count,
                'downloaded': downloaded_count,
                'success_rate': (found_count/total_targeted)*100
            },
            'found_cases': self.found_cases,
            'downloaded_cases': self.downloaded_cases,
            'missing_cases': missing_cases
        }

        try:
            self.gdrive.upload_file_content(
                content=json.dumps(report_data, indent=2),
                filename="landmark_scotus_web_scraping_report.json",
                folder_id=self.target_folder_id
            )
            logger.info(f"\n📊 Detailed scraping report saved to Google Drive")
        except Exception as e:
            logger.error(f"Error saving report: {e}")


def main():
    """Main entry point."""
    scraper = WebScrapeLandmarkSCOTUS()
    scraper.scrape_all_landmark_cases()


if __name__ == "__main__":
    main()

