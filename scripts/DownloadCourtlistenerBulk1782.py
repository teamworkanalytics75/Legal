#!/usr/bin/env python3
"""
CourtListener Bulk Data Downloader for Section 1782 Cases

Downloads CourtListener bulk data files and processes them to find all Section 1782 cases.
This bypasses API limitations and gives access to the complete database.
"""

import sys
import json
import logging
import csv
import gzip
import requests
from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime
import re
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class CourtListenerBulkDownloader:
    """Download and process CourtListener bulk data for Section 1782 cases."""

    def __init__(self):
        self.base_url = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data"
        self.data_dir = Path("data/courtlistener_bulk")
        self.data_dir.mkdir(exist_ok=True)
        self.section_1782_cases = []
        self.processed_files = set()

    def get_latest_bulk_data_urls(self) -> Dict[str, str]:
        """Get URLs for the latest bulk data files."""
        logger.info("🔍 Finding latest CourtListener bulk data files...")

        # Try different recent dates (bulk data is generated monthly)
        dates_to_try = [
            "2024-12-31",  # December 2024
            "2024-11-30",  # November 2024
            "2024-10-31",  # October 2024
            "2024-09-30",  # September 2024
            "2024-08-31",  # August 2024
        ]

        urls = {}
        for date in dates_to_try:
            try:
                # Test if opinions.csv exists for this date
                test_url = f"{self.base_url}/{date}/opinions.csv"
                response = requests.head(test_url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ Found bulk data for {date}")
                    urls = {
                        'opinions': f"{self.base_url}/{date}/opinions.csv",
                        'dockets': f"{self.base_url}/{date}/dockets.csv",
                        'courts': f"{self.base_url}/{date}/courts.csv",
                        'schema': f"{self.base_url}/{date}/schema.sql"
                    }
                    break
                else:
                    logger.info(f"❌ No bulk data for {date}")
            except Exception as e:
                logger.warning(f"⚠️ Error checking {date}: {e}")
                continue

        if not urls:
            logger.error("❌ Could not find any recent bulk data files")
            return {}

        logger.info(f"📊 Using bulk data URLs: {urls}")
        return urls

    def download_bulk_file(self, url: str, filename: str) -> bool:
        """Download a bulk data file."""
        filepath = self.data_dir / filename

        if filepath.exists():
            logger.info(f"📁 File already exists: {filename}")
            return True

        try:
            logger.info(f"⬇️ Downloading {filename}...")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"✅ Downloaded {filename}")
            return True

        except Exception as e:
            logger.error(f"❌ Error downloading {filename}: {e}")
            return False

    def process_opinions_csv(self, filepath: Path) -> List[Dict]:
        """Process opinions CSV to find Section 1782 cases."""
        logger.info(f"🔍 Processing opinions from {filepath.name}...")

        section_1782_cases = []
        section_1782_keywords = [
            'section 1782',
            '28 u.s.c. 1782',
            '28 usc 1782',
            '1782',
            'judicial assistance',
            'foreign discovery',
            'foreign tribunal',
            'intel corp',
            'brandi-dohrn',
            'euromepa',
            'macquarie',
            'del valle ruiz'
        ]

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in tqdm(reader, desc="Scanning opinions"):
                    # Check if this opinion contains Section 1782 keywords
                    plain_text = row.get('plain_text', '').lower()
                    html = row.get('html', '').lower()
                    html_lawbox = row.get('html_lawbox', '').lower()

                    # Combine all text fields
                    all_text = f"{plain_text} {html} {html_lawbox}"

                    # Check for Section 1782 keywords
                    found_keywords = []
                    for keyword in section_1782_keywords:
                        if keyword.lower() in all_text:
                            found_keywords.append(keyword)

                    if found_keywords:
                        case_data = {
                            'id': row.get('id'),
                            'cluster_id': row.get('cluster_id'),
                            'case_name': row.get('caseName'),
                            'absolute_url': row.get('absolute_url'),
                            'date_filed': row.get('dateFiled'),
                            'date_created': row.get('date_created'),
                            'date_modified': row.get('date_modified'),
                            'download_url': row.get('download_url'),
                            'plain_text': plain_text[:1000] + "..." if len(plain_text) > 1000 else plain_text,
                            'found_keywords': found_keywords,
                            'cite_count': row.get('citeCount', 0),
                            'page_count': row.get('page_count', 0),
                            'extracted_by_ocr': row.get('extracted_by_ocr', False),
                            'per_curiam': row.get('per_curiam', False)
                        }
                        section_1782_cases.append(case_data)

                        if len(section_1782_cases) % 100 == 0:
                            logger.info(f"📊 Found {len(section_1782_cases)} Section 1782 cases so far...")

        except Exception as e:
            logger.error(f"❌ Error processing {filepath}: {e}")
            return []

        logger.info(f"🎉 Found {len(section_1782_cases)} Section 1782 cases!")
        return section_1782_cases

    def enrich_with_docket_data(self, cases: List[Dict], dockets_file: Path) -> List[Dict]:
        """Enrich cases with docket information."""
        logger.info(f"🔍 Enriching cases with docket data from {dockets_file.name}...")

        # Load docket data
        dockets = {}
        try:
            with open(dockets_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dockets[row['id']] = row
        except Exception as e:
            logger.error(f"❌ Error loading dockets: {e}")
            return cases

        # Enrich cases with docket data
        enriched_cases = []
        for case in cases:
            cluster_id = case.get('cluster_id')
            if cluster_id and cluster_id in dockets:
                docket_data = dockets[cluster_id]
                case.update({
                    'docket_number': docket_data.get('docket_number'),
                    'court_id': docket_data.get('court_id'),
                    'case_name_full': docket_data.get('case_name'),
                    'date_filed_docket': docket_data.get('date_filed'),
                    'date_terminated': docket_data.get('date_terminated'),
                    'nature_of_suit': docket_data.get('nature_of_suit'),
                    'jurisdiction': docket_data.get('jurisdiction')
                })
            enriched_cases.append(case)

        logger.info(f"✅ Enriched {len(enriched_cases)} cases with docket data")
        return enriched_cases

    def save_results(self, cases: List[Dict]):
        """Save Section 1782 cases to files."""
        logger.info(f"💾 Saving {len(cases)} Section 1782 cases...")

        # Save detailed JSON
        json_file = self.data_dir / "section_1782_cases_detailed.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)

        # Save summary CSV
        csv_file = self.data_dir / "section_1782_cases_summary.csv"
        if cases:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'id', 'case_name', 'docket_number', 'court_id', 'date_filed',
                    'cite_count', 'page_count', 'found_keywords', 'download_url'
                ])
                writer.writeheader()
                for case in cases:
                    writer.writerow({
                        'id': case.get('id'),
                        'case_name': case.get('case_name'),
                        'docket_number': case.get('docket_number'),
                        'court_id': case.get('court_id'),
                        'date_filed': case.get('date_filed'),
                        'cite_count': case.get('cite_count'),
                        'page_count': case.get('page_count'),
                        'found_keywords': ', '.join(case.get('found_keywords', [])),
                        'download_url': case.get('download_url')
                    })

        logger.info(f"✅ Saved results to {json_file} and {csv_file}")

    def download_and_process(self):
        """Main method to download and process bulk data."""
        logger.info("🚀 Starting CourtListener bulk data download and processing...")

        # Get latest bulk data URLs
        urls = self.get_latest_bulk_data_urls()
        if not urls:
            logger.error("❌ Could not find bulk data URLs")
            return

        # Download bulk data files
        downloaded_files = {}
        for file_type, url in urls.items():
            filename = f"{file_type}.csv" if file_type != 'schema' else f"{file_type}.sql"
            if self.download_bulk_file(url, filename):
                downloaded_files[file_type] = self.data_dir / filename

        if 'opinions' not in downloaded_files:
            logger.error("❌ Could not download opinions file")
            return

        # Process opinions to find Section 1782 cases
        section_1782_cases = self.process_opinions_csv(downloaded_files['opinions'])

        if not section_1782_cases:
            logger.warning("⚠️ No Section 1782 cases found in bulk data")
            return

        # Enrich with docket data if available
        if 'dockets' in downloaded_files:
            section_1782_cases = self.enrich_with_docket_data(section_1782_cases, downloaded_files['dockets'])

        # Save results
        self.save_results(section_1782_cases)

        logger.info(f"🎉 Successfully processed bulk data and found {len(section_1782_cases)} Section 1782 cases!")

        return section_1782_cases


def main():
    """Main function."""
    logger.info("🎯 Starting CourtListener Bulk Data Download for Section 1782 Cases")

    try:
        downloader = CourtListenerBulkDownloader()
        cases = downloader.download_and_process()

        if cases:
            logger.info(f"🎉 Success! Found {len(cases)} Section 1782 cases in bulk data")
        else:
            logger.warning("⚠️ No Section 1782 cases found")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
