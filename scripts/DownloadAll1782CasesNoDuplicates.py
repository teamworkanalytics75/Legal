#!/usr/bin/env python3
"""
Download All Section 1782 Cases (No Duplicates) and Upload to Google Drive

Downloads all unique Section 1782 cases from CourtListener API and uploads them to Google Drive
with duplicate detection and live progress tracking.
"""

import sys
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set
import requests
from tqdm import tqdm

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

# CourtListener API configuration
COURTLISTENER_API_TOKEN = "1d6baa6881aa5b32acecff70866e5901a8c5bc18"
COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
HEADERS = {
    'Authorization': f'Token {COURTLISTENER_API_TOKEN}',
    'User-Agent': 'The MatrixLegalAI/2.0 (Research/Education)'
}


class Section1782Downloader:
    """Download all unique Section 1782 cases from CourtListener and upload to Google Drive."""

    def __init__(self):
        self.backup = GoogleDriveBackup()
        self.cases_dir = Path("data/section_1782_cases")
        self.cases_dir.mkdir(exist_ok=True)
        self.downloaded_cases = []
        self.failed_cases = []
        self.processed_case_ids = set()  # Track processed case IDs to avoid duplicates
        self.existing_files = set()  # Track existing files to avoid re-uploading

        # Load existing progress
        self.load_existing_progress()

    def load_existing_progress(self):
        """Load existing downloaded cases to avoid duplicates."""
        logger.info("🔍 Checking for existing downloaded cases...")

        # Check existing JSON files
        for json_file in self.cases_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)
                    if 'opinion_id' in case_data:
                        self.processed_case_ids.add(case_data['opinion_id'])
                    self.existing_files.add(json_file.name)
            except Exception as e:
                logger.warning(f"⚠️ Error reading existing file {json_file}: {e}")

        logger.info(f"📊 Found {len(self.processed_case_ids)} already processed cases")
        logger.info(f"📁 Found {len(self.existing_files)} existing files")

    def get_case_hash(self, case: Dict) -> str:
        """Create a unique hash for a case to detect duplicates."""
        # Use case name, citation, and date as the basis for uniqueness
        case_name = case.get('caseName', '')
        citation = case.get('citation', [])
        date_filed = case.get('dateFiled', '')

        # Create hash from key identifying fields
        hash_string = f"{case_name}|{citation}|{date_filed}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def get_all_section_1782_cases(self) -> List[Dict]:
        """Get all unique Section 1782 cases from CourtListener API."""
        logger.info("🔍 Fetching all Section 1782 cases from CourtListener...")

        all_cases = []
        seen_hashes = set()
        page = 1
        per_page = 100  # API limit

        while True:
            url = f"{COURTLISTENER_BASE_URL}/search/"
            params = {
                'q': 'section 1782',
                'format': 'json',
                'order_by': 'score desc',
                'stat_Precedential': 'on',
                'stat_Non-Precedential': 'on',
                'page': page,
                'per_page': per_page
            }

            try:
                response = requests.get(url, headers=HEADERS, params=params)
                response.raise_for_status()
                data = response.json()

                cases = data.get('results', [])
                if not cases:
                    break

                # Filter out duplicates
                unique_cases = []
                for case in cases:
                    case_hash = self.get_case_hash(case)
                    if case_hash not in seen_hashes:
                        seen_hashes.add(case_hash)
                        unique_cases.append(case)

                all_cases.extend(unique_cases)
                logger.info(f"📄 Page {page}: {len(cases)} total, {len(unique_cases)} unique (Total unique: {len(all_cases)})")

                page += 1

                # Rate limiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"❌ Error fetching page {page}: {e}")
                break

        logger.info(f"🎉 Total unique Section 1782 cases found: {len(all_cases)}")
        return all_cases

    def download_case_details(self, case: Dict) -> Optional[Dict]:
        """Download detailed case information including opinion text."""
        try:
            if not case.get('opinions') or len(case['opinions']) == 0:
                return None

            opinion_id = case['opinions'][0]['id']

            # Skip if already processed
            if opinion_id in self.processed_case_ids:
                return None

            opinion_url = f"{COURTLISTENER_BASE_URL}/opinions/{opinion_id}/"

            response = requests.get(opinion_url, headers=HEADERS)
            response.raise_for_status()
            opinion_data = response.json()

            # Combine case and opinion data
            case_details = {
                'case_name': case.get('caseName', ''),
                'citation': case.get('citation', []),
                'cite_count': case.get('citeCount', 0),
                'date_filed': case.get('dateFiled', ''),
                'court': case.get('court', ''),
                'resource_uri': case.get('resource_uri', ''),
                'absolute_url': case.get('absolute_url', ''),
                'plain_text': opinion_data.get('plain_text', ''),
                'download_url': opinion_data.get('download_url', ''),
                'html': opinion_data.get('html', ''),
                'html_lawbox': opinion_data.get('html_lawbox', ''),
                'html_columbia': opinion_data.get('html_columbia', ''),
                'html_anon_2020': opinion_data.get('html_anon_2020', ''),
                'xml_harvard': opinion_data.get('xml_harvard', ''),
                'html_with_citations': opinion_data.get('html_with_citations', ''),
                'extracted_by_ocr': opinion_data.get('extracted_by_ocr', False),
                'per_curiam': opinion_data.get('per_curiam', False),
                'date_created': opinion_data.get('date_created', ''),
                'date_modified': opinion_data.get('date_modified', ''),
                'sha1': opinion_data.get('sha1', ''),
                'page_count': opinion_data.get('page_count', 0),
                'download_count': opinion_data.get('download_count', 0),
                'opinion_id': opinion_id,
                'case_hash': self.get_case_hash(case)
            }

            return case_details

        except Exception as e:
            logger.error(f"❌ Error downloading case details for {case.get('caseName', 'Unknown')}: {e}")
            return None

    def save_case_to_file(self, case_details: Dict) -> str:
        """Save case details to a JSON file."""
        # Create safe filename
        case_name = case_details.get('case_name', 'Unknown_Case')
        safe_filename = "".join(c for c in case_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = safe_filename[:100]  # Limit length

        # Add citation if available
        if case_details.get('citation'):
            citation = case_details['citation'][0] if isinstance(case_details['citation'], list) else case_details['citation']
            safe_filename += f"_{citation}"

        filename = f"{safe_filename}.json"
        filepath = self.cases_dir / filename

        # Handle duplicate filenames
        counter = 1
        original_filepath = filepath
        while filepath.exists():
            name_part = original_filepath.stem
            filepath = self.cases_dir / f"{name_part}_{counter}.json"
            counter += 1

        # Save case data
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(case_details, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def upload_case_to_gdrive(self, filepath: str, case_details: Dict) -> bool:
        """Upload case file to Google Drive."""
        try:
            file_id = self.backup.upload_file(
                Path(filepath),
                topic="Section 1782 Cases"
            )
            return file_id is not None
        except Exception as e:
            logger.error(f"❌ Error uploading {filepath} to Google Drive: {e}")
            return False

    def download_all_cases(self):
        """Download all unique Section 1782 cases and upload to Google Drive."""
        logger.info("🚀 Starting download of all unique Section 1782 cases...")

        # Get all cases
        all_cases = self.get_all_section_1782_cases()

        if not all_cases:
            logger.error("❌ No cases found!")
            return

        # Filter out already processed cases
        new_cases = []
        for case in all_cases:
            if case.get('opinions') and len(case['opinions']) > 0:
                opinion_id = case['opinions'][0]['id']
                if opinion_id not in self.processed_case_ids:
                    new_cases.append(case)

        logger.info(f"📊 Total cases: {len(all_cases)}")
        logger.info(f"📊 Already processed: {len(all_cases) - len(new_cases)}")
        logger.info(f"📊 New cases to download: {len(new_cases)}")

        if not new_cases:
            logger.info("🎉 All cases already downloaded!")
            return

        # Create progress bar
        progress_bar = tqdm(total=len(new_cases), desc="Downloading new cases", unit="case")

        for i, case in enumerate(new_cases):
            try:
                # Download case details
                case_details = self.download_case_details(case)

                if case_details:
                    # Save to file
                    filepath = self.save_case_to_file(case_details)

                    # Upload to Google Drive
                    upload_success = self.upload_case_to_gdrive(filepath, case_details)

                    if upload_success:
                        self.downloaded_cases.append(case_details)
                        self.processed_case_ids.add(case_details['opinion_id'])
                        logger.info(f"✅ Case {i+1}/{len(new_cases)}: {case_details.get('case_name', 'Unknown')}")
                    else:
                        self.failed_cases.append(case)
                        logger.warning(f"⚠️ Upload failed for case {i+1}: {case.get('caseName', 'Unknown')}")
                else:
                    self.failed_cases.append(case)
                    logger.warning(f"⚠️ No details for case {i+1}: {case.get('caseName', 'Unknown')}")

                # Update progress
                progress_bar.update(1)

                # Rate limiting
                time.sleep(0.2)

            except Exception as e:
                logger.error(f"❌ Error processing case {i+1}: {e}")
                self.failed_cases.append(case)
                progress_bar.update(1)

        progress_bar.close()

        # Save summary
        self.save_download_summary()

        logger.info(f"🎉 Download complete!")
        logger.info(f"✅ Successfully downloaded: {len(self.downloaded_cases)} cases")
        logger.info(f"❌ Failed: {len(self.failed_cases)} cases")
        logger.info(f"📊 Total processed: {len(self.processed_case_ids)} cases")

    def save_download_summary(self):
        """Save download summary to file."""
        summary = {
            'total_cases_found': len(self.processed_case_ids) + len(self.failed_cases),
            'successfully_downloaded': len(self.downloaded_cases),
            'failed_downloads': len(self.failed_cases),
            'total_processed': len(self.processed_case_ids),
            'download_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'downloaded_cases': [
                {
                    'case_name': case.get('case_name', ''),
                    'citation': case.get('citation', []),
                    'court': case.get('court', ''),
                    'date_filed': case.get('date_filed', ''),
                    'cite_count': case.get('cite_count', 0),
                    'opinion_id': case.get('opinion_id', '')
                }
                for case in self.downloaded_cases
            ],
            'failed_cases': [
                {
                    'case_name': case.get('caseName', ''),
                    'citation': case.get('citation', []),
                    'court': case.get('court', ''),
                    'date_filed': case.get('dateFiled', '')
                }
                for case in self.failed_cases
            ]
        }

        summary_file = Path("data/section_1782_download_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Upload summary to Google Drive
        try:
            self.backup.upload_file(summary_file, topic="Section 1782 Dataset")
            logger.info("📄 Uploaded download summary to Google Drive")
        except Exception as e:
            logger.error(f"❌ Error uploading summary: {e}")


def main():
    """Main function."""
    logger.info("🎯 Starting Section 1782 Complete Dataset Download (No Duplicates)")

    try:
        downloader = Section1782Downloader()
        downloader.download_all_cases()

        logger.info("🎉 All downloads completed!")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

