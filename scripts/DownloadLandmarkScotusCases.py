#!/usr/bin/env python3
"""
Download Missing Landmark SCOTUS §1782 Cases

Targets the three most important missing SCOTUS cases:
1. Intel Corp. v. Advanced Micro Devices, Inc. (2004) - 542 U.S. 241
2. ZF Automotive US, Inc. v. Luxshare, Ltd. (2022) - 596 U.S. ___
3. Servotronics, Inc. v. Rolls-Royce PLC (2021) - 593 U.S. ___
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import time

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CourtListener client directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "download_case_law",
    Path(__file__).parent.parent / "document_ingestion" / "download_case_law.py"
)
cl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl_module)
CourtListenerClient = cl_module.CourtListenerClient

# Import Google Drive backup module
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


class LandmarkSCOTUSDownloader:
    """Downloads the missing landmark SCOTUS §1782 cases."""

    def __init__(self):
        """Initialize the downloader."""
        self.cl_client = CourtListenerClient()
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
                "search_terms": [
                    "Intel Corp",
                    "Advanced Micro Devices",
                    "AMD",
                    "542 U.S. 241",
                    "03-724",
                    "Intel Corporation",
                    "Advanced Micro Devices Inc"
                ]
            },
            "ZF Automotive US, Inc. v. Luxshare, Ltd.": {
                "year": "2022",
                "citation": "596 U.S. ___",
                "docket": "21-401",
                "search_terms": [
                    "ZF Automotive",
                    "Luxshare",
                    "596 U.S.",
                    "21-401",
                    "ZF Automotive US",
                    "Luxshare Ltd"
                ]
            },
            "Servotronics, Inc. v. Rolls-Royce PLC": {
                "year": "2021",
                "citation": "593 U.S. ___",
                "docket": "20-794",
                "search_terms": [
                    "Servotronics",
                    "Rolls-Royce",
                    "593 U.S.",
                    "20-794",
                    "Servotronics Inc",
                    "Rolls-Royce PLC"
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

    def search_case_multiple_strategies(self, case_name: str, case_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Search for a case using multiple strategies."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Searching for: {case_name}")
        logger.info(f"Citation: {case_info['citation']}")
        logger.info(f"Year: {case_info['year']}")
        logger.info(f"{'='*60}")

        # Strategy 1: Search by citation
        logger.info("Strategy 1: Searching by citation...")
        try:
            response = self.cl_client.search_opinions(
                keywords=[case_info['citation']],
                limit=10,
                include_non_precedential=True
            )
            if response and response.get('results'):
                for result in response['results']:
                    if self._is_matching_case(result, case_name, case_info):
                        logger.info(f"✅ Found via citation: {result.get('case_name')}")
                        return result
        except Exception as e:
            logger.error(f"Citation search error: {e}")

        # Strategy 2: Search by docket number
        logger.info("Strategy 2: Searching by docket number...")
        try:
            response = self.cl_client.search_opinions(
                keywords=[case_info['docket']],
                limit=10,
                include_non_precedential=True
            )
            if response and response.get('results'):
                for result in response['results']:
                    if self._is_matching_case(result, case_name, case_info):
                        logger.info(f"✅ Found via docket: {result.get('case_name')}")
                        return result
        except Exception as e:
            logger.error(f"Docket search error: {e}")

        # Strategy 3: Search by year + Supreme Court
        logger.info("Strategy 3: Searching by year + Supreme Court...")
        try:
            response = self.cl_client.search_opinions(
                keywords=[case_info['year'], "Supreme Court"],
                limit=20,
                include_non_precedential=True
            )
            if response and response.get('results'):
                for result in response['results']:
                    if self._is_matching_case(result, case_name, case_info):
                        logger.info(f"✅ Found via year + SCOTUS: {result.get('case_name')}")
                        return result
        except Exception as e:
            logger.error(f"Year + SCOTUS search error: {e}")

        # Strategy 4: Search by individual parties
        logger.info("Strategy 4: Searching by individual parties...")
        for term in case_info['search_terms'][:3]:  # Try first 3 search terms
            try:
                logger.info(f"  Trying: {term}")
                response = self.cl_client.search_opinions(
                    keywords=[term],
                    limit=20,
                    include_non_precedential=True
                )
                if response and response.get('results'):
                    for result in response['results']:
                        if self._is_matching_case(result, case_name, case_info):
                            logger.info(f"✅ Found via party search: {result.get('case_name')}")
                            return result
            except Exception as e:
                logger.error(f"Party search error for {term}: {e}")

        # Strategy 5: Broad search with multiple terms
        logger.info("Strategy 5: Broad search with multiple terms...")
        try:
            response = self.cl_client.search_opinions(
                keywords=case_info['search_terms'][:2],  # Use first 2 terms
                limit=30,
                include_non_precedential=True
            )
            if response and response.get('results'):
                for result in response['results']:
                    if self._is_matching_case(result, case_name, case_info):
                        logger.info(f"✅ Found via broad search: {result.get('case_name')}")
                        return result
        except Exception as e:
            logger.error(f"Broad search error: {e}")

        logger.warning(f"❌ No match found for: {case_name}")
        return None

    def _is_matching_case(self, result: Dict[str, Any], case_name: str, case_info: Dict[str, Any]) -> bool:
        """Check if a search result matches the target case."""
        result_name = result.get('case_name', '').lower()
        result_citation = result.get('citation', '').lower()
        result_court = result.get('court', '').lower()

        # Must be Supreme Court
        if 'supreme court' not in result_court and 'scotus' not in result_court:
            return False

        # Check for citation match
        if case_info['citation'].lower() in result_citation:
            return True

        # Check for docket match
        if case_info['docket'] in result_citation:
            return True

        # Check for party name matches
        for term in case_info['search_terms']:
            if term.lower() in result_name:
                return True

        return False

    def download_case_content(self, case_data: Dict[str, Any], case_name: str) -> bool:
        """Download and save case content."""
        try:
            case_id = case_data.get('id')
            logger.info(f"Downloading content for: {case_name}")

            # Try to get full opinion
            opinion = self.cl_client.get_opinion_by_id(str(case_id))
            if opinion:
                # Create clean filename
                clean_name = case_name.replace(' ', '_').replace('.', '').replace(',', '')
                filename = f"001_SCOTUS_{clean_name}.json"

                # Upload to target folder
                uploaded_id = self.gdrive.upload_file_content(
                    content=json.dumps(opinion, indent=2),
                    filename=filename,
                    folder_id=self.target_folder_id
                )

                if uploaded_id:
                    self.downloaded_cases.append({
                        'case_name': case_name,
                        'case_id': case_id,
                        'filename': filename,
                        'court': opinion.get('court', 'Supreme Court'),
                        'date_filed': opinion.get('date_filed', 'Unknown'),
                        'citation': opinion.get('citation', 'Unknown')
                    })

                    logger.info(f"✅ Successfully downloaded: {case_name}")
                    return True
                else:
                    logger.error(f"❌ Failed to upload: {case_name}")
                    return False
            else:
                logger.warning(f"⚠️  Could not download full content for: {case_name}")
                return False

        except Exception as e:
            logger.error(f"❌ Error downloading {case_name}: {e}")
            return False

    def download_all_landmark_cases(self):
        """Download all landmark SCOTUS cases."""
        logger.info("Starting download of landmark SCOTUS §1782 cases...")

        # Find target folder
        if not self.find_target_folder():
            return False

        # Process each landmark case
        for case_name, case_info in self.landmark_cases.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {case_name}")
            logger.info(f"{'='*80}")

            # Search for the case
            case_data = self.search_case_multiple_strategies(case_name, case_info)

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
                logger.warning(f"❌ Could not find: {case_name}")

            # Add delay between searches
            time.sleep(2)

        # Generate summary report
        self.generate_summary_report()
        return True

    def generate_summary_report(self):
        """Generate a summary report."""
        logger.info(f"\n{'='*80}")
        logger.info("LANDMARK SCOTUS CASE DOWNLOAD SUMMARY")
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
                logger.info(f"    Court: {case_data.get('court', 'Supreme Court')}")

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
            logger.info(f"⚠️  No landmark SCOTUS cases were successfully downloaded")
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
                filename="landmark_scotus_download_report.json",
                folder_id=self.target_folder_id
            )
            logger.info(f"\n📊 Detailed download report saved to Google Drive")
        except Exception as e:
            logger.error(f"Error saving report: {e}")


def main():
    """Main entry point."""
    downloader = LandmarkSCOTUSDownloader()
    downloader.download_all_landmark_cases()


if __name__ == "__main__":
    main()

