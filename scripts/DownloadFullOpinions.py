#!/usr/bin/env python3
"""
Download Full Case Opinions

Iterate through all cases in Google Drive folder and download full opinion text.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

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

# Import CourtListener client directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "download_case_law",
    Path(__file__).parent.parent / "document_ingestion" / "download_case_law.py"
)
cl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl_module)
CourtListenerClient = cl_module.CourtListenerClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FullOpinionDownloader:
    """Download full case opinions for all cases in Google Drive folder."""

    def __init__(self):
        """Initialize the downloader."""
        self.gdrive = GoogleDriveBackup()
        self.cl_client = CourtListenerClient()

        # Clean folder ID
        self.clean_folder_id = "1_wv4FZLhubYOO2ncRBcW68w4htt4N5Pl"

        # Track progress
        self.processed_cases = []
        self.failed_cases = []
        self.downloaded_opinions = []

    def get_case_id_from_filename(self, filename: str) -> Optional[str]:
        """Extract case ID from filename."""
        try:
            # Look for patterns like (ID_12345) or (ID_Misc. No. 2013-0987)
            import re

            # Pattern for numeric IDs
            numeric_match = re.search(r'\(ID_(\d+)\)', filename)
            if numeric_match:
                return numeric_match.group(1)

            # Pattern for misc case IDs
            misc_match = re.search(r'\(ID_(Misc\. No\. \d{4}-\d+)\)', filename)
            if misc_match:
                return misc_match.group(1)

            # Pattern for circuit case IDs
            circuit_match = re.search(r'\(ID_(\d{2}-\d+)\)', filename)
            if circuit_match:
                return circuit_match.group(1)

            logger.warning(f"Could not extract case ID from filename: {filename}")
            return None

        except Exception as e:
            logger.error(f"Error extracting case ID from {filename}: {e}")
            return None

    def download_case_opinion(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Download full case opinion by case ID."""
        try:
            logger.info(f"Downloading opinion for case ID: {case_id}")

            # Try to get the opinion by ID
            opinion = self.cl_client.get_opinion_by_id(case_id)
            if opinion:
                logger.info(f"Successfully downloaded opinion for case ID: {case_id}")
                return opinion
            else:
                logger.warning(f"No opinion found for case ID: {case_id}")
                return None

        except Exception as e:
            logger.error(f"Error downloading opinion for case ID {case_id}: {e}")
            return None

    def download_case_by_search(self, case_name: str) -> Optional[Dict[str, Any]]:
        """Try to find and download case by searching for case name."""
        try:
            logger.info(f"Searching for case: {case_name}")

            # Clean up case name for search
            search_terms = []

            # Extract key terms from case name
            if "In re" in case_name:
                search_terms.append("In re")
            if "Application" in case_name:
                search_terms.append("Application")
            if "1782" in case_name:
                search_terms.append("1782")

            # Add company/person names
            import re
            # Look for company names (capitalized words)
            company_matches = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', case_name)
            search_terms.extend(company_matches[:2])  # Take first 2 company names

            if not search_terms:
                search_terms = [case_name.split()[0]]  # Fallback to first word

            logger.info(f"Search terms: {search_terms}")

            # Search for the case
            response = self.cl_client.search_opinions(
                keywords=search_terms,
                limit=10,
                include_non_precedential=True
            )

            if response and response.get('results'):
                # Look for best match
                for result in response['results']:
                    result_name = result.get('case_name', '').lower()
                    if any(term.lower() in result_name for term in search_terms):
                        logger.info(f"Found potential match: {result.get('case_name')}")
                        return result

            logger.warning(f"No matching case found for: {case_name}")
            return None

        except Exception as e:
            logger.error(f"Error searching for case {case_name}: {e}")
            return None

    def process_single_case(self, file_metadata: Dict[str, Any]) -> bool:
        """Process a single case file."""
        filename = file_metadata['name']
        file_id = file_metadata['id']

        logger.info(f"Processing case: {filename}")

        try:
            # Extract case ID
            case_id = self.get_case_id_from_filename(filename)

            opinion_data = None

            if case_id:
                # Try direct download by case ID
                opinion_data = self.download_case_opinion(case_id)

            if not opinion_data:
                # Fallback to search by case name
                case_name = filename.replace('.json', '').replace('(ID_' + (case_id or 'unknown') + ')', '').strip()
                opinion_data = self.download_case_by_search(case_name)

            if opinion_data:
                # Save the full opinion data
                opinion_filename = f"full_opinion_{filename}"
                self.gdrive.upload_file_content(
                    content=json.dumps(opinion_data, indent=2),
                    filename=opinion_filename,
                    folder_id=self.clean_folder_id
                )

                self.downloaded_opinions.append({
                    'original_file': filename,
                    'opinion_file': opinion_filename,
                    'case_id': case_id,
                    'case_name': opinion_data.get('case_name', 'Unknown'),
                    'court': opinion_data.get('court', 'Unknown'),
                    'date_filed': opinion_data.get('date_filed', 'Unknown')
                })

                logger.info(f"✅ Successfully downloaded full opinion for: {filename}")
                return True
            else:
                self.failed_cases.append({
                    'filename': filename,
                    'case_id': case_id,
                    'reason': 'No opinion found'
                })
                logger.warning(f"❌ Failed to download opinion for: {filename}")
                return False

        except Exception as e:
            self.failed_cases.append({
                'filename': filename,
                'case_id': case_id,
                'reason': f'Error: {str(e)}'
            })
            logger.error(f"❌ Error processing {filename}: {e}")
            return False

    def download_all_opinions(self):
        """Download full opinions for all cases in the folder."""
        logger.info("Starting full opinion download process...")

        # Get all files from the folder
        query = f"'{self.clean_folder_id}' in parents and trashed=false"
        results = self.gdrive.service.files().list(
            q=query,
            fields='files(id, name, size, createdTime)'
        ).execute()
        files = results.get('files', [])
        logger.info(f"Found {len(files)} files to process")

        # Process each file
        for i, file_metadata in enumerate(files, 1):
            filename = file_metadata['name']

            # Skip if it's already a full opinion file
            if filename.startswith('full_opinion_'):
                logger.info(f"Skipping {filename} (already a full opinion)")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing case {i}/{len(files)}: {filename}")
            logger.info(f"{'='*60}")

            success = self.process_single_case(file_metadata)
            self.processed_cases.append({
                'filename': filename,
                'success': success,
                'index': i
            })

        # Generate summary report
        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate a summary report of the download process."""
        logger.info(f"\n{'='*80}")
        logger.info("FULL OPINION DOWNLOAD SUMMARY")
        logger.info(f"{'='*80}")

        total_cases = len(self.processed_cases)
        successful_downloads = len(self.downloaded_opinions)
        failed_downloads = len(self.failed_cases)

        logger.info(f"Total cases processed: {total_cases}")
        logger.info(f"Successful downloads: {successful_downloads}")
        logger.info(f"Failed downloads: {failed_downloads}")
        logger.info(f"Success rate: {(successful_downloads/total_cases)*100:.1f}%")

        if self.downloaded_opinions:
            logger.info(f"\n✅ Successfully Downloaded Opinions:")
            for opinion in self.downloaded_opinions[:10]:  # Show first 10
                logger.info(f"  - {opinion['case_name']} ({opinion['court']})")
            if len(self.downloaded_opinions) > 10:
                logger.info(f"  ... and {len(self.downloaded_opinions) - 10} more")

        if self.failed_cases:
            logger.info(f"\n❌ Failed Downloads:")
            for failure in self.failed_cases[:10]:  # Show first 10
                logger.info(f"  - {failure['filename']}: {failure['reason']}")
            if len(self.failed_cases) > 10:
                logger.info(f"  ... and {len(self.failed_cases) - 10} more failures")

        # Save detailed report
        report_data = {
            'summary': {
                'total_cases': total_cases,
                'successful_downloads': successful_downloads,
                'failed_downloads': failed_downloads,
                'success_rate': (successful_downloads/total_cases)*100
            },
            'downloaded_opinions': self.downloaded_opinions,
            'failed_cases': self.failed_cases,
            'processed_cases': self.processed_cases
        }

        # Upload report to Google Drive
        try:
            self.gdrive.upload_file_content(
                content=json.dumps(report_data, indent=2),
                filename="full_opinion_download_report.json",
                folder_id=self.clean_folder_id
            )
            logger.info(f"\n📊 Detailed report saved to Google Drive")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        logger.info(f"\n🎯 Next steps:")
        logger.info(f"1. Check Google Drive folder for full opinion files")
        logger.info(f"2. Review failed cases and retry if needed")
        logger.info(f"3. Use downloaded opinions for analysis")


def main():
    """Main entry point."""
    downloader = FullOpinionDownloader()
    downloader.download_all_opinions()


if __name__ == "__main__":
    main()
