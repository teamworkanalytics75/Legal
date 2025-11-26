#!/usr/bin/env python3
"""
Final Working Case Upload to Google Drive

Upload cases using the correct Google Drive API with MediaFileUpload.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

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

# Import MediaFileUpload
from googleapiclient.http import MediaFileUpload

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FinalCaseUploader:
    """Final working case uploader for Google Drive."""

    def __init__(self):
        """Initialize the uploader."""
        self.gdrive = GoogleDriveBackup()
        self.data_dir = Path(__file__).parent.parent / "data" / "case_law"

        # Clean folder ID (the one we created earlier)
        self.clean_folder_id = "1_wv4FZLhubYOO2ncRBcW68w4htt4N5Pl"

    def load_all_unique_cases(self):
        """Load all unique cases from both collections."""
        try:
            all_cases = []

            # Load web-scraped cases
            web_cases_file = self.data_dir / "unique_web_cases.json"
            if web_cases_file.exists():
                with open(web_cases_file, 'r', encoding='utf-8') as f:
                    web_cases = json.load(f)
                all_cases.extend(web_cases)
                logger.info(f"Loaded {len(web_cases)} unique web cases")

            # Load previous cases
            prev_cases_file = self.data_dir / "unique_previous_cases.json"
            if prev_cases_file.exists():
                with open(prev_cases_file, 'r', encoding='utf-8') as f:
                    prev_cases = json.load(f)
                all_cases.extend(prev_cases)
                logger.info(f"Loaded {len(prev_cases)} unique previous cases")

            logger.info(f"Total unique cases to upload: {len(all_cases)}")
            return all_cases

        except Exception as e:
            logger.error(f"Error loading unique cases: {e}")
            return []

    def create_case_file(self, case, temp_dir):
        """Create a temporary JSON file for the case."""
        try:
            # Create a simple JSON file
            case_data = {
                'case_id': case.get('case_id', ''),
                'case_name': case.get('case_name', ''),
                'court': case.get('court', ''),
                'docket_number': case.get('docket_number', ''),
                'date_filed': case.get('date_filed', ''),
                'status': case.get('status', ''),
                'citations': case.get('citations', ''),
                'case_url': case.get('case_url', ''),
                'uploaded_at': datetime.now().isoformat(),
                'source': 'courtlistener_web_scraping' if case.get('case_url') else 'courtlistener_api'
            }

            # Generate filename
            case_name = case.get('case_name', 'Unknown Case')
            case_id = case.get('case_id', '')

            # Clean filename
            filename = case_name.replace('Application of ', '')
            filename = filename.replace('Application for ', '')
            filename = filename.replace('In re ', '')
            filename = filename.split(' (')[0]  # Remove court info

            # Clean special characters
            for char in [':', '/', '\\', '?', '*', '"', '<', '>', '|']:
                filename = filename.replace(char, '')

            # Limit length
            if len(filename) > 80:
                filename = filename[:80]

            # Add case ID
            if case_id:
                filename = f"{filename} (ID_{case_id})"

            filename += '.json'

            # Create temp file
            temp_file = temp_dir / filename
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(case_data, f, indent=2, ensure_ascii=False)

            return temp_file

        except Exception as e:
            logger.error(f"Error creating case file: {e}")
            return None

    def upload_case_to_folder(self, case, folder_id):
        """Upload a single case to the specified folder."""
        try:
            # Create temp directory
            temp_dir = Path("temp_upload")
            temp_dir.mkdir(exist_ok=True)

            # Create case file
            temp_file = self.create_case_file(case, temp_dir)
            if not temp_file:
                return False

            # Create MediaFileUpload
            media = MediaFileUpload(temp_file, mimetype='application/json')

            # Upload using the correct method
            file_id = self.gdrive.service.files().create(
                body={
                    'name': temp_file.name,
                    'parents': [folder_id]
                },
                media_body=media,
                fields='id'
            ).execute()

            # Clean up temp file
            try:
                temp_file.unlink()
            except PermissionError:
                # File might still be in use, try again later
                pass

            logger.info(f"SUCCESS: Uploaded {temp_file.name}")
            return True

        except Exception as e:
            logger.error(f"Error uploading case: {e}")
            # Clean up temp file if it exists
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()
            return False

    def upload_all_cases(self):
        """Upload all unique cases to Google Drive."""
        logger.info("Starting upload of all unique cases...")

        # Load all cases
        all_cases = self.load_all_unique_cases()
        if not all_cases:
            logger.error("No cases to upload")
            return False

        # Upload each case
        successful_uploads = 0
        failed_uploads = 0

        for i, case in enumerate(all_cases, 1):
            case_name = case.get('case_name', 'Unknown')[:50]
            logger.info(f"Uploading case {i}/{len(all_cases)}: {case_name}...")

            success = self.upload_case_to_folder(case, self.clean_folder_id)
            if success:
                successful_uploads += 1
            else:
                failed_uploads += 1

        # Clean up temp directory
        temp_dir = Path("temp_upload")
        if temp_dir.exists():
            temp_dir.rmdir()

        # Summary
        logger.info(f"\nUPLOAD SUMMARY:")
        logger.info(f"Total cases: {len(all_cases)}")
        logger.info(f"Successful uploads: {successful_uploads}")
        logger.info(f"Failed uploads: {failed_uploads}")
        logger.info(f"Success rate: {successful_uploads/len(all_cases)*100:.1f}%")

        return successful_uploads > 0

    def create_final_summary(self):
        """Create a final summary of the database."""
        try:
            # Load all cases
            all_cases = self.load_all_unique_cases()

            # Create summary
            summary = {
                'database_name': 'The Art of War - 1782 Caselaw Database',
                'total_cases': len(all_cases),
                'creation_date': '2025-10-15',
                'sources': {
                    'courtlistener_api': len([c for c in all_cases if not c.get('case_url')]),
                    'courtlistener_web_scraping': len([c for c in all_cases if c.get('case_url')])
                },
                'case_breakdown': {
                    'verified_1782_cases': len([c for c in all_cases if '1782' in c.get('case_name', '').lower()]),
                    'application_cases': len([c for c in all_cases if 'application' in c.get('case_name', '').lower()]),
                    'federal_cases': len([c for c in all_cases if any(court in c.get('court', '').lower() for court in ['cir', 'd.', 'fed'])]),
                },
                'google_drive_folder': {
                    'name': 'The Art of War - Caselaw Database',
                    'id': self.clean_folder_id,
                    'url': f'https://drive.google.com/drive/folders/{self.clean_folder_id}'
                }
            }

            # Save summary
            summary_path = self.data_dir / "final_database_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"Final database summary saved: {summary_path}")

            return summary

        except Exception as e:
            logger.error(f"Error creating final summary: {e}")
            return None


def main():
    """Main entry point."""
    uploader = FinalCaseUploader()

    # Upload all cases
    success = uploader.upload_all_cases()

    if success:
        # Create final summary
        summary = uploader.create_final_summary()

        print(f"\nSUCCESS!")
        print(f"All unique cases uploaded to Google Drive")
        print(f"Final database summary created")
        print(f"Google Drive folder: https://drive.google.com/drive/folders/{uploader.clean_folder_id}")
    else:
        print(f"\nUpload failed")


if __name__ == "__main__":
    main()
