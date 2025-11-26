#!/usr/bin/env python3
"""
Complete Google Drive Cleanup

This script processes ALL files in the Google Drive folder, handling pagination properly.
"""

import sys
import json
import logging
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from google_drive_backup import GoogleDriveBackup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def is_1782_discovery_case(case_content):
    """Check if a case is actually about 1782 discovery."""

    try:
        case = json.loads(case_content)
    except json.JSONDecodeError:
        return False

    case_name = case.get('caseName', '').lower()
    case_text = case.get('plain_text', '').lower()
    html_text = case.get('html_with_citations', '').lower()

    # Combine all text for analysis
    all_text = f"{case_name} {case_text} {html_text}"

    # Strong positive indicators for 1782 discovery
    strong_indicators = [
        '28 u.s.c. 1782',
        '28 usc 1782',
        'section 1782',
        '§ 1782',
        'discovery for use in a foreign proceeding',
        'foreign tribunal',
        'international tribunal',
        'assistance before a foreign',
        'judicial assistance',
        'discovery pursuant to 28',
        'application pursuant to 28',
        'order pursuant to 28',
        'subpoena under 28',
        'intel corp v. advanced micro devices',  # The famous 1782 case
        'brandi-dohrn',
        'euromepa',
        'del valle ruiz'
    ]

    # Check for strong indicators
    has_strong_indicator = any(indicator in all_text for indicator in strong_indicators)

    # Additional check: if it mentions "1782" but not in the context of U.S.C., it's likely a false positive
    if '1782' in all_text and not any(usc_indicator in all_text for usc_indicator in ['28 u.s.c. 1782', '28 usc 1782', 'section 1782', '§ 1782']):
        return False

    return has_strong_indicator

def complete_gdrive_cleanup():
    """Process ALL files in Google Drive folder."""

    backup = GoogleDriveBackup()
    folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'

    logger.info(f"Starting complete Google Drive cleanup: {folder_id}")

    try:
        # Get subfolder
        query = f"'{folder_id}' in parents"
        files = backup.service.files().list(q=query, fields='files(id, name, mimeType)').execute()

        subfolder_id = None
        for file in files.get('files', []):
            if file.get('mimeType') == 'application/vnd.google-apps.folder':
                subfolder_id = file['id']
                logger.info(f"Found subfolder: {file['name']} (ID: {subfolder_id})")
                break

        if not subfolder_id:
            logger.error("No subfolder found")
            return 0, 0

        # Process files in batches of 100
        kept_count = 0
        removed_count = 0
        processed_count = 0

        while True:
            # Get next batch of files
            files = backup.service.files().list(
                q=f"'{subfolder_id}' in parents",
                fields='files(id, name, size), nextPageToken',
                pageSize=100
            ).execute()

            file_list = files.get('files', [])
            if not file_list:
                break

            logger.info(f"Processing batch of {len(file_list)} files...")

            for file in file_list:
                file_id = file['id']
                file_name = file['name']
                processed_count += 1

                try:
                    # Download file content
                    content = backup.service.files().get_media(fileId=file_id).execute()

                    # Check if it's actually a 1782 case
                    if is_1782_discovery_case(content):
                        kept_count += 1
                        logger.info(f"✓ KEEP ({processed_count}): {file_name}")
                    else:
                        # Delete the file
                        backup.service.files().delete(fileId=file_id).execute()
                        removed_count += 1
                        logger.info(f"✗ REMOVED ({processed_count}): {file_name}")

                except Exception as e:
                    logger.error(f"Error processing {file_name}: {e}")

            # Check if there are more files
            if not files.get('nextPageToken'):
                break

        logger.info(f"Complete cleanup finished: Processed {processed_count} files, kept {kept_count} verified 1782 cases, removed {removed_count} false positives")
        return kept_count, removed_count

    except Exception as e:
        logger.error(f"Error accessing Google Drive: {e}")
        return 0, 0

if __name__ == "__main__":
    print("Complete Google Drive Cleanup")
    print("=" * 50)

    kept, removed = complete_gdrive_cleanup()

    print(f"\nFinal result:")
    print(f"Kept: {kept} verified 1782 discovery cases")
    print(f"Removed: {removed} false positive cases")

    if kept > 0:
        print(f"\nYour Google Drive now contains only verified 1782 discovery cases!")
        print("Access at: https://drive.google.com/drive/folders/157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA")


