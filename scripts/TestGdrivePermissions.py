#!/usr/bin/env python3
"""
Test Google Drive Permissions and Clean Up

This script tests the new Google Drive permissions and cleans up false positives.
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

def test_and_cleanup():
    """Test Google Drive permissions and clean up files."""

    logger.info("Testing Google Drive permissions...")

    try:
        # Initialize Google Drive backup (this will re-authenticate with new scopes)
        backup = GoogleDriveBackup()
        logger.info("✓ Google Drive authentication successful with full permissions")

        # Get the folder ID for 1782_discovery
        folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'

        # First, find the subfolder
        query = f"'{folder_id}' in parents"
        files = backup.service.files().list(q=query, fields='files(id, name, size, mimeType)').execute()

        subfolder_id = None
        for file in files.get('files', []):
            if file.get('mimeType') == 'application/vnd.google-apps.folder':
                subfolder_id = file['id']
                logger.info(f"Found subfolder: {file['name']} (ID: {subfolder_id})")
                break

        if not subfolder_id:
            logger.error("No subfolder found")
            return 0, 0

        # Get first 20 files to test
        files = backup.service.files().list(
            q=f"'{subfolder_id}' in parents",
            fields='files(id, name, size)',
            pageSize=20
        ).execute()

        file_list = files.get('files', [])
        logger.info(f"Testing with {len(file_list)} files...")

        kept_count = 0
        removed_count = 0

        for file in file_list:
            file_id = file['id']
            file_name = file['name']

            try:
                # Download file content
                content = backup.service.files().get_media(fileId=file_id).execute()

                # Check if it's actually a 1782 case
                if is_1782_discovery_case(content):
                    kept_count += 1
                    logger.info(f"✓ KEEP: {file_name}")
                else:
                    # Delete the file
                    backup.service.files().delete(fileId=file_id).execute()
                    removed_count += 1
                    logger.info(f"✗ REMOVED: {file_name}")

            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")

        logger.info(f"Test complete: Kept {kept_count} verified 1782 cases, removed {removed_count} false positives")
        return kept_count, removed_count

    except Exception as e:
        logger.error(f"Error with Google Drive: {e}")
        return 0, 0

if __name__ == "__main__":
    print("Test Google Drive Permissions and Clean Up")
    print("=" * 50)

    kept, removed = test_and_cleanup()

    print(f"\nTest result:")
    print(f"Kept: {kept} verified 1782 discovery cases")
    print(f"Removed: {removed} false positive cases")

    if kept > 0 or removed > 0:
        print(f"\n✓ Google Drive permissions working correctly!")
        print("You can now run the full cleanup script.")
    else:
        print(f"\n✗ Google Drive permissions may need adjustment.")


