#!/usr/bin/env python3
"""
Create New Google Drive Folder with Clean 1782 Cases

This script creates a new folder and copies verified 1782 cases with proper filenames.
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

def create_clean_folder():
    """Create new folder and copy verified 1782 cases with proper names."""

    logger.info("Creating new clean Google Drive folder...")

    try:
        backup = GoogleDriveBackup()
        logger.info("Google Drive authentication successful")

        # Get the parent folder ID
        parent_folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'

        # Create new folder
        folder_metadata = {
            'name': 'The Matrix_1782_Discovery_Clean',
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }

        new_folder = backup.service.files().create(body=folder_metadata, fields='id').execute()
        new_folder_id = new_folder.get('id')
        logger.info(f"Created new folder: The Matrix_1782_Discovery_Clean (ID: {new_folder_id})")

        # Find the old subfolder
        query = f"'{parent_folder_id}' in parents"
        files = backup.service.files().list(q=query, fields='files(id, name, size, mimeType)').execute()

        old_subfolder_id = None
        for file in files.get('files', []):
            if file.get('mimeType') == 'application/vnd.google-apps.folder' and '1782_discovery' in file['name']:
                old_subfolder_id = file['id']
                logger.info(f"Found old subfolder: {file['name']} (ID: {old_subfolder_id})")
                break

        if not old_subfolder_id:
            logger.error("No old subfolder found")
            return

        # Get ALL files from old folder
        all_files = []
        page_token = None

        while True:
            files = backup.service.files().list(
                q=f"'{old_subfolder_id}' in parents",
                fields='files(id, name, size), nextPageToken',
                pageSize=100
            ).execute()

            all_files.extend(files.get('files', []))
            page_token = files.get('nextPageToken')

            if not page_token:
                break

        logger.info(f"Found {len(all_files)} files in old folder")

        copied_count = 0
        skipped_count = 0

        # Process each file
        for i, file in enumerate(all_files):
            file_id = file['id']
            file_name = file['name']

            try:
                # Download file content
                content = backup.service.files().get_media(fileId=file_id).execute()

                # Check if it's actually a 1782 case
                if is_1782_discovery_case(content):
                    # Parse the case to get clean name
                    case_data = json.loads(content)
                    case_name = case_data.get('caseName', file_name)

                    # Create clean filename
                    clean_name = "".join(c for c in case_name if c.isalnum() or c in (' ', '-', '_')).strip()
                    clean_name = clean_name[:100]  # Limit length
                    new_file_name = f"{clean_name}.json"

                    # Copy file to new folder
                    file_metadata = {
                        'name': new_file_name,
                        'parents': [new_folder_id]
                    }

                    backup.service.files().copy(fileId=file_id, body=file_metadata).execute()
                    copied_count += 1
                    logger.info(f"COPIED ({i+1}/{len(all_files)}): {new_file_name}")

                else:
                    skipped_count += 1
                    logger.info(f"SKIPPED ({i+1}/{len(all_files)}): {file_name} (not 1782)")

            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")

        logger.info(f"Copy complete: Copied {copied_count} verified 1782 cases, skipped {skipped_count} non-1782 cases")
        logger.info(f"New clean folder: https://drive.google.com/drive/folders/{new_folder_id}")

        return new_folder_id, copied_count

    except Exception as e:
        logger.error(f"Error: {e}")
        return None, 0

if __name__ == "__main__":
    print("Create New Clean Google Drive Folder")
    print("=" * 50)

    folder_id, copied_count = create_clean_folder()

    if folder_id:
        print(f"\nSUCCESS!")
        print(f"Created new clean folder with {copied_count} verified 1782 discovery cases")
        print(f"Access at: https://drive.google.com/drive/folders/{folder_id}")
        print(f"\nThe old folder can now be deleted if desired.")
    else:
        print(f"\nFAILED: Could not create clean folder")
