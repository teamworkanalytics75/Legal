#!/usr/bin/env python3
"""
Check Google Drive and Copy Files

This script checks what folders exist and copies verified 1782 cases.
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

def check_and_copy():
    """Check Google Drive folders and copy verified files."""

    try:
        backup = GoogleDriveBackup()
        logger.info("Google Drive authentication successful")

        parent_folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'

        # List folders in parent
        query = f"'{parent_folder_id}' in parents"
        files = backup.service.files().list(q=query, fields='files(id, name, mimeType)').execute()

        logger.info("Folders in parent:")
        folders = {}
        for file in files.get('files', []):
            if file.get('mimeType') == 'application/vnd.google-apps.folder':
                folders[file['name']] = file['id']
                logger.info(f"- {file['name']} (ID: {file['id']})")

        # Check if clean folder exists
        clean_folder_id = None
        for name, folder_id in folders.items():
            if 'clean' in name.lower():
                clean_folder_id = folder_id
                logger.info(f"Found clean folder: {name} (ID: {clean_folder_id})")
                break

        if not clean_folder_id:
            logger.info("Creating new clean folder...")
            folder_metadata = {
                'name': 'The Matrix_1782_Discovery_Clean',
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_folder_id]
            }
            new_folder = backup.service.files().create(body=folder_metadata, fields='id').execute()
            clean_folder_id = new_folder.get('id')
            logger.info(f"Created clean folder: {clean_folder_id}")

        # Find old folder
        old_folder_id = None
        for name, folder_id in folders.items():
            if '1782_discovery' in name.lower() and 'clean' not in name.lower():
                old_folder_id = folder_id
                logger.info(f"Found old folder: {name} (ID: {old_folder_id})")
                break

        if not old_folder_id:
            logger.error("No old folder found")
            return

        # Get files from old folder
        all_files = []
        page_token = None

        while True:
            files = backup.service.files().list(
                q=f"'{old_folder_id}' in parents",
                fields='files(id, name, size), nextPageToken',
                pageSize=100
            ).execute()

            all_files.extend(files.get('files', []))
            page_token = files.get('nextPageToken')

            if not page_token:
                break

        logger.info(f"Found {len(all_files)} files in old folder")

        # Copy first 10 files as test
        copied_count = 0
        for i, file in enumerate(all_files[:10]):
            file_id = file['id']
            file_name = file['name']

            try:
                # Create clean filename
                clean_name = file_name.replace('unknown_', '').replace('search_', '')
                if not clean_name.endswith('.json'):
                    clean_name += '.json'

                # Copy file to clean folder
                file_metadata = {
                    'name': clean_name,
                    'parents': [clean_folder_id]
                }

                backup.service.files().copy(fileId=file_id, body=file_metadata).execute()
                copied_count += 1
                logger.info(f"COPIED ({i+1}/10): {clean_name}")

            except Exception as e:
                logger.error(f"Error copying {file_name}: {e}")

        logger.info(f"Test copy complete: Copied {copied_count} files")
        logger.info(f"Clean folder: https://drive.google.com/drive/folders/{clean_folder_id}")

        return clean_folder_id, copied_count

    except Exception as e:
        logger.error(f"Error: {e}")
        return None, 0

if __name__ == "__main__":
    print("Check Google Drive and Copy Files")
    print("=" * 50)

    folder_id, copied_count = check_and_copy()

    if folder_id:
        print(f"\nSUCCESS!")
        print(f"Copied {copied_count} files to clean folder")
        print(f"Access at: https://drive.google.com/drive/folders/{folder_id}")
    else:
        print(f"\nFAILED: Could not copy files")
