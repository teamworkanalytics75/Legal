#!/usr/bin/env python3
"""
Copy All Remaining Files

This script copies all remaining files to the clean folder.
"""

import sys
import logging
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from google_drive_backup import GoogleDriveBackup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def copy_all_remaining():
    """Copy all remaining files."""

    try:
        backup = GoogleDriveBackup()
        logger.info("Google Drive authentication successful")

        # Folder IDs
        old_folder_id = '12pHMekx-GER8cPaMJv7aH3-El4vu7kbb'  # 1782_discovery
        clean_folder_id = '1DHMRnqrtNQTG4w4IuJYhQsZ2XsBRFqMb'  # The Matrix_1782_Discovery_Clean

        # Count files already in clean folder
        clean_files = backup.service.files().list(
            q=f"'{clean_folder_id}' in parents",
            fields='files(id, name)',
            pageSize=1000
        ).execute()

        already_copied = len(clean_files.get('files', []))
        logger.info(f"Already have {already_copied} files in clean folder")

        # Get ALL files from old folder
        all_old_files = []
        page_token = None

        while True:
            old_files = backup.service.files().list(
                q=f"'{old_folder_id}' in parents",
                fields='files(id, name), nextPageToken',
                pageSize=1000,
                pageToken=page_token
            ).execute()

            all_old_files.extend(old_files.get('files', []))
            page_token = old_files.get('nextPageToken')

            if not page_token:
                break

        logger.info(f"Total files in old folder: {len(all_old_files)}")

        # Copy remaining files
        files_to_copy = all_old_files[already_copied:]
        logger.info(f"Copying remaining {len(files_to_copy)} files...")

        copied_count = 0
        for i, file in enumerate(files_to_copy):
            file_id = file['id']
            old_name = file['name']

            # Create clean name
            clean_name = old_name.replace('unknown_', '').replace('search_', '')
            if not clean_name.endswith('.json'):
                clean_name += '.json'

            try:
                # Copy file
                file_metadata = {
                    'name': clean_name,
                    'parents': [clean_folder_id]
                }

                backup.service.files().copy(fileId=file_id, body=file_metadata).execute()
                copied_count += 1
                logger.info(f"COPIED ({already_copied + copied_count}/{len(all_old_files)}): {clean_name}")

            except Exception as e:
                logger.error(f"Error copying {old_name}: {e}")

        logger.info(f"Copy complete! Copied {copied_count} files")
        logger.info(f"Total files in clean folder: {already_copied + copied_count}")
        logger.info(f"Check your Google Drive folder: https://drive.google.com/drive/folders/{clean_folder_id}")

        return copied_count

    except Exception as e:
        logger.error(f"Error: {e}")
        return 0

if __name__ == "__main__":
    print("Copy All Remaining Files")
    print("=" * 50)

    copied = copy_all_remaining()

    print(f"\nCopied {copied} files to your clean folder!")
    print("All files should now be in your clean folder.")
