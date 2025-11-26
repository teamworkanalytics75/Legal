#!/usr/bin/env python3
"""
Copy All Files to Clean Folder

This script copies all files from old folder to new folder in batches.
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

def copy_all_files():
    """Copy all files from old folder to new folder."""

    try:
        backup = GoogleDriveBackup()
        logger.info("Google Drive authentication successful")

        # Folder IDs
        old_folder_id = '12pHMekx-GER8cPaMJv7aH3-El4vu7kbb'  # 1782_discovery
        clean_folder_id = '1DHMRnqrtNQTG4w4IuJYhQsZ2XsBRFqMb'  # The Matrix_1782_Discovery_Clean

        # Get ALL files from old folder
        all_files = []
        page_token = None

        while True:
            files = backup.service.files().list(
                q=f"'{old_folder_id}' in parents",
                fields='files(id, name), nextPageToken',
                pageSize=100
            ).execute()

            all_files.extend(files.get('files', []))
            page_token = files.get('nextPageToken')

            if not page_token:
                break

        logger.info(f"Found {len(all_files)} total files to copy")

        # Copy files in batches of 20
        batch_size = 20
        copied_count = 0

        for batch_start in range(0, len(all_files), batch_size):
            batch_end = min(batch_start + batch_size, len(all_files))
            batch_files = all_files[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end}")

            for i, file in enumerate(batch_files):
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
                    logger.info(f"COPIED ({copied_count}/{len(all_files)}): {clean_name}")

                except Exception as e:
                    logger.error(f"Error copying {old_name}: {e}")

            logger.info(f"Batch complete. Copied {copied_count} files so far.")

        logger.info(f"Copy complete! Total copied: {copied_count} files")
        logger.info(f"Check your Google Drive folder: https://drive.google.com/drive/folders/{clean_folder_id}")

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    print("Copy All Files to Clean Folder")
    print("=" * 50)

    copy_all_files()
