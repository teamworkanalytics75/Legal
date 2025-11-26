#!/usr/bin/env python3
"""
Copy Files in Small Batches

This script copies files in small batches to avoid timeouts.
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

def copy_batch(start_offset=5, batch_size=10):
    """Copy a small batch of files."""

    try:
        backup = GoogleDriveBackup()
        logger.info("Google Drive authentication successful")

        # Folder IDs
        old_folder_id = '12pHMekx-GER8cPaMJv7aH3-El4vu7kbb'  # 1782_discovery
        clean_folder_id = '1DHMRnqrtNQTG4w4IuJYhQsZ2XsBRFqMb'  # The Matrix_1782_Discovery_Clean

        # Get files starting from offset
        files = backup.service.files().list(
            q=f"'{old_folder_id}' in parents",
            fields='files(id, name)',
            pageSize=batch_size,
            orderBy='name'
        ).execute()

        file_list = files.get('files', [])

        # Skip the first start_offset files (already copied)
        files_to_copy = file_list[start_offset:start_offset + batch_size]

        logger.info(f"Copying batch: files {start_offset+1} to {start_offset+len(files_to_copy)}")

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
                logger.info(f"COPIED: {clean_name}")

            except Exception as e:
                logger.error(f"Error copying {old_name}: {e}")

        logger.info(f"Batch complete! Copied {copied_count} files")
        logger.info(f"Check your Google Drive folder: https://drive.google.com/drive/folders/{clean_folder_id}")

        return copied_count

    except Exception as e:
        logger.error(f"Error: {e}")
        return 0

if __name__ == "__main__":
    print("Copy Files in Small Batches")
    print("=" * 50)

    # Copy next 10 files (files 6-15)
    copied = copy_batch(start_offset=5, batch_size=10)

    print(f"\nCopied {copied} files to your clean folder!")
    print("Run this script again to copy more files.")
