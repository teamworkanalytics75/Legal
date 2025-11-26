#!/usr/bin/env python3
"""
Simple Google Drive File Counter

This script just counts files and shows what's in your Google Drive.
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

def count_gdrive_files():
    """Count files in Google Drive and show sample."""

    logger.info("Counting Google Drive files...")

    try:
        backup = GoogleDriveBackup()
        logger.info("Google Drive authentication successful")

        # Get the folder ID for 1782_discovery
        folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'

        # Find the subfolder
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
            return

        # Count files using pagination
        total_files = 0
        page_token = None

        logger.info("Counting files...")
        while True:
            files = backup.service.files().list(
                q=f"'{subfolder_id}' in parents",
                fields='files(id, name), nextPageToken',
                pageSize=1000  # Larger page size for counting
            ).execute()

            batch_files = files.get('files', [])
            total_files += len(batch_files)
            page_token = files.get('nextPageToken')

            if not page_token:
                break

            if total_files % 10000 == 0:
                logger.info(f"Counted {total_files} files so far...")

        logger.info(f"Total files in Google Drive: {total_files}")

        # Show first 10 files as sample
        logger.info("Sample files:")
        sample_files = backup.service.files().list(
            q=f"'{subfolder_id}' in parents",
            fields='files(name)',
            pageSize=10
        ).execute()

        for i, file in enumerate(sample_files.get('files', [])):
            logger.info(f"{i+1}. {file['name']}")

        return total_files

    except Exception as e:
        logger.error(f"Error: {e}")
        return 0

if __name__ == "__main__":
    print("Google Drive File Counter")
    print("=" * 50)

    total = count_gdrive_files()

    print(f"\nTotal files: {total}")

    if total > 100000:
        print(f"\nWARNING: You have {total} files!")
        print("This is way more than expected for 1782 cases.")
        print("You may want to:")
        print("1. Check if there are duplicate uploads")
        print("2. Consider starting fresh with a new folder")
        print("3. Or manually clean up the folder in Google Drive web interface")


