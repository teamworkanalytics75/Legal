#!/usr/bin/env python3
"""
Simple Google Drive File Copy

This script simply copies files from old folder to new folder with clean names.
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

def simple_copy():
    """Simple copy of files to new folder."""

    try:
        backup = GoogleDriveBackup()
        logger.info("Google Drive authentication successful")

        # Folder IDs
        parent_folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'
        old_folder_id = '12pHMekx-GER8cPaMJv7aH3-El4vu7kbb'  # 1782_discovery
        clean_folder_id = '1DHMRnqrtNQTG4w4IuJYhQsZ2XsBRFqMb'  # The Matrix_1782_Discovery_Clean

        logger.info(f"Old folder ID: {old_folder_id}")
        logger.info(f"Clean folder ID: {clean_folder_id}")

        # Get first 5 files from old folder
        files = backup.service.files().list(
            q=f"'{old_folder_id}' in parents",
            fields='files(id, name)',
            pageSize=5
        ).execute()

        file_list = files.get('files', [])
        logger.info(f"Found {len(file_list)} files to copy")

        # Copy each file
        for i, file in enumerate(file_list):
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
                logger.info(f"COPIED ({i+1}/{len(file_list)}): {clean_name}")

            except Exception as e:
                logger.error(f"Error copying {old_name}: {e}")

        logger.info(f"Copy complete!")
        logger.info(f"Check your Google Drive folder: https://drive.google.com/drive/folders/{clean_folder_id}")

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    print("Simple Google Drive File Copy")
    print("=" * 50)

    simple_copy()
