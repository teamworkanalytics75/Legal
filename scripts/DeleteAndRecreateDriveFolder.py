#!/usr/bin/env python3
"""
Delete Google Drive 1782 Folder

This script deletes the old cluttered Google Drive folder and creates a new clean one.
"""

import sys
import json
import logging
from pathlib import Path

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

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def delete_and_recreate_drive_folder():
    """Delete old Drive folder and create new clean one."""

    logger.info("="*80)
    logger.info("DELETING AND RECREATING GOOGLE DRIVE 1782 FOLDER")
    logger.info("="*80)

    try:
        # Initialize Google Drive backup
        backup = GoogleDriveBackup()

        # Old folder ID to delete
        old_folder_id = "157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA"

        logger.info(f"Deleting old folder: {old_folder_id}")

        # Delete the old folder
        try:
            backup.service.files().delete(fileId=old_folder_id).execute()
            logger.info("✅ Old folder deleted successfully")
        except Exception as e:
            logger.warning(f"Could not delete old folder (may not exist): {e}")

        # Create new clean folder
        new_folder_name = "The Art of War - Database"
        logger.info(f"Creating new folder: {new_folder_name}")

        new_folder_id = backup.create_backup_folder(new_folder_name)
        logger.info(f"✅ New folder created with ID: {new_folder_id}")

        logger.info("\n" + "="*80)
        logger.info("DRIVE FOLDER CLEANUP COMPLETE")
        logger.info("="*80)
        logger.info(f"Old folder ID: {old_folder_id} (deleted)")
        logger.info(f"New folder ID: {new_folder_id}")
        logger.info(f"New folder name: {new_folder_name}")
        logger.info(f"Direct link: https://drive.google.com/drive/folders/{new_folder_id}")

        return {
            'old_folder_id': old_folder_id,
            'new_folder_id': new_folder_id,
            'new_folder_name': new_folder_name
        }

    except Exception as e:
        logger.error(f"Error during Drive folder cleanup: {e}")
        return None


if __name__ == "__main__":
    delete_and_recreate_drive_folder()
