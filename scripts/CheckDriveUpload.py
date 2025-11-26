#!/usr/bin/env python3
"""
Check Google Drive Upload Status

Verify what was actually uploaded to Google Drive.
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


class DriveUploadChecker:
    """Check what was actually uploaded to Google Drive."""

    def __init__(self):
        """Initialize the checker."""
        self.gdrive = GoogleDriveBackup()

        # Clean folder ID (the one we created earlier)
        self.clean_folder_id = "1_wv4FZLhubYOO2ncRBcW68w4htt4N5Pl"

    def check_folder_exists(self):
        """Check if the folder exists."""
        try:
            folder_info = self.gdrive.service.files().get(
                fileId=self.clean_folder_id,
                fields='id, name, mimeType'
            ).execute()

            logger.info(f"Folder found: {folder_info.get('name')}")
            logger.info(f"Folder ID: {folder_info.get('id')}")
            logger.info(f"Folder URL: https://drive.google.com/drive/folders/{self.clean_folder_id}")
            return True

        except Exception as e:
            logger.error(f"Folder not found or error accessing: {e}")
            return False

    def list_files_in_folder(self):
        """List all files in the folder."""
        try:
            query = f"'{self.clean_folder_id}' in parents and trashed=false"
            results = self.gdrive.service.files().list(
                q=query,
                fields='files(id, name, size, createdTime)'
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in folder")

            if files:
                logger.info("Files in folder:")
                for i, file in enumerate(files, 1):
                    logger.info(f"  {i}. {file.get('name')} (ID: {file.get('id')})")
            else:
                logger.warning("No files found in folder!")

            return files

        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

    def check_upload_status(self):
        """Check the overall upload status."""
        logger.info("Checking Google Drive upload status...")

        # Check if folder exists
        folder_exists = self.check_folder_exists()
        if not folder_exists:
            logger.error("Target folder does not exist!")
            return False

        # List files
        files = self.list_files_in_folder()

        # Summary
        logger.info(f"\nUPLOAD STATUS SUMMARY:")
        logger.info(f"Folder exists: {folder_exists}")
        logger.info(f"Files uploaded: {len(files)}")
        logger.info(f"Expected files: 65")

        if len(files) == 65:
            logger.info("SUCCESS: All 65 files uploaded!")
        elif len(files) > 0:
            logger.warning(f"PARTIAL: Only {len(files)}/65 files uploaded")
        else:
            logger.error("FAILED: No files uploaded")

        return len(files) > 0


def main():
    """Main entry point."""
    checker = DriveUploadChecker()
    success = checker.check_upload_status()

    if success:
        print(f"\nGoogle Drive folder checked successfully!")
        print(f"Folder URL: https://drive.google.com/drive/folders/{checker.clean_folder_id}")
    else:
        print(f"\nGoogle Drive upload verification failed!")


if __name__ == "__main__":
    main()
