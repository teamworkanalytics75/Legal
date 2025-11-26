#!/usr/bin/env python3
"""
Clean Up Google Drive Duplicates

Remove duplicate files from the Google Drive folder.
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


class DriveDuplicateCleaner:
    """Clean up duplicate files in Google Drive folder."""

    def __init__(self):
        """Initialize the cleaner."""
        self.gdrive = GoogleDriveBackup()

        # Clean folder ID
        self.clean_folder_id = "1_wv4FZLhubYOO2ncRBcW68w4htt4N5Pl"

    def find_duplicates(self):
        """Find duplicate files by name."""
        try:
            query = f"'{self.clean_folder_id}' in parents and trashed=false"
            results = self.gdrive.service.files().list(
                q=query,
                fields='files(id, name, createdTime)'
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} total files")

            # Group files by name
            name_groups = {}
            for file in files:
                name = file.get('name')
                if name not in name_groups:
                    name_groups[name] = []
                name_groups[name].append(file)

            # Find duplicates
            duplicates = []
            unique_files = []

            for name, file_list in name_groups.items():
                if len(file_list) > 1:
                    # Sort by creation time, keep the newest
                    file_list.sort(key=lambda x: x.get('createdTime', ''), reverse=True)
                    unique_files.append(file_list[0])  # Keep the newest
                    duplicates.extend(file_list[1:])   # Mark others as duplicates
                    logger.info(f"Found {len(file_list)} copies of '{name}' - keeping newest")
                else:
                    unique_files.append(file_list[0])

            logger.info(f"Unique files: {len(unique_files)}")
            logger.info(f"Duplicate files to remove: {len(duplicates)}")

            return unique_files, duplicates

        except Exception as e:
            logger.error(f"Error finding duplicates: {e}")
            return [], []

    def delete_duplicates(self, duplicates):
        """Delete duplicate files."""
        deleted_count = 0
        failed_count = 0

        for duplicate in duplicates:
            try:
                self.gdrive.service.files().delete(fileId=duplicate['id']).execute()
                deleted_count += 1
                logger.info(f"Deleted duplicate: {duplicate['name']}")
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to delete {duplicate['name']}: {e}")

        logger.info(f"Deleted {deleted_count} duplicates, {failed_count} failed")
        return deleted_count, failed_count

    def clean_folder(self):
        """Clean up the folder by removing duplicates."""
        logger.info("Starting duplicate cleanup...")

        # Find duplicates
        unique_files, duplicates = self.find_duplicates()

        if not duplicates:
            logger.info("No duplicates found!")
            return True

        # Delete duplicates
        deleted_count, failed_count = self.delete_duplicates(duplicates)

        # Final status
        logger.info(f"\nCLEANUP SUMMARY:")
        logger.info(f"Original files: {len(unique_files) + len(duplicates)}")
        logger.info(f"Unique files kept: {len(unique_files)}")
        logger.info(f"Duplicates removed: {deleted_count}")
        logger.info(f"Failed deletions: {failed_count}")

        return deleted_count > 0


def main():
    """Main entry point."""
    cleaner = DriveDuplicateCleaner()
    success = cleaner.clean_folder()

    if success:
        print(f"\nDuplicate cleanup completed!")
        print(f"Google Drive folder: https://drive.google.com/drive/folders/{cleaner.clean_folder_id}")
    else:
        print(f"\nDuplicate cleanup failed!")


if __name__ == "__main__":
    main()
