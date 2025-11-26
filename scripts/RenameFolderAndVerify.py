#!/usr/bin/env python3
"""
Rename Google Drive Folder and Verify All Files

1. Rename the folder to "The Art of War - Caselaw Database"
2. Verify that we've checked all files in the folder
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


class FolderManager:
    """Manage Google Drive folder operations."""

    def __init__(self):
        """Initialize the manager."""
        self.gdrive = GoogleDriveBackup()
        self.folder_id = "12pHMekx-GER8cPaMJv7aH3-El4vu7kbb"

    def rename_folder(self, new_name):
        """Rename the Google Drive folder."""
        try:
            logger.info(f"Renaming folder to: {new_name}")

            # Update folder metadata
            folder_metadata = {
                'name': new_name
            }

            updated_folder = self.gdrive.service.files().update(
                fileId=self.folder_id,
                body=folder_metadata,
                fields='id, name'
            ).execute()

            logger.info(f"✅ Folder renamed successfully!")
            logger.info(f"New name: {updated_folder.get('name')}")
            logger.info(f"Folder ID: {updated_folder.get('id')}")

            return updated_folder

        except Exception as e:
            logger.error(f"❌ Error renaming folder: {e}")
            return None

    def verify_all_files(self):
        """Verify that we've checked all files in the folder."""
        try:
            logger.info("Verifying all files in the folder...")

            # Get all files
            results = self.gdrive.service.files().list(
                q=f"'{self.folder_id}' in parents",
                fields="files(id, name, mimeType, size, modifiedTime)",
                pageSize=1000
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in folder")

            # Categorize files
            json_files = []
            other_files = []

            for file_info in files:
                if file_info['name'].endswith('.json'):
                    json_files.append(file_info)
                else:
                    other_files.append(file_info)

            logger.info(f"JSON files: {len(json_files)}")
            logger.info(f"Other files: {len(other_files)}")

            # Show file breakdown
            logger.info("\n" + "="*60)
            logger.info("FILE VERIFICATION SUMMARY")
            logger.info("="*60)
            logger.info(f"Total files: {len(files)}")
            logger.info(f"JSON files: {len(json_files)}")
            logger.info(f"Other files: {len(other_files)}")

            if other_files:
                logger.info("\nNon-JSON files found:")
                for file_info in other_files:
                    logger.info(f"  - {file_info['name']} ({file_info['mimeType']})")

            # Show sample JSON files
            logger.info(f"\nSample JSON files (first 10):")
            for i, file_info in enumerate(json_files[:10], 1):
                logger.info(f"  {i}. {file_info['name']}")

            if len(json_files) > 10:
                logger.info(f"  ... and {len(json_files) - 10} more JSON files")

            return {
                'total_files': len(files),
                'json_files': len(json_files),
                'other_files': len(other_files),
                'files': files
            }

        except Exception as e:
            logger.error(f"Error verifying files: {e}")
            return None

    def run_operations(self):
        """Run all operations."""
        logger.info("Starting folder operations...")

        # Rename folder
        rename_result = self.rename_folder("The Art of War - Caselaw Database")
        if not rename_result:
            logger.error("Failed to rename folder")
            return

        # Verify all files
        verification_result = self.verify_all_files()
        if not verification_result:
            logger.error("Failed to verify files")
            return

        # Summary
        logger.info("\n" + "="*60)
        logger.info("OPERATIONS COMPLETE")
        logger.info("="*60)
        logger.info(f"✅ Folder renamed to: {rename_result.get('name')}")
        logger.info(f"✅ Files verified: {verification_result['total_files']} total")
        logger.info(f"   - JSON files: {verification_result['json_files']}")
        logger.info(f"   - Other files: {verification_result['other_files']}")

        return {
            'folder_renamed': True,
            'new_name': rename_result.get('name'),
            'files_verified': verification_result
        }


def main():
    """Main entry point."""
    manager = FolderManager()
    result = manager.run_operations()

    if result:
        print(f"\nSUCCESS!")
        print(f"Folder renamed to: {result['new_name']}")
        print(f"Files verified: {result['files_verified']['total_files']}")
        print(f"JSON files: {result['files_verified']['json_files']}")
        print(f"Other files: {result['files_verified']['other_files']}")


if __name__ == "__main__":
    main()
