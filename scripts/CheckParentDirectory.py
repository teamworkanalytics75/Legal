#!/usr/bin/env python3
"""
Check All Folders in Parent Directory

Check all folders in the parent directory to see if there are other
folders containing §1782 cases that we haven't accounted for.
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


class ParentDirectoryChecker:
    """Check all folders in the parent directory."""

    def __init__(self):
        """Initialize the checker."""
        self.gdrive = GoogleDriveBackup()
        # We need to find the parent directory of our target folder
        self.target_folder_id = "12pHMekx-GER8cPaMJv7aH3-El4vu7kbb"

    def find_parent_directory(self):
        """Find the parent directory of our target folder."""
        try:
            logger.info("Finding parent directory...")

            # Get the target folder's metadata
            folder_info = self.gdrive.service.files().get(
                fileId=self.target_folder_id,
                fields='id, name, parents'
            ).execute()

            parents = folder_info.get('parents', [])
            if parents:
                parent_id = parents[0]
                logger.info(f"Parent directory ID: {parent_id}")

                # Get parent directory info
                parent_info = self.gdrive.service.files().get(
                    fileId=parent_id,
                    fields='id, name'
                ).execute()

                logger.info(f"Parent directory name: {parent_info.get('name')}")
                return parent_id
            else:
                logger.warning("No parent directory found")
                return None

        except Exception as e:
            logger.error(f"Error finding parent directory: {e}")
            return None

    def list_all_folders_in_parent(self, parent_id):
        """List all folders in the parent directory."""
        try:
            logger.info(f"Listing all folders in parent directory...")

            # Get all folders in the parent directory
            results = self.gdrive.service.files().list(
                q=f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder'",
                fields="files(id, name, createdTime, modifiedTime)",
                pageSize=1000
            ).execute()

            folders = results.get('files', [])
            logger.info(f"Found {len(folders)} folders in parent directory")

            return folders

        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            return []

    def check_folder_contents(self, folder_id, folder_name):
        """Check contents of a specific folder."""
        try:
            # Get all files in the folder
            results = self.gdrive.service.files().list(
                q=f"'{folder_id}' in parents",
                fields="files(id, name, mimeType, size)",
                pageSize=1000
            ).execute()

            files = results.get('files', [])

            # Categorize files
            json_files = []
            other_files = []

            for file_info in files:
                if file_info['name'].endswith('.json'):
                    json_files.append(file_info)
                else:
                    other_files.append(file_info)

            return {
                'total_files': len(files),
                'json_files': len(json_files),
                'other_files': len(other_files),
                'files': files
            }

        except Exception as e:
            logger.error(f"Error checking folder {folder_name}: {e}")
            return None

    def run_check(self):
        """Run the complete check."""
        logger.info("Starting parent directory check...")

        # Find parent directory
        parent_id = self.find_parent_directory()
        if not parent_id:
            logger.error("Could not find parent directory")
            return None

        # List all folders
        folders = self.list_all_folders_in_parent(parent_id)
        if not folders:
            logger.error("No folders found in parent directory")
            return None

        # Check each folder
        logger.info("\n" + "="*80)
        logger.info("FOLDER ANALYSIS")
        logger.info("="*80)

        total_files = 0
        total_json_files = 0
        folder_summary = []

        for folder in folders:
            folder_id = folder['id']
            folder_name = folder['name']

            logger.info(f"\nChecking folder: {folder_name}")
            logger.info(f"Folder ID: {folder_id}")

            contents = self.check_folder_contents(folder_id, folder_name)
            if contents:
                logger.info(f"  Total files: {contents['total_files']}")
                logger.info(f"  JSON files: {contents['json_files']}")
                logger.info(f"  Other files: {contents['other_files']}")

                total_files += contents['total_files']
                total_json_files += contents['json_files']

                folder_summary.append({
                    'name': folder_name,
                    'id': folder_id,
                    'total_files': contents['total_files'],
                    'json_files': contents['json_files'],
                    'other_files': contents['other_files']
                })

        # Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        logger.info(f"Total folders found: {len(folders)}")
        logger.info(f"Total files across all folders: {total_files}")
        logger.info(f"Total JSON files across all folders: {total_json_files}")

        logger.info(f"\nFolder breakdown:")
        for folder_info in folder_summary:
            logger.info(f"  - {folder_info['name']}: {folder_info['json_files']} JSON files")

        return {
            'parent_id': parent_id,
            'total_folders': len(folders),
            'total_files': total_files,
            'total_json_files': total_json_files,
            'folders': folder_summary
        }


def main():
    """Main entry point."""
    checker = ParentDirectoryChecker()
    result = checker.run_check()

    if result:
        print(f"\nSUCCESS!")
        print(f"Found {result['total_folders']} folders")
        print(f"Total files: {result['total_files']}")
        print(f"Total JSON files: {result['total_json_files']}")

        print(f"\nFolder breakdown:")
        for folder_info in result['folders']:
            print(f"  - {folder_info['name']}: {folder_info['json_files']} JSON files")


if __name__ == "__main__":
    main()
