#!/usr/bin/env python3
"""
Google Drive Folder Verification

First, verify we're looking at the correct folder and get accurate file count.
Then clean up duplicates and false positives.
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


class DriveFolderChecker:
    """Check and clean Google Drive folder."""

    def __init__(self):
        """Initialize the checker."""
        self.gdrive = GoogleDriveBackup()
        self.folder_id = "1DHMRnqrtNQTG4w4IuJYhQsZ2XsBRFqMb"

        # Results
        self.folder_info = {}
        self.file_list = []

    def get_folder_info(self):
        """Get detailed information about the folder."""
        try:
            logger.info(f"Checking folder: {self.folder_id}")

            # Get folder metadata
            folder_metadata = self.gdrive.service.files().get(
                fileId=self.folder_id,
                fields="id, name, mimeType, createdTime, modifiedTime"
            ).execute()

            logger.info(f"Folder name: {folder_metadata.get('name', 'Unknown')}")
            logger.info(f"Folder ID: {folder_metadata.get('id', 'Unknown')}")
            logger.info(f"Created: {folder_metadata.get('createdTime', 'Unknown')}")
            logger.info(f"Modified: {folder_metadata.get('modifiedTime', 'Unknown')}")

            # Get all files in folder
            results = self.gdrive.service.files().list(
                q=f"'{self.folder_id}' in parents",
                fields="files(id, name, mimeType, size, modifiedTime, createdTime)",
                pageSize=1000  # Get up to 1000 files
            ).execute()

            files = results.get('files', [])
            logger.info(f"Total files found: {len(files)}")

            # Count by type
            json_files = [f for f in files if f.get('mimeType') == 'application/json']
            other_files = [f for f in files if f.get('mimeType') != 'application/json']

            logger.info(f"JSON files: {len(json_files)}")
            logger.info(f"Other files: {len(other_files)}")

            if other_files:
                logger.info("Non-JSON files found:")
                for f in other_files[:10]:  # Show first 10
                    logger.info(f"  - {f['name']} ({f.get('mimeType', 'Unknown')})")
                if len(other_files) > 10:
                    logger.info(f"  ... and {len(other_files) - 10} more")

            self.folder_info = {
                'folder_name': folder_metadata.get('name'),
                'folder_id': folder_metadata.get('id'),
                'total_files': len(files),
                'json_files': len(json_files),
                'other_files': len(other_files),
                'files': files
            }

            return self.folder_info

        except Exception as e:
            logger.error(f"Error getting folder info: {e}")
            return None

    def analyze_file_names(self):
        """Analyze file names for patterns and duplicates."""
        if not self.folder_info.get('files'):
            logger.error("No files to analyze")
            return

        files = self.folder_info['files']
        logger.info("\n" + "="*60)
        logger.info("FILE NAME ANALYSIS")
        logger.info("="*60)

        # Group by base name (remove case numbers)
        name_groups = {}
        for file_info in files:
            name = file_info['name']
            # Extract base name (remove case_XX_ prefix)
            if name.startswith('case_') and '_' in name:
                base_name = name.split('_', 2)[2] if len(name.split('_', 2)) > 2 else name
            else:
                base_name = name

            if base_name not in name_groups:
                name_groups[base_name] = []
            name_groups[base_name].append(file_info)

        # Find duplicates
        duplicates = {name: files for name, files in name_groups.items() if len(files) > 1}
        unique_files = {name: files[0] for name, files in name_groups.items() if len(files) == 1}

        logger.info(f"Unique file names: {len(unique_files)}")
        logger.info(f"Duplicate groups: {len(duplicates)}")

        if duplicates:
            logger.info("\nDuplicate files found:")
            total_duplicates = 0
            for base_name, file_list in duplicates.items():
                logger.info(f"  '{base_name}': {len(file_list)} copies")
                total_duplicates += len(file_list) - 1  # Count extras

                # Show file IDs for deletion
                for i, file_info in enumerate(file_list):
                    status = "KEEP" if i == 0 else "DELETE"
                    logger.info(f"    {status}: {file_info['id']} - {file_info['name']}")

            logger.info(f"\nTotal duplicate files to remove: {total_duplicates}")

        return {
            'unique_files': unique_files,
            'duplicates': duplicates,
            'total_duplicates': sum(len(files) - 1 for files in duplicates.values())
        }

    def check_folder_correctness(self):
        """Verify this is the correct folder with 219 JSON files."""
        logger.info("="*60)
        logger.info("FOLDER VERIFICATION")
        logger.info("="*60)

        folder_info = self.get_folder_info()
        if not folder_info:
            logger.error("Could not get folder information")
            return False

        # Check if this matches user's description
        json_count = folder_info['json_files']
        total_count = folder_info['total_files']

        logger.info(f"\nFolder verification:")
        logger.info(f"  Total files: {total_count}")
        logger.info(f"  JSON files: {json_count}")
        logger.info(f"  User expected: 219 JSON files")

        if json_count == 219:
            logger.info("✅ CORRECT: Folder has exactly 219 JSON files as expected")
            return True
        elif abs(json_count - 219) <= 5:
            logger.info(f"⚠️  CLOSE: Folder has {json_count} JSON files (expected 219, difference: {abs(json_count - 219)})")
            return True
        else:
            logger.error(f"❌ WRONG FOLDER: Expected 219 JSON files, found {json_count}")
            return False

    def run_check(self):
        """Run complete folder check."""
        logger.info("Google Drive Folder Verification")
        logger.info("="*60)
        logger.info("Checking if we're looking at the correct folder...")

        # First verify this is the right folder
        is_correct = self.check_folder_correctness()

        if not is_correct:
            logger.error("This doesn't appear to be the correct folder!")
            logger.error("Please verify the folder ID and try again.")
            return

        # Analyze file names for duplicates
        analysis = self.analyze_file_names()

        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Folder: {self.folder_info['folder_name']}")
        logger.info(f"Total files: {self.folder_info['total_files']}")
        logger.info(f"JSON files: {self.folder_info['json_files']}")
        logger.info(f"Unique files: {len(analysis['unique_files'])}")
        logger.info(f"Duplicate groups: {len(analysis['duplicates'])}")
        logger.info(f"Files to remove: {analysis['total_duplicates']}")

        if analysis['total_duplicates'] > 0:
            logger.info(f"\n⚠️  ACTION NEEDED: {analysis['total_duplicates']} duplicate files should be removed")
            logger.info("Next step: Run cleanup script to remove duplicates")
        else:
            logger.info("\n✅ CLEAN: No duplicates found")


def main():
    """Main entry point."""
    checker = DriveFolderChecker()
    checker.run_check()


if __name__ == "__main__":
    main()
