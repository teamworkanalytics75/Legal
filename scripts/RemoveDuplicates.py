#!/usr/bin/env python3
"""
Remove Duplicates and Clean File Names

This script removes duplicate files and standardizes file names.
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from google_drive_backup import GoogleDriveBackup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def remove_duplicates_and_clean_names():
    """Remove duplicate files and clean up file names."""

    logger.info("Starting duplicate removal and name cleanup...")

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

        # Get ALL files
        all_files = []
        page_token = None

        while True:
            files = backup.service.files().list(
                q=f"'{subfolder_id}' in parents",
                fields='files(id, name, size), nextPageToken',
                pageSize=100
            ).execute()

            all_files.extend(files.get('files', []))
            page_token = files.get('nextPageToken')

            if not page_token:
                break

        logger.info(f"Found {len(all_files)} total files")

        # Group files by case content to find duplicates
        case_groups = defaultdict(list)

        # Process first 200 files to find duplicates
        for file in all_files[:200]:
            try:
                # Download file content
                content = backup.service.files().get_media(fileId=file['id']).execute()
                case_data = json.loads(content)

                # Use case name as the key for grouping
                case_name = case_data.get('caseName', file['name'])
                case_groups[case_name].append({
                    'id': file['id'],
                    'name': file['name'],
                    'size': file.get('size', 0)
                })

            except Exception as e:
                logger.error(f"Error processing {file['name']}: {e}")

        # Find duplicates and clean up
        removed_count = 0
        renamed_count = 0

        for case_name, files_list in case_groups.items():
            if len(files_list) > 1:
                logger.info(f"Found {len(files_list)} duplicates for: {case_name}")

                # Keep the largest file (most complete)
                files_list.sort(key=lambda x: int(x['size']), reverse=True)
                keep_file = files_list[0]

                # Remove duplicates
                for duplicate_file in files_list[1:]:
                    try:
                        backup.service.files().delete(fileId=duplicate_file['id']).execute()
                        removed_count += 1
                        logger.info(f"REMOVED duplicate: {duplicate_file['name']}")
                    except Exception as e:
                        logger.error(f"Error removing duplicate {duplicate_file['name']}: {e}")

                # Rename the kept file to a clean name
                clean_name = "".join(c for c in case_name if c.isalnum() or c in (' ', '-', '_')).strip()
                clean_name = clean_name[:100]  # Limit length
                new_name = f"{clean_name}.json"

                if new_name != keep_file['name']:
                    try:
                        backup.service.files().update(
                            fileId=keep_file['id'],
                            body={'name': new_name}
                        ).execute()
                        renamed_count += 1
                        logger.info(f"RENAMED: {keep_file['name']} -> {new_name}")
                    except Exception as e:
                        logger.error(f"Error renaming {keep_file['name']}: {e}")

        logger.info(f"Cleanup complete: Removed {removed_count} duplicates, renamed {renamed_count} files")

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    print("Remove Duplicates and Clean File Names")
    print("=" * 50)

    remove_duplicates_and_clean_names()


