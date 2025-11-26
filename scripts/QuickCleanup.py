#!/usr/bin/env python3
"""
Simple Google Drive File Counter and Cleaner

This script just counts files and removes obvious false positives.
"""

import sys
import json
import logging
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from google_drive_backup import GoogleDriveBackup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def quick_cleanup():
    """Quick cleanup - just remove obvious false positives."""

    logger.info("Starting quick Google Drive cleanup...")

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

        # Get first 100 files to see what we're dealing with
        files = backup.service.files().list(
            q=f"'{subfolder_id}' in parents",
            fields='files(id, name, size)',
            pageSize=100
        ).execute()

        file_list = files.get('files', [])
        logger.info(f"Found {len(file_list)} files (showing first 100)")

        # Show first 20 files
        logger.info("First 20 files:")
        for i, file in enumerate(file_list[:20]):
            logger.info(f"{i+1}. {file['name']} ({file.get('size', 'unknown')} bytes)")

        # Count files with obvious false positive patterns
        false_positive_patterns = [
            'terry cusick',
            'seacoast national bank',
            'florida fourth district',
            '4d22-1782',
            'docket',
            'case number'
        ]

        removed_count = 0
        kept_count = 0

        # Process first 50 files for quick cleanup
        for file in file_list[:50]:
            file_name = file['name'].lower()

            # Check if it's an obvious false positive
            is_false_positive = any(pattern in file_name for pattern in false_positive_patterns)

            if is_false_positive:
                try:
                    backup.service.files().delete(fileId=file['id']).execute()
                    removed_count += 1
                    logger.info(f"REMOVED: {file['name']}")
                except Exception as e:
                    logger.error(f"Error removing {file['name']}: {e}")
            else:
                kept_count += 1
                logger.info(f"KEEP: {file['name']}")

        logger.info(f"Quick cleanup complete: Kept {kept_count}, removed {removed_count}")

    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    print("Quick Google Drive Cleanup")
    print("=" * 50)

    quick_cleanup()


