#!/usr/bin/env python3
"""
Upload Canonical 1782 Database to Google Drive

This script uploads the clean canonical database from "The Art of War - Database"
to the new Google Drive folder.
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


def upload_canonical_database():
    """Upload canonical database to Google Drive."""

    logger.info("="*80)
    logger.info("UPLOADING CANONICAL DATABASE TO GOOGLE DRIVE")
    logger.info("="*80)

    try:
        # Initialize Google Drive backup
        backup = GoogleDriveBackup()

        # Get the new folder ID (from the previous script output)
        new_folder_id = "1uQ9-7Y-iTkO7KYZ7yCkLp_9IzZljGbIb"

        # Canonical database directory
        canonical_dir = Path("data/case_law/The Art of War - Database")

        if not canonical_dir.exists():
            logger.error(f"Canonical directory not found: {canonical_dir}")
            return

        logger.info(f"Uploading from: {canonical_dir}")
        logger.info(f"Uploading to Drive folder: {new_folder_id}")

        uploaded_count = 0
        skipped_count = 0

        # Upload all JSON files
        for json_file in canonical_dir.glob("*.json"):
            try:
                # Skip non-case files
                if json_file.name in ['drive_verification_results.json']:
                    skipped_count += 1
                    logger.info(f"Skipped non-case file: {json_file.name}")
                    continue

                logger.info(f"Uploading: {json_file.name}")

                # Upload JSON file
                file_id = backup.upload_file_content(
                    json_file.read_text(encoding='utf-8'),
                    json_file.name,
                    new_folder_id
                )
                if file_id:
                    uploaded_count += 1

                # Also upload matching TXT file if it exists
                txt_file = json_file.with_suffix('.txt')
                if txt_file.exists():
                    logger.info(f"Uploading: {txt_file.name}")
                    txt_file_id = backup.upload_file_content(
                        txt_file.read_text(encoding='utf-8'),
                        txt_file.name,
                        new_folder_id
                    )
                    if txt_file_id:
                        uploaded_count += 1

            except Exception as e:
                logger.error(f"Error uploading {json_file.name}: {e}")

        logger.info("\n" + "="*80)
        logger.info("UPLOAD COMPLETE")
        logger.info("="*80)
        logger.info(f"Files uploaded: {uploaded_count}")
        logger.info(f"Files skipped: {skipped_count}")
        logger.info(f"Drive folder ID: {new_folder_id}")
        logger.info(f"Direct link: https://drive.google.com/drive/folders/{new_folder_id}")

        return {
            'uploaded_count': uploaded_count,
            'skipped_count': skipped_count,
            'folder_id': new_folder_id
        }

    except Exception as e:
        logger.error(f"Error during upload: {e}")
        return None


if __name__ == "__main__":
    upload_canonical_database()
