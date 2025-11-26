#!/usr/bin/env python3
"""
Upload Section 1782 Dataset to Google Drive

Upload the Section 1782 dataset files to Google Drive for backup.
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


def upload_section_1782_dataset():
    """Upload Section 1782 dataset to Google Drive."""

    logger.info("🚀 Starting Section 1782 dataset upload to Google Drive...")

    try:
        # Initialize Google Drive backup
        backup = GoogleDriveBackup()

        # Create Section 1782 folder
        folder_name = "Section 1782 Dataset"
        folder_id = backup.create_backup_folder(folder_name)
        logger.info(f"📁 Created folder: {folder_name}")

        # Files to upload
        files_to_upload = [
            {
                "local_path": "data/section_1782_top_50_cases.json",
                "description": "Top 50 Section 1782 cases with metadata"
            },
            {
                "local_path": "data/section_1782_dataset_summary.md",
                "description": "Comprehensive dataset summary and analysis"
            },
            {
                "local_path": "data/section_1782_cases/",
                "description": "Directory for individual case PDFs",
                "is_directory": True
            }
        ]

        uploaded_count = 0

        for file_info in files_to_upload:
            local_path = Path(file_info["local_path"])

            if not local_path.exists():
                logger.warning(f"⚠️  File not found: {local_path}")
                continue

            try:
                if file_info.get("is_directory", False):
                    # Upload directory contents
                    logger.info(f"📂 Uploading directory: {local_path}")
                    for file_path in local_path.rglob("*"):
                        if file_path.is_file():
                            backup.upload_file(
                                file_path,
                                topic="Section 1782 Cases"
                            )
                            uploaded_count += 1
                            logger.info(f"✅ Uploaded: {file_path.name}")
                else:
                    # Upload single file
                    logger.info(f"📄 Uploading file: {local_path}")
                    backup.upload_file(
                        local_path,
                        topic="Section 1782 Dataset"
                    )
                    uploaded_count += 1
                    logger.info(f"✅ Uploaded: {local_path.name}")

            except Exception as e:
                logger.error(f"❌ Failed to upload {local_path}: {e}")

        logger.info(f"🎉 Upload complete! {uploaded_count} files uploaded to Google Drive")
        logger.info(f"📁 Folder: {folder_name}")

        return True

    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False


if __name__ == "__main__":
    success = upload_section_1782_dataset()
    sys.exit(0 if success else 1)
