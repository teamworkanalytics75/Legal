#!/usr/bin/env python3
"""
The Matrix - Bulk Download 1782 Discovery Cases

Specialized script for downloading 1782 discovery cases from all federal courts.
Supports overnight mode for large-scale downloads with automatic Google Drive backup.
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from download_case_law import download_1782_federal_cases
from google_drive_backup import GoogleDriveBackup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/download_1782.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def setup_directories():
    """Ensure required directories exist."""
    Path("logs").mkdir(exist_ok=True)
    Path("data/case_law/1782_discovery").mkdir(parents=True, exist_ok=True)


def backup_to_gdrive(topic: str = "1782_discovery"):
    """Backup downloaded cases to Google Drive."""
    try:
        logger.info(f"Starting Google Drive backup for {topic}...")
        gdrive = GoogleDriveBackup()
        gdrive.create_backup_folder("The Matrix_1782_Discovery")

        local_dir = Path("data/case_law") / topic
        stats = gdrive.upload_directory(local_dir, topic)

        logger.info(f"Google Drive backup complete:")
        logger.info(f"  Uploaded: {stats['uploaded']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        logger.info(f"  Failed: {stats['failed']}")

        return True
    except Exception as e:
        logger.error(f"Google Drive backup failed: {e}")
        return False


def main():
    """Main entry point for 1782 bulk download."""
    parser = argparse.ArgumentParser(description="Download 1782 discovery cases")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test download (500 cases)"
    )
    parser.add_argument(
        "--overnight",
        action="store_true",
        help="Use overnight mode for large downloads"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=10000,
        help="Maximum cases to download"
    )
    parser.add_argument(
        "--date-after",
        default="2000-01-01",
        help="Download cases after this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup to Google Drive after download"
    )
    parser.add_argument(
        "--backup-only",
        action="store_true",
        help="Only backup existing files to Google Drive"
    )

    args = parser.parse_args()

    setup_directories()

    print("\n" + "="*60)
    print("The Matrix 1782 Discovery Bulk Download")
    print("="*60 + "\n")

    if args.backup_only:
        logger.info("Running Google Drive backup only...")
        success = backup_to_gdrive()
        if success:
            print("[OK] Backup completed successfully!")
        else:
            print("[ERROR] Backup failed - check logs")
        return

    # Determine download size
    if args.test:
        max_cases = 500
        logger.info("Running TEST download (500 cases)")
    else:
        max_cases = args.max_cases
        logger.info(f"Running FULL download ({max_cases} cases)")

    # Start download
    start_time = datetime.now()
    logger.info(f"Starting download at {start_time}")
    logger.info(f"Overnight mode: {args.overnight}")
    logger.info(f"Date filter: {args.date_after}")

    try:
        stats = download_1782_federal_cases(
            date_after=args.date_after,
            max_cases=max_cases,
            overnight_mode=args.overnight
        )

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        for topic, count in stats.items():
            print(f"{topic}: {count} cases")
        print(f"Total: {sum(stats.values())} cases")
        print(f"Duration: {duration}")
        print("="*60 + "\n")

        # Backup to Google Drive if requested
        if args.backup and sum(stats.values()) > 0:
            logger.info("Starting Google Drive backup...")
            backup_success = backup_to_gdrive()
            if backup_success:
                print("✅ Google Drive backup completed!")
            else:
                print("⚠️ Google Drive backup failed - check logs")

    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        print("\n⚠️ Download interrupted - checkpoint saved for resume")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        print(f"\n❌ Download failed: {e}")


if __name__ == "__main__":
    main()
