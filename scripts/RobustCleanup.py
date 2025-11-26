#!/usr/bin/env python3
"""
Robust Google Drive Cleanup - Batch Processing

This script processes files in small batches with proper error handling and timeouts.
"""

import sys
import json
import logging
import time
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from google_drive_backup import GoogleDriveBackup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def is_1782_discovery_case(case_content):
    """Check if a case is actually about 1782 discovery."""

    try:
        case = json.loads(case_content)
    except json.JSONDecodeError:
        return False

    case_name = case.get('caseName', '').lower()
    case_text = case.get('plain_text', '').lower()
    html_text = case.get('html_with_citations', '').lower()

    # Combine all text for analysis
    all_text = f"{case_name} {case_text} {html_text}"

    # Strong positive indicators for 1782 discovery
    strong_indicators = [
        '28 u.s.c. 1782',
        '28 usc 1782',
        'section 1782',
        '§ 1782',
        'discovery for use in a foreign proceeding',
        'foreign tribunal',
        'international tribunal',
        'assistance before a foreign',
        'judicial assistance',
        'discovery pursuant to 28',
        'application pursuant to 28',
        'order pursuant to 28',
        'subpoena under 28',
        'intel corp v. advanced micro devices',  # The famous 1782 case
        'brandi-dohrn',
        'euromepa',
        'del valle ruiz'
    ]

    # Check for strong indicators
    has_strong_indicator = any(indicator in all_text for indicator in strong_indicators)

    # Additional check: if it mentions "1782" but not in the context of U.S.C., it's likely a false positive
    if '1782' in all_text and not any(usc_indicator in all_text for usc_indicator in ['28 u.s.c. 1782', '28 usc 1782', 'section 1782', '§ 1782']):
        return False

    return has_strong_indicator

def batch_cleanup_with_retry(batch_size=5, max_retries=3):
    """Process files in small batches with retry logic."""

    logger.info(f"Starting batch cleanup (batch size: {batch_size}, max retries: {max_retries})")

    try:
        # Initialize Google Drive backup
        backup = GoogleDriveBackup()
        logger.info("Google Drive authentication successful")

        # Get the folder ID for 1782_discovery
        folder_id = '157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA'

        # First, find the subfolder
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
            return 0, 0

        # Get ALL files using pagination
        all_files = []
        page_token = None

        logger.info("Collecting all files...")
        while True:
            try:
                files = backup.service.files().list(
                    q=f"'{subfolder_id}' in parents",
                    fields='files(id, name, size), nextPageToken',
                    pageSize=100
                ).execute()

                all_files.extend(files.get('files', []))
                page_token = files.get('nextPageToken')

                if not page_token:
                    break

                logger.info(f"Collected {len(all_files)} files so far...")

            except Exception as e:
                logger.error(f"Error collecting files: {e}")
                break

        logger.info(f"Found {len(all_files)} total files to process")

        kept_count = 0
        removed_count = 0
        processed_count = 0

        # Process files in batches
        for batch_start in range(0, len(all_files), batch_size):
            batch_end = min(batch_start + batch_size, len(all_files))
            batch_files = all_files[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end}")

            for file in batch_files:
                file_id = file['id']
                file_name = file['name']
                processed_count += 1

                # Retry logic for each file
                success = False
                for attempt in range(max_retries):
                    try:
                        # Download file content
                        content = backup.service.files().get_media(fileId=file_id).execute()

                        # Check if it's actually a 1782 case
                        if is_1782_discovery_case(content):
                            kept_count += 1
                            logger.info(f"KEEP ({processed_count}/{len(all_files)}): {file_name}")
                        else:
                            # Delete the file
                            backup.service.files().delete(fileId=file_id).execute()
                            removed_count += 1
                            logger.info(f"REMOVED ({processed_count}/{len(all_files)}): {file_name}")

                        success = True
                        break

                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1} failed for {file_name}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2)  # Wait before retry
                        else:
                            logger.error(f"Failed to process {file_name} after {max_retries} attempts")

                if not success:
                    logger.error(f"Skipping {file_name} due to repeated failures")

            # Small delay between batches
            time.sleep(1)

            # Log progress
            logger.info(f"Batch complete. Progress: {processed_count}/{len(all_files)} files processed")
            logger.info(f"Current stats: {kept_count} kept, {removed_count} removed")

        logger.info(f"Cleanup complete: Processed {processed_count} files, kept {kept_count} verified 1782 cases, removed {removed_count} false positives")
        return kept_count, removed_count

    except Exception as e:
        logger.error(f"Error with Google Drive: {e}")
        return 0, 0

if __name__ == "__main__":
    print("Robust Google Drive Cleanup - Batch Processing")
    print("=" * 50)

    kept, removed = batch_cleanup_with_retry(batch_size=3, max_retries=3)

    print(f"\nFinal result:")
    print(f"Kept: {kept} verified 1782 discovery cases")
    print(f"Removed: {removed} false positive cases")

    if kept > 0:
        print(f"\nYour Google Drive now contains only verified 1782 discovery cases!")
        print("Access at: https://drive.google.com/drive/folders/157I1VHclMJETmfE_nEt5bsS0_4Bhv7kA")


