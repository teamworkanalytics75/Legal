#!/usr/bin/env python3
"""
Retrofit Hash Fields to Existing 1782 JSON Files

This script adds _matrix_case_hash and _matrix_primary_identifier fields to all
existing JSON files in the 1782_discovery folder that don't already have them.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CourtListener client
import importlib.util
spec = importlib.util.spec_from_file_location(
    "download_case_law",
    Path(__file__).parent.parent / "document_ingestion" / "download_case_law.py"
)
download_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_module)
CourtListenerClient = download_module.CourtListenerClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def retrofit_hashes_to_json_files():
    """Add hash fields to all JSON files that don't already have them."""

    logger.info("="*80)
    logger.info("RETROFITTING HASH FIELDS TO 1782 JSON FILES")
    logger.info("="*80)

    # Initialize client to get hash methods
    client = CourtListenerClient()

    base_dir = Path("data/case_law/1782_discovery")
    if not base_dir.exists():
        logger.error(f"Directory not found: {base_dir}")
        return

    # Track statistics
    total_files = 0
    files_processed = 0
    files_skipped = 0
    files_with_errors = 0

    # Process all JSON files
    for json_file in base_dir.rglob("*.json"):
        total_files += 1

        try:
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Skip if already has hash fields
            if '_matrix_case_hash' in data and '_matrix_primary_identifier' in data:
                files_skipped += 1
                logger.debug(f"Skipped (already has hashes): {json_file.name}")
                continue

            # Skip non-case files (reports, test files, etc.)
            if json_file.name in [
                '1782_case_count_research_report.json',
                'drive_verification_results.json'
            ] or json_file.parent.name == 'cap_test':
                files_skipped += 1
                logger.debug(f"Skipped (non-case file): {json_file.name}")
                continue

            # Compute hash and identifier
            case_hash = client._compute_case_hash(data)
            primary_id = client._determine_primary_identifier(data, case_hash)

            if not case_hash:
                logger.warning(f"Could not compute hash for: {json_file.name}")
                files_with_errors += 1
                continue

            # Add hash fields to data
            data['_matrix_case_hash'] = case_hash
            data['_matrix_primary_identifier'] = primary_id

            # Write back to file
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            files_processed += 1
            logger.info(f"Added hashes to: {json_file.name}")

        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {e}")
            files_with_errors += 1

    # Report results
    logger.info("\n" + "="*80)
    logger.info("RETROFIT COMPLETE")
    logger.info("="*80)
    logger.info(f"Total JSON files found: {total_files}")
    logger.info(f"Files processed (hashes added): {files_processed}")
    logger.info(f"Files skipped (already had hashes): {files_skipped}")
    logger.info(f"Files with errors: {files_with_errors}")

    return {
        'total_files': total_files,
        'files_processed': files_processed,
        'files_skipped': files_skipped,
        'files_with_errors': files_with_errors
    }


if __name__ == "__main__":
    retrofit_hashes_to_json_files()
