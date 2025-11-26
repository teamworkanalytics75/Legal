#!/usr/bin/env python3
"""
Analyze Duplicate Files by Content Hash

This script analyzes the 1782_discovery folder to identify duplicate files
based on their _matrix_case_hash values.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_duplicates():
    """Analyze duplicate files by content hash."""

    logger.info("="*80)
    logger.info("ANALYZING DUPLICATE FILES BY CONTENT HASH")
    logger.info("="*80)

    base = Path("data/case_law/1782_discovery")
    by_hash = defaultdict(list)
    files_without_hash = []

    # Scan all JSON files
    for path in base.rglob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            case_hash = data.get("_matrix_case_hash")
            if not case_hash:
                files_without_hash.append(path)
                continue

            by_hash[case_hash].append(path)

        except Exception as e:
            logger.error(f"Error reading {path}: {e}")

    # Report statistics
    total_files = len(list(base.rglob("*.json")))
    unique_hashes = len(by_hash)
    files_with_hash = sum(len(files) for files in by_hash.values())

    logger.info(f"Total JSON files: {total_files}")
    logger.info(f"Files with _matrix_case_hash: {files_with_hash}")
    logger.info(f"Files without _matrix_case_hash: {len(files_without_hash)}")
    logger.info(f"Unique content hashes: {unique_hashes}")

    # Find duplicates
    duplicates = {h: files for h, files in by_hash.items() if len(files) > 1}

    if duplicates:
        logger.info(f"\nFound {len(duplicates)} duplicate groups:")
        for h, files in duplicates.items():
            logger.info(f"\nHash {h[:12]}... -> {len(files)} copies:")
            for f in files:
                logger.info(f"   {f}")
    else:
        logger.info("\nNo duplicate content hashes found!")

    # Files without hash
    if files_without_hash:
        logger.info(f"\nFiles without _matrix_case_hash ({len(files_without_hash)}):")
        for f in files_without_hash[:10]:  # Show first 10
            logger.info(f"   {f}")
        if len(files_without_hash) > 10:
            logger.info(f"   ... and {len(files_without_hash) - 10} more")

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)

    return {
        'total_files': total_files,
        'unique_hashes': unique_hashes,
        'duplicates': duplicates,
        'files_without_hash': files_without_hash
    }


if __name__ == "__main__":
    analyze_duplicates()
