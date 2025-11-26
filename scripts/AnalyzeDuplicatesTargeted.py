#!/usr/bin/env python3
"""
Analyze Duplicates in Specific Directory

Modified version of analyze_duplicates.py that targets a specific directory.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_duplicates_in_directory(target_dir: str):
    """Analyze duplicate files by content hash in specific directory."""

    logger.info("="*80)
    logger.info(f"ANALYZING DUPLICATE FILES IN: {target_dir}")
    logger.info("="*80)

    base = Path(target_dir)
    if not base.exists():
        logger.error(f"Directory not found: {base}")
        return

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
        logger.info("\n✅ NO DUPLICATE CONTENT HASHES FOUND!")

    # Files without hash
    if files_without_hash:
        logger.info(f"\nFiles without _matrix_case_hash ({len(files_without_hash)}):")
        for f in files_without_hash:
            logger.info(f"   {f}")

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
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data/case_law/The Art of War - Database"
    analyze_duplicates_in_directory(target_dir)
