#!/usr/bin/env python3
"""
Normalize Legacy Files - Add Missing Hash Metadata

This script retrofits the missing _matrix_case_hash and _matrix_primary_identifier
fields to legacy JSON files using the CourtListenerClient's internal methods.
"""

import sys
import json
import logging
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CourtListener client directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "download_case_law",
    Path(__file__).parent.parent / "document_ingestion" / "download_case_law.py"
)
cl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl_module)
CourtListenerClient = cl_module.CourtListenerClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_legacy_files():
    """Normalize legacy files by adding missing hash metadata."""

    logger.info("="*80)
    logger.info("NORMALIZING LEGACY FILES - ADDING MISSING HASH METADATA")
    logger.info("="*80)

    # Initialize client to reuse its methods
    client = CourtListenerClient()
    topic_dir = client.local_dir / "1782_discovery"

    updated_count = 0
    skipped_count = 0
    error_count = 0

    # Process all JSON files
    for path in topic_dir.rglob("*.json"):
        try:
            # Skip analysis/report files
            if any(skip in path.name.lower() for skip in ['report', 'analysis', 'index', 'backup']):
                logger.info(f"⏭️  Skipping analysis file: {path.name}")
                skipped_count += 1
                continue

            # Load file data
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            # Skip if already normalized
            if "_matrix_case_hash" in data:
                logger.info(f"⏭️  Already normalized: {path.name}")
                skipped_count += 1
                continue

            # Compute missing metadata using client's methods
            case_hash = client._compute_case_hash(data)
            primary_id = client._determine_primary_identifier(data, case_hash)

            # Add metadata
            data["_matrix_case_hash"] = case_hash
            data["_matrix_primary_identifier"] = primary_id

            # Save updated file
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)

            updated_count += 1
            logger.info(f"✅ Updated: {path.name}")

        except Exception as e:
            logger.error(f"❌ Error processing {path.name}: {e}")
            error_count += 1

    # Summary
    logger.info("\n" + "="*80)
    logger.info("NORMALIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Files updated: {updated_count}")
    logger.info(f"Files skipped: {skipped_count}")
    logger.info(f"Errors: {error_count}")

    if updated_count > 0:
        logger.info("\n✅ Legacy files have been normalized with hash metadata")
        logger.info("Next step: Run analyze_duplicates.py to identify true duplicates")
    else:
        logger.info("\nℹ️  No files needed normalization")


if __name__ == "__main__":
    normalize_legacy_files()
