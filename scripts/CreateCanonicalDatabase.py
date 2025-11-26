#!/usr/bin/env python3
"""
Create Clean Canonical 1782 Database

This script creates a clean canonical folder "The Art of War - Database" containing
exactly one vetted copy per unique case, based on the hash analysis results.
"""

import sys
import json
import logging
import shutil
from pathlib import Path
from collections import defaultdict

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


def create_canonical_database():
    """Create clean canonical database with one copy per unique case."""

    logger.info("="*80)
    logger.info("CREATING CLEAN CANONICAL 1782 DATABASE")
    logger.info("="*80)

    # Initialize client for hash computation
    client = CourtListenerClient()

    # Define paths
    source_dir = Path("data/case_law/1782_discovery")
    canonical_dir = Path("data/case_law/The Art of War - Database")

    # Create canonical directory
    canonical_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created canonical directory: {canonical_dir}")

    # Group files by hash
    by_hash = defaultdict(list)
    files_without_hash = []

    # Scan all JSON files in source directory
    for json_file in source_dir.rglob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            case_hash = data.get("_matrix_case_hash")
            if not case_hash:
                files_without_hash.append(json_file)
                continue

            by_hash[case_hash].append(json_file)

        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}")

    logger.info(f"Found {len(by_hash)} unique hash groups")
    logger.info(f"Found {len(files_without_hash)} files without hash")

    # Process each hash group
    files_migrated = 0
    cluster_ids = set()

    for case_hash, files in by_hash.items():
        logger.info(f"\nProcessing hash group: {case_hash[:12]}... ({len(files)} copies)")

        # Find the best canonical copy (prefer case_XX_*.json files)
        canonical_file = None
        for file_path in files:
            file_name = file_path.name

            # Skip non-case files
            if file_name in [
                '1782_case_count_research_report.json',
                'drive_verification_results.json'
            ]:
                continue

            # Skip files in subdirectories
            if file_path.parent.name in ['cap_test', 'landmark_cases', 'phase1_binding', 'phase2_post_zf']:
                continue

            # Prefer case_XX_*.json files
            if file_name.startswith('case_') and file_name.endswith('.json'):
                canonical_file = file_path
                break

        # If no case_XX_*.json found, use the first non-search, non-unknown file
        if not canonical_file:
            for file_path in files:
                file_name = file_path.name

                # Skip search snapshots and unknown files
                if (file_name.startswith('search_') or
                    file_name.startswith('unknown_') or
                    file_name.startswith('case_') or
                    file_path.parent.name in ['cap_test', 'landmark_cases', 'phase1_binding', 'phase2_post_zf']):
                    continue

                canonical_file = file_path
                break

        # If still no canonical file, use the first file
        if not canonical_file:
            canonical_file = files[0]

        logger.info(f"  Selected canonical copy: {canonical_file.name}")

        # Copy JSON file to canonical directory
        dest_json = canonical_dir / canonical_file.name
        shutil.copy2(canonical_file, dest_json)
        files_migrated += 1

        # Copy matching TXT file if it exists
        txt_file = canonical_file.with_suffix('.txt')
        if txt_file.exists():
            dest_txt = canonical_dir / txt_file.name
            shutil.copy2(txt_file, dest_txt)
            logger.info(f"  Also copied: {txt_file.name}")

        # Extract cluster ID for validation
        try:
            with open(canonical_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cluster_id = data.get('cluster_id')
            if cluster_id:
                cluster_ids.add(cluster_id)
        except Exception as e:
            logger.warning(f"  Could not extract cluster ID from {canonical_file.name}: {e}")

    # Report results
    logger.info("\n" + "="*80)
    logger.info("CANONICAL DATABASE CREATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Files migrated to canonical directory: {files_migrated}")
    logger.info(f"Unique cluster IDs found: {len(cluster_ids)}")
    logger.info(f"Canonical directory: {canonical_dir}")

    # List cluster IDs for validation
    if cluster_ids:
        logger.info(f"\nCluster IDs in canonical database:")
        for cluster_id in sorted(cluster_ids):
            logger.info(f"  {cluster_id}")

    return {
        'files_migrated': files_migrated,
        'unique_clusters': len(cluster_ids),
        'canonical_dir': str(canonical_dir)
    }


if __name__ == "__main__":
    create_canonical_database()
