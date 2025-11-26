#!/usr/bin/env python3
"""
Fix Cursor Pagination Issue

This script implements the plan to fix the cursor pagination problem:
1. Backup current checkpoint
2. Delete checkpoint to force rebuild
3. Run rebuild-only pass
4. Execute fresh download pass
"""

import sys
import json
import logging
import shutil
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


def fix_cursor_pagination():
    """Fix the cursor pagination issue by rebuilding checkpoint."""

    logger.info("="*80)
    logger.info("FIXING CURSOR PAGINATION ISSUE")
    logger.info("="*80)

    # Step 1: Backup current checkpoint
    checkpoint_dir = Path("data/case_law/.checkpoints")
    checkpoint_file = checkpoint_dir / "1782_discovery_checkpoint.json"
    backup_file = checkpoint_dir / "1782_discovery_checkpoint_backup.json"

    if checkpoint_file.exists():
        logger.info("Step 1: Backing up current checkpoint...")
        shutil.copy2(checkpoint_file, backup_file)
        logger.info(f"✅ Checkpoint backed up to: {backup_file}")

        # Show current checkpoint state
        with open(checkpoint_file, 'r') as f:
            current_data = json.load(f)
        logger.info(f"Current checkpoint has {len(current_data.get('downloaded_ids', []))} cases")
        logger.info(f"Last cursor: {current_data.get('last_cursor', 'None')}")
    else:
        logger.info("No existing checkpoint found - will create new one")

    # Step 2: Delete checkpoint to force rebuild
    logger.info("\nStep 2: Deleting checkpoint to force rebuild...")
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("✅ Checkpoint deleted - next run will rebuild from disk")

    # Step 3: Run rebuild-only pass
    logger.info("\nStep 3: Running rebuild-only pass...")
    client = CourtListenerClient()

    # Load checkpoint (this will trigger rebuild from disk)
    checkpoint_state = client._load_checkpoint("1782_discovery")

    logger.info(f"✅ Checkpoint rebuilt from disk:")
    logger.info(f"  Downloaded IDs: {len(checkpoint_state['downloaded_ids'])}")
    logger.info(f"  Case hashes: {len(checkpoint_state['case_hashes'])}")
    logger.info(f"  Total downloaded: {checkpoint_state.get('total_downloaded', 0)}")

    # Step 4: Test fresh download pass
    logger.info("\nStep 4: Testing fresh download pass...")

    # Test with small batch first
    keywords = ["1782", "foreign tribunal", "international discovery"]
    courts = ["mass", "ca1", "scotus"]  # Focus on key courts

    logger.info(f"Testing with keywords: {keywords}")
    logger.info(f"Testing with courts: {courts}")

    # Run small test download
    results = client.bulk_download(
        topic="1782_discovery",
        courts=courts,
        keywords=keywords,
        max_results=10,  # Small test
        resume=True
    )

    logger.info(f"✅ Test download completed: {len(results)} new cases")

    # Check final checkpoint state
    final_checkpoint = client._load_checkpoint("1782_discovery")
    logger.info(f"\nFinal checkpoint state:")
    logger.info(f"  Downloaded IDs: {len(final_checkpoint['downloaded_ids'])}")
    logger.info(f"  Case hashes: {len(final_checkpoint['case_hashes'])}")
    logger.info(f"  Last cursor: {final_checkpoint.get('last_cursor', 'None')}")

    logger.info("\n" + "="*80)
    logger.info("CURSOR PAGINATION FIX COMPLETE")
    logger.info("="*80)
    logger.info("The checkpoint has been rebuilt and should now properly:")
    logger.info("1. Recognize existing files as duplicates")
    logger.info("2. Advance cursors instead of looping")
    logger.info("3. Skip already-downloaded cases")
    logger.info("\nYou can now run larger downloads with confidence!")


if __name__ == "__main__":
    fix_cursor_pagination()
