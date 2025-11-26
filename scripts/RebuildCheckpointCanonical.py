#!/usr/bin/env python3
"""
Rebuild Checkpoint for The Art of War - Database

This script rebuilds the checkpoint from the clean canonical database.
"""

import sys
import json
import logging
from pathlib import Path

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


def rebuild_checkpoint_for_canonical_db():
    """Rebuild checkpoint from The Art of War - Database."""

    logger.info("="*80)
    logger.info("REBUILDING CHECKPOINT FOR THE ART OF WAR - DATABASE")
    logger.info("="*80)

    # Initialize client
    client = CourtListenerClient()

    # Temporarily modify the client to use the canonical directory
    canonical_dir = Path("data/case_law/The Art of War - Database")

    # Create a temporary topic directory structure
    temp_topic_dir = client.local_dir / "1782_discovery"
    temp_topic_dir.mkdir(parents=True, exist_ok=True)

    # Copy files from canonical directory to temp topic directory
    logger.info(f"Copying files from {canonical_dir} to {temp_topic_dir}")
    for json_file in canonical_dir.glob("*.json"):
        shutil.copy2(json_file, temp_topic_dir / json_file.name)

        # Also copy matching TXT file if it exists
        txt_file = json_file.with_suffix('.txt')
        if txt_file.exists():
            shutil.copy2(txt_file, temp_topic_dir / txt_file.name)

    # Rebuild checkpoint from the temp directory
    logger.info("Rebuilding checkpoint from disk...")
    downloaded_ids, case_hashes = client._rebuild_checkpoint_from_disk("1782_discovery")

    # Create fresh checkpoint state
    checkpoint_state = {
        'topic': '1782_discovery',
        'downloaded_ids': list(downloaded_ids),
        'case_hashes': list(case_hashes),
        'count': len(downloaded_ids),
        'total_downloaded': len(downloaded_ids),
        'last_cursor': None,  # Start fresh for future downloads
        'version': 2,
        'last_updated': datetime.now().isoformat()
    }

    # Save the checkpoint
    client._save_checkpoint("1782_discovery", checkpoint_state)

    logger.info(f"\n" + "="*80)
    logger.info("CHECKPOINT REBUILD COMPLETE")
    logger.info("="*80)
    logger.info(f"Cases indexed: {len(downloaded_ids)}")
    logger.info(f"Unique hashes: {len(case_hashes)}")
    logger.info(f"Last cursor: None (fresh start)")
    logger.info(f"Checkpoint saved to: {client.checkpoint_dir / '1782_discovery_checkpoint.json'}")

    return {
        'cases_indexed': len(downloaded_ids),
        'unique_hashes': len(case_hashes),
        'checkpoint_path': str(client.checkpoint_dir / '1782_discovery_checkpoint.json')
    }


if __name__ == "__main__":
    import shutil
    from datetime import datetime
    rebuild_checkpoint_for_canonical_db()
