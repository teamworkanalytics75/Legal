#!/usr/bin/env python3
"""
Test New Cursor-Based Pagination System

Tests the updated downloader with cursor-based pagination and checkpoint rebuilding.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

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


class CursorPaginationTester:
    """Tests the new cursor-based pagination system."""

    def __init__(self):
        """Initialize the tester."""
        self.client = CourtListenerClient()

    def test_checkpoint_rebuild(self):
        """Test checkpoint rebuilding from existing files."""
        logger.info("Testing checkpoint rebuild for 1782_discovery...")

        # Load checkpoint to trigger rebuild
        checkpoint_state = self.client._load_checkpoint("1782_discovery")

        logger.info(f"Checkpoint state after rebuild:")
        logger.info(f"  Downloaded IDs: {len(checkpoint_state['downloaded_ids'])}")
        logger.info(f"  Case hashes: {len(checkpoint_state['case_hashes'])}")
        logger.info(f"  Last cursor: {checkpoint_state.get('last_cursor')}")
        logger.info(f"  Total downloaded: {checkpoint_state.get('total_downloaded', 0)}")

        return checkpoint_state

    def test_cursor_pagination(self, max_results: int = 50):
        """Test cursor-based pagination with a small sample."""
        logger.info(f"Testing cursor-based pagination with {max_results} results...")

        # Test with 1782 keywords
        keywords = ["1782", "foreign tribunal", "international discovery"]
        courts = ["mass", "ca1", "scotus"]  # Focus on key courts

        logger.info(f"Search parameters:")
        logger.info(f"  Keywords: {keywords}")
        logger.info(f"  Courts: {courts}")
        logger.info(f"  Max results: {max_results}")

        # Run bulk download with resume=True
        results = self.client.bulk_download(
            topic="test_cursor_pagination",
            courts=courts,
            keywords=keywords,
            max_results=max_results,
            resume=True
        )

        logger.info(f"Download completed:")
        logger.info(f"  Results downloaded: {len(results)}")

        # Check the checkpoint file
        checkpoint_file = self.client._get_checkpoint_file("test_cursor_pagination")
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            logger.info(f"Checkpoint file contents:")
            logger.info(f"  Version: {checkpoint_data.get('version')}")
            logger.info(f"  Downloaded IDs: {len(checkpoint_data.get('downloaded_ids', []))}")
            logger.info(f"  Case hashes: {len(checkpoint_data.get('case_hashes', []))}")
            logger.info(f"  Last cursor: {checkpoint_data.get('last_cursor')}")
            logger.info(f"  Total downloaded: {checkpoint_data.get('total_downloaded', 0)}")
            logger.info(f"  Last updated: {checkpoint_data.get('last_updated')}")

        return results

    def test_1782_discovery_resume(self, max_results: int = 100):
        """Test resuming 1782_discovery download."""
        logger.info(f"Testing 1782_discovery resume with {max_results} additional results...")

        # Load current checkpoint
        checkpoint_state = self.client._load_checkpoint("1782_discovery")
        initial_count = len(checkpoint_state['downloaded_ids'])

        logger.info(f"Current state:")
        logger.info(f"  Existing cases: {initial_count}")
        logger.info(f"  Last cursor: {checkpoint_state.get('last_cursor')}")

        # Run bulk download with resume=True
        keywords = ["1782", "foreign tribunal", "international discovery"]
        courts = ["mass", "ca1", "scotus", "d.d.c.", "s.d.n.y.", "n.d. cal."]

        results = self.client.bulk_download(
            topic="1782_discovery",
            courts=courts,
            keywords=keywords,
            max_results=max_results,
            resume=True
        )

        # Check final state
        final_checkpoint_state = self.client._load_checkpoint("1782_discovery")
        final_count = len(final_checkpoint_state['downloaded_ids'])

        logger.info(f"Final state:")
        logger.info(f"  Total cases: {final_count}")
        logger.info(f"  New cases added: {final_count - initial_count}")
        logger.info(f"  Results returned: {len(results)}")
        logger.info(f"  Last cursor: {final_checkpoint_state.get('last_cursor')}")

        return results

    def analyze_checkpoint_structure(self):
        """Analyze the structure of the checkpoint file."""
        logger.info("Analyzing checkpoint file structure...")

        checkpoint_file = self.client._get_checkpoint_file("1782_discovery")
        if not checkpoint_file.exists():
            logger.warning("No 1782_discovery checkpoint file found")
            return

        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        logger.info(f"Checkpoint file analysis:")
        logger.info(f"  File size: {checkpoint_file.stat().st_size} bytes")
        logger.info(f"  Version: {checkpoint_data.get('version')}")
        logger.info(f"  Topic: {checkpoint_data.get('topic')}")
        logger.info(f"  Downloaded IDs: {len(checkpoint_data.get('downloaded_ids', []))}")
        logger.info(f"  Case hashes: {len(checkpoint_data.get('case_hashes', []))}")
        logger.info(f"  Count: {checkpoint_data.get('count')}")
        logger.info(f"  Total downloaded: {checkpoint_data.get('total_downloaded')}")
        logger.info(f"  Last cursor: {checkpoint_data.get('last_cursor')}")
        logger.info(f"  Last updated: {checkpoint_data.get('last_updated')}")

        # Sample some IDs and hashes
        downloaded_ids = checkpoint_data.get('downloaded_ids', [])
        case_hashes = checkpoint_data.get('case_hashes', [])

        if downloaded_ids:
            logger.info(f"  Sample downloaded IDs: {downloaded_ids[:3]}...")
        if case_hashes:
            logger.info(f"  Sample case hashes: {case_hashes[:3]}...")

        return checkpoint_data


def main():
    """Main entry point."""
    tester = CursorPaginationTester()

    logger.info("="*80)
    logger.info("TESTING NEW CURSOR-BASED PAGINATION SYSTEM")
    logger.info("="*80)

    # Test 1: Checkpoint rebuild
    logger.info("\n1. Testing checkpoint rebuild...")
    checkpoint_state = tester.test_checkpoint_rebuild()

    # Test 2: Analyze checkpoint structure
    logger.info("\n2. Analyzing checkpoint structure...")
    tester.analyze_checkpoint_structure()

    # Test 3: Small cursor pagination test
    logger.info("\n3. Testing cursor pagination with small sample...")
    small_results = tester.test_cursor_pagination(max_results=20)

    # Test 4: Resume 1782_discovery
    logger.info("\n4. Testing 1782_discovery resume...")
    resume_results = tester.test_1782_discovery_resume(max_results=50)

    logger.info("\n" + "="*80)
    logger.info("CURSOR PAGINATION TESTING COMPLETE")
    logger.info("="*80)
    logger.info(f"Small test results: {len(small_results)} cases")
    logger.info(f"Resume test results: {len(resume_results)} cases")
    logger.info("Check the checkpoint files for detailed state information")


if __name__ == "__main__":
    main()
