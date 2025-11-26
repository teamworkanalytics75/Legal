#!/usr/bin/env python3
"""
Test CourtListener API Search

This script tests the CourtListener API with different search parameters
to verify it's working and find the best approach for D. Mass cases.
"""

import sys
import json
import logging
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly to avoid __init__.py issues
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


def test_api_searches():
    """Test different search approaches to find the best one."""

    client = CourtListenerClient()

    # Test 1: Search all federal courts for "1782" (should find cases)
    logger.info("=" * 60)
    logger.info("Test 1: Search all federal courts for '1782'")
    logger.info("=" * 60)

    response = client.search_opinions(
        keywords=["1782"],
        courts=["ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11"],
        limit=20,
        include_non_precedential=True
    )

    if response and 'results' in response:
        results = response['results']
        logger.info(f"Found {len(results)} results across all circuits")

        # Show first 5 results
        for i, case in enumerate(results[:5]):
            court = case.get('court', 'Unknown')
            case_name = case.get('caseName', 'Unknown')
            logger.info(f"{i+1}. [{court}] {case_name}")
    else:
        logger.warning("No results found")

    # Test 2: Search specifically for D. Mass cases
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Search D. Mass specifically")
    logger.info("=" * 60)

    response = client.search_opinions(
        keywords=["1782"],
        courts=["mad"],
        limit=20,
        include_non_precedential=True
    )

    if response and 'results' in response:
        results = response['results']
        logger.info(f"Found {len(results)} D. Mass results")

        for i, case in enumerate(results):
            case_name = case.get('caseName', 'Unknown')
            court = case.get('court', 'Unknown')
            logger.info(f"{i+1}. [{court}] {case_name}")
    else:
        logger.warning("No D. Mass results found")

    # Test 3: Search for "Intel Corp" (should find the foundational case)
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Search for 'Intel Corp' (foundational §1782 case)")
    logger.info("=" * 60)

    response = client.search_opinions(
        keywords=["Intel Corp"],
        limit=10,
        include_non_precedential=True
    )

    if response and 'results' in response:
        results = response['results']
        logger.info(f"Found {len(results)} Intel Corp results")

        for i, case in enumerate(results):
            case_name = case.get('caseName', 'Unknown')
            court = case.get('court', 'Unknown')
            logger.info(f"{i+1}. [{court}] {case_name}")
    else:
        logger.warning("No Intel Corp results found")

    # Test 4: Search for "ZF Automotive" (recent SCOTUS case)
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Search for 'ZF Automotive' (recent SCOTUS §1782 case)")
    logger.info("=" * 60)

    response = client.search_opinions(
        keywords=["ZF Automotive"],
        limit=10,
        include_non_precedential=True
    )

    if response and 'results' in response:
        results = response['results']
        logger.info(f"Found {len(results)} ZF Automotive results")

        for i, case in enumerate(results):
            case_name = case.get('caseName', 'Unknown')
            court = case.get('court', 'Unknown')
            logger.info(f"{i+1}. [{court}] {case_name}")
    else:
        logger.warning("No ZF Automotive results found")


if __name__ == "__main__":
    print("CourtListener API Search Test")
    print("=" * 60)

    test_api_searches()
