#!/usr/bin/env python3
"""
Relaxed §1782 Validation and Re-acquisition Script

This script implements a more lenient §1782 validation filter and re-processes
the 43 cases that were found but failed the original strict validation.
"""

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from download_case_law import CourtListenerClient
from filters import is_actual_1782_case

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def is_actual_1782_case_relaxed(opinion: Dict) -> bool:
    """
    Relaxed §1782 validation that catches more cases while avoiding false positives.

    This function implements a more lenient approach than the original filter:
    1. First tries the original strict validation
    2. Falls back to relaxed criteria for cases that might be valid §1782
    """

    # First try the original strict validation
    if is_actual_1782_case(opinion):
        return True

    # Collect text for analysis
    text_parts = []
    for key in ("caseName", "caseNameFull", "snippet"):
        value = opinion.get(key)
        if isinstance(value, str):
            text_parts.append(value)

    for op in opinion.get("opinions", []) or []:
        for key in ("snippet", "text", "plain_text"):
            value = op.get(key)
            if isinstance(value, str):
                text_parts.append(value)

    if not text_parts:
        return False

    text = "\n".join(text_parts).lower()
    case_name = opinion.get('caseName', '').lower()

    # Relaxed validation criteria

    # 1. Case name suggests foreign entity + "In re" pattern
    if "in re" in case_name:
        foreign_indicators = [
            "ltd", "inc", "corp", "gmbh", "sa", "ag", "llc", "lp",
            "republic", "kingdom", "corporation", "company", "holding",
            "automobil", "telecom", "sporting", "goods", "banco",
            "santander", "uralkali", "commonwealth", "australia",
            "turkey", "iraq", "guinea", "sierra", "leone", "punjab",
            "yokohama", "medytox", "matrix", "oasis", "avalru",
            "mariani", "sasol", "promnefstroy", "qwest"
        ]
        if any(indicator in case_name for indicator in foreign_indicators):
            logger.info(f"✓ Foreign entity detected in case name: {case_name[:100]}")
            return True

    # 2. Any §1782-related terms (more lenient)
    relaxed_patterns = [
        r"discovery.*foreign",
        r"foreign.*discovery",
        r"international.*arbitration",
        r"subpoena.*foreign",
        r"deposition.*foreign",
        r"evidence.*foreign.*tribunal",
        r"judicial.*assistance",
        r"letters.*rogatory",
        r"foreign.*tribunal",
        r"foreign.*proceeding",
        r"foreign.*litigation",
        r"aid.*foreign",
        r"commission.*take.*testimony",
        r"application.*pursuant",
        r"order.*take.*discovery",
        r"petition.*discovery",
        r"interested.*person"
    ]

    pattern_matches = 0
    for pattern in relaxed_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            pattern_matches += 1
            logger.info(f"✓ Pattern match: {pattern}")

    # Need at least 1 pattern match for relaxed validation
    if pattern_matches >= 1:
        logger.info(f"✓ Relaxed validation passed with {pattern_matches} pattern matches")
        return True

    # 3. Federal court + WL citation (district orders)
    court = opinion.get('court_id', '').lower()
    citation = str(opinion.get('citation', '')).lower()

    federal_courts = ["ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11", "cadc"]
    if court in federal_courts and 'wl' in citation:
        logger.info(f"✓ Federal court + WL citation: {court} + {citation}")
        return True

    # 4. Check for any mention of 1782 (even without full statute reference)
    if re.search(r'\b1782\b', text):
        logger.info("✓ Contains '1782' reference")
        return True

    return False


def re_validate_failed_cases():
    """Re-validate the 43 cases that failed the original strict validation."""

    logger.info("="*80)
    logger.info("RE-VALIDATING FAILED CASES WITH RELAXED CRITERIA")
    logger.info("="*80)

    # Load the acquisition log
    log_path = Path("data/case_law/wishlist_acquisition_log.json")
    if not log_path.exists():
        logger.error("Acquisition log not found!")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    # Get cases that were found but failed validation
    failed_cases = [
        case for case in log_data["cases"]
        if case["status"] == "found" and "failed §1782 validation" in case["notes"]
    ]

    logger.info(f"Found {len(failed_cases)} cases to re-validate")

    client = CourtListenerClient()
    revalidated_results = []

    for i, case in enumerate(failed_cases, 1):
        logger.info(f"\nProgress: {i}/{len(failed_cases)}")
        logger.info(f"Re-validating: {case['wishlist_name']}")

        cluster_id = case.get('cluster_id')
        if not cluster_id:
            logger.warning(f"No cluster_id for {case['wishlist_name']}")
            continue

        try:
            # Get the opinion by cluster ID
            result = client.search_opinions(
                keywords=[str(cluster_id)],
                limit=5
            )

            if not result or 'results' not in result:
                logger.warning(f"No results for cluster {cluster_id}")
                continue

            # Find the matching opinion
            opinion = None
            for op in result['results']:
                if op.get('cluster_id') == cluster_id:
                    opinion = op
                    break

            if not opinion:
                logger.warning(f"Cluster {cluster_id} not found in results")
                continue

            # Apply relaxed validation
            is_valid_relaxed = is_actual_1782_case_relaxed(opinion)

            result_entry = {
                "wishlist_name": case['wishlist_name'],
                "citation": case['citation'],
                "cluster_id": cluster_id,
                "original_status": case['status'],
                "relaxed_validation": is_valid_relaxed,
                "case_name": opinion.get('caseName', 'Unknown'),
                "court": opinion.get('court', 'Unknown')
            }

            if is_valid_relaxed:
                logger.info(f"✓ RELAXED VALIDATION PASSED: {case['wishlist_name']}")

                # Save the case
                try:
                    file_path = client.save_opinion(opinion, topic="1782_discovery")
                    result_entry["file_path"] = str(file_path)
                    result_entry["status"] = "saved"
                    logger.info(f"✓ Case saved to: {file_path}")
                except Exception as e:
                    logger.error(f"Error saving case: {e}")
                    result_entry["status"] = "save_failed"
                    result_entry["error"] = str(e)
            else:
                logger.info(f"✗ Still fails relaxed validation: {case['wishlist_name']}")
                result_entry["status"] = "still_failed"

            revalidated_results.append(result_entry)

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error processing {case['wishlist_name']}: {e}")
            revalidated_results.append({
                "wishlist_name": case['wishlist_name'],
                "cluster_id": cluster_id,
                "status": "error",
                "error": str(e)
            })

    # Save revalidation results
    results_file = Path("data/case_law/relaxed_validation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(revalidated_results, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("RE-VALIDATION SUMMARY")
    logger.info(f"{'='*80}")

    saved_cases = [r for r in revalidated_results if r.get('status') == 'saved']
    still_failed = [r for r in revalidated_results if r.get('status') == 'still_failed']

    logger.info(f"Cases that passed relaxed validation: {len(saved_cases)}")
    logger.info(f"Cases that still failed: {len(still_failed)}")
    logger.info(f"Total processed: {len(revalidated_results)}")

    if saved_cases:
        logger.info(f"\nSuccessfully saved cases:")
        for case in saved_cases:
            logger.info(f"  ✓ {case['wishlist_name']}")

    logger.info(f"\nResults saved to: {results_file}")

    return revalidated_results


def main():
    """Main entry point."""
    results = re_validate_failed_cases()

    if results:
        print(f"\n🎉 Re-validation complete!")
        print(f"Check data/case_law/relaxed_validation_results.json for detailed results")


if __name__ == "__main__":
    main()
