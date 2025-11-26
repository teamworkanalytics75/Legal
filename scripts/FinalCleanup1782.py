#!/usr/bin/env python3
"""
Final Cleanup of 1782 Cases

This script removes all files that are NOT actually about 1782 discovery and keeps only verified cases.
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def is_1782_discovery_case(case):
    """Check if a case is actually about 1782 discovery."""

    case_name = case.get('caseName', '').lower()
    case_text = case.get('plain_text', '').lower()
    html_text = case.get('html_with_citations', '').lower()

    # Combine all text for analysis
    all_text = f"{case_name} {case_text} {html_text}"

    # Strong positive indicators for 1782 discovery
    strong_indicators = [
        '28 u.s.c. 1782',
        '28 usc 1782',
        'section 1782',
        '§ 1782',
        'discovery for use in a foreign proceeding',
        'foreign tribunal',
        'international tribunal',
        'assistance before a foreign',
        'judicial assistance',
        'discovery pursuant to 28',
        'application pursuant to 28',
        'order pursuant to 28',
        'subpoena under 28',
        'intel corp v. advanced micro devices',  # The famous 1782 case
        'brandi-dohrn',
        'euromepa',
        'del valle ruiz'
    ]

    # Check for strong indicators
    has_strong_indicator = any(indicator in all_text for indicator in strong_indicators)

    # Additional check: if it mentions "1782" but not in the context of U.S.C., it's likely a false positive
    if '1782' in all_text and not any(usc_indicator in all_text for usc_indicator in ['28 u.s.c. 1782', '28 usc 1782', 'section 1782', '§ 1782']):
        return False

    return has_strong_indicator

def final_cleanup():
    """Remove all non-1782 cases and keep only verified 1782 discovery cases."""

    case_dir = Path('data/case_law/1782_discovery')
    if not case_dir.exists():
        logger.error("Case directory doesn't exist")
        return

    json_files = list(case_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to check")

    kept_count = 0
    removed_count = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                case = json.load(f)

            case_name = case.get('caseName', 'Unknown')

            # Check if it's actually a 1782 case
            if is_1782_discovery_case(case):
                kept_count += 1
                logger.info(f"✓ KEEP: {case_name}")
            else:
                # Remove the file
                json_file.unlink()
                removed_count += 1
                logger.info(f"✗ REMOVED: {case_name}")

                # Also remove corresponding TXT file if it exists
                txt_file = json_file.with_suffix('.txt')
                if txt_file.exists():
                    txt_file.unlink()

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")

    logger.info(f"Final cleanup complete: Kept {kept_count} verified 1782 cases, removed {removed_count} non-1782 cases")
    return kept_count

if __name__ == "__main__":
    print("Final Cleanup of 1782 Cases")
    print("=" * 50)

    kept = final_cleanup()

    print(f"\nFinal result: {kept} verified 1782 discovery cases remain")

    # List remaining files
    case_dir = Path('data/case_law/1782_discovery')
    if case_dir.exists():
        files = list(case_dir.glob("*.json"))
        print(f"\nRemaining files: {len(files)}")
        print("These are verified 1782 discovery cases:")
        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            print(f"  - {f.name}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")


