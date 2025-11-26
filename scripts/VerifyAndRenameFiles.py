#!/usr/bin/env python3
"""
Verify and Rename 1782 Case Files

This script checks if files are actually about 1782 discovery and renames them properly.
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

    # Positive indicators for 1782 discovery
    positive_indicators = [
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
        'intel corp v. advanced micro devices',
        'brandi-dohrn',
        'euromepa',
        'macquarie',
        'del valle ruiz'
    ]

    # Check for positive indicators
    has_positive = any(indicator in all_text for indicator in positive_indicators)

    return has_positive

def create_clean_filename(case):
    """Create a clean filename from case data."""

    case_name = case.get('caseName', 'Unknown')
    case_id = case.get('id', 'unknown')

    # Create safe filename
    safe_name = "".join(c for c in case_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name[:100]  # Limit length

    return f"{case_id}_{safe_name}.json"

def verify_and_rename_files():
    """Check files and rename them with proper case names."""

    case_dir = Path('data/case_law/1782_discovery')
    if not case_dir.exists():
        logger.error("Case directory doesn't exist")
        return

    json_files = list(case_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to check")

    verified_count = 0
    renamed_count = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                case = json.load(f)

            case_name = case.get('caseName', 'Unknown')

            # Check if it's actually a 1782 case
            if is_1782_discovery_case(case):
                verified_count += 1
                logger.info(f"✓ VERIFIED: {case_name}")

                # Create clean filename
                clean_filename = create_clean_filename(case)
                clean_path = case_dir / clean_filename

                # Only rename if the filename is different
                if json_file.name != clean_filename:
                    if not clean_path.exists():
                        json_file.rename(clean_path)
                        renamed_count += 1
                        logger.info(f"  Renamed: {json_file.name} → {clean_filename}")
                    else:
                        logger.info(f"  Skipped: {clean_filename} already exists")
                else:
                    logger.info(f"  Already clean: {json_file.name}")
            else:
                logger.warning(f"✗ NOT 1782: {case_name}")

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")

    logger.info(f"Verification complete: {verified_count} verified 1782 cases, {renamed_count} files renamed")

if __name__ == "__main__":
    print("Verify and Rename 1782 Case Files")
    print("=" * 50)

    verify_and_rename_files()


