#!/usr/bin/env python3
"""
Cleanup False Positive 1782 Cases

This script removes cases that mention "1782" but aren't actually about 28 U.S.C. § 1782 discovery.
"""

import sys
import json
import logging
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from download_case_law import CourtListenerClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def is_false_positive_1782(case):
    """Check if a case is a false positive (mentions 1782 but isn't about 1782 discovery)."""

    case_name = case.get('caseName', '').lower()
    case_text = case.get('plain_text', '').lower()
    html_text = case.get('html_with_citations', '').lower()

    # Combine all text for analysis
    all_text = f"{case_name} {case_text} {html_text}"

    # Strong indicators this is NOT a 1782 discovery case
    false_positive_indicators = [
        'docket no. 4d22-1782',
        'case no. 4d22-1782',
        'no. 4d22-1782',
        'docket number',
        'case number',
        'appeal no.',
        'appellate no.',
        'district court of appeal',
        'court of appeals',
        'state of florida',
        'people v.',
        'state v.',
        'united states v.',
        'criminal case',
        'criminal appeal',
        'terry cusick',
        'seacoast national bank',
        'price v. realty company',
        'martin misjuns',
        'zapet-alvarado',
        'road-con inc',
        'quantrell liquan mcdaniel',
        'hector ramos',
        'people v. hayon',
        'united states v. davis',
        'united states v. emanuele palma',
        'davis v. state of florida',
        'terry cusick v. seacoast',
        'state of iowa v. david michael stephen bloomer',
        'r.k. v. r.g.'
    ]

    # Check for false positive indicators
    is_false_positive = any(indicator in all_text for indicator in false_positive_indicators)

    if is_false_positive:
        logger.info(f"✗ FALSE POSITIVE: {case.get('caseName', 'Unknown')}")
    else:
        logger.info(f"✓ KEEP: {case.get('caseName', 'Unknown')}")

    return is_false_positive

def cleanup_false_positives():
    """Remove false positive cases from local storage."""

    case_dir = Path("data/case_law/1782_discovery")
    if not case_dir.exists():
        logger.error("Case directory doesn't exist")
        return 0

    json_files = list(case_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to check")

    removed_count = 0
    kept_count = 0

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                case = json.load(f)

            if is_false_positive_1782(case):
                # Remove the JSON file
                json_file.unlink()
                removed_count += 1
                logger.info(f"Removed: {json_file.name}")

                # Also remove corresponding TXT file if it exists
                txt_file = json_file.with_suffix('.txt')
                if txt_file.exists():
                    txt_file.unlink()
                    logger.info(f"Removed: {txt_file.name}")
            else:
                kept_count += 1

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")

    logger.info(f"Cleanup complete: Removed {removed_count} false positives, kept {kept_count} real 1782 cases")
    return removed_count

if __name__ == "__main__":
    print("Cleanup False Positive 1782 Cases")
    print("=" * 50)

    removed = cleanup_false_positives()

    print(f"\nRemoved {removed} false positive cases")
    print("These cases mentioned '1782' but weren't about 28 U.S.C. § 1782 discovery")

    # List remaining files
    case_dir = Path("data/case_law/1782_discovery")
    if case_dir.exists():
        files = list(case_dir.glob("*.json"))
        print(f"\nRemaining files: {len(files)}")
        print("These are actual 1782 discovery cases:")
        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            print(f"  - {f.name}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")


