#!/usr/bin/env python3
"""
Precise 1782 Discovery Filter Script

This script downloads cases and filters them to only include actual 1782 discovery cases.
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

    # Negative indicators (cases that mention 1782 but aren't about it)
    negative_indicators = [
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
        'criminal appeal'
    ]

    # Check for positive indicators
    has_positive = any(indicator in all_text for indicator in positive_indicators)

    # Check for negative indicators
    has_negative = any(indicator in all_text for indicator in negative_indicators)

    # Case is 1782 discovery if it has positive indicators and no strong negative indicators
    is_1782 = has_positive and not has_negative

    if is_1782:
        logger.info(f"✓ 1782 Discovery: {case.get('caseName', 'Unknown')}")
    else:
        logger.info(f"✗ Not 1782: {case.get('caseName', 'Unknown')}")

    return is_1782

def download_precise_1782_cases(max_cases_per_search=50):
    """Download cases using multiple search terms and filter for actual 1782 cases."""

    client = CourtListenerClient()

    # Search terms that are more likely to find 1782 cases
    search_terms = [
        ['28 USC 1782'],
        ['28 U.S.C. 1782'],
        ['foreign tribunal'],
        ['foreign proceeding'],
        ['international tribunal'],
        ['Intel Corp'],
        ['Brandi-Dohrn'],
        ['Euromepa'],
        ['Macquarie'],
        ['del valle ruiz'],
        ['discovery foreign'],
        ['assistance foreign'],
        ['judicial assistance'],
        ['application pursuant to 28'],
        ['order pursuant to 28']
    ]

    all_downloaded = 0
    all_cases = set()  # Track unique cases by ID

    for search_term in search_terms:
        logger.info(f"Searching with: {search_term[0]}")

        try:
            response = client.search_opinions(keywords=search_term, limit=max_cases_per_search)

            if not response or 'results' not in response:
                logger.warning(f"No response for search: {search_term[0]}")
                continue

            results = response['results']
            logger.info(f"Found {len(results)} cases for '{search_term[0]}'")

            search_downloaded = 0
            for i, case in enumerate(results):
                case_id = case.get('id', f'search_{search_term[0]}_{i}')

                # Skip if we've already processed this case
                if case_id in all_cases:
                    continue

                all_cases.add(case_id)

                # Filter for actual 1782 discovery cases
                if not is_1782_discovery_case(case):
                    continue

                try:
                    case_name = case.get('caseName', 'Unknown')

                    # Create safe filename
                    safe_name = "".join(c for c in case_name if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_name = safe_name[:100]

                    # Save JSON file
                    json_file = Path("data/case_law/1782_discovery") / f"{case_id}_{safe_name}.json"
                    json_file.parent.mkdir(parents=True, exist_ok=True)

                    # Skip if file already exists
                    if json_file.exists():
                        logger.info(f"Skipping existing file: {json_file.name}")
                        continue

                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(case, f, indent=2, ensure_ascii=False)

                    # Save text file if available
                    text = case.get('plain_text') or case.get('html_with_citations', '')
                    if text:
                        text_file = Path("data/case_law/1782_discovery") / f"{case_id}_{safe_name}.txt"
                        with open(text_file, 'w', encoding='utf-8') as f:
                            f.write(text)

                    search_downloaded += 1
                    all_downloaded += 1
                    logger.info(f"Downloaded {all_downloaded}: {case_name}")

                except Exception as e:
                    logger.error(f"Error downloading case {case_id}: {e}")

            logger.info(f"Search '{search_term[0]}' complete: {search_downloaded} new 1782 cases")

        except Exception as e:
            logger.error(f"Error with search '{search_term[0]}': {e}")

    logger.info(f"Precise 1782 search complete: {all_downloaded} actual 1782 cases saved")
    return all_downloaded

if __name__ == "__main__":
    print("Precise 1782 Discovery Download")
    print("=" * 50)

    # Download cases using multiple search terms with filtering
    downloaded = download_precise_1782_cases(max_cases_per_search=20)

    print(f"\nDownloaded {downloaded} actual 1782 discovery cases to:")
    print("data/case_law/1782_discovery/")

    # List the files
    case_dir = Path("data/case_law/1782_discovery")
    if case_dir.exists():
        files = list(case_dir.glob("*.json"))
        print(f"\nTotal files: {len(files)}")
        print("Recent files:")
        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            print(f"  - {f.name}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")


