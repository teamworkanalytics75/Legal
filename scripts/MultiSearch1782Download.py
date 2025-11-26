#!/usr/bin/env python3
"""
Multi-Search 1782 Download Script

This script searches multiple terms to find more 1782 cases.
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

def download_multi_search_1782_cases(max_cases_per_search=50):
    """Download 1782 cases using multiple search terms."""

    client = CourtListenerClient()

    # Multiple search terms that find 1782-related cases
    search_terms = [
        ['1782'],
        ['28 USC 1782'],
        ['28 U.S.C. 1782'],
        ['foreign tribunal'],
        ['Intel Corp'],
        ['Brandi-Dohrn'],
        ['Euromepa'],
        ['Macquarie'],
        ['del valle ruiz'],
        ['foreign proceeding'],
        ['international tribunal'],
        ['assistance foreign'],
        ['discovery foreign'],
        ['usc 1782']
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

            logger.info(f"Search '{search_term[0]}' complete: {search_downloaded} new cases")

        except Exception as e:
            logger.error(f"Error with search '{search_term[0]}': {e}")

    logger.info(f"Multi-search complete: {all_downloaded} total cases saved")
    return all_downloaded

if __name__ == "__main__":
    print("Multi-Search 1782 Download")
    print("=" * 50)

    # Download cases using multiple search terms
    downloaded = download_multi_search_1782_cases(max_cases_per_search=20)

    print(f"\nDownloaded {downloaded} cases to:")
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


