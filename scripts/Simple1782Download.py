#!/usr/bin/env python3
"""
Simple 1782 Download Script - Actually Downloads Cases

This script directly downloads 1782 cases without complex filtering.
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

def download_1782_cases_direct(max_cases=50, batch_size=20):
    """Download 1782 cases directly without complex filtering."""

    client = CourtListenerClient()

    # Simple search that we know works
    logger.info(f"Searching for {max_cases} 1782 cases with 'usc 1782'...")

    all_downloaded = 0
    offset = 0

    while all_downloaded < max_cases:
        # Calculate how many to request in this batch
        remaining = max_cases - all_downloaded
        current_batch_size = min(batch_size, remaining)

        logger.info(f"Batch: requesting {current_batch_size} cases (offset {offset})")

        response = client.search_opinions(keywords=['1782'], limit=current_batch_size, offset=offset)

        if not response or 'results' not in response:
            logger.error("No response from API")
            break

        results = response['results']
        if not results:
            logger.info("No more results available")
            break

        logger.info(f"Found {len(results)} cases in this batch")

        batch_downloaded = 0
        for i, case in enumerate(results):
            try:
                # Save the case directly
                case_id = case.get('id', f'case_{offset + i}')
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

                batch_downloaded += 1
                all_downloaded += 1
                logger.info(f"Downloaded {all_downloaded}/{max_cases}: {case_name}")

            except Exception as e:
                logger.error(f"Error downloading case {offset + i}: {e}")

        logger.info(f"Batch complete: {batch_downloaded} new cases downloaded")

        # Move to next batch
        offset += len(results)

        # If we got fewer results than requested, we've reached the end
        if len(results) < current_batch_size:
            logger.info("Reached end of available results")
            break

    logger.info(f"Download complete: {all_downloaded} cases saved")
    return all_downloaded

if __name__ == "__main__":
    print("Simple 1782 Download - Batch Approach")
    print("=" * 50)

    # Download 500 cases in batches of 50
    downloaded = download_1782_cases_direct(max_cases=500, batch_size=50)

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
