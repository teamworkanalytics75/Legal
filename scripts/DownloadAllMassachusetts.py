"""
The Matrix - Download ALL Massachusetts cases efficiently.

This downloads by court (not by keyword) which is much faster.
Then you filter locally with SQL queries.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, 'document_ingestion')

from download_case_law import CourtListenerClient
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("download_all_ma.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def download_all_by_court(court_id: str, date_after: str = "2010-01-01", max_cases: int = 5000):
    """Download all cases from a specific court."""
    client = CourtListenerClient()

    logger.info(f"Downloading ALL cases from court: {court_id}")
    logger.info(f"Date after: {date_after}, Max: {max_cases}")

    # Download without keyword filtering - much faster!
    cases = client.bulk_download(
        topic=f"court_{court_id}",
        courts=[court_id],
        keywords=None, # No keywords = all cases from this court
        date_after=date_after,
        max_results=max_cases
    )

    logger.info(f"Downloaded {len(cases)} cases from {court_id}")
    return cases


def main():
    """Download all Massachusetts courts."""

    # Massachusetts courts
    courts = {
        'mass': 'Massachusetts Supreme Judicial Court',
        'massappct': 'Massachusetts Appeals Court',
        'ca1': '1st Circuit Court of Appeals',
        'mad': 'District of Massachusetts',
    }

    print("\n" + "="*60)
    print("WITCHWEB - Download All Massachusetts Cases")
    print("="*60 + "\n")

    total = 0

    for court_id, court_name in courts.items():
        print(f"\n{'='*60}")
        print(f"Court: {court_name} ({court_id})")
        print(f"{'='*60}")

        cases = download_all_by_court(court_id, date_after="2010-01-01", max_cases=5000)
        total += len(cases)

        print(f"Downloaded: {len(cases)} cases")

    print("\n" + "="*60)
    print(f"TOTAL: {total} cases downloaded")
    print("="*60)

    print("\nNext steps:")
    print("1. Import to database:")
    print(" py run_case_law_pipeline.py --import")
    print()
    print("2. Filter with SQL:")
    print(" SELECT * FROM case_law WHERE opinion_text LIKE '%pseudonym%'")
    print()


if __name__ == "__main__":
    main()

