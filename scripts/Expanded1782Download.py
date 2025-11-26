#!/usr/bin/env python3
"""
Expanded 1782 Download Strategy

This script implements a multi-pronged approach to download section 1782 discovery cases
using expanded keywords, date ranges, and court-specific searches.
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CourtListener client
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


class Expanded1782Downloader:
    """Expanded section 1782 case downloader with multi-search strategy."""

    def __init__(self, topic: str = "1782_discovery", apply_filters: bool = True):
        """Initialize the downloader."""
        self.client = CourtListenerClient()
        self.topic = topic
        self.apply_filters = apply_filters

        # High-volume districts for section 1782 cases
        self.target_districts = [
            'mad',   # D. Mass
            'nys',   # SDNY
            'nye',   # EDNY
            'cand',  # ND Cal
            'cacd',  # CD Cal
            'flsd',  # SD Fla
            'txsd',  # SD Tex
            'ilnd',  # ND Ill
            'wawd',  # WD Wash
            'nynd',  # ND NY
        ]

        # Circuit courts
        self.target_circuits = [
            'ca1',   # 1st Circuit
            'ca2',   # 2nd Circuit
            'ca9',   # 9th Circuit
            'ca11',  # 11th Circuit
            'ca5',   # 5th Circuit
        ]

        # Targeted keyword combinations (balanced precision/recall)
        self.keyword_combinations = [
            ['"28 u.s.c. 1782"', '"judicial assistance"'],
            ['"section 1782"', '"foreign tribunal"'],
            ['"take discovery for use in"', '"section 1782"'],
            ['"application to take discovery"', '"foreign tribunal"'],
            ['"application pursuant to 28 u.s.c. 1782"', '"foreign proceeding"'],
            ['"order pursuant to 28 u.s.c. 1782"', '"discovery"'],
            ['"application for judicial assistance"', '"section 1782"'],
            ['"discovery in aid of foreign proceedings"', '"section 1782"'],
            ['"discovery for use in"', '"foreign proceeding"'],
            ['"for use in a foreign or international tribunal"', '"interested person"'],
            ['"letters rogatory"', '"section 1782"'],
            ['"letters of request"', '"section 1782"'],
            ['"request for judicial assistance"', '"section 1782"'],
            ['"order to take discovery"', '"foreign tribunal"'],
            ['"petition for discovery"', '"foreign proceeding"'],
            ['"petition to depose"', '"foreign tribunal"'],
            ['"petition to obtain discovery"', '"foreign proceeding"'],
            ['"commission to take testimony"', '"foreign proceeding"'],
            ['"commissioner"', '"foreign tribunal"'],
            ['"subpoena for use in"', '"foreign proceeding"'],
            ['"subpoena under 28 u.s.c. 1782"', '"order"'],
            ['"aid of foreign litigation"', '"order to take discovery"'],
            ['"foreign evidence gathering"', '"section 1782"'],
            ['"international arbitration"', '"section 1782"'],
            ['"intel factors"', '"section 1782"'],
            ['"brandi-dohrn"', '"section 1782"'],
            ['"zf automotive"', '"section 1782"'],
            ['"alixpartners"', '"section 1782"']
        ]

        # Track progress
        self.total_downloaded = 0
        self.duplicates_skipped = 0
        self.cursor_loops = 0

    def download_with_expanded_strategy(self, target_cases: int = 200):
        """Download cases using expanded multi-search strategy."""

        logger.info("=" * 80)
        logger.info("EXPANDED 1782 DOWNLOAD STRATEGY")
        logger.info("=" * 80)
        logger.info(f"Target: {target_cases} total cases")
        logger.info(f"Dataset topic: {self.topic}")
        logger.info(f"Section 1782 filter enabled: {self.apply_filters}")

        # Reset counters for each run
        self.total_downloaded = 0
        self.duplicates_skipped = 0
        self.cursor_loops = 0

        # Load existing checkpoint
        checkpoint_state = self.client._load_checkpoint(self.topic)
        existing_hashes = set(checkpoint_state.get('case_hashes', set()))
        existing_count = len(existing_hashes) if existing_hashes else len(checkpoint_state.get('downloaded_ids', []))

        logger.info(f"Starting with existing checkpoint: {existing_count} cases")

        logger.info(f"Existing cases: {existing_count}")
        logger.info(f"Need to find: {target_cases - existing_count} new cases")

        # Strategy 1: Broader date range searches
        logger.info("\n" + "="*60)
        logger.info("STRATEGY 1: BROADER DATE RANGE SEARCHES")
        logger.info("="*60)

        self._run_date_range_searches(existing_hashes, target_cases)

        # Strategy 2: District-specific searches
        logger.info("\n" + "="*60)
        logger.info("STRATEGY 2: DISTRICT-SPECIFIC SEARCHES")
        logger.info("="*60)

        self._run_district_searches(existing_hashes, target_cases)

        # Strategy 3: Circuit-specific searches
        logger.info("\n" + "="*60)
        logger.info("STRATEGY 3: CIRCUIT-SPECIFIC SEARCHES")
        logger.info("="*60)

        self._run_circuit_searches(existing_hashes, target_cases)

        # Strategy 4: Keyword sweeps with alternate phrasing
        logger.info("\n" + "="*60)
        logger.info("STRATEGY 4: KEYWORD-SPECIFIC SWEEPS")
        logger.info("="*60)

        self._run_keyword_sweeps(existing_hashes, target_cases)

        # Final summary
        logger.info("\n" + "="*80)
        logger.info("EXPANDED DOWNLOAD COMPLETE")
        logger.info("="*80)
        logger.info(f"Total cases downloaded: {self.total_downloaded}")
        logger.info(f"Duplicates skipped: {self.duplicates_skipped}")
        logger.info(f"Cursor loops detected: {self.cursor_loops}")

        return {
            'total_downloaded': self.total_downloaded,
            'duplicates_skipped': self.duplicates_skipped,
            'cursor_loops': self.cursor_loops
        }

    def _run_date_range_searches(self, existing_hashes: Set[str], target_cases: int):
        """Run searches with broader date ranges."""

        date_ranges = [
            ("1950-01-01", "1980-01-01"),  # Early cases
            ("1980-01-01", "2000-01-01"),  # Pre-Intel Corp era
            ("2000-01-01", "2010-01-01"),  # Intel Corp era
            ("2010-01-01", "2020-01-01"),  # Modern era
            ("2020-01-01", "2025-01-01"),  # Recent cases
        ]

        for start_date, end_date in date_ranges:
            if self.total_downloaded >= target_cases:
                break

            logger.info(f"Searching {start_date} to {end_date}")

            for keywords in self.keyword_combinations[:3]:  # Use first 3 keyword sets
                if self.total_downloaded >= target_cases:
                    break

                try:
                    opinions = self.client.bulk_download(
                        topic=self.topic,
                        courts=None,  # Search all courts
                        keywords=keywords,
                        date_after=start_date,
                        max_results=25,  # Smaller batches to avoid cursor loops
                        resume=True,
                        apply_filters=self.apply_filters
                    )

                    new_cases = self._process_downloaded_opinions(opinions, existing_hashes)
                    logger.info(f"  Found {new_cases} new cases with keywords: {keywords}")

                    time.sleep(2)  # Rate limiting

                except Exception as e:
                    logger.error(f"Error in date range search {start_date}-{end_date}: {e}")

    def _run_district_searches(self, existing_hashes: Set[str], target_cases: int):
        """Run district-specific searches."""

        for district in self.target_districts:
            if self.total_downloaded >= target_cases:
                break

            logger.info(f"Searching district: {district}")

            try:
                opinions = self.client.bulk_download(
                    topic=self.topic,
                    courts=[district],
                    keywords=self.keyword_combinations[0],
                    date_after="1950-01-01",
                    max_results=25,
                    resume=True,
                    apply_filters=self.apply_filters
                )

                new_cases = self._process_downloaded_opinions(opinions, existing_hashes)
                logger.info(f"  Found {new_cases} new cases in district {district}")

                time.sleep(2)  # Rate limiting

            except Exception as e:
                logger.error(f"Error in district search {district}: {e}")

    def _run_circuit_searches(self, existing_hashes: Set[str], target_cases: int):
        """Run circuit-specific searches."""

        for circuit in self.target_circuits:
            if self.total_downloaded >= target_cases:
                break

            logger.info(f"Searching circuit: {circuit}")

            try:
                opinions = self.client.bulk_download(
                    topic=self.topic,
                    courts=[circuit],
                    keywords=self.keyword_combinations[1],
                    date_after="1950-01-01",
                    max_results=25,
                    resume=True,
                    apply_filters=self.apply_filters
                )

                new_cases = self._process_downloaded_opinions(opinions, existing_hashes)
                logger.info(f"  Found {new_cases} new cases in circuit {circuit}")

                time.sleep(2)  # Rate limiting

            except Exception as e:
                logger.error(f"Error in circuit search {circuit}: {e}")

    def _run_keyword_sweeps(self, existing_hashes: Set[str], target_cases: int):
        """Run sweeps over the extended keyword combinations."""

        for index, keywords in enumerate(self.keyword_combinations):
            if self.total_downloaded >= target_cases:
                break

            # First three keyword sets are already covered in the date sweeps.
            if index < 3:
                continue

            logger.info(f"Keyword sweep {index + 1}: {keywords}")

            try:
                opinions = self.client.bulk_download(
                    topic=self.topic,
                    courts=None,
                    keywords=keywords,
                    date_after="1950-01-01",
                    max_results=25,
                    resume=True,
                    apply_filters=self.apply_filters
                )

                new_cases = self._process_downloaded_opinions(opinions, existing_hashes)
                logger.info(f"  Found {new_cases} new cases with keywords: {keywords}")

                time.sleep(2)

            except Exception as e:
                logger.error(f"Error in keyword sweep {keywords}: {e}")

    def _process_downloaded_opinions(self, opinions: List[Dict], existing_hashes: Set[str]) -> int:
        """Process downloaded opinions and count new cases."""

        new_cases = 0

        for opinion in opinions:
            case_hash = self.client._compute_case_hash(opinion)

            if case_hash in existing_hashes:
                self.duplicates_skipped += 1
                continue

            # Save the new case
            try:
                primary_id = self.client._determine_primary_identifier(opinion, case_hash)
                self.client.save_opinion(opinion, self.topic, case_hash, primary_id)

                existing_hashes.add(case_hash)
                new_cases += 1
                self.total_downloaded += 1

            except Exception as e:
                logger.error(f"Error saving opinion: {e}")

        return new_cases


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Expanded section 1782 downloader")
    parser.add_argument("--target", type=int, default=200, help="Desired total cases after download")
    parser.add_argument(
        "--topic",
        default="1782_discovery",
        help="Dataset topic name (used for local storage and checkpoints)."
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Disable the section 1782 filter and ingest all search hits."
    )
    args = parser.parse_args()

    downloader = Expanded1782Downloader(topic=args.topic, apply_filters=not args.skip_filter)
    results = downloader.download_with_expanded_strategy(target_cases=args.target)

    print(f"\nFinal Results:")
    print(f"Total downloaded: {results['total_downloaded']}")
    print(f"Duplicates skipped: {results['duplicates_skipped']}")
    print(f"Cursor loops: {results['cursor_loops']}")


if __name__ == "__main__":
    main()
