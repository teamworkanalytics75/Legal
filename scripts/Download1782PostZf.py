#!/usr/bin/env python3
"""
Phase 2: Download Post-ZF Automotive §1782 Cases (2022-present)

This script downloads all §1782 cases from 2022-present nationwide,
focusing on high-volume districts and active circuits to capture
the post-ZF Automotive regime change.

Target: 200-300 cases
Timeline: 1 week
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/download_1782_phase2.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Phase2Downloader:
    """Download Phase 2: All post-ZF Automotive §1782 cases (2022-present)."""

    def __init__(self):
        """Initialize the Phase 2 downloader."""
        self.client = CourtListenerClient()
        self.base_dir = Path("data/case_law/1782_discovery/phase2_post_zf")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # High-volume districts for §1782 cases
        self.high_volume_districts = [
            "nysd",  # Southern District of New York
            "nyed",  # Eastern District of New York
            "cand",  # Northern District of California
            "cacd",  # Central District of California
            "casd",  # Southern District of California
            "ded",   # District of Delaware
            "mad",   # District of Massachusetts
            "txnd",  # Northern District of Texas
            "txsd",  # Southern District of Texas
            "flsd",  # Southern District of Florida
        ]

        # Active circuit courts
        self.active_circuits = [
            "ca2",   # Second Circuit (NY)
            "ca9",   # Ninth Circuit (CA)
            "ca11",  # Eleventh Circuit (FL)
            "ca3",   # Third Circuit (DE)
            "ca1",   # First Circuit (MA)
        ]

        # ZF Automotive decision date (June 13, 2022)
        self.zf_date = "2022-06-13"

        self.stats = {
            "districts": {},
            "circuits": {},
            "total": 0
        }

    def download_high_volume_districts(self) -> int:
        """
        Download §1782 cases from high-volume districts (2022-present).

        Returns:
            Number of cases downloaded.
        """
        logger.info("=" * 60)
        logger.info("Downloading High-Volume District Cases (2022-present)")
        logger.info("=" * 60)

        total_downloaded = 0

        for district in self.high_volume_districts:
            logger.info(f"\n--- {district.upper()} ---")
            district_downloaded = 0

            # Search terms that should find §1782 cases
            search_terms = [
                "1782",
                "foreign tribunal",
                "international tribunal",
                "judicial assistance",
                "In re Application",
                "In re Request",
                "pursuant to 28"
            ]

            seen_ids = set()

            for term in search_terms:
                logger.info(f"Searching {district} for: {term}")

                response = self.client.search_opinions(
                    keywords=[term],
                    courts=[district],
                    date_filed_after=self.zf_date,
                    limit=50,
                    include_non_precedential=True
                )

                if not response or 'results' not in response:
                    continue

                results = response['results']
                logger.info(f"Found {len(results)} results for '{term}'")

                for case in results:
                    case_id = case.get('id', 'unknown')

                    # Skip duplicates
                    if case_id in seen_ids:
                        continue
                    seen_ids.add(case_id)

                    case_name = case.get('caseName', 'Unknown')

                    # Verify it's actually a §1782 case
                    case_text = case.get('plain_text', '') or case.get('html_with_citations', '')
                    if not self._is_1782_case(case_text, case_name):
                        continue

                    # Save the case
                    filename = f"{district}_{case_id}_{self._clean_filename(case_name)}.json"
                    filepath = self.base_dir / filename

                    if filepath.exists():
                        continue

                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(case, f, indent=2, ensure_ascii=False)

                    # Save text if available
                    if case_text:
                        text_filepath = self.base_dir / f"{district}_{case_id}_{self._clean_filename(case_name)}.txt"
                        with open(text_filepath, 'w', encoding='utf-8') as f:
                            f.write(case_text)

                    district_downloaded += 1
                    total_downloaded += 1
                    logger.info(f"Downloaded: [{district}] {case_name}")

            self.stats["districts"][district] = district_downloaded
            logger.info(f"{district.upper()} complete: {district_downloaded} cases")

        logger.info(f"\nHigh-volume districts complete: {total_downloaded} total cases")
        return total_downloaded

    def download_active_circuits(self) -> int:
        """
        Download §1782 cases from active circuit courts (2022-present).

        Returns:
            Number of cases downloaded.
        """
        logger.info("\n" + "=" * 60)
        logger.info("Downloading Active Circuit Court Cases (2022-present)")
        logger.info("=" * 60)

        total_downloaded = 0

        for circuit in self.active_circuits:
            logger.info(f"\n--- {circuit.upper()} ---")
            circuit_downloaded = 0

            # Search terms for circuit courts
            search_terms = [
                "1782",
                "foreign tribunal",
                "international tribunal",
                "judicial assistance",
                "Intel Corp",
                "ZF Automotive"
            ]

            seen_ids = set()

            for term in search_terms:
                logger.info(f"Searching {circuit} for: {term}")

                response = self.client.search_opinions(
                    keywords=[term],
                    courts=[circuit],
                    date_filed_after=self.zf_date,
                    limit=50,
                    include_non_precedential=True
                )

                if not response or 'results' not in response:
                    continue

                results = response['results']
                logger.info(f"Found {len(results)} results for '{term}'")

                for case in results:
                    case_id = case.get('id', 'unknown')

                    # Skip duplicates
                    if case_id in seen_ids:
                        continue
                    seen_ids.add(case_id)

                    case_name = case.get('caseName', 'Unknown')

                    # Verify it's actually a §1782 case
                    case_text = case.get('plain_text', '') or case.get('html_with_citations', '')
                    if not self._is_1782_case(case_text, case_name):
                        continue

                    # Save the case
                    filename = f"{circuit}_{case_id}_{self._clean_filename(case_name)}.json"
                    filepath = self.base_dir / filename

                    if filepath.exists():
                        continue

                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(case, f, indent=2, ensure_ascii=False)

                    # Save text if available
                    if case_text:
                        text_filepath = self.base_dir / f"{circuit}_{case_id}_{self._clean_filename(case_name)}.txt"
                        with open(text_filepath, 'w', encoding='utf-8') as f:
                            f.write(case_text)

                    circuit_downloaded += 1
                    total_downloaded += 1
                    logger.info(f"Downloaded: [{circuit}] {case_name}")

            self.stats["circuits"][circuit] = circuit_downloaded
            logger.info(f"{circuit.upper()} complete: {circuit_downloaded} cases")

        logger.info(f"\nActive circuits complete: {total_downloaded} total cases")
        return total_downloaded

    def _is_1782_case(self, case_text: str, case_name: str) -> bool:
        """Verify if a case is actually about §1782 discovery."""
        import re

        combined = f"{case_name} {case_text}".lower()

        # Strong indicators
        strong_indicators = [
            r"28 u\.s\.c\. ?§? ?1782",
            r"28 usc ?§? ?1782",
            r"section 1782",
            r"discovery for use in (a )?foreign",
            r"foreign or international tribunal",
            r"intel corp",  # Foundational case
            r"zf automotive",  # Recent SCOTUS case
        ]

        # False positive patterns (docket numbers)
        false_positives = [
            r"\b\d{1,2}d\d{2}-1782\b",  # e.g., "4D22-1782"
            r"case number.*1782",
            r"docket.*1782",
        ]

        # Must have strong indicator AND not be false positive
        has_indicator = any(re.search(ind, combined) for ind in strong_indicators)
        is_false_positive = any(re.search(fp, combined) for fp in false_positives)

        return has_indicator and not is_false_positive

    def _clean_filename(self, name: str) -> str:
        """Clean a case name for use as a filename."""
        clean = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_'))
        clean = clean[:100].strip()
        return clean

    def run(self) -> Dict[str, int]:
        """
        Run Phase 2 download: All post-ZF Automotive cases (2022-present).

        Returns:
            Download statistics.
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Post-ZF Automotive Download (2022-present)")
        logger.info("=" * 60)
        logger.info(f"Started at: {datetime.now()}")
        logger.info(f"ZF Automotive date: {self.zf_date}")

        try:
            # Download high-volume districts
            district_cases = self.download_high_volume_districts()

            # Download active circuits
            circuit_cases = self.download_active_circuits()

            # Calculate totals
            self.stats["total"] = district_cases + circuit_cases

            # Print summary
            logger.info("=" * 60)
            logger.info("PHASE 2 DOWNLOAD COMPLETE")
            logger.info("=" * 60)
            logger.info("District Cases:")
            for district, count in self.stats["districts"].items():
                logger.info(f"  {district.upper()}: {count}")
            logger.info("Circuit Cases:")
            for circuit, count in self.stats["circuits"].items():
                logger.info(f"  {circuit.upper()}: {count}")
            logger.info(f"Total cases: {self.stats['total']}")
            logger.info(f"Completed at: {datetime.now()}")
            logger.info("=" * 60)

            return self.stats

        except Exception as e:
            logger.error(f"Error during Phase 2 download: {e}", exc_info=True)
            return self.stats


def main():
    """Main entry point."""
    print("Phase 2: Post-ZF Automotive §1782 Download (2022-present)")
    print("=" * 60)

    downloader = Phase2Downloader()
    stats = downloader.run()

    print("\nDownload Summary:")
    print("Districts:")
    for district, count in stats["districts"].items():
        print(f"  {district.upper()}: {count}")
    print("Circuits:")
    for circuit, count in stats["circuits"].items():
        print(f"  {circuit.upper()}: {count}")
    print(f"Total: {stats['total']} cases")
    print("\nFiles saved to: data/case_law/1782_discovery/phase2_post_zf/")


if __name__ == "__main__":
    main()
