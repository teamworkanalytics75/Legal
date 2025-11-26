#!/usr/bin/env python3
"""
Phase 1: Download D. Mass + First Circuit + SCOTUS §1782 Cases

This script downloads all binding authority for Massachusetts §1782 filings:
- SCOTUS cases (Intel Corp, ZF Automotive, etc.)
- First Circuit appeals
- All D. Mass §1782 cases (published + unreported + docket entries)

Target: 50-100 cases
Timeline: 1-2 days
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
        logging.FileHandler("logs/download_1782_phase1.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Phase1Downloader:
    """Download Phase 1: All binding authority for D. Mass §1782 filings."""

    def __init__(self):
        """Initialize the Phase 1 downloader."""
        self.client = CourtListenerClient()
        self.base_dir = Path("data/case_law/1782_discovery/phase1_binding")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each court
        self.scotus_dir = self.base_dir / "scotus"
        self.ca1_dir = self.base_dir / "ca1"
        self.mad_dir = self.base_dir / "mad"

        for dir_path in [self.scotus_dir, self.ca1_dir, self.mad_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "scotus": 0,
            "ca1": 0,
            "mad": 0,
            "total": 0
        }

    def download_scotus_cases(self) -> int:
        """
        Download all SCOTUS §1782 cases.

        Expected cases:
        - Intel Corp v. AMD (2004)
        - ZF Automotive v. Luxshare (2022)
        - AlixPartners v. Fund for Protection (2022)

        Returns:
            Number of cases downloaded.
        """
        logger.info("=" * 60)
        logger.info("Downloading SCOTUS §1782 Cases")
        logger.info("=" * 60)

        # Search for SCOTUS cases mentioning §1782
        # Use simpler search terms - CourtListener API is sensitive to exact phrases
        search_terms = [
            "1782"
        ]

        downloaded = 0
        for term in search_terms:
            logger.info(f"Searching SCOTUS for: {term}")

            response = self.client.search_opinions(
                keywords=[term],
                courts=["scotus"],
                limit=100,
                include_non_precedential=True
            )

            if not response or 'results' not in response:
                logger.warning(f"No results for term: {term}")
                continue

            results = response['results']
            logger.info(f"Found {len(results)} results for '{term}'")

            for case in results:
                case_id = case.get('id', 'unknown')
                case_name = case.get('caseName', 'Unknown')

                # Save case
                filename = f"scotus_{case_id}_{self._clean_filename(case_name)}.json"
                filepath = self.scotus_dir / filename

                if filepath.exists():
                    logger.info(f"Skipping existing: {case_name}")
                    continue

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(case, f, indent=2, ensure_ascii=False)

                downloaded += 1
                logger.info(f"Downloaded SCOTUS case: {case_name}")

        self.stats["scotus"] = downloaded
        logger.info(f"SCOTUS download complete: {downloaded} cases")
        return downloaded

    def download_ca1_cases(self) -> int:
        """
        Download all First Circuit §1782 cases.

        Returns:
            Number of cases downloaded.
        """
        logger.info("=" * 60)
        logger.info("Downloading First Circuit §1782 Cases")
        logger.info("=" * 60)

        # Search for CA1 cases mentioning §1782
        search_terms = [
            "1782"
        ]

        downloaded = 0
        for term in search_terms:
            logger.info(f"Searching CA1 for: {term}")

            response = self.client.search_opinions(
                keywords=[term],
                courts=["ca1"],
                limit=100,
                include_non_precedential=True
            )

            if not response or 'results' not in response:
                logger.warning(f"No results for term: {term}")
                continue

            results = response['results']
            logger.info(f"Found {len(results)} results for '{term}'")

            for case in results:
                case_id = case.get('id', 'unknown')
                case_name = case.get('caseName', 'Unknown')

                # Save case
                filename = f"ca1_{case_id}_{self._clean_filename(case_name)}.json"
                filepath = self.ca1_dir / filename

                if filepath.exists():
                    logger.info(f"Skipping existing: {case_name}")
                    continue

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(case, f, indent=2, ensure_ascii=False)

                downloaded += 1
                logger.info(f"Downloaded CA1 case: {case_name}")

        self.stats["ca1"] = downloaded
        logger.info(f"First Circuit download complete: {downloaded} cases")
        return downloaded

    def download_dmass_cases(self) -> int:
        """
        Download all D. Mass §1782 cases.

        Uses multiple search strategies:
        1. Keyword search: "28 U.S.C. 1782"
        2. Case name patterns: "In re", "Application", "Pursuant to 28"
        3. Docket number pattern: "mc-" (miscellaneous cases)

        Returns:
            Number of cases downloaded.
        """
        logger.info("=" * 60)
        logger.info("Downloading D. Mass §1782 Cases")
        logger.info("=" * 60)

        # Strategy 1: Keyword search
        search_terms = [
            "1782"
        ]

        downloaded = 0
        seen_ids = set()

        for term in search_terms:
            logger.info(f"Searching D. Mass for: {term}")

            response = self.client.search_opinions(
                keywords=[term],
                courts=["mad"],
                limit=100,
                include_non_precedential=True
            )

            if not response or 'results' not in response:
                logger.warning(f"No results for term: {term}")
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
                    logger.info(f"Skipping non-§1782 case: {case_name}")
                    continue

                # Save case
                filename = f"mad_{case_id}_{self._clean_filename(case_name)}.json"
                filepath = self.mad_dir / filename

                if filepath.exists():
                    logger.info(f"Skipping existing: {case_name}")
                    continue

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(case, f, indent=2, ensure_ascii=False)

                # Save text file
                if case_text:
                    text_filepath = self.mad_dir / f"mad_{case_id}_{self._clean_filename(case_name)}.txt"
                    with open(text_filepath, 'w', encoding='utf-8') as f:
                        f.write(case_text)

                downloaded += 1
                logger.info(f"Downloaded D. Mass case: {case_name}")

        self.stats["mad"] = downloaded
        logger.info(f"D. Mass download complete: {downloaded} cases")
        return downloaded

    def _is_1782_case(self, case_text: str, case_name: str) -> bool:
        """
        Verify if a case is actually about §1782 discovery.

        Args:
            case_text: Full text of the case.
            case_name: Name of the case.

        Returns:
            True if this is a §1782 case, False otherwise.
        """
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
        """
        Clean a case name for use as a filename.

        Args:
            name: Case name.

        Returns:
            Cleaned filename.
        """
        # Remove invalid characters
        clean = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_'))
        # Limit length
        clean = clean[:100].strip()
        return clean

    def run(self) -> Dict[str, int]:
        """
        Run Phase 1 download: SCOTUS + CA1 + D. Mass.

        Returns:
            Download statistics.
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Binding Authority Download")
        logger.info("=" * 60)
        logger.info(f"Started at: {datetime.now()}")

        try:
            # Download SCOTUS cases
            self.download_scotus_cases()

            # Download First Circuit cases
            self.download_ca1_cases()

            # Download D. Mass cases
            self.download_dmass_cases()

            # Calculate totals
            self.stats["total"] = sum([
                self.stats["scotus"],
                self.stats["ca1"],
                self.stats["mad"]
            ])

            # Print summary
            logger.info("=" * 60)
            logger.info("PHASE 1 DOWNLOAD COMPLETE")
            logger.info("=" * 60)
            logger.info(f"SCOTUS cases: {self.stats['scotus']}")
            logger.info(f"First Circuit cases: {self.stats['ca1']}")
            logger.info(f"D. Mass cases: {self.stats['mad']}")
            logger.info(f"Total cases: {self.stats['total']}")
            logger.info(f"Completed at: {datetime.now()}")
            logger.info("=" * 60)

            return self.stats

        except Exception as e:
            logger.error(f"Error during Phase 1 download: {e}", exc_info=True)
            return self.stats


def main():
    """Main entry point."""
    print("Phase 1: D. Mass + First Circuit + SCOTUS §1782 Download")
    print("=" * 60)

    downloader = Phase1Downloader()
    stats = downloader.run()

    print("\nDownload Summary:")
    print(f"SCOTUS: {stats['scotus']} cases")
    print(f"First Circuit: {stats['ca1']} cases")
    print(f"D. Mass: {stats['mad']} cases")
    print(f"Total: {stats['total']} cases")
    print("\nFiles saved to: data/case_law/1782_discovery/phase1_binding/")


if __name__ == "__main__":
    main()

