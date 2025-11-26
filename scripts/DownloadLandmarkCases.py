#!/usr/bin/env python3
"""
Download Known Landmark §1782 Cases

This script downloads specific known §1782 cases by name,
then searches for related cases to build a comprehensive dataset.
"""

import sys
import json
import logging
from pathlib import Path

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LandmarkCaseDownloader:
    """Download known landmark §1782 cases and related cases."""

    def __init__(self):
        """Initialize the downloader."""
        self.client = CourtListenerClient()
        self.base_dir = Path("data/case_law/1782_discovery/landmark_cases")
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Known landmark §1782 cases
        self.landmark_cases = [
            # SCOTUS cases
            "Intel Corp. v. Advanced Micro Devices",
            "ZF Automotive U. S., Inc. v. Luxshare, Ltd.",
            "AlixPartners, LLP v. Fund for Protection of Investor Rights",

            # Circuit court cases
            "Brandi-Dohrn v. IKB Deutsche Industriebank AG",
            "Euromepa S.A. v. R. Esmerian, Inc.",
            "In re del Valle Ruiz",
            "In re Application of Chevron Corp.",
            "In re Application of O'Keeffe",

            # D. Mass cases (if any exist)
            "In re Peruvian Sporting Goods",
            "In re Hand Held Products Inc.",
        ]

        self.downloaded_count = 0

    def download_landmark_cases(self):
        """Download known landmark cases by name."""
        logger.info("=" * 60)
        logger.info("Downloading Landmark §1782 Cases")
        logger.info("=" * 60)

        for case_name in self.landmark_cases:
            logger.info(f"Searching for: {case_name}")

            # Search for the case by name
            response = self.client.search_opinions(
                keywords=[case_name],
                limit=10,
                include_non_precedential=True
            )

            if not response or 'results' not in response:
                logger.warning(f"No results for: {case_name}")
                continue

            results = response['results']
            logger.info(f"Found {len(results)} results for '{case_name}'")

            # Download the best match (first result)
            if results:
                case = results[0]
                case_id = case.get('id', 'unknown')
                found_name = case.get('caseName', 'Unknown')
                court = case.get('court', 'Unknown')

                # Save the case
                filename = f"landmark_{case_id}_{self._clean_filename(found_name)}.json"
                filepath = self.base_dir / filename

                if filepath.exists():
                    logger.info(f"Skipping existing: {found_name}")
                    continue

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(case, f, indent=2, ensure_ascii=False)

                # Save text if available
                text = case.get('plain_text') or case.get('html_with_citations', '')
                if text:
                    text_filepath = self.base_dir / f"landmark_{case_id}_{self._clean_filename(found_name)}.txt"
                    with open(text_filepath, 'w', encoding='utf-8') as f:
                        f.write(text)

                self.downloaded_count += 1
                logger.info(f"Downloaded: [{court}] {found_name}")

    def search_related_cases(self):
        """Search for cases that cite or are related to landmark cases."""
        logger.info("\n" + "=" * 60)
        logger.info("Searching for Related §1782 Cases")
        logger.info("=" * 60)

        # Search terms that should find §1782 cases
        search_terms = [
            "discovery for use in foreign proceeding",
            "foreign tribunal",
            "international tribunal",
            "judicial assistance",
            "28 USC 1782",
            "section 1782",
            "In re Application",
            "In re Request",
            "pursuant to 28"
        ]

        seen_ids = set()

        for term in search_terms:
            logger.info(f"Searching for: {term}")

            response = self.client.search_opinions(
                keywords=[term],
                limit=50,
                include_non_precedential=True
            )

            if not response or 'results' not in response:
                logger.warning(f"No results for: {term}")
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
                court = case.get('court', 'Unknown')

                # Verify it's actually a §1782 case
                case_text = case.get('plain_text', '') or case.get('html_with_citations', '')
                if not self._is_1782_case(case_text, case_name):
                    continue

                # Save the case
                filename = f"related_{case_id}_{self._clean_filename(case_name)}.json"
                filepath = self.base_dir / filename

                if filepath.exists():
                    continue

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(case, f, indent=2, ensure_ascii=False)

                # Save text if available
                if case_text:
                    text_filepath = self.base_dir / f"related_{case_id}_{self._clean_filename(case_name)}.txt"
                    with open(text_filepath, 'w', encoding='utf-8') as f:
                        f.write(case_text)

                self.downloaded_count += 1
                logger.info(f"Downloaded: [{court}] {case_name}")

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

    def run(self):
        """Run the landmark case download."""
        logger.info("Starting Landmark §1782 Case Download")
        logger.info(f"Target cases: {len(self.landmark_cases)}")

        try:
            # Download landmark cases
            self.download_landmark_cases()

            # Search for related cases
            self.search_related_cases()

            logger.info("=" * 60)
            logger.info("LANDMARK CASE DOWNLOAD COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total cases downloaded: {self.downloaded_count}")
            logger.info(f"Files saved to: {self.base_dir}")

        except Exception as e:
            logger.error(f"Error during download: {e}", exc_info=True)


def main():
    """Main entry point."""
    print("Landmark §1782 Case Download")
    print("=" * 60)

    downloader = LandmarkCaseDownloader()
    downloader.run()

    print(f"\nDownloaded {downloader.downloaded_count} landmark and related cases")
    print(f"Files saved to: data/case_law/1782_discovery/landmark_cases/")


if __name__ == "__main__":
    main()
