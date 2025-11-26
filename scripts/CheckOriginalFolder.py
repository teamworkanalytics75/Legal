#!/usr/bin/env python3
"""
Check Original Folder for More Cases

Check the original folder the user referenced to see if we missed cases
and compare with our current clean folder.
"""

import sys
import json
import logging
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Google Drive backup module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "google_drive_backup",
    Path(__file__).parent.parent / "document_ingestion" / "google_drive_backup.py"
)
gdrive_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdrive_module)
GoogleDriveBackup = gdrive_module.GoogleDriveBackup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OriginalFolderChecker:
    """Check the original folder for more cases."""

    def __init__(self):
        """Initialize the checker."""
        self.gdrive = GoogleDriveBackup()
        self.original_folder_id = "12pHMekx-GER8cPaMJv7aH3-El4vu7kbb"
        self.clean_folder_id = "1NjD3rHIAtCjzrVSbIQfJMB3-XLtIXG71"

        # Results
        self.original_cases = []
        self.clean_cases = []
        self.missing_cases = []

    def is_actual_1782_discovery(self, case_text: str, case_name: str) -> bool:
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
            r"alixpartners",  # Recent SCOTUS case
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

    def get_case_content_hash(self, file_info):
        """Get a hash of the case content for deduplication."""
        try:
            file_id = file_info['id']

            # Download file content
            request = self.gdrive.service.files().get_media(fileId=file_id)
            content = request.execute()

            # Parse JSON and extract key content
            try:
                case_data = json.loads(content.decode('utf-8'))

                # Create hash based on case name and key content
                case_name = case_data.get('caseName', '')
                plain_text = case_data.get('plain_text', '') or case_data.get('html_with_citations', '')

                # Use first 1000 chars of text for hash (enough to identify duplicates)
                text_sample = plain_text[:1000] if plain_text else ''

                # Create hash
                import hashlib
                content_hash = hashlib.md5(f"{case_name}|{text_sample}".encode()).hexdigest()

                return {
                    'file_info': file_info,
                    'case_name': case_name,
                    'content_hash': content_hash,
                    'court': case_data.get('court', 'Unknown'),
                    'date_filed': case_data.get('date_filed', 'Unknown'),
                    'plain_text': plain_text
                }

            except json.JSONDecodeError:
                # Not a JSON file
                case_name = file_info['name']
                content_hash = hashlib.md5(case_name.encode()).hexdigest()

                return {
                    'file_info': file_info,
                    'case_name': case_name,
                    'content_hash': content_hash,
                    'court': 'Unknown',
                    'date_filed': 'Unknown',
                    'plain_text': ''
                }

        except Exception as e:
            logger.error(f"Error processing {file_info['name']}: {e}")
            return None

    def analyze_folder(self, folder_id, folder_name):
        """Analyze a folder for §1782 cases."""
        try:
            logger.info(f"Analyzing {folder_name} folder...")

            # Get all files in folder
            results = self.gdrive.service.files().list(
                q=f"'{folder_id}' in parents",
                fields="files(id, name, mimeType, size, modifiedTime)",
                pageSize=1000
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in {folder_name}")

            # Process each file
            verified_cases = []
            false_positives = []

            for i, file_info in enumerate(files, 1):
                logger.info(f"Processing file {i}/{len(files)}: {file_info['name']}")

                case_info = self.get_case_content_hash(file_info)
                if not case_info:
                    continue

                # Verify if it's actually a §1782 case
                is_1782 = self.is_actual_1782_discovery(case_info['plain_text'], case_info['case_name'])

                if is_1782:
                    verified_cases.append(case_info)
                    logger.info(f"✅ Verified §1782: {case_info['case_name']}")
                else:
                    false_positives.append(case_info)
                    logger.info(f"❌ False positive: {case_info['case_name']}")

            logger.info(f"{folder_name} - Verified §1782 cases: {len(verified_cases)}")
            logger.info(f"{folder_name} - False positives: {len(false_positives)}")

            return verified_cases, false_positives

        except Exception as e:
            logger.error(f"Error analyzing {folder_name} folder: {e}")
            return [], []

    def find_missing_cases(self, original_cases, clean_cases):
        """Find cases in original that are missing from clean folder."""
        # Create sets of content hashes for comparison
        original_hashes = {case['content_hash'] for case in original_cases}
        clean_hashes = {case['content_hash'] for case in clean_cases}

        # Find missing cases
        missing_hashes = original_hashes - clean_hashes
        missing_cases = [case for case in original_cases if case['content_hash'] in missing_hashes]

        logger.info(f"Missing cases: {len(missing_cases)}")

        return missing_cases

    def run_comparison(self):
        """Run complete comparison between original and clean folders."""
        logger.info("Checking Original Folder vs Clean Folder")
        logger.info("="*60)

        # Analyze original folder
        original_cases, original_false_positives = self.analyze_folder(
            self.original_folder_id, "Original"
        )

        # Analyze clean folder
        clean_cases, clean_false_positives = self.analyze_folder(
            self.clean_folder_id, "Clean"
        )

        # Find missing cases
        missing_cases = self.find_missing_cases(original_cases, clean_cases)

        # Summary
        logger.info("\n" + "="*60)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"Original folder verified cases: {len(original_cases)}")
        logger.info(f"Clean folder verified cases: {len(clean_cases)}")
        logger.info(f"Missing cases: {len(missing_cases)}")

        if missing_cases:
            logger.info(f"\nMissing cases found:")
            for case in missing_cases:
                logger.info(f"  - {case['case_name']}")

        # Show all original cases
        logger.info(f"\nAll original folder cases:")
        for i, case in enumerate(original_cases, 1):
            logger.info(f"  {i}. {case['case_name']}")

        return {
            'original_cases': len(original_cases),
            'clean_cases': len(clean_cases),
            'missing_cases': len(missing_cases),
            'missing_case_list': missing_cases
        }


def main():
    """Main entry point."""
    checker = OriginalFolderChecker()
    result = checker.run_comparison()

    if result:
        print(f"\nCOMPARISON COMPLETE!")
        print(f"Original folder cases: {result['original_cases']}")
        print(f"Clean folder cases: {result['clean_cases']}")
        print(f"Missing cases: {result['missing_cases']}")

        if result['missing_cases'] > 0:
            print(f"\nWe missed {result['missing_cases']} cases!")
            print("Need to add them to the clean folder.")


if __name__ == "__main__":
    main()
