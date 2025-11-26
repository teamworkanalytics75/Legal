#!/usr/bin/env python3
"""
Analyze What We Actually Have

Let me check what cases we actually have and why my deduplication was so aggressive.
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


class CaseAnalyzer:
    """Analyze what cases we actually have."""

    def __init__(self):
        """Initialize the analyzer."""
        self.gdrive = GoogleDriveBackup()
        self.original_folder_id = "12pHMekx-GER8cPaMJv7aH3-El4vu7kbb"

        # Results
        self.verified_cases = []
        self.case_groups = {}

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

    def get_case_info(self, file_info):
        """Get case information for analysis."""
        try:
            file_id = file_info['id']

            # Download file content
            request = self.gdrive.service.files().get_media(fileId=file_id)
            content = request.execute()

            # Parse JSON and extract key content
            try:
                case_data = json.loads(content.decode('utf-8'))

                case_name = case_data.get('caseName', '')
                plain_text = case_data.get('plain_text', '') or case_data.get('html_with_citations', '')

                # Extract key identifiers
                case_id = case_data.get('id', '')
                docket_number = case_data.get('docketNumber', '')
                court = case_data.get('court', '')
                date_filed = case_data.get('date_filed', 'Unknown')

                return {
                    'file_info': file_info,
                    'case_name': case_name,
                    'case_id': case_id,
                    'docket_number': docket_number,
                    'court': court,
                    'date_filed': date_filed,
                    'plain_text': plain_text
                }

            except json.JSONDecodeError:
                # Not a JSON file
                case_name = file_info['name']

                return {
                    'file_info': file_info,
                    'case_name': case_name,
                    'case_id': '',
                    'docket_number': '',
                    'court': 'Unknown',
                    'date_filed': 'Unknown',
                    'plain_text': ''
                }

        except Exception as e:
            logger.error(f"Error processing {file_info['name']}: {e}")
            return None

    def analyze_cases(self):
        """Analyze all cases to understand what we have."""
        try:
            logger.info("Analyzing all cases in original folder...")

            # Get all files
            results = self.gdrive.service.files().list(
                q=f"'{self.original_folder_id}' in parents",
                fields="files(id, name, mimeType, size, modifiedTime)",
                pageSize=1000
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in original folder")

            # Process each file
            verified_cases = []

            for i, file_info in enumerate(files, 1):
                logger.info(f"Processing file {i}/{len(files)}: {file_info['name']}")

                case_info = self.get_case_info(file_info)
                if not case_info:
                    continue

                # Verify if it's actually a §1782 case
                is_1782 = self.is_actual_1782_discovery(case_info['plain_text'], case_info['case_name'])

                if is_1782:
                    verified_cases.append(case_info)
                    logger.info(f"✅ Verified §1782: {case_info['case_name']}")
                else:
                    logger.info(f"❌ False positive: {case_info['case_name']}")

            logger.info(f"Total verified §1782 cases: {len(verified_cases)}")

            # Group by case name to see duplicates
            case_groups = {}
            for case in verified_cases:
                case_name = case['case_name']
                if case_name not in case_groups:
                    case_groups[case_name] = []
                case_groups[case_name].append(case)

            # Show analysis
            logger.info("\n" + "="*60)
            logger.info("CASE ANALYSIS")
            logger.info("="*60)
            logger.info(f"Total verified §1782 cases: {len(verified_cases)}")
            logger.info(f"Unique case names: {len(case_groups)}")

            logger.info("\nCase groups (showing duplicates):")
            for case_name, cases in case_groups.items():
                if len(cases) > 1:
                    logger.info(f"\n{case_name} ({len(cases)} copies):")
                    for i, case in enumerate(cases, 1):
                        logger.info(f"  {i}. ID: {case['case_id']}, Docket: {case['docket_number']}, Court: {case['court']}")
                else:
                    logger.info(f"\n{case_name} (1 copy)")
                    logger.info(f"  ID: {cases[0]['case_id']}, Docket: {cases[0]['docket_number']}, Court: {cases[0]['court']}")

            return verified_cases, case_groups

        except Exception as e:
            logger.error(f"Error analyzing cases: {e}")
            return [], {}


def main():
    """Main entry point."""
    analyzer = CaseAnalyzer()
    verified_cases, case_groups = analyzer.analyze_cases()

    if verified_cases:
        print(f"\nANALYSIS COMPLETE!")
        print(f"Total verified §1782 cases: {len(verified_cases)}")
        print(f"Unique case names: {len(case_groups)}")
        print(f"Duplicates found: {len(verified_cases) - len(case_groups)}")


if __name__ == "__main__":
    main()
