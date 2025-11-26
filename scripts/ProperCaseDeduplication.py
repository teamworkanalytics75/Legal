#!/usr/bin/env python3
"""
Proper Case Deduplication

Use multiple factors to identify truly unique cases:
- Case name
- Docket number
- Court
- Date filed
- Content hash of full text
"""

import sys
import json
import logging
import hashlib
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


class ProperCaseDeduplicator:
    """Proper deduplication using multiple factors."""

    def __init__(self):
        """Initialize the deduplicator."""
        self.gdrive = GoogleDriveBackup()
        self.original_folder_id = "12pHMekx-GER8cPaMJv7aH3-El4vu7kbb"

        # Results
        self.verified_cases = []
        self.unique_cases = []

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
        """Get comprehensive case information."""
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

                # Extract comprehensive metadata
                case_id = case_data.get('id', '')
                docket_number = case_data.get('docketNumber', '')
                court = case_data.get('court', '')
                date_filed = case_data.get('date_filed', 'Unknown')
                date_created = case_data.get('date_created', 'Unknown')
                absolute_url = case_data.get('absolute_url', '')

                # Create content hash for deduplication
                content_hash = hashlib.md5(plain_text.encode('utf-8')).hexdigest() if plain_text else ''

                # Create unique identifier combining multiple factors
                unique_id = f"{case_name}|{docket_number}|{court}|{date_filed}|{content_hash}"

                return {
                    'file_info': file_info,
                    'case_name': case_name,
                    'case_id': case_id,
                    'docket_number': docket_number,
                    'court': court,
                    'date_filed': date_filed,
                    'date_created': date_created,
                    'absolute_url': absolute_url,
                    'plain_text': plain_text,
                    'content_hash': content_hash,
                    'unique_id': unique_id
                }

            except json.JSONDecodeError:
                # Not a JSON file
                case_name = file_info['name']
                content_hash = hashlib.md5(case_name.encode('utf-8')).hexdigest()
                unique_id = f"{case_name}||Unknown|Unknown||{content_hash}"

                return {
                    'file_info': file_info,
                    'case_name': case_name,
                    'case_id': '',
                    'docket_number': '',
                    'court': 'Unknown',
                    'date_filed': 'Unknown',
                    'date_created': 'Unknown',
                    'absolute_url': '',
                    'plain_text': '',
                    'content_hash': content_hash,
                    'unique_id': unique_id
                }

        except Exception as e:
            logger.error(f"Error processing {file_info['name']}: {e}")
            return None

    def analyze_cases_properly(self):
        """Analyze cases with proper multi-factor deduplication."""
        try:
            logger.info("Analyzing cases with proper multi-factor deduplication...")

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

            # Proper deduplication using multiple factors
            unique_cases = []
            seen_unique_ids = set()
            duplicate_groups = {}

            for case in verified_cases:
                unique_id = case['unique_id']

                if unique_id not in seen_unique_ids:
                    seen_unique_ids.add(unique_id)
                    unique_cases.append(case)
                    logger.info(f"✅ Unique case: {case['case_name']} ({case['docket_number']}, {case['court']})")
                else:
                    # Group duplicates for analysis
                    if unique_id not in duplicate_groups:
                        duplicate_groups[unique_id] = []
                    duplicate_groups[unique_id].append(case)
                    logger.info(f"🔄 Duplicate: {case['case_name']} ({case['docket_number']}, {case['court']})")

            logger.info(f"Unique cases after proper deduplication: {len(unique_cases)}")

            # Show duplicate analysis
            if duplicate_groups:
                logger.info("\n" + "="*60)
                logger.info("DUPLICATE ANALYSIS")
                logger.info("="*60)

                for unique_id, duplicates in duplicate_groups.items():
                    logger.info(f"\nDuplicate group ({len(duplicates)} copies):")
                    logger.info(f"  Case: {duplicates[0]['case_name']}")
                    logger.info(f"  Docket: {duplicates[0]['docket_number']}")
                    logger.info(f"  Court: {duplicates[0]['court']}")
                    logger.info(f"  Date: {duplicates[0]['date_filed']}")

            # Show unique cases
            logger.info("\n" + "="*60)
            logger.info("UNIQUE CASES FOUND")
            logger.info("="*60)

            for i, case in enumerate(unique_cases, 1):
                logger.info(f"{i}. {case['case_name']}")
                logger.info(f"   Docket: {case['docket_number']}")
                logger.info(f"   Court: {case['court']}")
                logger.info(f"   Date: {case['date_filed']}")
                logger.info(f"   Case ID: {case['case_id']}")
                logger.info(f"   URL: {case['absolute_url']}")
                logger.info("")

            return verified_cases, unique_cases, duplicate_groups

        except Exception as e:
            logger.error(f"Error analyzing cases: {e}")
            return [], [], {}


def main():
    """Main entry point."""
    deduplicator = ProperCaseDeduplicator()
    verified_cases, unique_cases, duplicate_groups = deduplicator.analyze_cases_properly()

    if verified_cases:
        print(f"\nPROPER DEDUPLICATION COMPLETE!")
        print(f"Total verified §1782 cases: {len(verified_cases)}")
        print(f"Unique cases after proper deduplication: {len(unique_cases)}")
        print(f"Duplicates removed: {len(verified_cases) - len(unique_cases)}")
        print(f"Duplicate groups: {len(duplicate_groups)}")

        if len(unique_cases) >= 200:
            print(f"✅ SUCCESS: {len(unique_cases)} unique §1782 cases (≥200 target met)")
        else:
            print(f"⚠️  WARNING: Only {len(unique_cases)} unique §1782 cases (need {200-len(unique_cases)} more for 200 target)")


if __name__ == "__main__":
    main()
