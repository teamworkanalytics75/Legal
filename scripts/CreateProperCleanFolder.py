#!/usr/bin/env python3
"""
Create Proper Clean Folder with All 86 Cases

Create a clean folder with all 86 verified §1782 cases from the original folder,
properly deduplicated without being overly aggressive.
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


class ProperCleanFolderCreator:
    """Create a proper clean folder with all 86 verified §1782 cases."""

    def __init__(self):
        """Initialize the creator."""
        self.gdrive = GoogleDriveBackup()
        self.original_folder_id = "12pHMekx-GER8cPaMJv7aH3-El4vu7kbb"
        self.final_folder_id = None

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
        """Get case information for deduplication."""
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

                # Create a more sophisticated hash based on case name and key identifiers
                import hashlib

                # Extract key identifiers for better deduplication
                case_id = case_data.get('id', '')
                docket_number = case_data.get('docketNumber', '')
                court = case_data.get('court', '')

                # Create hash based on case name and key identifiers
                hash_input = f"{case_name}|{case_id}|{docket_number}|{court}"
                content_hash = hashlib.md5(hash_input.encode()).hexdigest()

                return {
                    'file_info': file_info,
                    'case_name': case_name,
                    'content_hash': content_hash,
                    'case_id': case_id,
                    'docket_number': docket_number,
                    'court': court,
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
                    'case_id': '',
                    'docket_number': '',
                    'court': 'Unknown',
                    'date_filed': 'Unknown',
                    'plain_text': ''
                }

        except Exception as e:
            logger.error(f"Error processing {file_info['name']}: {e}")
            return None

    def get_all_verified_cases(self):
        """Get all verified §1782 cases from the original folder."""
        try:
            logger.info("Getting all verified §1782 cases from original folder...")

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
            return verified_cases

        except Exception as e:
            logger.error(f"Error getting verified cases: {e}")
            return []

    def deduplicate_cases(self, verified_cases):
        """Deduplicate cases using a more sophisticated approach."""
        logger.info("Deduplicating cases...")

        # Group by content hash
        hash_groups = {}
        for case in verified_cases:
            content_hash = case['content_hash']
            if content_hash not in hash_groups:
                hash_groups[content_hash] = []
            hash_groups[content_hash].append(case)

        # For each group, keep the best case (prefer cases with more metadata)
        unique_cases = []
        duplicates_removed = 0

        for content_hash, cases in hash_groups.items():
            if len(cases) == 1:
                unique_cases.append(cases[0])
            else:
                # Choose the best case from the group
                best_case = max(cases, key=lambda c: (
                    len(c['case_id']),  # Prefer cases with case IDs
                    len(c['docket_number']),  # Prefer cases with docket numbers
                    len(c['court']),  # Prefer cases with court info
                    len(c['plain_text'])  # Prefer cases with more text
                ))
                unique_cases.append(best_case)
                duplicates_removed += len(cases) - 1
                logger.info(f"Removed {len(cases) - 1} duplicates of: {best_case['case_name']}")

        logger.info(f"Unique cases: {len(unique_cases)}")
        logger.info(f"Duplicates removed: {duplicates_removed}")

        return unique_cases

    def create_final_folder(self):
        """Create the final clean folder."""
        try:
            logger.info("Creating final clean folder...")

            folder_metadata = {
                'name': 'WitchWeb_1782_Discovery_Complete',
                'mimeType': 'application/vnd.google-apps.folder'
            }

            folder = self.gdrive.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()

            self.final_folder_id = folder.get('id')
            logger.info(f"Created final clean folder: {self.final_folder_id}")

            return self.final_folder_id

        except Exception as e:
            logger.error(f"Error creating final folder: {e}")
            return None

    def copy_unique_cases(self, unique_cases):
        """Copy unique cases to the final folder."""
        if not self.final_folder_id:
            logger.error("No final folder ID available")
            return

        logger.info(f"Copying {len(unique_cases)} unique cases to final folder...")

        copied_count = 0
        for i, case in enumerate(unique_cases, 1):
            try:
                logger.info(f"Copying case {i}/{len(unique_cases)}: {case['case_name']}")

                # Copy file to final folder
                copied_file = self.gdrive.service.files().copy(
                    fileId=case['file_info']['id'],
                    body={
                        'name': f"{case['case_name']}.json",
                        'parents': [self.final_folder_id]
                    }
                ).execute()

                copied_count += 1
                logger.info(f"✅ Copied: {case['case_name']}")

            except Exception as e:
                logger.error(f"❌ Failed to copy {case['case_name']}: {e}")

        logger.info(f"Successfully copied {copied_count}/{len(unique_cases)} cases")
        return copied_count

    def run_creation(self):
        """Run complete creation process."""
        logger.info("Creating Proper Clean Folder with All 86 Cases")
        logger.info("="*60)

        # Get all verified cases
        verified_cases = self.get_all_verified_cases()
        if not verified_cases:
            logger.error("No verified cases found")
            return

        # Deduplicate cases
        unique_cases = self.deduplicate_cases(verified_cases)

        # Create final folder
        final_folder_id = self.create_final_folder()
        if not final_folder_id:
            logger.error("Failed to create final folder")
            return

        # Copy unique cases
        copied_count = self.copy_unique_cases(unique_cases)

        # Summary
        logger.info("\n" + "="*60)
        logger.info("CREATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Original folder verified cases: {len(verified_cases)}")
        logger.info(f"Unique cases after deduplication: {len(unique_cases)}")
        logger.info(f"Duplicates removed: {len(verified_cases) - len(unique_cases)}")
        logger.info(f"Final folder ID: {final_folder_id}")
        logger.info(f"Final folder URL: https://drive.google.com/drive/folders/{final_folder_id}")
        logger.info(f"Successfully copied: {copied_count} unique cases")

        if copied_count >= 200:
            logger.info(f"✅ SUCCESS: {copied_count} unique §1782 cases (≥200 target met)")
        else:
            logger.info(f"⚠️  WARNING: Only {copied_count} unique §1782 cases (need {200-copied_count} more for 200 target)")

        return {
            'final_folder_id': final_folder_id,
            'unique_cases': copied_count,
            'duplicates_removed': len(verified_cases) - len(unique_cases)
        }


def main():
    """Main entry point."""
    creator = ProperCleanFolderCreator()
    result = creator.run_creation()

    if result:
        print(f"\nPROPER CLEAN FOLDER CREATED!")
        print(f"Folder ID: {result['final_folder_id']}")
        print(f"URL: https://drive.google.com/drive/folders/{result['final_folder_id']}")
        print(f"Unique cases: {result['unique_cases']}")
        print(f"Duplicates removed: {result['duplicates_removed']}")


if __name__ == "__main__":
    main()
