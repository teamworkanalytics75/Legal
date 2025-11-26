#!/usr/bin/env python3
"""
Create Clean Google Drive Folder

Create a new clean folder with only the 45 unique §1782 cases,
removing all 130 duplicates and false positives.
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


class CleanFolderCreator:
    """Create a clean folder with only unique §1782 cases."""

    def __init__(self):
        """Initialize the creator."""
        self.gdrive = GoogleDriveBackup()
        self.source_folder_id = "1DHMRnqrtNQTG4w4IuJYhQsZ2XsBRFqMb"
        self.clean_folder_id = None

        # Results
        self.unique_cases = []
        self.false_positives = []

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

    def get_unique_cases(self):
        """Get the list of unique cases from the source folder."""
        try:
            logger.info("Getting unique cases from source folder...")

            # Get all files
            results = self.gdrive.service.files().list(
                q=f"'{self.source_folder_id}' in parents",
                fields="files(id, name, mimeType, size, modifiedTime)",
                pageSize=1000
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in source folder")

            # Group by base name (remove case numbers)
            name_groups = {}
            for file_info in files:
                name = file_info['name']
                # Extract base name (remove case_XX_ prefix)
                if name.startswith('case_') and '_' in name:
                    base_name = name.split('_', 2)[2] if len(name.split('_', 2)) > 2 else name
                else:
                    base_name = name

                if base_name not in name_groups:
                    name_groups[base_name] = []
                name_groups[base_name].append(file_info)

            # Get unique cases (first file from each group)
            unique_cases = []
            for base_name, file_list in name_groups.items():
                # Take the first file (keep the original)
                unique_file = file_list[0]
                unique_cases.append(unique_file)

            logger.info(f"Identified {len(unique_cases)} unique cases")
            return unique_cases

        except Exception as e:
            logger.error(f"Error getting unique cases: {e}")
            return []

    def verify_case_content(self, file_info):
        """Verify if a case is actually about §1782 discovery."""
        try:
            file_id = file_info['id']
            file_name = file_info['name']

            # Download file content
            request = self.gdrive.service.files().get_media(fileId=file_id)
            content = request.execute()

            # Try to parse as JSON
            try:
                case_data = json.loads(content.decode('utf-8'))
                case_name = case_data.get('caseName', 'Unknown')
                case_text = case_data.get('plain_text', '') or case_data.get('html_with_citations', '')

                # Verify if it's actually a §1782 case
                is_1782 = self.is_actual_1782_discovery(case_text, case_name)

                return {
                    'file_info': file_info,
                    'case_name': case_name,
                    'is_1782': is_1782,
                    'court': case_data.get('court', 'Unknown'),
                    'date_filed': case_data.get('date_filed', 'Unknown')
                }

            except json.JSONDecodeError:
                # Not a JSON file, might be text
                case_text = content.decode('utf-8')
                is_1782 = self.is_actual_1782_discovery(case_text, file_name)

                return {
                    'file_info': file_info,
                    'case_name': file_name,
                    'is_1782': is_1782,
                    'court': 'Unknown',
                    'date_filed': 'Unknown'
                }

        except Exception as e:
            logger.error(f"Error verifying {file_info['name']}: {e}")
            return {
                'file_info': file_info,
                'case_name': 'Error',
                'is_1782': False,
                'court': 'Error',
                'date_filed': 'Error',
                'error': str(e)
            }

    def create_clean_folder(self):
        """Create a new clean folder."""
        try:
            logger.info("Creating new clean folder...")

            folder_metadata = {
                'name': 'WitchWeb_1782_Discovery_Verified',
                'mimeType': 'application/vnd.google-apps.folder'
            }

            folder = self.gdrive.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()

            self.clean_folder_id = folder.get('id')
            logger.info(f"Created clean folder: {self.clean_folder_id}")

            return self.clean_folder_id

        except Exception as e:
            logger.error(f"Error creating clean folder: {e}")
            return None

    def copy_verified_cases(self, verified_cases):
        """Copy verified §1782 cases to the clean folder."""
        if not self.clean_folder_id:
            logger.error("No clean folder ID available")
            return

        logger.info(f"Copying {len(verified_cases)} verified cases to clean folder...")

        copied_count = 0
        for i, case in enumerate(verified_cases, 1):
            try:
                logger.info(f"Copying case {i}/{len(verified_cases)}: {case['case_name']}")

                # Copy file to clean folder
                copied_file = self.gdrive.service.files().copy(
                    fileId=case['file_info']['id'],
                    body={
                        'name': f"{case['case_name']}.json",
                        'parents': [self.clean_folder_id]
                    }
                ).execute()

                copied_count += 1
                logger.info(f"✅ Copied: {case['case_name']}")

            except Exception as e:
                logger.error(f"❌ Failed to copy {case['case_name']}: {e}")

        logger.info(f"Successfully copied {copied_count}/{len(verified_cases)} cases")

    def run_cleanup(self):
        """Run complete cleanup process."""
        logger.info("Creating Clean Google Drive Folder")
        logger.info("="*60)

        # Get unique cases
        unique_cases = self.get_unique_cases()
        if not unique_cases:
            logger.error("No unique cases found")
            return

        # Verify each unique case
        logger.info("Verifying unique cases...")
        verified_cases = []
        false_positives = []

        for i, case in enumerate(unique_cases, 1):
            logger.info(f"Verifying case {i}/{len(unique_cases)}: {case['name']}")

            verification = self.verify_case_content(case)

            if verification.get('is_1782', False):
                verified_cases.append(verification)
                logger.info(f"✅ Verified §1782: {verification['case_name']}")
            else:
                false_positives.append(verification)
                logger.info(f"❌ False positive: {verification['case_name']}")

        # Create clean folder
        clean_folder_id = self.create_clean_folder()
        if not clean_folder_id:
            logger.error("Failed to create clean folder")
            return

        # Copy verified cases
        self.copy_verified_cases(verified_cases)

        # Summary
        logger.info("\n" + "="*60)
        logger.info("CLEANUP SUMMARY")
        logger.info("="*60)
        logger.info(f"Source folder files: 219")
        logger.info(f"Unique cases: {len(unique_cases)}")
        logger.info(f"Verified §1782 cases: {len(verified_cases)}")
        logger.info(f"False positives: {len(false_positives)}")
        logger.info(f"Clean folder ID: {clean_folder_id}")
        logger.info(f"Clean folder URL: https://drive.google.com/drive/folders/{clean_folder_id}")

        if len(verified_cases) >= 200:
            logger.info(f"✅ SUCCESS: {len(verified_cases)} verified §1782 cases (≥200 target met)")
        else:
            logger.info(f"⚠️  WARNING: Only {len(verified_cases)} verified §1782 cases (need {200-len(verified_cases)} more for 200 target)")

        # Show false positives
        if false_positives:
            logger.info(f"\nFalse positives found:")
            for case in false_positives:
                logger.info(f"  - {case['case_name']}")

        return {
            'clean_folder_id': clean_folder_id,
            'verified_cases': len(verified_cases),
            'false_positives': len(false_positives)
        }


def main():
    """Main entry point."""
    creator = CleanFolderCreator()
    result = creator.run_cleanup()

    if result:
        print(f"\n🎉 CLEAN FOLDER CREATED!")
        print(f"📁 Folder ID: {result['clean_folder_id']}")
        print(f"🔗 URL: https://drive.google.com/drive/folders/{result['clean_folder_id']}")
        print(f"📊 Verified §1782 cases: {result['verified_cases']}")
        print(f"❌ False positives removed: {result['false_positives']}")


if __name__ == "__main__":
    main()
