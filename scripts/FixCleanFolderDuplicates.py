#!/usr/bin/env python3
"""
Fix Clean Google Drive Folder - Remove Duplicates

The clean folder still has duplicates. This script will:
1. Analyze the clean folder for duplicates
2. Create a truly clean folder with only unique cases
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


class DuplicateRemover:
    """Remove duplicates from the clean folder."""

    def __init__(self):
        """Initialize the remover."""
        self.gdrive = GoogleDriveBackup()
        self.clean_folder_id = "1xfIdlbCnhxqE-nicSg-zaHY8XBu7r2bR"
        self.final_folder_id = None

        # Results
        self.unique_cases = []
        self.duplicates_removed = 0

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
                    'date_filed': case_data.get('date_filed', 'Unknown')
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
                    'date_filed': 'Unknown'
                }

        except Exception as e:
            logger.error(f"Error processing {file_info['name']}: {e}")
            return None

    def analyze_clean_folder(self):
        """Analyze the clean folder for duplicates."""
        try:
            logger.info("Analyzing clean folder for duplicates...")

            # Get all files in clean folder
            results = self.gdrive.service.files().list(
                q=f"'{self.clean_folder_id}' in parents",
                fields="files(id, name, mimeType, size, modifiedTime)",
                pageSize=1000
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in clean folder")

            # Process each file to get content hashes
            case_data = []
            for i, file_info in enumerate(files, 1):
                logger.info(f"Processing file {i}/{len(files)}: {file_info['name']}")

                case_info = self.get_case_content_hash(file_info)
                if case_info:
                    case_data.append(case_info)

            # Group by content hash to find duplicates
            hash_groups = {}
            for case in case_data:
                content_hash = case['content_hash']
                if content_hash not in hash_groups:
                    hash_groups[content_hash] = []
                hash_groups[content_hash].append(case)

            # Identify unique cases and duplicates
            unique_cases = []
            duplicates = []

            for content_hash, cases in hash_groups.items():
                if len(cases) == 1:
                    unique_cases.append(cases[0])
                else:
                    # Keep the first case, mark others as duplicates
                    unique_cases.append(cases[0])
                    duplicates.extend(cases[1:])
                    logger.info(f"Found {len(cases)} duplicates of: {cases[0]['case_name']}")

            logger.info(f"Unique cases: {len(unique_cases)}")
            logger.info(f"Duplicates to remove: {len(duplicates)}")

            return unique_cases, duplicates

        except Exception as e:
            logger.error(f"Error analyzing clean folder: {e}")
            return [], []

    def create_final_folder(self):
        """Create the final truly clean folder."""
        try:
            logger.info("Creating final clean folder...")

            folder_metadata = {
                'name': 'WitchWeb_1782_Discovery_Final',
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
        """Copy only unique cases to the final folder."""
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

        logger.info(f"Successfully copied {copied_count}/{len(unique_cases)} unique cases")
        return copied_count

    def run_deduplication(self):
        """Run complete deduplication process."""
        logger.info("Fixing Clean Google Drive Folder - Removing Duplicates")
        logger.info("="*60)

        # Analyze current clean folder
        unique_cases, duplicates = self.analyze_clean_folder()

        if not unique_cases:
            logger.error("No unique cases found")
            return

        # Create final folder
        final_folder_id = self.create_final_folder()
        if not final_folder_id:
            logger.error("Failed to create final folder")
            return

        # Copy unique cases
        copied_count = self.copy_unique_cases(unique_cases)

        # Summary
        logger.info("\n" + "="*60)
        logger.info("DEDUPLICATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Original clean folder files: {len(unique_cases) + len(duplicates)}")
        logger.info(f"Unique cases: {len(unique_cases)}")
        logger.info(f"Duplicates removed: {len(duplicates)}")
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
            'duplicates_removed': len(duplicates)
        }


def main():
    """Main entry point."""
    remover = DuplicateRemover()
    result = remover.run_deduplication()

    if result:
        print(f"\nCLEAN FOLDER FIXED!")
        print(f"Folder ID: {result['final_folder_id']}")
        print(f"URL: https://drive.google.com/drive/folders/{result['final_folder_id']}")
        print(f"Unique cases: {result['unique_cases']}")
        print(f"Duplicates removed: {result['duplicates_removed']}")


if __name__ == "__main__":
    main()
