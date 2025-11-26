#!/usr/bin/env python3
"""
Google Drive §1782 Case Verification

Check the Google Drive folder to verify all cases are actually about §1782 discovery
and ensure we have a perfect starting set of 200+ verified cases.
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


class DriveCaseVerifier:
    """Verify §1782 cases in Google Drive folder."""

    def __init__(self):
        """Initialize the verifier."""
        self.gdrive = GoogleDriveBackup()
        self.folder_id = "1DHMRnqrtNQTG4w4IuJYhQsZ2XsBRFqMb"

        # Verification results
        self.verification_results = {
            'total_files': 0,
            'verified_1782': 0,
            'false_positives': 0,
            'verification_errors': 0,
            'case_details': []
        }

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

    def list_folder_contents(self):
        """List all files in the Google Drive folder."""
        try:
            logger.info(f"Listing contents of folder: {self.folder_id}")

            # Get folder contents
            results = self.gdrive.service.files().list(
                q=f"'{self.folder_id}' in parents",
                fields="files(id, name, mimeType, size, modifiedTime)"
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in folder")

            return files

        except Exception as e:
            logger.error(f"Error listing folder contents: {e}")
            return []

    def download_and_verify_file(self, file_info):
        """Download and verify a single file."""
        file_id = file_info['id']
        file_name = file_info['name']

        try:
            logger.info(f"Verifying: {file_name}")

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

                case_detail = {
                    'file_name': file_name,
                    'case_name': case_name,
                    'is_1782': is_1782,
                    'court': case_data.get('court', 'Unknown'),
                    'date_filed': case_data.get('date_filed', 'Unknown'),
                    'file_size': file_info.get('size', 'Unknown')
                }

                return case_detail

            except json.JSONDecodeError:
                # Not a JSON file, might be text
                case_text = content.decode('utf-8')
                is_1782 = self.is_actual_1782_discovery(case_text, file_name)

                case_detail = {
                    'file_name': file_name,
                    'case_name': file_name,
                    'is_1782': is_1782,
                    'court': 'Unknown',
                    'date_filed': 'Unknown',
                    'file_size': file_info.get('size', 'Unknown')
                }

                return case_detail

        except Exception as e:
            logger.error(f"Error verifying {file_name}: {e}")
            return {
                'file_name': file_name,
                'case_name': 'Error',
                'is_1782': False,
                'court': 'Error',
                'date_filed': 'Error',
                'file_size': file_info.get('size', 'Unknown'),
                'error': str(e)
            }

    def verify_all_cases(self):
        """Verify all cases in the Google Drive folder."""
        logger.info("Starting verification of all cases in Google Drive folder")
        logger.info("=" * 60)

        # List all files
        files = self.list_folder_contents()
        if not files:
            logger.error("No files found in folder")
            return

        self.verification_results['total_files'] = len(files)
        logger.info(f"Verifying {len(files)} files...")

        # Verify each file
        for i, file_info in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}")

            case_detail = self.download_and_verify_file(file_info)
            self.verification_results['case_details'].append(case_detail)

            # Update counters
            if case_detail.get('is_1782', False):
                self.verification_results['verified_1782'] += 1
            elif 'error' in case_detail:
                self.verification_results['verification_errors'] += 1
            else:
                self.verification_results['false_positives'] += 1

        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate verification summary."""
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 60)

        total = self.verification_results['total_files']
        verified = self.verification_results['verified_1782']
        false_pos = self.verification_results['false_positives']
        errors = self.verification_results['verification_errors']

        logger.info(f"Total files: {total}")
        logger.info(f"Verified §1782 cases: {verified}")
        logger.info(f"False positives: {false_pos}")
        logger.info(f"Verification errors: {errors}")
        logger.info(f"Success rate: {(verified/total)*100:.1f}%")

        # Check if we have 200+ verified cases
        if verified >= 200:
            logger.info(f"✅ SUCCESS: You have {verified} verified §1782 cases (≥200 target met)")
        else:
            logger.info(f"⚠️  WARNING: Only {verified} verified §1782 cases (need {200-verified} more for 200 target)")

        # Show false positives
        if false_pos > 0:
            logger.info(f"\nFalse positives found:")
            for case in self.verification_results['case_details']:
                if not case.get('is_1782', False) and 'error' not in case:
                    logger.info(f"  - {case['file_name']}: {case['case_name']}")

        # Show errors
        if errors > 0:
            logger.info(f"\nVerification errors:")
            for case in self.verification_results['case_details']:
                if 'error' in case:
                    logger.info(f"  - {case['file_name']}: {case['error']}")

        # Save detailed results
        self.save_results()

    def save_results(self):
        """Save verification results to file."""
        results_file = Path("data/case_law/1782_discovery/drive_verification_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.verification_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed results saved to: {results_file}")

    def run_verification(self):
        """Run complete verification process."""
        try:
            self.verify_all_cases()

            logger.info("=" * 60)
            logger.info("VERIFICATION COMPLETE")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error during verification: {e}", exc_info=True)


def main():
    """Main entry point."""
    print("Google Drive §1782 Case Verification")
    print("=" * 60)
    print("Verifying all cases are actually about §1782 discovery")
    print("=" * 60)

    verifier = DriveCaseVerifier()
    verifier.run_verification()


if __name__ == "__main__":
    main()
