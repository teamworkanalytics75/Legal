#!/usr/bin/env python3
"""
Check for Major SCOTUS §1782 Cases

Search through our Google Drive files for the landmark SCOTUS cases.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

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


class SCOTUSCaseChecker:
    """Check for major SCOTUS §1782 cases in our collection."""

    def __init__(self):
        """Initialize the checker."""
        self.gdrive = GoogleDriveBackup()

        # Clean folder ID
        self.clean_folder_id = "1_wv4FZLhubYOO2ncRBcW68w4htt4N5Pl"

        # Major SCOTUS §1782 cases to look for
        self.major_scotus_cases = {
            "Intel Corp. v. Advanced Micro Devices, Inc.": {
                "year": "2004",
                "citation": "542 U.S. 241",
                "keywords": ["Intel", "Advanced Micro Devices", "AMD", "542 U.S. 241"]
            },
            "ZF Automotive US, Inc. v. Luxshare, Ltd.": {
                "year": "2022",
                "citation": "596 U.S. ___",
                "keywords": ["ZF Automotive", "Luxshare", "596 U.S."]
            },
            "Servotronics, Inc. v. Rolls-Royce PLC": {
                "year": "2021",
                "citation": "593 U.S. ___",
                "keywords": ["Servotronics", "Rolls-Royce", "593 U.S."]
            },
            "Kiobel v. Royal Dutch Petroleum Co.": {
                "year": "2013",
                "citation": "569 U.S. 108",
                "keywords": ["Kiobel", "Royal Dutch Petroleum", "569 U.S. 108"]
            },
            "Morrison v. National Australia Bank Ltd.": {
                "year": "2010",
                "citation": "561 U.S. 247",
                "keywords": ["Morrison", "National Australia Bank", "561 U.S. 247"]
            }
        }

        self.found_cases = {}
        self.missing_cases = {}

    def search_for_case(self, case_name: str, case_info: Dict[str, Any], content: str) -> bool:
        """Search for a specific SCOTUS case in the content."""
        content_lower = content.lower()

        # Check for case name variations
        name_variations = [
            case_name.lower(),
            case_name.replace("Corp.", "Corporation").lower(),
            case_name.replace("Inc.", "Incorporated").lower(),
            case_name.replace("Ltd.", "Limited").lower(),
            case_name.replace("PLC", "Public Limited Company").lower()
        ]

        # Check for keywords
        for keyword in case_info["keywords"]:
            if keyword.lower() in content_lower:
                return True

        # Check for name variations
        for variation in name_variations:
            if variation in content_lower:
                return True

        # Check for year
        if case_info["year"] in content:
            return True

        return False

    def check_all_files(self):
        """Check all files for major SCOTUS cases."""
        logger.info("Searching for major SCOTUS §1782 cases...")

        # Get all files from the folder
        query = f"'{self.clean_folder_id}' in parents and trashed=false"
        results = self.gdrive.service.files().list(
            q=query,
            fields='files(id, name, size, createdTime)'
        ).execute()
        files = results.get('files', [])

        logger.info(f"Checking {len(files)} files for SCOTUS cases...")

        # Process each file
        for i, file_metadata in enumerate(files, 1):
            filename = file_metadata['name']

            # Skip analysis files
            if filename.endswith('_report.json') or filename.startswith('full_opinion_'):
                continue

            logger.info(f"Checking file {i}/{len(files)}: {filename}")

            try:
                # Download file content
                content = self.gdrive.download_file_content(file_metadata['id'])

                if not content:
                    continue

                # Check each major SCOTUS case
                for case_name, case_info in self.major_scotus_cases.items():
                    if self.search_for_case(case_name, case_info, content):
                        if case_name not in self.found_cases:
                            self.found_cases[case_name] = {
                                'case_info': case_info,
                                'found_in_files': []
                            }

                        self.found_cases[case_name]['found_in_files'].append({
                            'filename': filename,
                            'file_id': file_metadata['id']
                        })

                        logger.info(f"✅ Found {case_name} in {filename}")

            except Exception as e:
                logger.error(f"Error checking {filename}: {e}")

        # Identify missing cases
        for case_name in self.major_scotus_cases:
            if case_name not in self.found_cases:
                self.missing_cases[case_name] = self.major_scotus_cases[case_name]

        # Generate report
        self.generate_scotus_report()

    def generate_scotus_report(self):
        """Generate a report on SCOTUS case coverage."""
        logger.info(f"\n{'='*80}")
        logger.info("SCOTUS §1782 CASE COVERAGE REPORT")
        logger.info(f"{'='*80}")

        logger.info(f"Major SCOTUS cases checked: {len(self.major_scotus_cases)}")
        logger.info(f"Cases found: {len(self.found_cases)}")
        logger.info(f"Cases missing: {len(self.missing_cases)}")
        logger.info(f"Coverage: {(len(self.found_cases)/len(self.major_scotus_cases))*100:.1f}%")

        if self.found_cases:
            logger.info(f"\n✅ FOUND SCOTUS CASES:")
            for case_name, data in self.found_cases.items():
                case_info = data['case_info']
                files = data['found_in_files']
                logger.info(f"  - {case_name} ({case_info['year']}, {case_info['citation']})")
                logger.info(f"    Found in {len(files)} file(s):")
                for file_info in files:
                    logger.info(f"      • {file_info['filename']}")

        if self.missing_cases:
            logger.info(f"\n❌ MISSING SCOTUS CASES:")
            for case_name, case_info in self.missing_cases.items():
                logger.info(f"  - {case_name} ({case_info['year']}, {case_info['citation']})")
                logger.info(f"    Keywords: {', '.join(case_info['keywords'])}")

        # Additional analysis
        logger.info(f"\n📊 ANALYSIS:")
        if len(self.found_cases) == 0:
            logger.info("❌ No major SCOTUS §1782 cases found in our collection!")
            logger.info("This suggests our collection may be missing the most important cases.")
        elif len(self.found_cases) < len(self.major_scotus_cases):
            logger.info(f"⚠️  Partial coverage - missing {len(self.missing_cases)} key SCOTUS cases")
        else:
            logger.info("✅ Complete coverage of major SCOTUS §1782 cases!")

        # Save detailed report
        report_data = {
            'summary': {
                'total_scotus_cases_checked': len(self.major_scotus_cases),
                'cases_found': len(self.found_cases),
                'cases_missing': len(self.missing_cases),
                'coverage_percent': (len(self.found_cases)/len(self.major_scotus_cases))*100
            },
            'found_cases': self.found_cases,
            'missing_cases': self.missing_cases,
            'major_scotus_cases_reference': self.major_scotus_cases
        }

        try:
            self.gdrive.upload_file_content(
                content=json.dumps(report_data, indent=2),
                filename="scotus_1782_coverage_report.json",
                folder_id=self.clean_folder_id
            )
            logger.info(f"\n📊 Detailed SCOTUS coverage report saved to Google Drive")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        logger.info(f"\n🎯 RECOMMENDATIONS:")
        if len(self.missing_cases) > 0:
            logger.info("1. Consider downloading missing SCOTUS cases from other sources")
            logger.info("2. Focus on Intel Corp. v. AMD (2004) - the landmark case")
            logger.info("3. Include ZF Automotive (2022) for recent developments")
        else:
            logger.info("1. Our collection has good SCOTUS coverage")
            logger.info("2. Focus on analyzing the cases we have")
            logger.info("3. Consider expanding to circuit court cases")


def main():
    """Main entry point."""
    checker = SCOTUSCaseChecker()
    checker.check_all_files()


if __name__ == "__main__":
    main()
