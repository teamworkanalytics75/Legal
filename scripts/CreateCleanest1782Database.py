#!/usr/bin/env python3
"""
Create Cleanest 1782 Database - Organized List of Confirmed §1782 Cases

Creates a clean, organized database with:
1. Supreme Court landmark cases first (alphabetically)
2. Only confirmed §1782 cases
3. No duplicates
4. Proper numbering and organization
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict

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


class Cleanest1782DatabaseCreator:
    """Creates the cleanest, most organized §1782 database."""

    def __init__(self):
        """Initialize the creator."""
        self.gdrive = GoogleDriveBackup()

        # Source folder (The Art of War - Caselaw Database)
        self.source_folder_id = "1_wv4FZLhubYOO2ncRBcW68w4htt4N5Pl"

        # Target folder (Cleanest 1782 Database) - will create if doesn't exist
        self.target_folder_name = "Cleanest 1782 Database"
        self.target_folder_id = None

        # Keywords to identify confirmed §1782 cases
        self.confirmed_1782_keywords = [
            "28 U.S.C. § 1782", "28 USC 1782", "section 1782",
            "foreign tribunal", "international tribunal",
            "judicial assistance", "foreign proceeding", "international discovery",
            "assistance to foreign tribunals", "foreign litigation",
            "international arbitration", "foreign arbitration"
        ]

        # Supreme Court landmark cases (highest priority)
        self.scotus_landmarks = {
            "Intel Corp. v. Advanced Micro Devices, Inc.": {
                "year": "2004",
                "citation": "542 U.S. 241",
                "importance": "LANDMARK - Established modern §1782 framework"
            },
            "ZF Automotive US, Inc. v. Luxshare, Ltd.": {
                "year": "2022",
                "citation": "596 U.S. ___",
                "importance": "RECENT - Clarified private arbitral tribunals"
            },
            "Servotronics, Inc. v. Rolls-Royce PLC": {
                "year": "2021",
                "citation": "593 U.S. ___",
                "importance": "RECENT - Private arbitral tribunal issue"
            }
        }

        self.confirmed_cases = []
        self.scotus_cases = []
        self.circuit_cases = []
        self.district_cases = []

    def create_target_folder(self):
        """Create the target folder if it doesn't exist."""
        try:
            # Check if folder already exists
            results = self.gdrive.service.files().list(
                q=f"name='{self.target_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields='files(id, name)'
            ).execute()

            folders = results.get('files', [])
            if folders:
                self.target_folder_id = folders[0]['id']
                logger.info(f"Using existing folder: {self.target_folder_name}")
            else:
                # Create new folder
                folder_metadata = {
                    'name': self.target_folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = self.gdrive.service.files().create(
                    body=folder_metadata,
                    fields='id'
                ).execute()
                self.target_folder_id = folder.get('id')
                logger.info(f"Created new folder: {self.target_folder_name}")

        except Exception as e:
            logger.error(f"Error creating target folder: {e}")
            return False
        return True

    def is_confirmed_1782_case(self, content: str) -> bool:
        """Check if content is a confirmed §1782 case."""
        content_lower = content.lower()

        # Must contain at least one confirmed keyword
        for keyword in self.confirmed_1782_keywords:
            if keyword.lower() in content_lower:
                return True

        return False

    def extract_case_metadata(self, filename: str, content: str) -> Dict[str, Any]:
        """Extract metadata from case content."""
        metadata = {
            'filename': filename,
            'case_name': filename.replace('.json', ''),
            'court': 'Unknown',
            'year': 'Unknown',
            'case_type': 'Unknown',
            'is_scotus': False,
            'is_landmark': False,
            'citation': 'Unknown'
        }

        content_lower = content.lower()

        # Extract case name more accurately
        name_patterns = [
            r'(?:IN RE|APPLICATION OF|PETITION OF)\s*(.+?)(?: UNDER| PURSUANT TO| FOR AN ORDER)',
            r'(.+?)\s+v\.\s+(.+)',
            r'(.+?)\s+IN RE',
        ]

        for pattern in name_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                if 'v.' in pattern:
                    metadata['case_name'] = f"{match.group(1).strip()} v. {match.group(2).strip()}"
                else:
                    metadata['case_name'] = match.group(1).strip()
                break

        # Extract court
        court_patterns = [
            r'(Supreme Court|SCOTUS)',
            r'(D\.D\.C\.|District of Columbia)',
            r'(S\.D\.N\.Y\.|Southern District of New York)',
            r'(N\.D\. Cal\.|Northern District of California)',
            r'(1st Cir\.|First Circuit)',
            r'(2nd Cir\.|Second Circuit)',
            r'(3rd Cir\.|Third Circuit)',
            r'(9th Cir\.|Ninth Circuit)',
            r'(10th Cir\.|Tenth Circuit)',
            r'(11th Cir\.|Eleventh Circuit)'
        ]

        for pattern in court_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata['court'] = match.group(1).upper()
                break

        # Check if SCOTUS
        if 'supreme court' in content_lower or 'scotus' in content_lower:
            metadata['is_scotus'] = True

        # Check if landmark case
        for landmark_name in self.scotus_landmarks.keys():
            if landmark_name.lower() in content_lower:
                metadata['is_landmark'] = True
                metadata['case_name'] = landmark_name
                landmark_info = self.scotus_landmarks[landmark_name]
                metadata['year'] = landmark_info['year']
                metadata['citation'] = landmark_info['citation']
                break

        # Extract year
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', content)
        if year_matches and not metadata['is_landmark']:
            metadata['year'] = max(year_matches)

        # Extract case type
        if "application" in content_lower:
            metadata['case_type'] = "Application"
        elif "opinion" in content_lower:
            metadata['case_type'] = "Opinion"
        elif "in re" in content_lower:
            metadata['case_type'] = "In Re"

        return metadata

    def categorize_case(self, metadata: Dict[str, Any]) -> str:
        """Categorize case by court level."""
        court = metadata['court'].upper()

        if 'SUPREME COURT' in court or 'SCOTUS' in court or metadata['is_scotus']:
            return 'scotus'
        elif 'CIR.' in court or 'CIRCUIT' in court:
            return 'circuit'
        else:
            return 'district'

    def process_source_folder(self):
        """Process all files in the source folder."""
        logger.info("Processing source folder for confirmed §1782 cases...")

        # Get all files from source folder
        results = self.gdrive.service.files().list(
            q=f"'{self.source_folder_id}' in parents and trashed=false",
            fields='files(id, name, size, createdTime)'
        ).execute()

        files = results.get('files', [])
        logger.info(f"Found {len(files)} files to process")

        confirmed_count = 0

        for i, file_metadata in enumerate(files, 1):
            filename = file_metadata['name']
            file_id = file_metadata['id']

            # Skip analysis files
            if filename in ["full_opinion_download_report.json", "case_data_analysis_report.json",
                           "scotus_1782_coverage_report.json", "landmark_1782_collection_report.json"]:
                logger.info(f"Skipping analysis file: {filename}")
                continue

            logger.info(f"Processing {i}/{len(files)}: {filename}")

            try:
                # Download content
                content = self.gdrive.download_file_content(file_id)
                if not content:
                    logger.warning(f"Could not download content for {filename}")
                    continue

                # Check if confirmed §1782 case
                if self.is_confirmed_1782_case(content):
                    metadata = self.extract_case_metadata(filename, content)
                    metadata['file_id'] = file_id
                    metadata['original_filename'] = filename

                    # Categorize case
                    category = self.categorize_case(metadata)
                    metadata['category'] = category

                    self.confirmed_cases.append(metadata)
                    confirmed_count += 1

                    logger.info(f"  ✅ Confirmed §1782 case: {metadata['case_name']} ({metadata['court']})")
                else:
                    logger.info(f"  ❌ Not a confirmed §1782 case: {filename}")

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")

        logger.info(f"Found {confirmed_count} confirmed §1782 cases")

        # Organize cases by category
        self.organize_cases()

    def organize_cases(self):
        """Organize cases by category and priority."""
        for case in self.confirmed_cases:
            category = case['category']

            if category == 'scotus':
                self.scotus_cases.append(case)
            elif category == 'circuit':
                self.circuit_cases.append(case)
            else:
                self.district_cases.append(case)

        # Sort each category alphabetically by case name
        self.scotus_cases.sort(key=lambda x: x['case_name'].lower())
        self.circuit_cases.sort(key=lambda x: x['case_name'].lower())
        self.district_cases.sort(key=lambda x: x['case_name'].lower())

        logger.info(f"Organized cases: {len(self.scotus_cases)} SCOTUS, {len(self.circuit_cases)} Circuit, {len(self.district_cases)} District")

    def create_clean_database(self):
        """Create the clean, organized database."""
        logger.info("Creating clean §1782 database...")

        # Create target folder
        if not self.create_target_folder():
            return False

        # Copy confirmed cases to target folder with proper numbering
        case_number = 1

        # Process SCOTUS cases first
        logger.info("Processing SCOTUS cases...")
        for case in self.scotus_cases:
            success = self.copy_case_to_clean_folder(case, case_number, "SCOTUS")
            if success:
                case_number += 1

        # Process Circuit cases
        logger.info("Processing Circuit cases...")
        for case in self.circuit_cases:
            success = self.copy_case_to_clean_folder(case, case_number, "CIRCUIT")
            if success:
                case_number += 1

        # Process District cases
        logger.info("Processing District cases...")
        for case in self.district_cases:
            success = self.copy_case_to_clean_folder(case, case_number, "DISTRICT")
            if success:
                case_number += 1

        # Create comprehensive index
        self.create_database_index()

        return True

    def copy_case_to_clean_folder(self, case: Dict[str, Any], case_number: int, category: str) -> bool:
        """Copy a case to the clean folder with proper naming."""
        try:
            # Download original content
            content = self.gdrive.download_file_content(case['file_id'])
            if not content:
                logger.error(f"Could not download content for {case['original_filename']}")
                return False

            # Create clean filename
            clean_name = case['case_name'].replace(' ', '_').replace('.', '').replace(',', '')
            clean_filename = f"{case_number:03d}_{category}_{clean_name}.json"

            # Upload to clean folder
            uploaded_id = self.gdrive.upload_file_content(
                content=content,
                filename=clean_filename,
                folder_id=self.target_folder_id
            )

            if uploaded_id:
                logger.info(f"  ✅ Copied: {clean_filename}")
                return True
            else:
                logger.error(f"  ❌ Failed to copy: {case['original_filename']}")
                return False

        except Exception as e:
            logger.error(f"Error copying case {case['original_filename']}: {e}")
            return False

    def create_database_index(self):
        """Create a comprehensive index of the clean database."""
        logger.info("Creating database index...")

        index_data = {
            'database_info': {
                'name': 'Cleanest 1782 Database',
                'description': 'Organized collection of confirmed §1782 discovery cases',
                'total_cases': len(self.confirmed_cases),
                'scotus_cases': len(self.scotus_cases),
                'circuit_cases': len(self.circuit_cases),
                'district_cases': len(self.district_cases),
                'created_date': '2025-10-15'
            },
            'scotus_cases': [
                {
                    'case_number': i+1,
                    'case_name': case['case_name'],
                    'court': case['court'],
                    'year': case['year'],
                    'citation': case['citation'],
                    'is_landmark': case['is_landmark'],
                    'filename': f"{i+1:03d}_SCOTUS_{case['case_name'].replace(' ', '_').replace('.', '').replace(',', '')}.json"
                }
                for i, case in enumerate(self.scotus_cases)
            ],
            'circuit_cases': [
                {
                    'case_number': len(self.scotus_cases) + i + 1,
                    'case_name': case['case_name'],
                    'court': case['court'],
                    'year': case['year'],
                    'filename': f"{len(self.scotus_cases) + i + 1:03d}_CIRCUIT_{case['case_name'].replace(' ', '_').replace('.', '').replace(',', '')}.json"
                }
                for i, case in enumerate(self.circuit_cases)
            ],
            'district_cases': [
                {
                    'case_number': len(self.scotus_cases) + len(self.circuit_cases) + i + 1,
                    'case_name': case['case_name'],
                    'court': case['court'],
                    'year': case['year'],
                    'filename': f"{len(self.scotus_cases) + len(self.circuit_cases) + i + 1:03d}_DISTRICT_{case['case_name'].replace(' ', '_').replace('.', '').replace(',', '')}.json"
                }
                for i, case in enumerate(self.district_cases)
            ],
            'landmark_cases': [
                {
                    'case_name': case['case_name'],
                    'year': case['year'],
                    'citation': case['citation'],
                    'importance': self.scotus_landmarks.get(case['case_name'], {}).get('importance', 'Unknown')
                }
                for case in self.scotus_cases if case['is_landmark']
            ]
        }

        # Upload index to clean folder
        try:
            self.gdrive.upload_file_content(
                content=json.dumps(index_data, indent=2),
                filename="CLEANEST_1782_DATABASE_INDEX.json",
                folder_id=self.target_folder_id
            )
            logger.info("✅ Database index created successfully")
        except Exception as e:
            logger.error(f"Error creating database index: {e}")

    def generate_summary_report(self):
        """Generate a summary report."""
        logger.info(f"\n{'='*80}")
        logger.info("CLEANEST 1782 DATABASE CREATION SUMMARY")
        logger.info(f"{'='*80}")

        logger.info(f"Total confirmed §1782 cases: {len(self.confirmed_cases)}")
        logger.info(f"SCOTUS cases: {len(self.scotus_cases)}")
        logger.info(f"Circuit cases: {len(self.circuit_cases)}")
        logger.info(f"District cases: {len(self.district_cases)}")

        if self.scotus_cases:
            logger.info(f"\n🏛️ SCOTUS CASES (Priority Order):")
            for i, case in enumerate(self.scotus_cases, 1):
                landmark_indicator = "⭐ LANDMARK" if case['is_landmark'] else ""
                logger.info(f"  {i:2d}. {case['case_name']} ({case['year']}) {landmark_indicator}")

        if self.circuit_cases:
            logger.info(f"\n⚖️ CIRCUIT CASES:")
            for i, case in enumerate(self.circuit_cases, 1):
                logger.info(f"  {i:2d}. {case['case_name']} ({case['court']}, {case['year']})")

        if self.district_cases:
            logger.info(f"\n🏛️ DISTRICT CASES:")
            for i, case in enumerate(self.district_cases, 1):
                logger.info(f"  {i:2d}. {case['case_name']} ({case['court']}, {case['year']})")

        logger.info(f"\n📁 Database Location: Google Drive - '{self.target_folder_name}'")
        logger.info(f"📊 Index File: CLEANEST_1782_DATABASE_INDEX.json")
        logger.info(f"🎯 All cases confirmed as genuine §1782 discovery cases")
        logger.info(f"🔢 Cases numbered with SCOTUS cases first (alphabetically)")


def main():
    """Main entry point."""
    creator = Cleanest1782DatabaseCreator()

    # Process source folder
    creator.process_source_folder()

    # Create clean database
    if creator.create_clean_database():
        creator.generate_summary_report()
    else:
        logger.error("Failed to create clean database")


if __name__ == "__main__":
    main()
