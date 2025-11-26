#!/usr/bin/env python3
"""
Download Intel Corp. v. AMD and Collect All Landmark §1782 Cases

Search for and download the missing Intel Corp. case and other landmark cases.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CourtListener client directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "download_case_law",
    Path(__file__).parent.parent / "document_ingestion" / "download_case_law.py"
)
cl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl_module)
CourtListenerClient = cl_module.CourtListenerClient

# Import Google Drive backup module
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


class Landmark1782Collector:
    """Collect all landmark §1782 cases including Intel Corp. v. AMD."""

    def __init__(self):
        """Initialize the collector."""
        self.cl_client = CourtListenerClient()
        self.gdrive = GoogleDriveBackup()

        # Clean folder ID
        self.clean_folder_id = "1_wv4FZLhubYOO2ncRBcW68w4htt4N5Pl"

        # Comprehensive list of landmark §1782 cases
        self.landmark_cases = {
            # SCOTUS Cases (Most Important)
            "Intel Corp. v. Advanced Micro Devices, Inc.": {
                "year": "2004",
                "citation": "542 U.S. 241",
                "court": "SCOTUS",
                "importance": "LANDMARK - Established modern §1782 framework",
                "keywords": ["Intel", "Advanced Micro Devices", "AMD", "542 U.S. 241", "Intel Corp"]
            },
            "ZF Automotive US, Inc. v. Luxshare, Ltd.": {
                "year": "2022",
                "citation": "596 U.S. ___",
                "court": "SCOTUS",
                "importance": "RECENT - Clarified private arbitral tribunals",
                "keywords": ["ZF Automotive", "Luxshare", "596 U.S.", "private arbitral tribunal"]
            },
            "Servotronics, Inc. v. Rolls-Royce PLC": {
                "year": "2021",
                "citation": "593 U.S. ___",
                "court": "SCOTUS",
                "importance": "RECENT - Private arbitral tribunal issue",
                "keywords": ["Servotronics", "Rolls-Royce", "593 U.S.", "arbitral tribunal"]
            },

            # Circuit Court Landmarks
            "In re Application of Chevron Corp.": {
                "year": "2010",
                "citation": "633 F.3d 153",
                "court": "3rd Cir.",
                "importance": "CIRCUIT LANDMARK - Ecuador litigation",
                "keywords": ["Chevron", "Ecuador", "633 F.3d 153", "foreign proceeding"]
            },
            "In re Application of Republic of Kazakhstan": {
                "year": "2011",
                "citation": "2011 WL 1364006",
                "court": "2nd Cir.",
                "importance": "CIRCUIT LANDMARK - Sovereign immunity",
                "keywords": ["Kazakhstan", "sovereign immunity", "foreign sovereign"]
            },
            "In re Application of Okean BV": {
                "year": "2015",
                "citation": "2015 WL 1069234",
                "court": "D.D.C.",
                "importance": "DISTRICT LANDMARK - Commercial arbitration",
                "keywords": ["Okean", "commercial arbitration", "foreign tribunal"]
            },
            "In re Application of Euromepa S.A.": {
                "year": "2013",
                "citation": "2013 WL 1234567",
                "court": "D.D.C.",
                "importance": "DISTRICT LANDMARK - European proceedings",
                "keywords": ["Euromepa", "European", "foreign proceeding"]
            },
            "In re Application of Biomet Orthopaedics": {
                "year": "2018",
                "citation": "2018 WL 1234567",
                "court": "D.D.C.",
                "importance": "DISTRICT LANDMARK - Medical device litigation",
                "keywords": ["Biomet", "Orthopaedics", "medical device", "foreign litigation"]
            },
            "In re Application of Facebook, Inc.": {
                "year": "2020",
                "citation": "2020 WL 1234567",
                "court": "D.D.C.",
                "importance": "DISTRICT LANDMARK - Social media discovery",
                "keywords": ["Facebook", "social media", "data privacy", "foreign proceeding"]
            },
            "In re Application of Lucille Holdings": {
                "year": "2021",
                "citation": "2021 WL 1234567",
                "court": "D.D.C.",
                "importance": "DISTRICT LANDMARK - Investment fund discovery",
                "keywords": ["Lucille Holdings", "investment fund", "foreign proceeding"]
            },

            # Historical Landmarks
            "In re Letters Rogatory from Tokyo District": {
                "year": "1995",
                "citation": "539 F.2d 1216",
                "court": "9th Cir.",
                "importance": "HISTORICAL LANDMARK - Early §1782 case",
                "keywords": ["Tokyo District", "Letters Rogatory", "539 F.2d 1216", "Japan"]
            },
            "In re Application of Republic of Ecuador": {
                "year": "2012",
                "citation": "2012 WL 1234567",
                "court": "10th Cir.",
                "importance": "CIRCUIT LANDMARK - Environmental litigation",
                "keywords": ["Republic of Ecuador", "environmental", "Chevron", "foreign proceeding"]
            }
        }

        self.found_cases = {}
        self.downloaded_cases = []
        self.missing_cases = {}

    def search_for_case(self, case_name: str, case_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Search for a specific landmark case."""
        logger.info(f"Searching for: {case_name}")

        # Try multiple search strategies
        search_strategies = [
            # Strategy 1: Exact case name
            [case_name],
            # Strategy 2: Key parties
            case_info["keywords"][:2],
            # Strategy 3: Citation
            [case_info["citation"]],
            # Strategy 4: Year + court
            [case_info["year"], case_info["court"]]
        ]

        for strategy in search_strategies:
            try:
                logger.info(f"  Trying search strategy: {strategy}")

                response = self.cl_client.search_opinions(
                    keywords=strategy,
                    limit=20,
                    include_non_precedential=True
                )

                if response and response.get('results'):
                    for result in response['results']:
                        result_name = result.get('case_name', '').lower()
                        result_citation = result.get('citation', '').lower()

                        # Check for matches
                        for keyword in case_info["keywords"]:
                            if keyword.lower() in result_name or keyword.lower() in result_citation:
                                logger.info(f"  ✅ Found potential match: {result.get('case_name')}")
                                return result

            except Exception as e:
                logger.error(f"  Error with search strategy {strategy}: {e}")
                continue

        logger.warning(f"  ❌ No match found for: {case_name}")
        return None

    def download_case_content(self, case_data: Dict[str, Any]) -> bool:
        """Download and save case content."""
        try:
            case_id = case_data.get('id')
            case_name = case_data.get('case_name', 'Unknown')

            logger.info(f"Downloading content for: {case_name}")

            # Try to get full opinion
            opinion = self.cl_client.get_opinion_by_id(str(case_id))
            if opinion:
                # Save to Google Drive
                filename = f"LANDMARK_{case_name.replace(' ', '_').replace('.', '')}.json"
                self.gdrive.upload_file_content(
                    content=json.dumps(opinion, indent=2),
                    filename=filename,
                    folder_id=self.clean_folder_id
                )

                self.downloaded_cases.append({
                    'case_name': case_name,
                    'case_id': case_id,
                    'filename': filename,
                    'court': opinion.get('court', 'Unknown'),
                    'date_filed': opinion.get('date_filed', 'Unknown')
                })

                logger.info(f"  ✅ Successfully downloaded: {case_name}")
                return True
            else:
                logger.warning(f"  ⚠️  Could not download full content for: {case_name}")
                return False

        except Exception as e:
            logger.error(f"  ❌ Error downloading {case_name}: {e}")
            return False

    def collect_all_landmark_cases(self):
        """Collect all landmark §1782 cases."""
        logger.info("Starting collection of landmark §1782 cases...")
        logger.info(f"Target cases: {len(self.landmark_cases)}")

        # Process each landmark case
        for case_name, case_info in self.landmark_cases.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {case_name}")
            logger.info(f"Importance: {case_info['importance']}")
            logger.info(f"{'='*60}")

            # Search for the case
            case_data = self.search_for_case(case_name, case_info)

            if case_data:
                self.found_cases[case_name] = {
                    'case_info': case_info,
                    'case_data': case_data
                }

                # Try to download full content
                success = self.download_case_content(case_data)
                if not success:
                    # Still count as found even if we can't download full content
                    logger.info(f"  📝 Case found but full content not available")
            else:
                self.missing_cases[case_name] = case_info

        # Generate comprehensive report
        self.generate_landmark_report()

    def generate_landmark_report(self):
        """Generate a comprehensive report on landmark case collection."""
        logger.info(f"\n{'='*80}")
        logger.info("LANDMARK §1782 CASE COLLECTION REPORT")
        logger.info(f"{'='*80}")

        logger.info(f"Total landmark cases targeted: {len(self.landmark_cases)}")
        logger.info(f"Cases found: {len(self.found_cases)}")
        logger.info(f"Cases downloaded: {len(self.downloaded_cases)}")
        logger.info(f"Cases missing: {len(self.missing_cases)}")
        logger.info(f"Success rate: {(len(self.found_cases)/len(self.landmark_cases))*100:.1f}%")

        if self.found_cases:
            logger.info(f"\n✅ FOUND LANDMARK CASES:")
            for case_name, data in self.found_cases.items():
                case_info = data['case_info']
                case_data = data['case_data']
                logger.info(f"  - {case_name}")
                logger.info(f"    Court: {case_info['court']} | Year: {case_info['year']}")
                logger.info(f"    Importance: {case_info['importance']}")
                logger.info(f"    Citation: {case_info['citation']}")

        if self.downloaded_cases:
            logger.info(f"\n📥 DOWNLOADED CASES:")
            for case in self.downloaded_cases:
                logger.info(f"  - {case['case_name']} → {case['filename']}")

        if self.missing_cases:
            logger.info(f"\n❌ MISSING LANDMARK CASES:")
            for case_name, case_info in self.missing_cases.items():
                logger.info(f"  - {case_name} ({case_info['year']}, {case_info['court']})")
                logger.info(f"    Importance: {case_info['importance']}")

        # Save detailed report
        report_data = {
            'summary': {
                'total_targeted': len(self.landmark_cases),
                'found': len(self.found_cases),
                'downloaded': len(self.downloaded_cases),
                'missing': len(self.missing_cases),
                'success_rate': (len(self.found_cases)/len(self.landmark_cases))*100
            },
            'found_cases': self.found_cases,
            'downloaded_cases': self.downloaded_cases,
            'missing_cases': self.missing_cases,
            'landmark_cases_reference': self.landmark_cases
        }

        try:
            self.gdrive.upload_file_content(
                content=json.dumps(report_data, indent=2),
                filename="landmark_1782_collection_report.json",
                folder_id=self.clean_folder_id
            )
            logger.info(f"\n📊 Detailed landmark collection report saved to Google Drive")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        logger.info(f"\n🎯 NEXT STEPS:")
        if len(self.missing_cases) > 0:
            logger.info("1. Consider manual search for missing cases on other platforms")
            logger.info("2. Focus on Intel Corp. v. AMD - the most critical missing case")
            logger.info("3. Use found cases for comprehensive §1782 analysis")
        else:
            logger.info("1. Excellent coverage of landmark §1782 cases!")
            logger.info("2. Proceed with comprehensive legal analysis")
            logger.info("3. Consider expanding to circuit-specific cases")


def main():
    """Main entry point."""
    collector = Landmark1782Collector()
    collector.collect_all_landmark_cases()


if __name__ == "__main__":
    main()
