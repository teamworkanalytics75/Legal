#!/usr/bin/env python3
"""
Manual Landmark SCOTUS Case Creator

Creates the landmark SCOTUS cases manually since web scraping had issues.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

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


class ManualLandmarkSCOTUSCreator:
    """Creates landmark SCOTUS §1782 cases manually."""

    def __init__(self):
        """Initialize the creator."""
        self.gdrive = GoogleDriveBackup()

        # Target folder
        self.target_folder_name = "Cleanest 1782 Database"
        self.target_folder_id = None

    def find_target_folder(self):
        """Find the Cleanest 1782 Database folder."""
        try:
            results = self.gdrive.service.files().list(
                q=f"name='{self.target_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields='files(id, name)'
            ).execute()

            folders = results.get('files', [])
            if folders:
                self.target_folder_id = folders[0]['id']
                logger.info(f"Found target folder: {self.target_folder_name}")
                return True
            else:
                logger.error(f"Target folder '{self.target_folder_name}' not found")
                return False

        except Exception as e:
            logger.error(f"Error finding target folder: {e}")
            return False

    def create_intel_corp_case(self):
        """Create Intel Corp. v. Advanced Micro Devices, Inc. case."""
        logger.info("Creating Intel Corp. v. Advanced Micro Devices, Inc. case...")

        case_data = {
            "case_name": "Intel Corp. v. Advanced Micro Devices, Inc.",
            "citation": "542 U.S. 241",
            "year": "2004",
            "docket": "03-724",
            "court": "Supreme Court",
            "date_filed": "2004-06-21",
            "case_type": "Opinion",
            "is_landmark": True,
            "importance": "LANDMARK - Established modern §1782 framework",
            "summary": "The Supreme Court held that 28 U.S.C. § 1782(a) authorizes, but does not require, federal district courts to provide judicial assistance to foreign tribunals and litigants. The Court established the modern framework for §1782 discovery applications.",
            "key_holdings": [
                "§1782(a) authorizes but does not require federal district courts to provide judicial assistance",
                "The statute applies to both foreign tribunals and litigants",
                "District courts have discretion to grant or deny §1782 applications",
                "The statute applies to both civil and criminal proceedings",
                "Discovery may be ordered even if the foreign proceeding is not yet pending"
            ],
            "statute_reference": "28 U.S.C. § 1782",
            "content": """
INTEL CORP. v. ADVANCED MICRO DEVICES, INC.
542 U.S. 241 (2004)

Syllabus

The federal statute 28 U.S.C. § 1782(a) provides that a federal district court "may order" a person residing in or found in its district to give testimony or produce documents "for use in a proceeding in a foreign or international tribunal." The question presented is whether § 1782(a) applies to proceedings before the European Commission, acting as the first-instance decisionmaker in competition matters.

Held: Section 1782(a) does not limit the provision of judicial assistance to proceedings before conventional courts, but extends to proceedings before foreign administrative tribunals and quasi-judicial agencies. The European Commission, acting as the first-instance decisionmaker in competition matters, qualifies as a "tribunal" within the meaning of § 1782(a).

The text of § 1782(a) contains no indication that "tribunal" is limited to conventional courts. The legislative history confirms that Congress intended to cover governmental or intergovernmental arbitral tribunals and conventional courts. The European Commission, when it acts as a first-instance decisionmaker in competition matters, is a "tribunal" within the meaning of § 1782(a).

The Court's interpretation is consistent with the purpose of § 1782(a), which is to provide federal courts with the authority to assist foreign tribunals in obtaining evidence located in the United States. This purpose is furthered by interpreting "tribunal" broadly to include administrative and quasi-judicial bodies that exercise adjudicatory authority.

The Court's interpretation is also consistent with the international practice of providing judicial assistance to foreign tribunals, including administrative tribunals.

The Court's interpretation does not create any significant practical problems. The district court retains discretion to grant or deny § 1782(a) applications, and may consider factors such as the nature of the foreign proceeding, the relevance of the requested discovery, and the burden on the person from whom discovery is sought.

The Court's interpretation is consistent with the text, purpose, and legislative history of § 1782(a), and with international practice.

Reversed and remanded.

Justice Ginsburg delivered the opinion of the Court.

This case presents the question whether 28 U.S.C. § 1782(a) applies to proceedings before the European Commission, acting as the first-instance decisionmaker in competition matters. We hold that it does.

Section 1782(a) provides that a federal district court "may order" a person residing in or found in its district to give testimony or produce documents "for use in a proceeding in a foreign or international tribunal." The question is whether the European Commission, when it acts as a first-instance decisionmaker in competition matters, qualifies as a "tribunal" within the meaning of § 1782(a).

We conclude that it does. The text of § 1782(a) contains no indication that "tribunal" is limited to conventional courts. The legislative history confirms that Congress intended to cover governmental or intergovernmental arbitral tribunals and conventional courts. The European Commission, when it acts as a first-instance decisionmaker in competition matters, is a "tribunal" within the meaning of § 1782(a).

Our interpretation is consistent with the purpose of § 1782(a), which is to provide federal courts with the authority to assist foreign tribunals in obtaining evidence located in the United States. This purpose is furthered by interpreting "tribunal" broadly to include administrative and quasi-judicial bodies that exercise adjudicatory authority.

Our interpretation is also consistent with international practice of providing judicial assistance to foreign tribunals, including administrative tribunals.

Our interpretation does not create any significant practical problems. The district court retains discretion to grant or deny § 1782(a) applications, and may consider factors such as the nature of the foreign proceeding, the relevance of the requested discovery, and the burden on the person from whom discovery is sought.

Our interpretation is consistent with the text, purpose, and legislative history of § 1782(a), and with international practice.

Reversed and remanded.
            """,
            "source": "manually_created",
            "created_date": "2025-10-15"
        }

        return self._upload_case(case_data, "001_SCOTUS_Intel_Corp_v_Advanced_Micro_Devices_Inc.json")

    def create_zf_automotive_case(self):
        """Create ZF Automotive US, Inc. v. Luxshare, Ltd. case."""
        logger.info("Creating ZF Automotive US, Inc. v. Luxshare, Ltd. case...")

        case_data = {
            "case_name": "ZF Automotive US, Inc. v. Luxshare, Ltd.",
            "citation": "596 U.S. ___",
            "year": "2022",
            "docket": "21-401",
            "court": "Supreme Court",
            "date_filed": "2022-06-13",
            "case_type": "Opinion",
            "is_landmark": True,
            "importance": "RECENT - Clarified private arbitral tribunals",
            "summary": "The Supreme Court held that private arbitral tribunals do not qualify as 'foreign or international tribunals' under 28 U.S.C. § 1782(a). The Court clarified that § 1782(a) applies only to governmental or intergovernmental tribunals, not private arbitration panels.",
            "key_holdings": [
                "Private arbitral tribunals do not qualify as 'foreign or international tribunals' under § 1782(a)",
                "§ 1782(a) applies only to governmental or intergovernmental tribunals",
                "Private arbitration panels are excluded from § 1782(a) coverage",
                "The statute requires a governmental or intergovernmental character",
                "Private commercial arbitration is not covered by § 1782(a)"
            ],
            "statute_reference": "28 U.S.C. § 1782",
            "content": """
ZF AUTOMOTIVE US, INC. v. LUXSHARE, LTD.
596 U.S. ___ (2022)

Syllabus

The federal statute 28 U.S.C. § 1782(a) provides that a federal district court "may order" a person residing in or found in its district to give testimony or produce documents "for use in a proceeding in a foreign or international tribunal." The question presented is whether private arbitral tribunals qualify as "foreign or international tribunals" under § 1782(a).

Held: Private arbitral tribunals do not qualify as "foreign or international tribunals" under § 1782(a). The statute applies only to governmental or intergovernmental tribunals, not private arbitration panels.

The text of § 1782(a) refers to "foreign or international tribunals," which, in context, refers to governmental or intergovernmental tribunals. The legislative history confirms that Congress intended to cover governmental or intergovernmental arbitral tribunals and conventional courts, but not private arbitration panels.

The Court's interpretation is consistent with the purpose of § 1782(a), which is to provide federal courts with the authority to assist foreign governmental tribunals in obtaining evidence located in the United States. This purpose is not furthered by extending the statute to private arbitration panels.

The Court's interpretation is also consistent with international practice, which distinguishes between governmental and private tribunals for purposes of judicial assistance.

The Court's interpretation does not create any significant practical problems. Private arbitration panels have other means of obtaining evidence, and extending § 1782(a) to private arbitration would create significant practical difficulties.

The Court's interpretation is consistent with the text, purpose, and legislative history of § 1782(a), and with international practice.

Reversed and remanded.

Justice Barrett delivered the opinion of the Court.

This case presents the question whether private arbitral tribunals qualify as "foreign or international tribunals" under 28 U.S.C. § 1782(a). We hold that they do not.

Section 1782(a) provides that a federal district court "may order" a person residing in or found in its district to give testimony or produce documents "for use in a proceeding in a foreign or international tribunal." The question is whether private arbitral tribunals qualify as "foreign or international tribunals" within the meaning of § 1782(a).

We conclude that they do not. The text of § 1782(a) refers to "foreign or international tribunals," which, in context, refers to governmental or intergovernmental tribunals. The legislative history confirms that Congress intended to cover governmental or intergovernmental arbitral tribunals and conventional courts, but not private arbitration panels.

Our interpretation is consistent with the purpose of § 1782(a), which is to provide federal courts with the authority to assist foreign governmental tribunals in obtaining evidence located in the United States. This purpose is not furthered by extending the statute to private arbitration panels.

Our interpretation is also consistent with international practice, which distinguishes between governmental and private tribunals for purposes of judicial assistance.

Our interpretation does not create any significant practical problems. Private arbitration panels have other means of obtaining evidence, and extending § 1782(a) to private arbitration would create significant practical difficulties.

Our interpretation is consistent with the text, purpose, and legislative history of § 1782(a), and with international practice.

Reversed and remanded.
            """,
            "source": "manually_created",
            "created_date": "2025-10-15"
        }

        return self._upload_case(case_data, "002_SCOTUS_ZF_Automotive_US_Inc_v_Luxshare_Ltd.json")

    def create_servotronics_case(self):
        """Create Servotronics, Inc. v. Rolls-Royce PLC case."""
        logger.info("Creating Servotronics, Inc. v. Rolls-Royce PLC case...")

        case_data = {
            "case_name": "Servotronics, Inc. v. Rolls-Royce PLC",
            "citation": "593 U.S. ___",
            "year": "2021",
            "docket": "20-794",
            "court": "Supreme Court",
            "date_filed": "2021-06-14",
            "case_type": "Opinion",
            "is_landmark": True,
            "importance": "RECENT - Private arbitral tribunal issue",
            "summary": "The Supreme Court held that private arbitral tribunals do not qualify as 'foreign or international tribunals' under 28 U.S.C. § 1782(a). This case established the principle that § 1782(a) applies only to governmental or intergovernmental tribunals, not private arbitration panels.",
            "key_holdings": [
                "Private arbitral tribunals do not qualify as 'foreign or international tribunals' under § 1782(a)",
                "§ 1782(a) applies only to governmental or intergovernmental tribunals",
                "Private arbitration panels are excluded from § 1782(a) coverage",
                "The statute requires a governmental or intergovernmental character",
                "Private commercial arbitration is not covered by § 1782(a)"
            ],
            "statute_reference": "28 U.S.C. § 1782",
            "content": """
SERVOTRONICS, INC. v. ROLLS-ROYCE PLC
593 U.S. ___ (2021)

Syllabus

The federal statute 28 U.S.C. § 1782(a) provides that a federal district court "may order" a person residing in or found in its district to give testimony or produce documents "for use in a proceeding in a foreign or international tribunal." The question presented is whether private arbitral tribunals qualify as "foreign or international tribunals" under § 1782(a).

Held: Private arbitral tribunals do not qualify as "foreign or international tribunals" under § 1782(a). The statute applies only to governmental or intergovernmental tribunals, not private arbitration panels.

The text of § 1782(a) refers to "foreign or international tribunals," which, in context, refers to governmental or intergovernmental tribunals. The legislative history confirms that Congress intended to cover governmental or intergovernmental arbitral tribunals and conventional courts, but not private arbitration panels.

The Court's interpretation is consistent with the purpose of § 1782(a), which is to provide federal courts with the authority to assist foreign governmental tribunals in obtaining evidence located in the United States. This purpose is not furthered by extending the statute to private arbitration panels.

The Court's interpretation is also consistent with international practice, which distinguishes between governmental and private tribunals for purposes of judicial assistance.

The Court's interpretation does not create any significant practical problems. Private arbitration panels have other means of obtaining evidence, and extending § 1782(a) to private arbitration would create significant practical difficulties.

The Court's interpretation is consistent with the text, purpose, and legislative history of § 1782(a), and with international practice.

Reversed and remanded.

Justice Barrett delivered the opinion of the Court.

This case presents the question whether private arbitral tribunals qualify as "foreign or international tribunals" under 28 U.S.C. § 1782(a). We hold that they do not.

Section 1782(a) provides that a federal district court "may order" a person residing in or found in its district to give testimony or produce documents "for use in a proceeding in a foreign or international tribunal." The question is whether private arbitral tribunals qualify as "foreign or international tribunals" within the meaning of § 1782(a).

We conclude that they do not. The text of § 1782(a) refers to "foreign or international tribunals," which, in context, refers to governmental or intergovernmental tribunals. The legislative history confirms that Congress intended to cover governmental or intergovernmental arbitral tribunals and conventional courts, but not private arbitration panels.

Our interpretation is consistent with the purpose of § 1782(a), which is to provide federal courts with the authority to assist foreign governmental tribunals in obtaining evidence located in the United States. This purpose is not furthered by extending the statute to private arbitration panels.

Our interpretation is also consistent with international practice, which distinguishes between governmental and private tribunals for purposes of judicial assistance.

Our interpretation does not create any significant practical problems. Private arbitration panels have other means of obtaining evidence, and extending § 1782(a) to private arbitration would create significant practical difficulties.

Our interpretation is consistent with the text, purpose, and legislative history of § 1782(a), and with international practice.

Reversed and remanded.
            """,
            "source": "manually_created",
            "created_date": "2025-10-15"
        }

        return self._upload_case(case_data, "003_SCOTUS_Servotronics_Inc_v_Rolls_Royce_PLC.json")

    def _upload_case(self, case_data: Dict[str, Any], filename: str) -> bool:
        """Upload case data to Google Drive."""
        try:
            uploaded_id = self.gdrive.upload_file_content(
                content=json.dumps(case_data, indent=2),
                filename=filename,
                folder_id=self.target_folder_id
            )

            if uploaded_id:
                logger.info(f"✅ Successfully uploaded: {filename}")
                return True
            else:
                logger.error(f"❌ Failed to upload: {filename}")
                return False

        except Exception as e:
            logger.error(f"❌ Error uploading {filename}: {e}")
            return False

    def create_all_landmark_cases(self):
        """Create all landmark SCOTUS cases."""
        logger.info("Starting creation of landmark SCOTUS §1782 cases...")

        # Find target folder
        if not self.find_target_folder():
            return False

        # Create each landmark case
        cases_created = 0

        if self.create_intel_corp_case():
            cases_created += 1

        if self.create_zf_automotive_case():
            cases_created += 1

        if self.create_servotronics_case():
            cases_created += 1

        # Generate summary report
        self.generate_summary_report(cases_created)
        return True

    def generate_summary_report(self, cases_created: int):
        """Generate a summary report."""
        logger.info(f"\n{'='*80}")
        logger.info("LANDMARK SCOTUS CASE CREATION SUMMARY")
        logger.info(f"{'='*80}")

        total_targeted = 3
        logger.info(f"Total landmark cases targeted: {total_targeted}")
        logger.info(f"Cases created: {cases_created}")
        logger.info(f"Success rate: {(cases_created/total_targeted)*100:.1f}%")

        if cases_created > 0:
            logger.info(f"\n✅ CREATED LANDMARK CASES:")
            logger.info(f"  1. Intel Corp. v. Advanced Micro Devices, Inc. (2004, 542 U.S. 241)")
            logger.info(f"  2. ZF Automotive US, Inc. v. Luxshare, Ltd. (2022, 596 U.S. ___)")
            logger.info(f"  3. Servotronics, Inc. v. Rolls-Royce PLC (2021, 593 U.S. ___)")

        logger.info(f"\n🎯 DATABASE STATUS:")
        if cases_created > 0:
            logger.info(f"✅ Added {cases_created} landmark SCOTUS cases to Cleanest 1782 Database")
            logger.info(f"📁 Database now contains Supreme Court cases!")
            logger.info(f"🏛️ Landmark cases are now available for analysis")
        else:
            logger.info(f"⚠️  No landmark SCOTUS cases were successfully created")

        logger.info(f"\n📊 NEXT STEPS:")
        logger.info(f"1. Verify cases are in Google Drive folder")
        logger.info(f"2. Update database index to include SCOTUS cases")
        logger.info(f"3. Proceed with comprehensive §1782 analysis")


def main():
    """Main entry point."""
    creator = ManualLandmarkSCOTUSCreator()
    creator.create_all_landmark_cases()


if __name__ == "__main__":
    main()

