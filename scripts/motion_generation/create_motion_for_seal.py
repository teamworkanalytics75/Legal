"""Setup script for Google Docs integration.

This script helps you set up Google service account credentials
and creates a motion for seal pseudonym in your Google Drive folder.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_google_credentials():
    """Help user set up Google service account credentials."""
    print("Google Docs Integration Setup")
    print("=" * 50)

    # Check if credentials already exist
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        print(f"Google credentials found at: {creds_path}")
        return creds_path

    print("\nTo use Google Docs integration, you need to:")
    print("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
    print("2. Create a new project or select existing one")
    print("3. Enable Google Docs API and Google Drive API")
    print("4. Create a Service Account:")
    print("   - Go to IAM & Admin > Service Accounts")
    print("   - Click 'Create Service Account'")
    print("   - Give it a name like 'writer-agents-service'")
    print("   - Grant it 'Editor' role for Google Drive")
    print("5. Create and download JSON key file")
    print("6. Place the JSON file in your project directory")

    print("\nQuick setup options:")
    print("A) I have the JSON file - let me specify the path")
    print("B) I need to create credentials first")
    print("C) Skip Google Docs for now, create local document")

    choice = input("\nEnter your choice (A/B/C): ").strip().upper()

    if choice == "A":
        json_path = input("Enter the full path to your service account JSON file: ").strip()
        if Path(json_path).exists():
            # Set environment variable
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path
            print(f"Credentials set to: {json_path}")
            return json_path
        else:
            print(f"File not found: {json_path}")
            return None

    elif choice == "B":
        print("\nAfter creating your service account:")
        print("1. Download the JSON key file")
        print("2. Run this script again and choose option A")
        print("3. Or set the GOOGLE_APPLICATION_CREDENTIALS environment variable")
        return None

    elif choice == "C":
        print("Will create local document instead")
        return None

    else:
        print("Invalid choice")
        return None

def create_motion_for_seal_pseudonym():
    """Create a motion for seal pseudonym document."""
    print("\nCreating Motion for Seal Pseudonym")
    print("=" * 50)

    # Import the necessary modules
    try:
        from writer_agents.code.google_docs_bridge import create_google_docs_bridge
        from writer_agents.code.google_docs_formatter import GoogleDocsFormatter
        from writer_agents.code.document_tracker import create_document_tracker
        from writer_agents.code.tasks import WriterDeliverable, DraftSection, PlanDirective, ReviewFindings

        print("Google Docs modules imported successfully")

    except ImportError as e:
        print(f"Failed to import modules: {e}")
        print("Make sure you're running from the project root directory")
        return False

    # Check if Google credentials are available
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not Path(creds_path).exists():
        print("Google credentials not found")
        print("Run setup_google_credentials() first or set GOOGLE_APPLICATION_CREDENTIALS")
        return False

    try:
        # Initialize Google Docs components
        print("Initializing Google Docs bridge...")
        bridge = create_google_docs_bridge()

        print("Initializing document formatter...")
        formatter = GoogleDocsFormatter()

        print("Initializing document tracker...")
        doc_tracker = create_document_tracker()

        # Create motion content
        print("Creating motion content...")

        # Create WriterDeliverable for motion for seal pseudonym
        deliverable = WriterDeliverable(
            plan=PlanDirective(
                objective="Draft motion for seal pseudonym under Federal Rule of Civil Procedure 5.2",
                deliverable_format="Federal court motion",
                tone="Professional and persuasive",
                style_constraints=[
                    "Follow Federal Rules of Civil Procedure",
                    "Include proper legal citations",
                    "Address privacy concerns",
                    "Demonstrate good cause for sealing"
                ],
                citation_expectations="Use proper Bluebook citation format"
            ),
            sections=[
                DraftSection(
                    section_id="caption",
                    title="Caption",
                    body="UNITED STATES DISTRICT COURT\nFOR THE DISTRICT OF MASSACHUSETTS\n\n[PLAINTIFF],\nPlaintiff,\n\nv.\n\n[DEFENDANT],\nDefendant.\n\nCivil Action No. [CASE NUMBER]"
                ),
                DraftSection(
                    section_id="motion_title",
                    title="Motion Title",
                    body="MOTION FOR LEAVE TO FILE UNDER SEAL AND TO PROCEED UNDER PSEUDONYM"
                ),
                DraftSection(
                    section_id="introduction",
                    title="Introduction",
                    body="""Plaintiff respectfully moves this Court for leave to file certain documents under seal and to proceed under pseudonym pursuant to Federal Rule of Civil Procedure 5.2 and this Court's Local Rules. This motion is made on the grounds that the documents contain highly sensitive personal information that, if publicly disclosed, would cause irreparable harm to Plaintiff's privacy and safety."""
                ),
                DraftSection(
                    section_id="factual_background",
                    title="Factual Background",
                    body="""Plaintiff is an individual who has been subjected to [describe the sensitive circumstances that warrant pseudonym treatment]. The nature of Plaintiff's claims involves highly personal and private matters that, if disclosed publicly, would cause significant embarrassment, harassment, and potential safety concerns.

The documents Plaintiff seeks to file under seal contain:
1. Highly sensitive personal information
2. Private communications and records
3. Information that could be used to identify Plaintiff's location or circumstances
4. Details that, if public, could subject Plaintiff to harassment or retaliation"""
                ),
                DraftSection(
                    section_id="legal_standard",
                    title="Legal Standard",
                    body="""Federal Rule of Civil Procedure 5.2(a) provides that a filing made with the court must include certain identifying information unless the court orders otherwise. Rule 5.2(a)(3) allows the court to order that a party be identified only by a pseudonym if the court determines that the party's interest in privacy or safety outweighs the public's interest in disclosure.

Courts have granted motions to proceed under pseudonym when:
1. The case involves highly sensitive personal matters
2. Public disclosure would cause significant embarrassment or harassment
3. The plaintiff's safety or privacy interests outweigh the public's right to know
4. The underlying claims involve matters of extreme personal sensitivity

See, e.g., Doe v. City of New York, 15 F. Supp. 3d 411 (S.D.N.Y. 2014); Doe v. Stegall, 653 F.2d 180 (5th Cir. 1981)."""
                ),
                DraftSection(
                    section_id="good_cause",
                    title="Good Cause for Sealing",
                    body="""Plaintiff has demonstrated good cause for sealing the sensitive documents and proceeding under pseudonym based on the following factors:

1. **Extreme Personal Sensitivity**: The matters at issue involve highly personal and private circumstances that are not appropriate for public disclosure.

2. **Risk of Harassment**: Public disclosure of Plaintiff's identity and the sensitive details of this case would likely subject Plaintiff to harassment, embarrassment, and potential retaliation.

3. **Safety Concerns**: The nature of the underlying facts creates legitimate safety concerns for Plaintiff if identity and details were publicly known.

4. **Privacy Interests**: Plaintiff's fundamental privacy interests in these highly personal matters outweigh any public interest in disclosure.

5. **Minimal Public Interest**: The public's interest in knowing Plaintiff's identity is minimal compared to the significant privacy and safety interests at stake."""
                ),
                DraftSection(
                    section_id="proposed_protective_order",
                    title="Proposed Protective Order",
                    body="""Plaintiff proposes the following protective measures:

1. **Pseudonym Usage**: Plaintiff shall be identified as "Jane Doe" or "John Doe" in all public filings and court proceedings.

2. **Sealed Documents**: The following documents shall be filed under seal:
   - Complaint (with redacted public version)
   - Any exhibits containing sensitive personal information
   - Discovery materials containing private information
   - Any other documents containing sensitive details

3. **Redacted Public Versions**: Where appropriate, Plaintiff will file redacted public versions of documents that remove sensitive identifying information while preserving the legal arguments and claims.

4. **Confidentiality Designations**: All discovery materials containing sensitive information shall be designated as "CONFIDENTIAL" and subject to the protective order."""
                ),
                DraftSection(
                    section_id="conclusion",
                    title="Conclusion",
                    body="""For the foregoing reasons, Plaintiff respectfully requests that this Court:

1. Grant Plaintiff leave to proceed under the pseudonym "[PLAINTIFF PSEUDONYM]"
2. Allow Plaintiff to file sensitive documents under seal
3. Permit Plaintiff to file redacted public versions where appropriate
4. Enter a protective order governing the handling of confidential information

This relief is necessary to protect Plaintiff's privacy, safety, and dignity while ensuring that the judicial process can proceed fairly and efficiently.

Respectfully submitted,

[ATTORNEY NAME]
[LAW FIRM]
[ADDRESS]
[PHONE]
[EMAIL]

Certificate of Service
I hereby certify that on [DATE], I served the foregoing motion on [DEFENDANT'S ATTORNEY] via [METHOD OF SERVICE]."""
                )
            ],
            edited_document="Motion for Seal Pseudonym - Complete Document",
            reviews=[
                ReviewFindings(
                    section_id="general",
                    severity="info",
                    message="Motion drafted with proper legal structure",
                    suggestions="Review specific facts and customize for your case"
                )
            ],
            metadata={
                "workflow_type": "motion_drafting",
                "document_type": "motion_for_seal_pseudonym",
                "court": "federal_district_massachusetts",
                "rule_citation": "FRCP_5.2"
            }
        )

        # Format the document
        print("Formatting document for Google Docs...")
        formatted_content = formatter.format_deliverable(deliverable, "motion")

        # Create document in Google Drive
        print("Creating document in Google Drive...")
        folder_id = "1MZwep4pb9M52lSLLGQAd3quslA8A5iBu"  # Your specified folder
        doc_id, doc_url = bridge.create_document(
            title="Motion for Seal Pseudonym - Draft",
            folder_id=folder_id
        )

        # Update document with content
        print("Updating document with content...")
        bridge.update_document(doc_id, formatted_content, "Motion for Seal Pseudonym - Draft")

        # Store document record
        print("Recording document in tracker...")
        case_id = f"motion_seal_pseudonym_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        doc_tracker.create_document_record(
            case_id=case_id,
            google_doc_id=doc_id,
            doc_url=doc_url,
            folder_id=folder_id,
            title="Motion for Seal Pseudonym - Draft",
            case_summary="Motion for leave to file under seal and proceed under pseudonym pursuant to FRCP 5.2",
            metadata={
                "workflow_type": "motion_drafting",
                "document_type": "motion_for_seal_pseudonym",
                "created_by": "writer_agents_system"
            }
        )

        print("\nSUCCESS!")
        print("=" * 50)
        print(f"Document created: {doc_url}")
        print(f"Document ID: {doc_id}")
        print(f"Folder: https://drive.google.com/drive/folders/{folder_id}")
        print(f"Case ID: {case_id}")

        print("\nNext Steps:")
        print("1. Open the document in Google Docs")
        print("2. Customize the facts and details for your specific case")
        print("3. Add your attorney information and signature")
        print("4. Review and file with the court")

        return True

    except Exception as e:
        print(f"Error creating document: {e}")
        logger.error(f"Failed to create motion: {e}")
        return False

def create_local_motion():
    """Create a local version of the motion if Google Docs is not available."""
    print("Creating local motion document...")

    motion_content = """MOTION FOR LEAVE TO FILE UNDER SEAL AND TO PROCEED UNDER PSEUDONYM

UNITED STATES DISTRICT COURT
FOR THE DISTRICT OF MASSACHUSETTS

[PLAINTIFF],
Plaintiff,

v.

[DEFENDANT],
Defendant.

Civil Action No. [CASE NUMBER]

MOTION FOR LEAVE TO FILE UNDER SEAL AND TO PROCEED UNDER PSEUDONYM

Plaintiff respectfully moves this Court for leave to file certain documents under seal and to proceed under pseudonym pursuant to Federal Rule of Civil Procedure 5.2 and this Court's Local Rules.

I. INTRODUCTION

This motion is made on the grounds that the documents contain highly sensitive personal information that, if publicly disclosed, would cause irreparable harm to Plaintiff's privacy and safety.

II. FACTUAL BACKGROUND

Plaintiff is an individual who has been subjected to [describe the sensitive circumstances]. The nature of Plaintiff's claims involves highly personal and private matters that, if disclosed publicly, would cause significant embarrassment, harassment, and potential safety concerns.

III. LEGAL STANDARD

Federal Rule of Civil Procedure 5.2(a) provides that a filing made with the court must include certain identifying information unless the court orders otherwise. Rule 5.2(a)(3) allows the court to order that a party be identified only by a pseudonym if the court determines that the party's interest in privacy or safety outweighs the public's interest in disclosure.

IV. GOOD CAUSE FOR SEALING

Plaintiff has demonstrated good cause for sealing based on:
1. Extreme personal sensitivity of the matters
2. Risk of harassment and embarrassment
3. Legitimate safety concerns
4. Fundamental privacy interests
5. Minimal public interest in disclosure

V. REQUESTED RELIEF

Plaintiff respectfully requests that this Court:
1. Grant Plaintiff leave to proceed under pseudonym
2. Allow filing of sensitive documents under seal
3. Permit filing of redacted public versions
4. Enter appropriate protective order

Respectfully submitted,

[ATTORNEY NAME]
[LAW FIRM]
[ADDRESS]
[PHONE]
[EMAIL]

Certificate of Service
I hereby certify that on [DATE], I served the foregoing motion on [DEFENDANT'S ATTORNEY] via [METHOD OF SERVICE].
"""

    # Save to local file
    output_path = Path("motion_for_seal_pseudonym_draft.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(motion_content)

    print(f"Local motion created: {output_path}")
    print("Edit this file to customize for your specific case")

def main():
    """Main function to run the setup and document creation."""
    print("Google Docs Motion Creator")
    print("=" * 50)

    # Setup credentials
    creds_path = setup_google_credentials()

    if creds_path:
        # Create the motion
        success = create_motion_for_seal_pseudonym()
        if success:
            print("\nMotion for Seal Pseudonym created successfully!")
        else:
            print("\nFailed to create motion")
    else:
        print("\nCreating local document instead...")
        # Create a local version
        create_local_motion()

if __name__ == "__main__":
    main()
