#!/usr/bin/env python3
"""
Clear the master Google Doc and test the fix for content deletion.

This script will:
1. Connect to the Google Doc
2. Clear all existing content (231 pages)
3. Insert a simple placeholder message
4. Verify the fix is working
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "writer_agents" / "code"))
sys.path.insert(0, str(project_root / "writer_agents"))
sys.path.insert(0, str(project_root))

from code.google_docs_bridge import create_google_docs_bridge
import os

# Master Draft Document ID
DOC_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"
DOC_URL = f"https://docs.google.com/document/d/{DOC_ID}/edit"

def find_credentials():
    """Find Google credentials file in common locations."""
    # Check environment variable first
    env_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_creds and Path(env_creds).exists():
        return env_creds
    
    # Check common locations
    possible_paths = [
        project_root / "document_ingestion" / "credentials.json",
        project_root / "writer_agents" / "credentials.json",
        project_root / "credentials.json",
        project_root.parent / "document_ingestion" / "credentials.json",
    ]
    
    # Also check for client_secret files
    for path in [project_root / "document_ingestion", project_root / "writer_agents", project_root]:
        if path.exists():
            for file in path.glob("client_secret_*.json"):
                return str(file)
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

def clear_and_test():
    """Clear the document and test the fix."""
    try:
        print("=" * 80)
        print("CLEARING GOOGLE DOC - Testing Content Deletion Fix")
        print("=" * 80)
        print(f"\nDocument ID: {DOC_ID}")
        print(f"Document URL: {DOC_URL}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Find credentials
        print("[1/5] Finding Google credentials...")
        credentials_path = find_credentials()
        if credentials_path:
            print(f"   [OK] Found credentials: {credentials_path}")
        else:
            print("   [WARNING] No credentials file found, checking environment variable...")
            if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                print(f"   [OK] Using credentials from environment: {credentials_path}")
            else:
                raise ValueError(
                    "Google credentials not found. Please set GOOGLE_APPLICATION_CREDENTIALS "
                    "environment variable or place credentials.json in document_ingestion/"
                )

        # Initialize bridge
        print("\n[2/5] Initializing Google Docs bridge...")
        bridge = create_google_docs_bridge(credentials_path=credentials_path)
        print("   [OK] Bridge initialized")

        # Get current document info
        print("\n[3/5] Checking current document state...")
        doc = bridge.docs_service.documents().get(documentId=DOC_ID).execute()
        body = doc.get("body", {})
        end_index = body.get("endIndex", 1)
        content_elements = body.get("content", [])
        
        print(f"   Current document length: {end_index} characters")
        print(f"   Content elements: {len(content_elements)}")
        
        # Estimate pages (rough: ~2000 chars per page)
        estimated_pages = end_index / 2000 if end_index > 1 else 0
        
        if end_index > 1:
            print(f"   Estimated pages: ~{estimated_pages:.1f}")
            print(f"   [INFO] Document has content - will be cleared")
        else:
            print(f"   [INFO] Document appears empty")

        # Create placeholder content
        print("\n[4/5] Preparing placeholder content...")
        if end_index > 1:
            lines = [
                f"Document Cleared - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "This document has been cleared and is ready for a fresh draft.",
                "",
                "The content deletion fix has been applied. The next motion generation will replace this message with the actual draft content.",
                "",
                f"Previous content ({end_index} characters, ~{estimated_pages:.1f} pages) has been removed."
            ]
        else:
            lines = [
                f"Document Cleared - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "This document has been cleared and is ready for a fresh draft.",
                "",
                "The content deletion fix has been applied. The next motion generation will replace this message with the actual draft content.",
                "",
                "Document was already empty or minimal."
            ]
        
        # Format as Google Docs content
        content = []
        for line in lines:
            content.append({
                "type": "paragraph",
                "text": line
            })
        
        print(f"   [OK] Created {len(content)} content elements")

        # Update document (this will clear old content and insert new)
        print("\n[5/5] Updating document (clearing old content + inserting placeholder)...")
        bridge.update_document(DOC_ID, content, title="Motion for Seal and Pseudonym - Master Draft")
        print("   [OK] Document updated successfully!")

        # Verify
        print("\n[VERIFY] Checking document after update...")
        import time
        time.sleep(2)  # Wait for Google Docs to sync
        
        doc_after = bridge.docs_service.documents().get(documentId=DOC_ID).execute()
        body_after = doc_after.get("body", {})
        end_index_after = body_after.get("endIndex", 1)
        content_elements_after = body_after.get("content", [])
        
        print(f"   New document length: {end_index_after} characters")
        print(f"   Content elements: {len(content_elements_after)}")
        
        if end_index_after < end_index:
            reduction = end_index - end_index_after
            print(f"   [SUCCESS] Content reduced by {reduction} characters")
            print(f"   [SUCCESS] Fix is working - old content was cleared!")
        else:
            print(f"   [WARNING] Document length didn't decrease - may need investigation")

        print("\n" + "=" * 80)
        print("[SUCCESS] Document cleared and ready for fresh draft!")
        print("=" * 80)
        print(f"\nView your document: {DOC_URL}")
        print("\nNext steps:")
        print("  1. Run a motion generation workflow")
        print("  2. The new draft will replace this placeholder")
        print("  3. Content will be properly cleared before each update")
        
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to clear document: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = clear_and_test()
    sys.exit(0 if success else 1)

