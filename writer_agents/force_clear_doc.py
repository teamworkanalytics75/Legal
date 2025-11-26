#!/usr/bin/env python3
"""
Force clear the Google Doc by calculating actual text length and deleting everything.
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

DOC_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"

def find_credentials():
    """Find Google credentials file."""
    env_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_creds and Path(env_creds).exists():
        return env_creds
    
    possible_paths = [
        project_root / "document_ingestion" / "credentials.json",
        project_root / "writer_agents" / "credentials.json",
        project_root / "credentials.json",
    ]
    
    for path in [project_root / "document_ingestion", project_root / "writer_agents", project_root]:
        if path.exists():
            for file in path.glob("client_secret_*.json"):
                return str(file)
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

def force_clear():
    """Force clear the document by calculating actual length."""
    try:
        print("=" * 80)
        print("FORCE CLEARING GOOGLE DOC")
        print("=" * 80)
        
        credentials_path = find_credentials()
        if not credentials_path:
            credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        
        if not credentials_path:
            raise ValueError("No credentials found")
        
        bridge = create_google_docs_bridge(credentials_path=credentials_path)
        
        # Get document
        print("\n[1] Fetching document...")
        doc = bridge.docs_service.documents().get(documentId=DOC_ID).execute()
        body = doc.get("body", {})
        content_elements = body.get("content", [])
        
        print(f"   Content elements: {len(content_elements)}")
        
        # Calculate actual text length
        print("\n[2] Calculating actual document length...")
        total_chars = 0
        max_end_index = 1
        
        for element in content_elements:
            # Get endIndex from element
            elem_end = element.get("endIndex", 1)
            if elem_end > max_end_index:
                max_end_index = elem_end
            
            # Count actual text
            if "paragraph" in element:
                para = element.get("paragraph", {})
                for elem in para.get("elements", []):
                    if "textRun" in elem:
                        text = elem["textRun"].get("content", "")
                        total_chars += len(text)
        
        body_end = body.get("endIndex", 1)
        calculated_end = max(max_end_index, body_end, total_chars + 100)
        
        print(f"   Total text characters: {total_chars:,}")
        print(f"   Max element endIndex: {max_end_index:,}")
        print(f"   Body endIndex: {body_end:,}")
        print(f"   Calculated endIndex: {calculated_end:,}")
        
        # Delete everything
        print(f"\n[3] Deleting all content (1 to {calculated_end - 1})...")
        delete_request = {
            "deleteContentRange": {
                "range": {
                    "segmentId": "",
                    "startIndex": 1,
                    "endIndex": calculated_end - 1
                }
            }
        }
        
        response = bridge.docs_service.documents().batchUpdate(
            documentId=DOC_ID,
            body={"requests": [delete_request]}
        ).execute()
        
        print(f"   [OK] Delete request executed")
        print(f"   Response: {response.get('replies', [])}")
        
        # Wait a moment
        import time
        time.sleep(2)
        
        # Verify
        print("\n[4] Verifying deletion...")
        doc_after = bridge.docs_service.documents().get(documentId=DOC_ID).execute()
        body_after = doc_after.get("body", {})
        elements_after = body_after.get("content", [])
        
        total_chars_after = 0
        for element in elements_after:
            if "paragraph" in element:
                para = element.get("paragraph", {})
                for elem in para.get("elements", []):
                    if "textRun" in elem:
                        text = elem["textRun"].get("content", "")
                        total_chars_after += len(text)
        
        print(f"   Elements after: {len(elements_after)}")
        print(f"   Text chars after: {total_chars_after:,}")
        
        if total_chars_after < total_chars:
            print(f"   [SUCCESS] Document cleared! ({total_chars:,} -> {total_chars_after:,} chars)")
        else:
            print(f"   [WARNING] Document may not be fully cleared")
        
        # Insert placeholder
        print("\n[5] Inserting placeholder...")
        placeholder = [
            {"type": "paragraph", "text": f"Document Force Cleared - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"},
            {"type": "paragraph", "text": ""},
            {"type": "paragraph", "text": "This document has been force cleared and is ready for a fresh draft."}
        ]
        
        bridge.update_document(DOC_ID, placeholder)
        print("   [OK] Placeholder inserted")
        
        print("\n" + "=" * 80)
        print("[SUCCESS] Force clear completed!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = force_clear()
    sys.exit(0 if success else 1)

