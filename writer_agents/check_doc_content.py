#!/usr/bin/env python3
"""Check actual document content in Google Docs."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "writer_agents" / "code"))
sys.path.insert(0, str(project_root / "writer_agents"))
sys.path.insert(0, str(project_root))

from code.google_docs_bridge import create_google_docs_bridge
import os

DOC_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"

def find_credentials():
    env_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_creds and Path(env_creds).exists():
        return env_creds
    
    for path in [project_root / "document_ingestion", project_root / "writer_agents", project_root]:
        if path.exists():
            for file in path.glob("client_secret_*.json"):
                return str(file)
    return None

def check_content():
    print("=" * 80)
    print("CHECKING ACTUAL DOCUMENT CONTENT")
    print("=" * 80)
    
    credentials_path = find_credentials() or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    bridge = create_google_docs_bridge(credentials_path=credentials_path)
    
    print("\n[1] Fetching document...")
    doc = bridge.docs_service.documents().get(documentId=DOC_ID).execute()
    body = doc.get("body", {})
    content_elements = body.get("content", [])
    
    print(f"   Content elements: {len(content_elements)}")
    
    print("\n[2] Extracting actual text content...")
    full_text = ""
    text_parts = []
    
    for i, element in enumerate(content_elements[:50]):  # First 50 elements
        if "paragraph" in element:
            para = element.get("paragraph", {})
            para_text = ""
            for elem in para.get("elements", []):
                if "textRun" in elem:
                    text = elem["textRun"].get("content", "")
                    para_text += text
                    full_text += text
            
            if para_text.strip():
                text_parts.append(f"Element {i}: {repr(para_text[:100])}")
    
    print(f"\n[3] Text Summary:")
    print(f"   Total text length: {len(full_text)} characters")
    print(f"   First 500 characters:")
    print("   " + "-" * 76)
    print("   " + full_text[:500].replace("\n", "\n   "))
    print("   " + "-" * 76)
    
    if len(full_text) < 100:
        print(f"\n⚠️  WARNING: Document appears to be empty or nearly empty!")
        print(f"   Only {len(full_text)} characters found")
    else:
        print(f"\n✅ Document has content ({len(full_text)} characters)")
    
    print(f"\n[4] Sample text parts (first 10 non-empty):")
    for part in text_parts[:10]:
        print(f"   {part}")
    
    print("\n" + "=" * 80)
    return len(full_text) > 0

if __name__ == "__main__":
    has_content = check_content()
    sys.exit(0 if has_content else 1)

