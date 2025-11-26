#!/usr/bin/env python3
"""Test update with full logging to see what's happening."""

import sys
import logging
from pathlib import Path

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "writer_agents" / "code"))
sys.path.insert(0, str(project_root / "writer_agents"))

from code.google_docs_bridge import GoogleDocsBridge

DOC_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"

print(f"Testing update with full logging: {DOC_ID}")
print("=" * 80)

bridge = GoogleDocsBridge()

# Test content
test_content = [
    {"type": "paragraph", "text": "[LIVE TEST] Testing update mechanism with logging enabled."},
    {"type": "paragraph", "text": f"Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"},
    {"type": "paragraph", "text": "If you see this, the update is working correctly."}
]

try:
    print("\n[1] Calling update_document...")
    bridge.update_document(DOC_ID, test_content)
    print("[2] update_document() returned successfully")

    print("\n[3] Waiting 3 seconds for Google Docs to sync...")
    import time
    time.sleep(3)

    print("[4] Checking document content...")
    doc = bridge.docs_service.documents().get(documentId=DOC_ID).execute()
    content_parts = []
    for element in doc.get("body", {}).get("content", []):
        if "paragraph" in element:
            para = element["paragraph"]
            for text_run in para.get("elements", []):
                if "textRun" in text_run:
                    text = text_run["textRun"].get("content", "")
                    if text.strip():
                        content_parts.append(text.strip())

    content = "\n".join(content_parts)
    print(f"\n[5] Document content after update:")
    print("-" * 80)
    print(content[:300])
    print("-" * 80)

    if "[LIVE TEST]" in content:
        print("\n[SUCCESS] Document was updated correctly!")
    else:
        print("\n[WARNING] Document doesn't contain test content - update may have failed")

except Exception as e:
    print(f"\n[ERROR] Update failed: {e}")
    import traceback
    traceback.print_exc()

