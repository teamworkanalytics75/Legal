#!/usr/bin/env python3
"""Test direct document update to see if it works."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "writer_agents" / "code"))
sys.path.insert(0, str(project_root / "writer_agents"))

from code.google_docs_bridge import GoogleDocsBridge

DOC_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"

print(f"Testing direct update to document: {DOC_ID}")
print("=" * 80)

bridge = GoogleDocsBridge()

# Test content
test_content = [
    {"type": "paragraph", "text": "[TEST UPDATE] This is a test update from the workflow system."},
    {"type": "paragraph", "text": "If you see this, the update mechanism is working."},
    {"type": "paragraph", "text": "The workflow should replace this with actual draft content soon."}
]

try:
    print("\nAttempting to update document...")
    bridge.update_document(DOC_ID, test_content)
    print("[SUCCESS] Document updated successfully!")
    print("\nCheck your Google Doc to see the test content.")
except Exception as e:
    print(f"\n[ERROR] Update failed: {e}")
    import traceback
    traceback.print_exc()

