#!/usr/bin/env python3
"""Debug document structure to understand why delete isn't working."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "writer_agents" / "code"))
sys.path.insert(0, str(project_root / "writer_agents"))

from code.google_docs_bridge import GoogleDocsBridge

DOC_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"

print(f"Debugging document structure: {DOC_ID}")
print("=" * 80)

bridge = GoogleDocsBridge()

try:
    doc = bridge.docs_service.documents().get(documentId=DOC_ID).execute()

    body = doc.get("body", {})
    end_index = body.get("endIndex", 1)

    print(f"\nDocument endIndex: {end_index}")
    print(f"Document body keys: {list(body.keys())}")

    # Count content elements
    content_elements = body.get("content", [])
    print(f"\nNumber of content elements: {len(content_elements)}")

    # Show structure of all elements with their indices
    for i, element in enumerate(content_elements):
        start_idx = element.get("startIndex", "N/A")
        end_idx = element.get("endIndex", "N/A")
        print(f"\nElement {i}: startIndex={start_idx}, endIndex={end_idx}")
        print(f"  Keys: {list(element.keys())}")
        if "paragraph" in element:
            para = element["paragraph"]
            print(f"  Type: paragraph")
            text_content = ""
            for elem in para.get("elements", []):
                if "textRun" in elem:
                    text_content += elem["textRun"].get("content", "")
            print(f"  Text: {repr(text_content[:60])}")
        elif "sectionBreak" in element:
            print(f"  Type: sectionBreak")
        elif "table" in element:
            print(f"  Type: table")

    # Calculate actual text length
    text_length = 0
    for element in content_elements:
        if "paragraph" in element:
            for text_run in element["paragraph"].get("elements", []):
                if "textRun" in text_run:
                    text_length += len(text_run["textRun"].get("content", ""))

    print(f"\nCalculated text length: {text_length}")
    print(f"Document endIndex: {end_index}")
    print(f"Difference: {end_index - text_length}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

