#!/usr/bin/env python3
"""Quick script to check workflow phase from Google Doc content."""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "writer_agents" / "code"))
sys.path.insert(0, str(project_root / "writer_agents"))

try:
    from code.google_docs_bridge import GoogleDocsBridge

    # Known document ID
    DOC_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"

    print(f"Checking workflow status for document: {DOC_ID}")
    print("=" * 80)

    # Initialize bridge
    bridge = GoogleDocsBridge()

    # Get document content
    try:
        doc = bridge.docs_service.documents().get(documentId=DOC_ID).execute()

        # Extract text content
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

        # Check for phase indicators
        phases = ["EXPLORE", "RESEARCH", "PLAN", "DRAFT", "VALIDATE", "REVIEW", "REFINE", "COMMIT"]
        found_phases = []
        for phase in phases:
            if phase in content.upper():
                found_phases.append(phase)

        print(f"\nDocument Content Preview (first 500 chars):")
        print("-" * 80)
        print(content[:500])
        print("-" * 80)

        print(f"\nPhase Indicators Found: {found_phases if found_phases else 'None'}")

        # Check for draft content
        has_draft_content = len(content) > 200 and not all(c in "le: top_features_count: top_feature: master_draft_mode: true" for c in content[:200])

        if "DRAFT" in content.upper() or "LIVE UPDATE" in content.upper():
            print("\n[OK] DRAFT PHASE DETECTED!")
            if has_draft_content:
                print("   [OK] Actual draft content is present")
            else:
                print("   [WAIT] Still waiting for draft content to appear")
        elif any(p in content.upper() for p in ["EXPLORE", "RESEARCH", "PLAN"]):
            print("\n[WAIT] Still in early phases (EXPLORE/RESEARCH/PLAN)")
        else:
            print("\n[?] Phase unclear from document content")

        print(f"\nContent Length: {len(content)} characters")
        print(f"Content Lines: {len(content.split(chr(10)))} lines")

    except Exception as e:
        print(f"\n[ERROR] Error reading document: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

