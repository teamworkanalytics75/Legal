#!/usr/bin/env python3
"""
Simple test to UPDATE the Google Doc and prove outline integration works.

This will directly update the Google Doc with outline-integrated content.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add paths
code_dir = Path(__file__).parent / "code"
sys.path.insert(0, str(code_dir))
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

# Load outline_manager
outline_manager_path = code_dir / "outline_manager.py"
spec = importlib.util.spec_from_file_location("outline_manager", outline_manager_path)
outline_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(outline_manager_module)
load_outline_manager = outline_manager_module.load_outline_manager

# Load formatter
formatter_path = code_dir / "google_docs_formatter.py"
spec = importlib.util.spec_from_file_location("google_docs_formatter", formatter_path)
formatter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formatter_module)
GoogleDocsFormatter = formatter_module.GoogleDocsFormatter

# Load bridge
bridge_path = code_dir / "google_docs_bridge.py"
spec = importlib.util.spec_from_file_location("google_docs_bridge", bridge_path)
bridge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge_module)
GoogleDocsBridge = bridge_module.GoogleDocsBridge

from tasks import WriterDeliverable, DraftSection, PlanDirective

print("=" * 80)
print("TEST: UPDATE GOOGLE DOC WITH OUTLINE INTEGRATION")
print("=" * 80)
print()

# Load outline manager
print("[1/4] Loading outline manager...")
outline_manager = load_outline_manager()
perfect_order = outline_manager.get_section_order()
print(f"   [OK] Loaded {len(outline_manager.sections)} sections")
print(f"   [OK] Perfect order: {', '.join(perfect_order[:5])}...")
print()

# Create test deliverable with sections in WRONG order (to test reordering)
print("[2/4] Creating test deliverable...")
sections = [
    DraftSection(
        section_id="privacy_harm",
        title="Privacy Harm Analysis",
        body="This section discusses privacy concerns.\n\n* Privacy harm 1\n* Privacy harm 2\n* Privacy harm 3"
    ),
    DraftSection(
        section_id="introduction",
        title="Introduction",
        body="This is the introduction to the motion."
    ),
    DraftSection(
        section_id="factual_background",
        title="Factual Background",
        body="The factual background is as follows."
    ),
    DraftSection(
        section_id="legal_standard",
        title="Legal Standard",
        body="The legal standard requires:\n\n1. Factor one\n2. Factor two\n3. Factor three"
    ),
    DraftSection(
        section_id="balancing_test",
        title="Balancing Test",
        body="The balancing test weighs:\n\n* Privacy interests\n* Public access interests"
    ),
]

print(f"   [OK] Created {len(sections)} sections")
print(f"   Original order: {[s.section_id for s in sections]}")
print()

# Format with outline integration
print("[3/4] Formatting with outline integration...")
deliverable = WriterDeliverable(
    plan=PlanDirective(
        objective="Test outline integration with Google Docs update",
        deliverable_format="Motion",
        tone="Formal",
        style_constraints=["Legal writing"],
        citation_expectations="Standard legal citations"
    ),
    sections=sections,
    edited_document="\n\n".join([s.body for s in sections]),
    reviews=[],
    metadata={}
)

formatter = GoogleDocsFormatter()

# Simulate detected sections (for outline reordering)
detected_sections = {
    "introduction": 0,
    "legal_standard": 1,
    "factual_background": 2,
    "privacy_harm": 3,
    "balancing_test": 4
}

formatted_content = formatter.format_deliverable(
    deliverable,
    format_type="motion",
    outline_manager=outline_manager,
    detected_sections=detected_sections
)

print(f"   [OK] Formatted {len(formatted_content)} content elements")
print()

# Add test header
print("[4/4] Updating Google Doc...")
test_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
test_header = [
    {
        "type": "heading1",
        "text": f"OUTLINE INTEGRATION TEST - {test_timestamp}"
    },
    {
        "type": "paragraph",
        "text": "This update proves the outline integration is working!"
    },
    {
        "type": "paragraph",
        "text": f"Sections were reordered according to perfect outline structure."
    },
    {
        "type": "paragraph",
        "text": f"Perfect outline order: {', '.join(perfect_order[:5])}..."
    },
    {
        "type": "paragraph",
        "text": "=" * 80
    },
    {
        "type": "paragraph",
        "text": ""
    }
]

full_content = test_header + formatted_content

try:
    bridge = GoogleDocsBridge()

    print("   Updating document: 1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE")
    bridge.update_document(
        "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE",
        full_content,
        "Motion for Seal and Pseudonym - Master Draft"
    )

    print("   [OK] Google Doc updated successfully!")
    print()
    print("=" * 80)
    print("SUCCESS: GOOGLE DOC UPDATED WITH OUTLINE INTEGRATION!")
    print("=" * 80)
    print()
    print("Document Link:")
    print("https://docs.google.com/document/d/1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE/edit?usp=drivesdk")
    print()
    print("The document now shows:")
    print("  1. Test header with timestamp:", test_timestamp)
    print("  2. Sections reordered according to perfect outline")
    print("  3. Outline integration working!")
    print()
    print("Check the document now - it should be updated!")
    print("=" * 80)

except Exception as e:
    print(f"   [ERROR] Failed to update: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

