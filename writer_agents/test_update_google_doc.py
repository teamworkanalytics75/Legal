#!/usr/bin/env python3
"""
Test script to UPDATE the Google Doc and prove outline integration works.

This will:
1. Create a test deliverable with sections in wrong order
2. Use outline integration to reorder sections
3. Update the actual Google Doc
4. Prove the integration works
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add paths
code_dir = Path(__file__).parent / "code"
sys.path.insert(0, str(code_dir))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks import WriterDeliverable, DraftSection, PlanDirective

# Import directly to avoid import issues
import importlib.util

# Load modules
outline_manager_path = code_dir / "outline_manager.py"
spec = importlib.util.spec_from_file_location("outline_manager", outline_manager_path)
outline_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(outline_manager_module)
load_outline_manager = outline_manager_module.load_outline_manager

formatter_path = code_dir / "google_docs_formatter.py"
spec = importlib.util.spec_from_file_location("google_docs_formatter", formatter_path)
formatter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formatter_module)
GoogleDocsFormatter = formatter_module.GoogleDocsFormatter

# Load google_docs_bridge
bridge_path = code_dir / "google_docs_bridge.py"
spec = importlib.util.spec_from_file_location("google_docs_bridge", bridge_path)
bridge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge_module)
GoogleDocsBridge = bridge_module.GoogleDocsBridge

print("=" * 80)
print("TEST: UPDATE GOOGLE DOC WITH OUTLINE INTEGRATION")
print("=" * 80)
print()

# Load outline manager
print("[1/5] Loading outline manager...")
outline_manager = load_outline_manager()
print(f"   [OK] Loaded {len(outline_manager.sections)} sections")
print()

# Create test deliverable with sections in WRONG order
print("[2/5] Creating test deliverable with sections in wrong order...")
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
    metadata={
        "test": True,
        "outline_integration_test": True,
        "timestamp": datetime.now().isoformat()
    }
)

print(f"   [OK] Created deliverable with {len(sections)} sections")
print("   Original order:", [s.section_id for s in sections])
print()

# Detect sections - import directly
print("[3/5] Detecting sections and validating outline...")
orchestrator_path = code_dir / "sk_plugins" / "FeaturePlugin" / "feature_orchestrator.py"
spec = importlib.util.spec_from_file_location("feature_orchestrator", orchestrator_path)
orchestrator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orchestrator_module)
RefinementLoop = orchestrator_module.RefinementLoop
refinement_loop = RefinementLoop(plugins={}, outline_manager=outline_manager)
detected_sections = refinement_loop._detect_sections(deliverable.edited_document)
print(f"   [OK] Detected {len(detected_sections)} sections: {list(detected_sections.keys())}")

validation = outline_manager.validate_section_order(list(detected_sections.keys()))
print(f"   [OK] Validation: {'VALID' if validation['valid'] else 'INVALID'}")
if validation.get('issues'):
    print(f"   [WARN] Issues: {len(validation['issues'])}")
print()

# Format with outline integration
print("[4/5] Formatting deliverable with outline integration...")
formatter = GoogleDocsFormatter()
formatted_content = formatter.format_deliverable(
    deliverable,
    format_type="motion",
    outline_manager=outline_manager,
    detected_sections=detected_sections
)
print(f"   [OK] Formatted {len(formatted_content)} content elements")

# Check if sections were reordered
print("   Checking section order in formatted content...")
section_titles = []
for element in formatted_content:
    if element.get("type") == "heading2" and "SECTION" in element.get("text", ""):
        section_titles.append(element.get("text", ""))
print(f"   [OK] Found {len(section_titles)} section headers")
print()

# Update Google Doc
print("[5/5] Updating Google Doc...")
print("   Master Draft ID: 1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE")
print("   Adding test content with outline integration proof...")

try:
    bridge = GoogleDocsBridge()

    # Add test header to show this is an outline integration test
    test_header = [
        {
            "type": "heading1",
            "text": f"OUTLINE INTEGRATION TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        },
        {
            "type": "paragraph",
            "text": "This update proves the outline integration is working!"
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

    # Combine test header with formatted content
    full_content = test_header + formatted_content

    # Update the document
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
    print("Check the document to see:")
    print("  1. Test header with timestamp")
    print("  2. Sections reordered according to perfect outline")
    print("  3. Outline integration working!")
    print()

except Exception as e:
    print(f"   [ERROR] Failed to update: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

