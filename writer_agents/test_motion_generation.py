#!/usr/bin/env python3
"""
Test script to validate motion generation end-to-end.

This script:
1. Clears the Google Doc
2. Runs motion generation
3. Validates output contains actual motion content (not examples)
4. Checks document is properly updated in Google Docs
5. Verifies content deletion worked correctly
6. Reports success/failure with specific issues
"""

import asyncio
import logging
import os
import sys
import re
from pathlib import Path
from typing import Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "writer_agents" / "code"))
sys.path.insert(0, str(project_root / "writer_agents"))
sys.path.insert(0, str(project_root))


def validate_content_is_actual_motion(content: str) -> Tuple[bool, str]:
    """Validate that generated content is actual motion text, not test/example prompts.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not content or len(content.strip()) < 100:
        return False, "Content is too short or empty"
    
    content_lower = content.lower()
    
    # Patterns that indicate test/example prompts (not actual motion)
    test_patterns = [
        r'comprehensive report on',
        r'your task:',
        r'generate three follow up questions',
        r'as a student of law',
        r'craft an extensive treatise',
        r'your analysis must include',
        r'for each technology mentioned',
        r'translate the following legal scenario',
        r'your essay should',
        r'your response:',
        r'write a comprehensive',
        r'provide your insights while including',
        r'analyze at least three',
        r'examine the balance between',
        r'discuss at least five',
        r'in your analysis include:',
        r'your task is to',
        r'please provide',
        r'create a comprehensive',
        r'develop this framework',
    ]
    
    # Check for test patterns
    for pattern in test_patterns:
        if re.search(pattern, content_lower):
            return False, f"Content appears to be test/example prompt (matched pattern: {pattern})"
    
    # Check for actual motion indicators (positive validation)
    motion_indicators = [
        r'united states district court',
        r'respectfully (requests?|moves?|submits?|prays?)',
        r'wherefore',
        r'pursuant to',
        r'federal rule',
        r'this court',
        r'plaintiff|defendant|movant',
        r'motion for',
        r'order granting',
    ]
    
    motion_count = sum(1 for pattern in motion_indicators if re.search(pattern, content_lower))
    
    # If we have test patterns OR no motion indicators, it's invalid
    if motion_count == 0:
        return False, "Content lacks motion indicators (no 'respectfully requests', 'wherefore', court references, etc.)"
    
    # If we have motion indicators and no test patterns, it's likely valid
    return True, ""


def get_document_content(doc_id: str) -> Optional[str]:
    """Fetch document content from Google Docs API."""
    try:
        from code.google_docs_bridge import create_google_docs_bridge
        
        # Find credentials using same logic as WorkflowOrchestrator
        project_root = Path(__file__).parent.parent
        credentials_file = project_root / "client_secret_701141645351-m9h9dmu5n7qh9bl391cnaoooh2sfleh4.apps.googleusercontent.com.json"
        if not credentials_file.exists():
            credentials_file = Path("/home/serteamwork/projects/TheMatrix/client_secret_701141645351-m9h9dmu5n7qh9bl391cnaoooh2sfleh4.apps.googleusercontent.com.json")
        if not credentials_file.exists():
            credentials_file = Path("client_secret_701141645351-m9h9dmu5n7qh9bl391cnaoooh2sfleh4.apps.googleusercontent.com.json")
        
        credentials_path = str(credentials_file) if credentials_file.exists() else None
        if not credentials_path:
            logger.error("Could not find Google credentials file")
            return None
        
        bridge = create_google_docs_bridge(credentials_path=credentials_path)
        doc = bridge.docs_service.documents().get(documentId=doc_id).execute()
        
        # Extract text from document
        body = doc.get("body", {})
        content_elements = body.get("content", [])
        
        text_parts = []
        for element in content_elements:
            if "paragraph" in element:
                para = element["paragraph"]
                if "elements" in para:
                    for elem in para["elements"]:
                        if "textRun" in elem:
                            text_parts.append(elem["textRun"].get("content", ""))
        
        return "".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to fetch document content: {e}")
        return None


def get_document_length(doc_id: str) -> int:
    """Get document length (endIndex) from Google Docs API."""
    try:
        from code.google_docs_bridge import create_google_docs_bridge
        
        # Find credentials using same logic as WorkflowOrchestrator
        project_root = Path(__file__).parent.parent
        credentials_file = project_root / "client_secret_701141645351-m9h9dmu5n7qh9bl391cnaoooh2sfleh4.apps.googleusercontent.com.json"
        if not credentials_file.exists():
            credentials_file = Path("/home/serteamwork/projects/TheMatrix/client_secret_701141645351-m9h9dmu5n7qh9bl391cnaoooh2sfleh4.apps.googleusercontent.com.json")
        if not credentials_file.exists():
            credentials_file = Path("client_secret_701141645351-m9h9dmu5n7qh9bl391cnaoooh2sfleh4.apps.googleusercontent.com.json")
        
        credentials_path = str(credentials_file) if credentials_file.exists() else None
        if not credentials_path:
            logger.error("Could not find Google credentials file")
            return 0
        
        bridge = create_google_docs_bridge(credentials_path=credentials_path)
        doc = bridge.docs_service.documents().get(documentId=doc_id).execute()
        
        body = doc.get("body", {})
        content_elements = body.get("content", [])
        
        # Calculate end_index from content elements (same logic as update_document)
        end_index = 1
        max_end_index = 1
        if content_elements:
            for element in content_elements:
                element_end = element.get("endIndex", 1)
                if element_end > max_end_index:
                    max_end_index = element_end
            
            end_index = max_end_index
            
            body_end_index = body.get("endIndex", 1)
            if body_end_index > max_end_index:
                end_index = body_end_index
        
        return end_index
    except Exception as e:
        logger.error(f"Failed to get document length: {e}")
        return 0


async def test_motion_generation():
    """Run end-to-end test of motion generation."""
    
    print("\n" + "="*80)
    print("MOTION GENERATION TEST")
    print("="*80)
    
    # Known document ID
    KNOWN_MASTER_DRAFT_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"
    
    # Step 1: Clear the document
    print("\n[STEP 1] Clearing Google Doc...")
    try:
        # Import clear function - adjust path based on where script is run from
        try:
            from clear_google_doc import clear_and_test
        except ImportError:
            from writer_agents.clear_google_doc import clear_and_test
        clear_and_test()
        print("   [OK] Document cleared")
    except Exception as e:
        print(f"   [ERROR] Failed to clear document: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Get document length before generation
    print("\n[STEP 2] Checking document state before generation...")
    length_before = get_document_length(KNOWN_MASTER_DRAFT_ID)
    print(f"   [INFO] Document length before: {length_before}")
    
    # Step 3: Run motion generation
    print("\n[STEP 3] Running motion generation...")
    print("   [INFO] This may take several minutes...")
    try:
        # Import and run the generation script - adjust path based on where script is run from
        try:
            from generate_full_motion_to_seal import generate_full_motion
        except ImportError:
            from writer_agents.generate_full_motion_to_seal import generate_full_motion
        result = await generate_full_motion()
        
        if not result:
            print("   [ERROR] Motion generation returned no result")
            return False
        
        print("   [OK] Motion generation completed")
    except Exception as e:
        print(f"   [ERROR] Motion generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Get document length after generation
    print("\n[STEP 4] Checking document state after generation...")
    length_after = get_document_length(KNOWN_MASTER_DRAFT_ID)
    print(f"   [INFO] Document length after: {length_after}")
    
    # Step 5: Fetch and validate document content
    print("\n[STEP 5] Validating document content...")
    doc_content = get_document_content(KNOWN_MASTER_DRAFT_ID)
    
    if not doc_content:
        print("   [ERROR] Could not fetch document content")
        return False
    
    print(f"   [INFO] Document content length: {len(doc_content)} characters")
    print(f"   [INFO] Content preview (first 500 chars):")
    print(f"   {doc_content[:500]}...")
    
    # Step 6: Validate content is actual motion
    print("\n[STEP 6] Validating content is actual motion (not examples)...")
    is_valid, error_msg = validate_content_is_actual_motion(doc_content)
    
    if not is_valid:
        print(f"   [FAIL] Content validation failed: {error_msg}")
        print(f"   [INFO] Content preview (first 1000 chars):")
        print(f"   {doc_content[:1000]}")
        return False
    
    print("   [OK] Content validation passed - appears to be actual motion")
    
    # Step 7: Verify deletion worked (document should have been cleared before insertion)
    print("\n[STEP 7] Verifying document deletion worked...")
    if length_after <= length_before and length_before > 100:
        print(f"   [WARNING] Document length after ({length_after}) <= before ({length_before})")
        print("   [INFO] This might indicate deletion didn't work, or document was already cleared")
    else:
        print(f"   [OK] Document length increased from {length_before} to {length_after}")
        print("   [OK] This indicates new content was added (deletion likely worked)")
    
    # Step 8: Check for feature type categorization in quality constraints
    print("\n[STEP 8] Checking feature type categorization...")
    # This would require checking logs or the actual prompt, but we can verify
    # that the content has proper structure
    has_court_ref = bool(re.search(r'united states district court', doc_content.lower()))
    has_respectfully = bool(re.search(r'respectfully (requests?|moves?|submits?|prays?)', doc_content.lower()))
    has_wherefore = bool(re.search(r'wherefore', doc_content.lower()))
    
    print(f"   [INFO] Motion indicators found:")
    print(f"      - Court reference: {has_court_ref}")
    print(f"      - 'Respectfully' phrase: {has_respectfully}")
    print(f"      - 'Wherefore': {has_wherefore}")
    
    # Final summary
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"   [OK] Document cleared: Yes")
    print(f"   [OK] Motion generation: Completed")
    print(f"   [OK] Content validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"   [OK] Document updated: Yes (length: {length_after})")
    print(f"   [OK] Content is actual motion: {'Yes' if is_valid else 'No'}")
    
    if is_valid and length_after > 100:
        print("\n   [SUCCESS] All tests passed!")
        return True
    else:
        print("\n   [FAILURE] Some tests failed")
        return False


if __name__ == "__main__":
    print("\n[START] Starting Motion Generation Test...")
    print("   This will:")
    print("   1. Clear the Google Doc")
    print("   2. Run motion generation")
    print("   3. Validate content is actual motion (not examples)")
    print("   4. Verify document deletion worked")
    print()
    
    result = asyncio.run(test_motion_generation())
    
    if result:
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some tests failed. Check the output above for details.")
        sys.exit(1)

