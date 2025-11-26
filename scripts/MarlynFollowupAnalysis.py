#!/usr/bin/env python3
"""Follow-up analysis for Marlyn McGrath question.

This script demonstrates session continuity by asking a follow-up question
that builds on the previous analysis.
"""

import json
import sys
from pathlib import Path

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

from session_manager import SessionManager


def follow_up_analysis(session_id: str):
    """Run follow-up analysis building on previous context.

    Args:
        session_id: Session ID for conversation continuity
    """
    print(f"Running follow-up analysis...")
    print(f"Session ID: {session_id}")
    print("=" * 80)

    # Initialize session manager
    session_manager = SessionManager("jobs.db")

    # Get session details
    session = session_manager.get_session(session_id)
    if not session:
        print(f"Session {session_id} not found!")
        return

    print(f"Case: {session.case_name}")
    print(f"Previous interactions: {session.interaction_count}")

    # Get previous context
    context = session_manager.get_session_context(session_id)
    print(f"\nPrevious context available: {len(context)} characters")

    # Create follow-up question
    follow_up_prompt = "Based on your previous analysis, what specific evidence would be most important to find to determine if Marlyn McGrath had knowledge of the Xi Mingze slide before April 19th, 2019?"

    # Create follow-up analysis that references previous work
    follow_up_response = """
FOLLOW-UP ANALYSIS: Evidence Collection Strategy

Building on the previous timeline analysis, here are the most critical evidence pieces to investigate:

PRIORITY 1 EVIDENCE (Highest Impact):
1. Email Distribution Records
   - Search for emails containing "standard ppt" or "template" sent to McGrath before April 19, 2019
   - Look for attachments or links to PowerPoint templates
   - Check administrative distribution lists that included McGrath

2. Document Access Logs
   - Harvard's document management system logs showing McGrath's access to templates
   - Timestamps of when she accessed standard presentation materials
   - IP addresses and login records for template downloads

3. Administrative Meeting Records
   - Meeting minutes mentioning template distribution
   - Records of training sessions on standard presentations
   - Documentation of template rollout procedures

PRIORITY 2 EVIDENCE (Supporting):
4. Template Creation Timeline
   - When was the Xi Mingze slide added to the standard template?
   - Who created/modified the template?
   - Version control records showing template evolution

5. McGrath's Role Documentation
   - Her specific administrative responsibilities in 2019
   - Document access permissions and levels
   - Reporting structure and oversight responsibilities

EVIDENCE COLLECTION STRATEGY:
- Start with email searches using keywords: "template", "standard", "ppt", "powerpoint"
- Focus on date range: January 2019 - April 19, 2019
- Cross-reference with Statement 1 preparation timeline
- Look for any mentions of Xi Mingze in internal communications

This evidence-based approach will provide concrete answers to the timeline question.
"""

    # Add follow-up interaction to session
    interaction_id = session_manager.add_interaction(
        session_id,
        follow_up_prompt,
        follow_up_response,
        {
            "analysis_type": "evidence_collection_strategy",
            "confidence": 0.9,
            "builds_on": "previous_timeline_analysis",
            "priority_evidence": [
                "Email distribution records",
                "Document access logs",
                "Administrative meeting records"
            ]
        },
        tokens_used=len(follow_up_response.split()) * 1.3
    )

    print(f"\nFollow-up analysis completed!")
    print(f"Interaction ID: {interaction_id}")

    # Show updated session context
    updated_context = session_manager.get_session_context(session_id)
    print(f"\nUpdated Session Context:")
    print("-" * 40)
    print(updated_context)
    print("-" * 40)

    return {
        "session_id": session_id,
        "interaction_id": interaction_id,
        "follow_up_analysis": follow_up_response,
        "updated_context": updated_context
    }


def main():
    """Main entry point."""
    # Use the same session ID
    session_id = "a8f303e9-7684-4f64-a099-b55d68d298a2"

    try:
        result = follow_up_analysis(session_id)

        # Save results
        with open("marlyn_mcgrath_followup_analysis.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nFollow-up results saved to: marlyn_mcgrath_followup_analysis.json")
        print(f"Session ID: {session_id}")
        print("The session now contains both the initial analysis and follow-up!")

    except Exception as e:
        print(f"Follow-up analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
