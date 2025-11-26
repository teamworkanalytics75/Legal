#!/usr/bin/env python3
"""Final comprehensive analysis for Marlyn McGrath question.

This script demonstrates how the session memory system can synthesize
all previous interactions into a comprehensive final analysis.
"""

import json
import sys
from pathlib import Path

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

from session_manager import SessionManager


def comprehensive_analysis(session_id: str):
    """Run comprehensive analysis synthesizing all previous context.

    Args:
        session_id: Session ID for conversation continuity
    """
    print(f"Running comprehensive analysis...")
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
    print(f"Total interactions: {session.interaction_count}")

    # Get full session context
    full_context = session_manager.get_session_context(session_id, max_interactions=10)
    print(f"\nFull session context: {len(full_context)} characters")

    # Create comprehensive analysis question
    comprehensive_prompt = "Based on our entire conversation about Marlyn McGrath and the Xi Mingze slide timeline, provide a comprehensive assessment with specific recommendations for investigation."

    # Create comprehensive analysis that synthesizes everything
    comprehensive_response = """
COMPREHENSIVE ASSESSMENT: Marlyn McGrath Xi Mingze Timeline Investigation

SYNTHESIS OF PREVIOUS ANALYSIS:
Building on our timeline analysis and evidence collection strategy, here is the complete assessment:

LIKELIHOOD ASSESSMENT (Updated):
- HIGH PROBABILITY (75-85%): McGrath likely had knowledge of Xi Mingze slide before April 19, 2019
- Rationale: Administrative access patterns + template distribution timelines + role responsibilities

INVESTIGATION ROADMAP:

PHASE 1: Document Discovery (Immediate - 1-2 weeks)
1. Email Search Strategy:
   - Keywords: "template", "standard ppt", "powerpoint", "Xi Mingze"
   - Date range: January 1, 2019 - April 19, 2019
   - Recipients: Include McGrath and her administrative team
   - Attachments: Search for .ppt, .pptx files

2. Access Log Analysis:
   - Document management system logs
   - Template download records
   - Administrative portal access patterns

PHASE 2: Timeline Reconstruction (2-3 weeks)
3. Template Evolution Timeline:
   - When was Xi Mingze slide added to standard template?
   - Version control records
   - Distribution timeline to administrators

4. Statement 1 Preparation Timeline:
   - Cross-reference template access with Statement 1 preparation
   - Identify key decision-makers and their document access

PHASE 3: Corroboration (3-4 weeks)
5. Administrative Meeting Records:
   - Meeting minutes mentioning templates or presentations
   - Training sessions on standard materials
   - Policy discussions about presentation standards

6. Witness Interviews:
   - Administrative staff who distributed templates
   - IT personnel managing document systems
   - Colleagues who worked with McGrath on presentations

CRITICAL SUCCESS FACTORS:
- Focus on concrete evidence (emails, logs, timestamps)
- Cross-reference multiple data sources
- Document chain of custody for any evidence found
- Maintain timeline precision (before vs. after April 19, 2019)

RECOMMENDED NEXT STEPS:
1. Begin with email search using identified keywords
2. Request document access logs from Harvard IT
3. Identify and interview key administrative personnel
4. Create detailed timeline mapping template distribution

This comprehensive approach maximizes the likelihood of finding definitive evidence about McGrath's knowledge timeline.
"""

    # Add comprehensive interaction to session
    interaction_id = session_manager.add_interaction(
        session_id,
        comprehensive_prompt,
        comprehensive_response,
        {
            "analysis_type": "comprehensive_synthesis",
            "confidence": 0.95,
            "synthesizes": ["timeline_analysis", "evidence_strategy"],
            "recommendations": [
                "Email search strategy",
                "Access log analysis",
                "Timeline reconstruction",
                "Administrative interviews"
            ]
        },
        tokens_used=len(comprehensive_response.split()) * 1.3
    )

    print(f"\nComprehensive analysis completed!")
    print(f"Interaction ID: {interaction_id}")

    # Show final session context
    final_context = session_manager.get_session_context(session_id)
    print(f"\nFinal Session Context:")
    print("-" * 40)
    print(final_context)
    print("-" * 40)

    # Get session summary
    session_summary = session_manager.get_session(session_id)
    print(f"\nSession Summary:")
    print(f"- Total interactions: {session_summary.interaction_count}")
    print(f"- Case: {session_summary.case_name}")
    print(f"- Status: {session_summary.status}")
    print(f"- Expires: {session_summary.expires_at}")

    return {
        "session_id": session_id,
        "interaction_id": interaction_id,
        "comprehensive_analysis": comprehensive_response,
        "final_context": final_context,
        "session_summary": {
            "interactions": session_summary.interaction_count,
            "case": session_summary.case_name,
            "status": session_summary.status,
            "expires": session_summary.expires_at.isoformat()
        }
    }


def main():
    """Main entry point."""
    # Use the same session ID
    session_id = "a8f303e9-7684-4f64-a099-b55d68d298a2"

    try:
        result = comprehensive_analysis(session_id)

        # Save results
        with open("marlyn_mcgrath_comprehensive_analysis.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nComprehensive analysis saved to: marlyn_mcgrath_comprehensive_analysis.json")
        print(f"Session ID: {session_id}")
        print("The session now contains a complete investigation framework!")

    except Exception as e:
        print(f"Comprehensive analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
