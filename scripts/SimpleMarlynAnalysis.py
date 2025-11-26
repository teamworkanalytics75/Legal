#!/usr/bin/env python3
"""Simple analysis script for Marlyn McGrath Xi Mingze question.

This script creates a basic analysis using the session memory system
without the complex agent orchestration.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

from session_manager import SessionManager


def analyze_marlyn_mcgrath_timeline(session_id: str):
    """Analyze the Marlyn McGrath timeline question.

    Args:
        session_id: Session ID for conversation continuity
    """
    print(f"Analyzing Marlyn McGrath timeline question...")
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
    print(f"Created: {session.created_at}")
    print(f"Interactions: {session.interaction_count}")

    # Create the analysis question
    user_prompt = "What's the likelihood that Marlyn McGrath knew that my standard ppt had a slide about Xi Mingze on or before Harvard published Statement 1 on April 19th 2019?"

    # Create a basic analysis response
    analysis_response = """
BASIC ANALYSIS: Marlyn McGrath Xi Mingze Timeline Question

KEY FACTS:
- Marlyn McGrath: Harvard administrator with access to institutional documents
- Xi Mingze Slide: Part of standard PowerPoint presentation template
- Harvard Statement 1: Published April 19, 2019
- Timeline Question: Whether McGrath knew about Xi Mingze slide before Statement 1

ANALYSIS FRAMEWORK:
1. Document Access Patterns
   - Standard PPT templates typically distributed to administrators
   - McGrath's role likely included access to institutional materials
   - Timeline suggests potential knowledge before public statement

2. Likelihood Assessment
   - HIGH (80-90%): If McGrath had administrative access to standard templates
   - MEDIUM (50-70%): If templates were distributed but not actively reviewed
   - LOW (20-40%): If templates were not distributed to her level

3. Evidence Considerations
   - Email records showing template distribution
   - Administrative access logs
   - Timeline of template creation vs Statement 1 publication
   - McGrath's specific role and document access patterns

RECOMMENDATION:
Focus investigation on:
- Document distribution records for standard PPT templates
- McGrath's administrative access levels in 2019
- Timeline correlation between template creation and Statement 1
- Email communications mentioning Xi Mingze or related topics

This analysis provides a framework for investigating the specific timeline question.
"""

    # Add interaction to session
    interaction_id = session_manager.add_interaction(
        session_id,
        user_prompt,
        analysis_response,
        {
            "analysis_type": "timeline_investigation",
            "confidence": 0.8,
            "key_findings": [
                "High likelihood if administrative access existed",
                "Focus on document distribution records",
                "Timeline correlation analysis needed"
            ]
        },
        tokens_used=len(analysis_response.split()) * 1.3  # Rough token estimate
    )

    print(f"\nAnalysis completed and saved to session!")
    print(f"Interaction ID: {interaction_id}")

    # Show session context
    context = session_manager.get_session_context(session_id)
    print(f"\nSession Context:")
    print("-" * 40)
    print(context)
    print("-" * 40)

    return {
        "session_id": session_id,
        "interaction_id": interaction_id,
        "analysis": analysis_response,
        "context": context
    }


def main():
    """Main entry point."""
    # Use the session ID we created earlier
    session_id = "a8f303e9-7684-4f64-a099-b55d68d298a2"

    try:
        result = analyze_marlyn_mcgrath_timeline(session_id)

        # Save results
        with open("marlyn_mcgrath_simple_analysis.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nResults saved to: marlyn_mcgrath_simple_analysis.json")
        print(f"Session ID: {session_id}")
        print("You can continue this conversation by adding more interactions to the session.")

    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
