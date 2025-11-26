#!/usr/bin/env python3
"""Run analysis on Marlyn McGrath Xi Mingze question using session memory.

This script analyzes the specific question about whether Marlyn McGrath knew
about the Xi Mingze slide before Harvard published Statement 1 on April 19th, 2019.
"""

import asyncio
import sys
from pathlib import Path

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

from master_supervisor import MasterSupervisor, SupervisorConfig
from session_manager import SessionManager
from insights import CaseInsights


async def analyze_marlyn_mcgrath_question(session_id: str):
    """Analyze the Marlyn McGrath Xi Mingze question.

    Args:
        session_id: Session ID for conversation continuity
    """
    print(f"Starting analysis for session: {session_id}")
    print("Question: What's the likelihood that Marlyn McGrath knew that my standard ppt had a slide about Xi Mingze on or before Harvard published Statement 1 on April 19th 2019?")
    print("=" * 80)

    # Create supervisor with session
    config = SupervisorConfig()
    supervisor = MasterSupervisor(
        session=None,
        config=config,
        session_id=session_id
    )

    # Create insights focused on the Marlyn McGrath timeline question
    insights = CaseInsights(
        evidence_nodes=[
            {"id": "marlyn_mcgrath", "type": "person", "strength": 0.9, "role": "Harvard administrator"},
            {"id": "xi_mingze_slide", "type": "document", "strength": 0.8, "content": "PowerPoint slide about Xi Mingze"},
            {"id": "standard_ppt", "type": "document", "strength": 0.7, "content": "Standard presentation template"},
            {"id": "harvard_statement_1", "type": "document", "strength": 0.9, "date": "April 19, 2019"},
            {"id": "timeline_analysis", "type": "analysis", "strength": 0.8, "focus": "Knowledge timeline"}
        ],
        causal_relationships=[
            {"from": "marlyn_mcgrath", "to": "xi_mingze_slide", "strength": 0.6, "type": "knowledge"},
            {"from": "standard_ppt", "to": "xi_mingze_slide", "strength": 0.8, "type": "contains"},
            {"from": "marlyn_mcgrath", "to": "harvard_statement_1", "strength": 0.7, "type": "involvement"},
            {"from": "xi_mingze_slide", "to": "harvard_statement_1", "strength": 0.5, "type": "influence"}
        ],
        strength_scores={
            "marlyn_mcgrath": 0.9,
            "xi_mingze_slide": 0.8,
            "standard_ppt": 0.7,
            "harvard_statement_1": 0.9,
            "timeline_analysis": 0.8
        },
        session_context=""
    )

    # Run analysis with session context
    result = await supervisor.execute_with_session(
        user_prompt="What's the likelihood that Marlyn McGrath knew that my standard ppt had a slide about Xi Mingze on or before Harvard published Statement 1 on April 19th 2019?",
        insights=insights,
        summary="Analysis of Marlyn McGrath's knowledge timeline regarding Xi Mingze slide"
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    # Display results
    print(f"Total tokens used: {result.get('execution_stats', {}).get('total_tokens', 0)}")
    print(f"Analysis phases completed: {len(result.get('phase_results', {}))}")

    if 'final_text' in result:
        print(f"\nFINAL ANALYSIS:")
        print("-" * 40)
        print(result['final_text'])
        print("-" * 40)

    # Show session context that was used
    session_context = supervisor.get_session_context()
    if session_context:
        print(f"\nSESSION CONTEXT USED:")
        print("-" * 40)
        print(session_context[:500] + "..." if len(session_context) > 500 else session_context)
        print("-" * 40)

    return result


async def main():
    """Main entry point."""
    # Use the session ID we created earlier
    session_id = "a8f303e9-7684-4f64-a099-b55d68d298a2"

    try:
        result = await analyze_marlyn_mcgrath_question(session_id)

        # Save results
        import json
        with open("marlyn_mcgrath_analysis_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\nResults saved to: marlyn_mcgrath_analysis_results.json")
        print(f"Session ID: {session_id}")
        print("You can continue this conversation using continue_session.py")

    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
