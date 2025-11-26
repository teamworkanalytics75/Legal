#!/usr/bin/env python3
"""Run full The Matrix atomic agent analysis on Marlyn McGrath question.

This script uses the complete 49-agent system with session memory
to provide deep legal analysis of the McGrath timeline question.
"""

import asyncio
import sys
from pathlib import Path

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

# Import with proper module structure
import os
os.chdir(str(Path(__file__).parent.parent / "writer_agents" / "code"))

try:
    from master_supervisor import MasterSupervisor, SupervisorConfig
    from session_manager import SessionManager
    from insights import CaseInsights
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Let me try a different approach...")

    # Try importing individual components
    try:
        import session_manager
        print("✅ SessionManager imported")
    except Exception as e2:
        print(f"❌ SessionManager failed: {e2}")

    try:
        import master_supervisor
        print("✅ MasterSupervisor imported")
    except Exception as e3:
        print(f"❌ MasterSupervisor failed: {e3}")

    sys.exit(1)


async def run_full_mcgrath_analysis(session_id: str):
    """Run full atomic agent analysis on McGrath question.

    Args:
        session_id: Session ID for conversation continuity
    """
    print(f"🚀 Starting FULL The Matrix atomic agent analysis...")
    print(f"Session ID: {session_id}")
    print("=" * 80)

    # Create supervisor with session
    config = SupervisorConfig()
    supervisor = MasterSupervisor(
        session=None,
        config=config,
        session_id=session_id
    )

    # Create comprehensive insights for McGrath analysis
    insights = CaseInsights(
        evidence_nodes=[
            {"id": "marlyn_mcgrath", "type": "person", "strength": 0.9, "role": "Harvard administrator"},
            {"id": "xi_mingze_slide", "type": "document", "strength": 0.8, "content": "PowerPoint slide about Xi Mingze"},
            {"id": "standard_ppt", "type": "document", "strength": 0.7, "content": "Standard presentation template"},
            {"id": "harvard_statement_1", "type": "document", "strength": 0.9, "date": "April 19, 2019"},
            {"id": "timeline_analysis", "type": "analysis", "strength": 0.8, "focus": "Knowledge timeline"},
            {"id": "administrative_access", "type": "evidence", "strength": 0.7, "category": "access_patterns"},
            {"id": "template_distribution", "type": "evidence", "strength": 0.6, "category": "distribution_timeline"}
        ],
        causal_relationships=[
            {"from": "marlyn_mcgrath", "to": "xi_mingze_slide", "strength": 0.6, "type": "knowledge"},
            {"from": "standard_ppt", "to": "xi_mingze_slide", "strength": 0.8, "type": "contains"},
            {"from": "marlyn_mcgrath", "to": "harvard_statement_1", "strength": 0.7, "type": "involvement"},
            {"from": "xi_mingze_slide", "to": "harvard_statement_1", "strength": 0.5, "type": "influence"},
            {"from": "administrative_access", "to": "marlyn_mcgrath", "strength": 0.8, "type": "enables"},
            {"from": "template_distribution", "to": "marlyn_mcgrath", "strength": 0.7, "type": "provides"}
        ],
        strength_scores={
            "marlyn_mcgrath": 0.9,
            "xi_mingze_slide": 0.8,
            "standard_ppt": 0.7,
            "harvard_statement_1": 0.9,
            "timeline_analysis": 0.8,
            "administrative_access": 0.7,
            "template_distribution": 0.6
        },
        session_context=""
    )

    # Run full atomic agent analysis with session context
    result = await supervisor.execute_with_session(
        user_prompt="What's the likelihood that Marlyn McGrath knew that my standard ppt had a slide about Xi Mingze on or before Harvard published Statement 1 on April 19th 2019? Provide comprehensive legal analysis with specific evidence recommendations.",
        insights=insights,
        summary="Full atomic agent analysis of Marlyn McGrath Xi Mingze timeline question"
    )

    print("\n" + "=" * 80)
    print("🎯 FULL ATOMIC AGENT ANALYSIS COMPLETE")
    print("=" * 80)

    # Display comprehensive results
    print(f"Total tokens used: {result.get('execution_stats', {}).get('total_tokens', 0)}")
    print(f"Research tokens: {result.get('execution_stats', {}).get('research_tokens', 0)}")
    print(f"Drafting tokens: {result.get('execution_stats', {}).get('drafting_tokens', 0)}")
    print(f"Citation tokens: {result.get('execution_stats', {}).get('citation_tokens', 0)}")
    print(f"QA tokens: {result.get('execution_stats', {}).get('qa_tokens', 0)}")

    if 'final_text' in result:
        print(f"\n📋 COMPREHENSIVE LEGAL ANALYSIS:")
        print("-" * 60)
        print(result['final_text'])
        print("-" * 60)

    # Show session context that was used
    session_context = supervisor.get_session_context()
    if session_context:
        print(f"\n🧠 SESSION CONTEXT USED:")
        print("-" * 40)
        print(session_context[:800] + "..." if len(session_context) > 800 else session_context)
        print("-" * 40)

    return result


async def main():
    """Main entry point."""
    # Use the existing session ID
    session_id = "a8f303e9-7684-4f64-a099-b55d68d298a2"

    try:
        result = await run_full_mcgrath_analysis(session_id)

        # Save results
        import json
        with open("marlyn_mcgrath_full_analysis.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n💾 Results saved to: marlyn_mcgrath_full_analysis.json")
        print(f"🆔 Session ID: {session_id}")
        print("🎉 Full atomic agent analysis completed successfully!")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
