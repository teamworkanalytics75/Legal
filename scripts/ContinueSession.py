#!/usr/bin/env python3
"""Continue an existing The Matrix session.

This script allows you to continue a conversation with the The Matrix agent system
using an existing session ID, maintaining context from previous interactions.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

# Import with absolute imports
from master_supervisor import MasterSupervisor, SupervisorConfig
from session_manager import SessionManager


async def continue_session(
    session_id: str,
    user_prompt: str,
    db_path: str = "jobs.db",
    model: str = "gpt-4o-mini"
) -> dict:
    """Continue a session with a new user prompt.

    Args:
        session_id: Existing session ID
        user_prompt: User's follow-up prompt
        db_path: Path to SQLite database
        model: Model to use for agents

    Returns:
        Analysis results
    """
    # Verify session exists
    session_manager = SessionManager(db_path)
    session = session_manager.get_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found or expired")

    # Create supervisor with session
    config = SupervisorConfig()
    supervisor = MasterSupervisor(
        session=None,  # We don't need the SQLite session for this
        config=config,
        session_id=session_id
    )

    # Create mock insights (you would normally load these from your BN analysis)
    from insights import CaseInsights
    insights = CaseInsights(
        evidence_nodes=[],
        causal_relationships=[],
        strength_scores={},
        session_context=""  # Will be populated by execute_with_session
    )

    # Execute with session context
    result = await supervisor.execute_with_session(
        user_prompt=user_prompt,
        insights=insights,
        summary=f"Follow-up analysis for {session.case_name}"
    )

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Continue a The Matrix session")
    parser.add_argument("session_id", help="Session ID to continue")
    parser.add_argument("prompt", help="User prompt for follow-up analysis")
    parser.add_argument("--db", default="jobs.db", help="Database path (default: jobs.db)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use (default: gpt-4o-mini)")
    parser.add_argument("--output", help="Output file for results (JSON)")

    args = parser.parse_args()

    try:
        # Run the async function
        result = asyncio.run(continue_session(
            args.session_id,
            args.prompt,
            args.db,
            args.model
        ))

        print("Analysis completed successfully!")
        print(f"Total tokens used: {result.get('execution_stats', {}).get('total_tokens', 0)}")

        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Results saved to: {args.output}")
        else:
            print("\nFinal text:")
            print(result.get('final_text', 'No text generated'))

    except Exception as e:
        print(f"Error continuing session: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
