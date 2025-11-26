#!/usr/bin/env python3
"""Test session memory with EsuWiki case follow-up scenario.

This script demonstrates the session memory system by:
1. Creating a session for the EsuWiki case
2. Running initial analysis
3. Running follow-up questions that build on previous context
4. Verifying that agents remember previous interactions
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

from master_supervisor import MasterSupervisor, SupervisorConfig
from session_manager import SessionManager
from insights import CaseInsights


class EsuWikiSessionTester:
    """Test session memory with EsuWiki case scenario."""

    def __init__(self, db_path: str = "jobs.db"):
        """Initialize tester.

        Args:
            db_path: Database path
        """
        self.db_path = db_path
        self.session_manager = SessionManager(db_path, default_expiry_days=7)

    async def create_esuwiki_session(self) -> str:
        """Create a session for the EsuWiki case.

        Returns:
            Session ID
        """
        print("Creating EsuWiki case session...")

        session_id = self.session_manager.create_session("EsuWiki Case Analysis")
        print(f"✓ Session created: {session_id}")
        return session_id

    async def run_initial_analysis(self, session_id: str) -> dict:
        """Run initial analysis of the EsuWiki case.

        Args:
            session_id: Session ID

        Returns:
            Analysis results
        """
        print("\nRunning initial EsuWiki analysis...")

        # Create supervisor with session
        config = SupervisorConfig()
        supervisor = MasterSupervisor(
            session=None,
            config=config,
            session_id=session_id
        )

        # Create mock insights for EsuWiki case
        insights = CaseInsights(
            evidence_nodes=[
                {"id": "esuwiki_platform", "type": "platform", "strength": 0.8},
                {"id": "user_content", "type": "content", "strength": 0.7},
                {"id": "dmca_notice", "type": "legal_action", "strength": 0.9},
            ],
            causal_relationships=[
                {"from": "esuwiki_platform", "to": "user_content", "strength": 0.8},
                {"from": "user_content", "to": "dmca_notice", "strength": 0.9},
            ],
            strength_scores={
                "esuwiki_platform": 0.8,
                "user_content": 0.7,
                "dmca_notice": 0.9,
            },
            session_context=""
        )

        # Run initial analysis
        result = await supervisor.execute_with_session(
            user_prompt="Analyze the EsuWiki case focusing on the connection between the platform and DMCA issues",
            insights=insights,
            summary="Initial analysis of EsuWiki case focusing on platform-DMCA connection"
        )

        print(f"✓ Initial analysis completed")
        print(f"  Tokens used: {result.get('execution_stats', {}).get('total_tokens', 0)}")
        print(f"  Final text length: {len(result.get('final_text', ''))}")

        return result

    async def run_follow_up_analysis(self, session_id: str) -> dict:
        """Run follow-up analysis building on previous context.

        Args:
            session_id: Session ID

        Returns:
            Follow-up analysis results
        """
        print("\nRunning follow-up analysis with session context...")

        # Create supervisor with session
        config = SupervisorConfig()
        supervisor = MasterSupervisor(
            session=None,
            config=config,
            session_id=session_id
        )

        # Create mock insights for follow-up
        insights = CaseInsights(
            evidence_nodes=[
                {"id": "esuwiki_platform", "type": "platform", "strength": 0.8},
                {"id": "user_content", "type": "content", "strength": 0.7},
                {"id": "dmca_notice", "type": "legal_action", "strength": 0.9},
                {"id": "fair_use", "type": "defense", "strength": 0.6},
            ],
            causal_relationships=[
                {"from": "esuwiki_platform", "to": "user_content", "strength": 0.8},
                {"from": "user_content", "to": "dmca_notice", "strength": 0.9},
                {"from": "fair_use", "to": "dmca_notice", "strength": -0.7},  # Negative relationship
            ],
            strength_scores={
                "esuwiki_platform": 0.8,
                "user_content": 0.7,
                "dmca_notice": 0.9,
                "fair_use": 0.6,
            },
            session_context=""
        )

        # Run follow-up analysis
        result = await supervisor.execute_with_session(
            user_prompt="Based on your previous analysis, what are the strongest defenses EsuWiki could raise against the DMCA claims?",
            insights=insights,
            summary="Follow-up analysis focusing on EsuWiki defenses"
        )

        print(f"✓ Follow-up analysis completed")
        print(f"  Tokens used: {result.get('execution_stats', {}).get('total_tokens', 0)}")
        print(f"  Final text length: {len(result.get('final_text', ''))}")

        return result

    async def verify_session_context(self, session_id: str) -> None:
        """Verify that session context is properly maintained.

        Args:
            session_id: Session ID
        """
        print("\nVerifying session context...")

        # Get session context
        context = self.session_manager.get_session_context(session_id)

        print(f"Session context length: {len(context)} characters")
        print("Context preview:")
        print("-" * 40)
        print(context[:500] + "..." if len(context) > 500 else context)
        print("-" * 40)

        # Verify context contains expected elements
        assert "EsuWiki Case Analysis" in context, "Context should include case name"
        assert "DMCA" in context, "Context should include DMCA references"
        assert "platform" in context.lower(), "Context should include platform references"

        # Get session details
        session = self.session_manager.get_session(session_id)
        print(f"\nSession details:")
        print(f"  Case: {session.case_name}")
        print(f"  Interactions: {session.interaction_count}")
        print(f"  Status: {session.status}")
        print(f"  Created: {session.created_at}")
        print(f"  Last active: {session.last_active}")

        print("✓ Session context verified successfully")

    async def test_session_continuity(self, session_id: str) -> None:
        """Test that agents can access session context.

        Args:
            session_id: Session ID
        """
        print("\nTesting agent session continuity...")

        # Create supervisor with session
        config = SupervisorConfig()
        supervisor = MasterSupervisor(
            session=None,
            config=config,
            session_id=session_id
        )

        # Get session context that agents would see
        agent_context = supervisor.get_session_context()

        print(f"Agent context length: {len(agent_context)} characters")
        print("Agent context preview:")
        print("-" * 40)
        print(agent_context[:300] + "..." if len(agent_context) > 300 else agent_context)
        print("-" * 40)

        # Verify context is available to agents
        assert len(agent_context) > 0, "Agents should have access to session context"

        print("✓ Agent session continuity verified")

    async def run_complete_test(self) -> None:
        """Run complete EsuWiki session memory test."""
        print("Starting EsuWiki session memory test...")
        print("=" * 60)

        try:
            # Step 1: Create session
            session_id = await self.create_esuwiki_session()

            # Step 2: Run initial analysis
            initial_result = await self.run_initial_analysis(session_id)

            # Step 3: Run follow-up analysis
            followup_result = await self.run_follow_up_analysis(session_id)

            # Step 4: Verify session context
            await self.verify_session_context(session_id)

            # Step 5: Test agent continuity
            await self.test_session_continuity(session_id)

            print("=" * 60)
            print("✓ EsuWiki session memory test completed successfully!")
            print(f"\nSession ID: {session_id}")
            print("You can now use this session for follow-up interactions.")

            # Save results
            results = {
                "session_id": session_id,
                "initial_analysis": {
                    "tokens_used": initial_result.get('execution_stats', {}).get('total_tokens', 0),
                    "text_length": len(initial_result.get('final_text', ''))
                },
                "followup_analysis": {
                    "tokens_used": followup_result.get('execution_stats', {}).get('total_tokens', 0),
                    "text_length": len(followup_result.get('final_text', ''))
                },
                "total_interactions": self.session_manager.get_session(session_id).interaction_count
            }

            with open("esuwiki_session_test_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"\nResults saved to: esuwiki_session_test_results.json")

        except Exception as e:
            print(f"✗ EsuWiki session test failed: {e}")
            raise


async def main():
    """Main entry point."""
    tester = EsuWikiSessionTester()

    try:
        await tester.run_complete_test()
    except Exception as e:
        print(f"EsuWiki session test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
