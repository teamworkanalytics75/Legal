#!/usr/bin/env python3
"""Test session memory functionality.

This script verifies that session memory works correctly by:
1. Creating a session
2. Running first analysis
3. Running follow-up using session context
4. Verifying context is preserved
5. Testing expiry mechanism
"""

import asyncio
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

from master_supervisor import MasterSupervisor, SupervisorConfig
from session_manager import SessionManager
from insights import CaseInsights


class SessionMemoryTester:
    """Test session memory functionality."""

    def __init__(self, db_path: str = None):
        """Initialize tester.

        Args:
            db_path: Database path (uses temp file if None)
        """
        if db_path is None:
            self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            self.db_path = self.temp_db.name
            self.temp_db.close()
        else:
            self.db_path = db_path
            self.temp_db = None

        self.session_manager = SessionManager(self.db_path, default_expiry_days=1)  # 1 day for testing

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_db:
            Path(self.db_path).unlink(missing_ok=True)

    async def test_session_creation(self) -> str:
        """Test creating a new session.

        Returns:
            Session ID
        """
        print("Testing session creation...")

        session_id = self.session_manager.create_session("Test Case - Session Memory")
        session = self.session_manager.get_session(session_id)

        assert session is not None, "Session should be created"
        assert session.case_name == "Test Case - Session Memory", "Case name should match"
        assert session.status == "active", "Session should be active"

        print(f"✓ Session created: {session_id}")
        return session_id

    async def test_session_interaction(self, session_id: str) -> None:
        """Test adding interactions to a session.

        Args:
            session_id: Session ID
        """
        print("Testing session interaction...")

        # Add first interaction
        interaction_id = self.session_manager.add_interaction(
            session_id,
            "What are the key legal issues in this case?",
            "The main legal issues involve contract breach and intellectual property disputes.",
            {"analysis": "preliminary", "confidence": 0.8},
            tokens_used=150
        )

        assert interaction_id > 0, "Interaction should be added"

        # Add second interaction
        interaction_id2 = self.session_manager.add_interaction(
            session_id,
            "Can you elaborate on the contract breach?",
            "The contract breach involves failure to deliver services as specified in section 3.2.",
            {"analysis": "detailed", "confidence": 0.9},
            tokens_used=200
        )

        assert interaction_id2 > interaction_id, "Second interaction should have higher ID"

        # Verify session updated
        session = self.session_manager.get_session(session_id)
        assert session.interaction_count == 2, "Session should have 2 interactions"

        print("✓ Session interactions added successfully")

    async def test_session_context(self, session_id: str) -> None:
        """Test retrieving session context.

        Args:
            session_id: Session ID
        """
        print("Testing session context retrieval...")

        context = self.session_manager.get_session_context(session_id)

        assert context, "Context should not be empty"
        assert "Test Case - Session Memory" in context, "Context should include case name"
        assert "What are the key legal issues" in context, "Context should include user prompts"
        assert "contract breach" in context, "Context should include agent responses"

        print("✓ Session context retrieved successfully")
        print(f"Context preview: {context[:200]}...")

    async def test_session_expiry(self, session_id: str) -> None:
        """Test session expiry mechanism.

        Args:
            session_id: Session ID
        """
        print("Testing session expiry...")

        # Manually expire the session by updating expires_at
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Set expiry to past date
        past_date = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute(
            "UPDATE sessions SET expires_at = ? WHERE id = ?",
            (past_date, session_id)
        )
        conn.commit()
        conn.close()

        # Run expiry check
        expired_count = self.session_manager.expire_old_sessions()
        assert expired_count >= 1, "At least one session should be expired"

        # Verify session is expired
        session = self.session_manager.get_session(session_id)
        assert session.status == "expired", "Session should be marked as expired"

        print("✓ Session expiry working correctly")

    async def test_master_supervisor_integration(self) -> None:
        """Test MasterSupervisor integration with sessions."""
        print("Testing MasterSupervisor integration...")

        # Create a new session
        session_id = self.session_manager.create_session("Integration Test Case")

        # Create supervisor with session
        config = SupervisorConfig()
        supervisor = MasterSupervisor(
            session=None,
            config=config,
            session_id=session_id
        )

        # Verify session methods exist
        assert hasattr(supervisor, 'create_session'), "Supervisor should have create_session method"
        assert hasattr(supervisor, 'get_session_context'), "Supervisor should have get_session_context method"
        assert hasattr(supervisor, 'execute_with_session'), "Supervisor should have execute_with_session method"

        # Test session context retrieval
        context = supervisor.get_session_context()
        assert context == "", "Empty session should have empty context"

        print("✓ MasterSupervisor integration working correctly")

    async def run_all_tests(self) -> None:
        """Run all session memory tests."""
        print("Starting session memory tests...")
        print("=" * 50)

        try:
            # Test 1: Session creation
            session_id = await self.test_session_creation()

            # Test 2: Session interactions
            await self.test_session_interaction(session_id)

            # Test 3: Session context
            await self.test_session_context(session_id)

            # Test 4: Session expiry
            await self.test_session_expiry(session_id)

            # Test 5: MasterSupervisor integration
            await self.test_master_supervisor_integration()

            print("=" * 50)
            print("✓ All session memory tests passed!")

        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise
        finally:
            self.cleanup()


async def main():
    """Main entry point."""
    tester = SessionMemoryTester()

    try:
        await tester.run_all_tests()
    except Exception as e:
        print(f"Session memory tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
