#!/usr/bin/env python3
"""Create a new session for The Matrix agent conversations.

This script creates a new session and returns the session ID for use
in follow-up interactions with the agent system.
"""

import argparse
import sys
from pathlib import Path

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

# Import with absolute imports
import job_persistence
from session_manager import SessionManager


def create_session(case_name: str, db_path: str = "jobs.db", expiry_days: int = 7) -> str:
    """Create a new session.

    Args:
        case_name: Name/description of the case
        db_path: Path to SQLite database
        expiry_days: Session expiry in days

    Returns:
        Session ID
    """
    session_manager = SessionManager(db_path, expiry_days)
    session_id = session_manager.create_session(case_name)
    return session_id


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create a new The Matrix session")
    parser.add_argument("case_name", help="Name/description of the case")
    parser.add_argument("--db", default="jobs.db", help="Database path (default: jobs.db)")
    parser.add_argument("--expiry", type=int, default=7, help="Session expiry in days (default: 7)")

    args = parser.parse_args()

    try:
        session_id = create_session(args.case_name, args.db, args.expiry)
        print(f"Session created successfully!")
        print(f"Session ID: {session_id}")
        print(f"Case: {args.case_name}")
        print(f"Expires in: {args.expiry} days")
        print(f"\nUse this session ID with continue_session.py for follow-up interactions.")

    except Exception as e:
        print(f"Error creating session: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
