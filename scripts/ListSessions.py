#!/usr/bin/env python3
"""List active The Matrix sessions.

This script shows all active sessions and their details, allowing you
to see which conversations are ongoing and their status.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

# Import with absolute imports
from session_manager import SessionManager


def list_sessions(db_path: str = "jobs.db", show_expired: bool = False) -> None:
    """List all sessions.

    Args:
        db_path: Path to SQLite database
        show_expired: Whether to include expired sessions
    """
    session_manager = SessionManager(db_path)

    # Get active sessions
    active_sessions = session_manager.list_active_sessions()

    if not active_sessions:
        print("No active sessions found.")
        return

    print(f"Active Sessions ({len(active_sessions)}):")
    print("=" * 80)

    for session in active_sessions:
        # Calculate time since last activity
        now = datetime.now()
        last_active = session.last_active
        time_since = now - last_active

        # Format time since
        if time_since.days > 0:
            time_str = f"{time_since.days}d ago"
        elif time_since.seconds > 3600:
            hours = time_since.seconds // 3600
            time_str = f"{hours}h ago"
        elif time_since.seconds > 60:
            minutes = time_since.seconds // 60
            time_str = f"{minutes}m ago"
        else:
            time_str = "just now"

        # Show session details
        print(f"Session ID: {session.id}")
        print(f"Case: {session.case_name}")
        print(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"Last Active: {last_active.strftime('%Y-%m-%d %H:%M')} ({time_str})")
        print(f"Interactions: {session.interaction_count}")
        print(f"Expires: {session.expires_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"Status: {session.status}")
        print("-" * 40)

    # Show session statistics
    total_interactions = sum(s.interaction_count for s in active_sessions)
    print(f"\nSummary:")
    print(f"Total active sessions: {len(active_sessions)}")
    print(f"Total interactions: {total_interactions}")

    # Show expired sessions if requested
    if show_expired:
        print(f"\nExpired Sessions:")
        print("=" * 80)
        # Note: This would require adding a method to get expired sessions
        print("(Expired session listing not yet implemented)")


def archive_session(session_id: str, db_path: str = "jobs.db") -> None:
    """Archive a session.

    Args:
        session_id: Session ID to archive
        db_path: Path to SQLite database
    """
    session_manager = SessionManager(db_path)

    success = session_manager.archive_session(session_id)
    if success:
        print(f"Session {session_id} archived successfully.")
    else:
        print(f"Failed to archive session {session_id}. Session may not exist or already archived.")


def cleanup_expired(db_path: str = "jobs.db", days_old: int = 30) -> None:
    """Clean up old expired sessions.

    Args:
        db_path: Path to SQLite database
        days_old: Delete sessions expired more than this many days ago
    """
    session_manager = SessionManager(db_path)

    deleted_count = session_manager.cleanup_expired_sessions(days_old)
    print(f"Cleaned up {deleted_count} expired sessions older than {days_old} days.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="List The Matrix sessions")
    parser.add_argument("--db", default="jobs.db", help="Database path (default: jobs.db)")
    parser.add_argument("--expired", action="store_true", help="Show expired sessions")
    parser.add_argument("--archive", help="Archive a specific session by ID")
    parser.add_argument("--cleanup", type=int, metavar="DAYS", help="Clean up expired sessions older than DAYS")

    args = parser.parse_args()

    try:
        if args.archive:
            archive_session(args.archive, args.db)
        elif args.cleanup:
            cleanup_expired(args.db, args.cleanup)
        else:
            list_sessions(args.db, args.expired)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
