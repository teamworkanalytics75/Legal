"""Session management for The Matrix agents.

Provides persistent session memory enabling conversation continuity
across multiple interactions with the agent system.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Note: JobManager import removed as it's not actually used in this class


@dataclass
class Session:
    """Represents a conversation session."""

    id: str
    case_name: str
    created_at: datetime
    last_active: datetime
    expires_at: datetime
    status: str  # active, archived, expired
    interaction_count: int = 0


@dataclass
class SessionInteraction:
    """Represents a single interaction within a session."""

    id: int
    session_id: str
    interaction_num: int
    user_prompt: str
    agent_response: str
    execution_results: Dict[str, Any]
    tokens_used: int
    timestamp: datetime


class SessionManager:
    """Manages persistent session memory for agent conversations."""

    def __init__(self, db_path: str, default_expiry_days: int = 7):
        """Initialize session manager.

        Args:
            db_path: Path to SQLite database
            default_expiry_days: Default session expiry in days
        """
        self.db_path = db_path
        self.default_expiry_days = default_expiry_days
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create session tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                case_name TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                last_active DATETIME NOT NULL,
                expires_at DATETIME NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                interaction_count INTEGER DEFAULT 0
            )
        """)

        # Create session_interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                interaction_num INTEGER NOT NULL,
                user_prompt TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                execution_results TEXT NOT NULL,
                tokens_used INTEGER DEFAULT 0,
                timestamp DATETIME NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_status
            ON sessions(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_expires
            ON sessions(expires_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_session
            ON session_interactions(session_id)
        """)

        conn.commit()
        conn.close()

    def create_session(self, case_name: str) -> str:
        """Create a new session.

        Args:
            case_name: Name/description of the case

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(days=self.default_expiry_days)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sessions (id, case_name, created_at, last_active, expires_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, case_name, now, now, expires_at, 'active'))

        conn.commit()
        conn.close()

        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session object or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, case_name, created_at, last_active, expires_at, status, interaction_count
            FROM sessions WHERE id = ?
        """, (session_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return Session(
            id=row[0],
            case_name=row[1],
            created_at=datetime.fromisoformat(row[2]),
            last_active=datetime.fromisoformat(row[3]),
            expires_at=datetime.fromisoformat(row[4]),
            status=row[5],
            interaction_count=row[6]
        )

    def add_interaction(
        self,
        session_id: str,
        user_prompt: str,
        agent_response: str,
        execution_results: Dict[str, Any],
        tokens_used: int = 0
    ) -> int:
        """Add an interaction to a session.

        Args:
            session_id: Session ID
            user_prompt: User's input
            agent_response: Agent's response
            execution_results: Full execution results
            tokens_used: Number of tokens consumed

        Returns:
            Interaction ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get next interaction number
        cursor.execute("""
            SELECT COALESCE(MAX(interaction_num), 0) + 1
            FROM session_interactions WHERE session_id = ?
        """, (session_id,))
        interaction_num = cursor.fetchone()[0]

        # Insert interaction
        cursor.execute("""
            INSERT INTO session_interactions
            (session_id, interaction_num, user_prompt, agent_response,
             execution_results, tokens_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            interaction_num,
            user_prompt,
            agent_response,
            json.dumps(execution_results),
            tokens_used,
            datetime.now()
        ))

        # Update session last_active and interaction_count
        cursor.execute("""
            UPDATE sessions
            SET last_active = ?, interaction_count = interaction_count + 1
            WHERE id = ?
        """, (datetime.now(), session_id))

        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return interaction_id

    def get_session_context(
        self,
        session_id: str,
        max_interactions: int = 10,
        max_tokens: int = 5000
    ) -> str:
        """Get formatted session context for agents.

        Args:
            session_id: Session ID
            max_interactions: Maximum interactions to include
            max_tokens: Maximum tokens for context

        Returns:
            Formatted context string
        """
        session = self.get_session(session_id)
        if not session:
            return ""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent interactions
        cursor.execute("""
            SELECT interaction_num, user_prompt, agent_response, timestamp
            FROM session_interactions
            WHERE session_id = ?
            ORDER BY interaction_num DESC
            LIMIT ?
        """, (session_id, max_interactions))

        interactions = cursor.fetchall()
        conn.close()

        if not interactions:
            return ""

        # Format context
        context_parts = [
            f"**SESSION CONTEXT - {session.case_name}**",
            f"Session ID: {session_id}",
            f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Interactions: {session.interaction_count}",
            ""
        ]

        # Add interactions (most recent first)
        for interaction_num, user_prompt, agent_response, timestamp in reversed(interactions):
            timestamp_str = datetime.fromisoformat(timestamp).strftime('%H:%M')

            context_parts.extend([
                f"**Interaction {interaction_num} ({timestamp_str}):**",
                f"User: {user_prompt[:200]}{'...' if len(user_prompt) > 200 else ''}",
                f"Agent: {agent_response[:300]}{'...' if len(agent_response) > 300 else ''}",
                ""
            ])

        context = "\n".join(context_parts)

        # Truncate if too long
        if len(context) > max_tokens * 4:  # Rough token estimate
            context = context[:max_tokens * 4] + "\n\n[Context truncated...]"

        return context

    def expire_old_sessions(self) -> int:
        """Mark expired sessions as expired.

        Returns:
            Number of sessions expired
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE sessions
            SET status = 'expired'
            WHERE status = 'active' AND expires_at < ?
        """, (datetime.now(),))

        expired_count = cursor.rowcount
        conn.commit()
        conn.close()

        return expired_count

    def archive_session(self, session_id: str) -> bool:
        """Archive a session.

        Args:
            session_id: Session ID

        Returns:
            True if archived successfully
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE sessions
            SET status = 'archived'
            WHERE id = ? AND status = 'active'
        """, (session_id,))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return success

    def list_active_sessions(self) -> List[Session]:
        """List all active sessions.

        Returns:
            List of active sessions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, case_name, created_at, last_active, expires_at, status, interaction_count
            FROM sessions
            WHERE status = 'active'
            ORDER BY last_active DESC
        """)

        sessions = []
        for row in cursor.fetchall():
            sessions.append(Session(
                id=row[0],
                case_name=row[1],
                created_at=datetime.fromisoformat(row[2]),
                last_active=datetime.fromisoformat(row[3]),
                expires_at=datetime.fromisoformat(row[4]),
                status=row[5],
                interaction_count=row[6]
            ))

        conn.close()
        return sessions

    def get_session_interactions(self, session_id: str) -> List[SessionInteraction]:
        """Get all interactions for a session.

        Args:
            session_id: Session ID

        Returns:
            List of interactions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, session_id, interaction_num, user_prompt, agent_response,
                   execution_results, tokens_used, timestamp
            FROM session_interactions
            WHERE session_id = ?
            ORDER BY interaction_num
        """, (session_id,))

        interactions = []
        for row in cursor.fetchall():
            interactions.append(SessionInteraction(
                id=row[0],
                session_id=row[1],
                interaction_num=row[2],
                user_prompt=row[3],
                agent_response=row[4],
                execution_results=json.loads(row[5]),
                tokens_used=row[6],
                timestamp=datetime.fromisoformat(row[7])
            ))

        conn.close()
        return interactions

    def cleanup_expired_sessions(self, days_old: int = 30) -> int:
        """Delete old expired sessions to free space.

        Args:
            days_old: Delete sessions expired more than this many days ago

        Returns:
            Number of sessions deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete interactions first (foreign key constraint)
        cursor.execute("""
            DELETE FROM session_interactions
            WHERE session_id IN (
                SELECT id FROM sessions
                WHERE status = 'expired' AND expires_at < ?
            )
        """, (cutoff_date,))

        # Delete sessions
        cursor.execute("""
            DELETE FROM sessions
            WHERE status = 'expired' AND expires_at < ?
        """, (cutoff_date,))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count
