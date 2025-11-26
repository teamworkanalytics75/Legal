"""Job persistence layer for atomic agent orchestration.

Provides SQLite-based job tracking, execution monitoring, and artifact storage
for the distributed cognition architecture.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Job:
    """Represents a single atomic task in the system."""

    id: str
    phase: str # research, drafting, citation, qa
    type: str # agent type (e.g., CitationFinderAgent)
    payload_ref: str # JSON serialized task data
    status: str # queued, running, succeeded, failed, dead
    priority: int = 0
    budget_tokens: Optional[int] = None
    budget_seconds: Optional[float] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    session_id: Optional[str] = None  # NEW: Link to session

    # LangChain tracking fields
    langchain_enabled: bool = False
    langchain_queries_count: int = 0
    langchain_cost_estimate: float = 0.0

    # Premium mode tracking fields
    premium_mode: bool = False
    premium_agents_used: int = 0
    estimated_premium_cost: float = 0.0


@dataclass
class Run:
    """Represents an execution attempt of a job."""

    id: str
    job_id: str
    agent_name: str
    status: str # running, succeeded, failed
    tokens_in: int = 0
    tokens_out: int = 0
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    heartbeat_at: Optional[str] = None


@dataclass
class Artifact:
    """Represents an output artifact from a job."""

    id: str
    job_id: str
    kind: str # draft, outline, citation, verification, etc.
    uri_or_blob: str # file path or inline data
    hash: str # SHA-256 of content
    created_at: Optional[str] = None


@dataclass
class Event:
    """Represents a lifecycle event for a job."""

    id: str
    ts: str
    job_id: str
    type: str # created, started, retry, succeeded, failed, escalated
    payload_json: str


class JobManager:
    """Manages job lifecycle and persistence."""

    def __init__(self, db_path: str | Path = "jobs.db") -> None:
        """Initialize job manager with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create job tracking tables if they don't exist."""
        conn = self._get_connection()

        # Jobs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                type TEXT NOT NULL,
                payload_ref TEXT,
                status TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                budget_tokens INTEGER,
                budget_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                ended_at TIMESTAMP
            )
        """)

        # Runs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                status TEXT NOT NULL,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                duration_seconds REAL,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                heartbeat_at TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            )
        """)

        # Artifacts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                uri_or_blob TEXT,
                hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            )
        """)

        # Events table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                job_id TEXT NOT NULL,
                type TEXT NOT NULL,
                payload_json TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            )
        """)

        # Session integration - add session_id to jobs table
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN session_id TEXT")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        # LangChain integration - add LangChain tracking columns
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN langchain_enabled INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN langchain_queries_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN langchain_cost_estimate REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        # Premium mode integration - add premium tracking columns
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN premium_mode INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN premium_agents_used INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN estimated_premium_cost REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            # Column already exists
            pass

        # Create indexes for common queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON jobs(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_phase
            ON jobs(phase, status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_job
            ON events(job_id, ts)
        """)

        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def create_job(
        self,
        phase: str,
        agent_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        budget_tokens: Optional[int] = None,
        budget_seconds: Optional[float] = None,
        session_id: Optional[str] = None,  # NEW: Link to session
        langchain_enabled: bool = False,  # NEW: LangChain tracking
        langchain_queries_count: int = 0,  # NEW: LangChain tracking
        langchain_cost_estimate: float = 0.0,  # NEW: LangChain tracking
        premium_mode: bool = False,  # NEW: Premium mode tracking
        premium_agents_used: int = 0,  # NEW: Premium mode tracking
        estimated_premium_cost: float = 0.0,  # NEW: Premium mode tracking
    ) -> str:
        """Create a new job and return its ID.

        Args:
            phase: Workflow phase (research, drafting, citation, qa)
            agent_type: Type of atomic agent to execute
            payload: Task input data
            priority: Job priority (higher = sooner)
            budget_tokens: Maximum tokens allowed
            budget_seconds: Maximum time allowed in seconds
            session_id: Optional session ID to link job to session

        Returns:
            Job ID (UUID)
        """
        job_id = str(uuid.uuid4())
        payload_json = json.dumps(payload, default=str)

        conn = self._get_connection()
        conn.execute("""
            INSERT INTO jobs (
                id, phase, type, payload_ref, status, priority,
                budget_tokens, budget_seconds, session_id,
                langchain_enabled, langchain_queries_count, langchain_cost_estimate,
                premium_mode, premium_agents_used, estimated_premium_cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id, phase, agent_type, payload_json, "queued", priority,
            budget_tokens, budget_seconds, session_id,
            langchain_enabled, langchain_queries_count, langchain_cost_estimate,
            premium_mode, premium_agents_used, estimated_premium_cost
        ))
        conn.commit()

        # Log creation event
        self.log_event(job_id, "created", {"agent_type": agent_type, "phase": phase, "session_id": session_id})

        return job_id

    def update_status(
        self,
        job_id: str,
        status: str,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> None:
        """Update job status.

        Args:
            job_id: Job identifier
            status: New status (queued, running, succeeded, failed, dead)
            started_at: Timestamp when job started
            ended_at: Timestamp when job ended
        """
        conn = self._get_connection()

        updates = ["status = ?"]
        values = [status]

        if started_at:
            updates.append("started_at = ?")
            values.append(started_at.isoformat())

        if ended_at:
            updates.append("ended_at = ?")
            values.append(ended_at.isoformat())

        values.append(job_id)

        conn.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
            values
        )
        conn.commit()

    def log_event(
        self,
        job_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an event for a job.

        Args:
            job_id: Job identifier
            event_type: Event type (created, started, retry, succeeded, failed, escalated)
            payload: Optional event data
        """
        event_id = str(uuid.uuid4())
        payload_json = json.dumps(payload or {}, default=str)

        conn = self._get_connection()
        conn.execute("""
            INSERT INTO events (id, job_id, type, payload_json)
            VALUES (?, ?, ?, ?)
        """, (event_id, job_id, event_type, payload_json))
        conn.commit()

    def get_pending_jobs(
        self,
        phase: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Job]:
        """Get pending jobs ordered by priority.

        Args:
            phase: Filter by phase (optional)
            limit: Maximum number of jobs to return

        Returns:
            List of pending Job objects
        """
        conn = self._get_connection()

        query = """
            SELECT * FROM jobs
            WHERE status IN ('queued', 'running')
        """
        params: List[Any] = []

        if phase:
            query += " AND phase = ?"
            params.append(phase)

        query += " ORDER BY priority DESC, created_at ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)

        jobs = []
        for row in cursor.fetchall():
            jobs.append(Job(**dict(row)))

        return jobs

    def get_jobs_by_session(self, session_id: str) -> List[Job]:
        """Get all jobs for a specific session.

        Args:
            session_id: Session ID

        Returns:
            List of Job objects for the session
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM jobs
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))

        jobs = []
        for row in cursor.fetchall():
            jobs.append(Job(**dict(row)))

        return jobs

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for a session.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with session job statistics
        """
        conn = self._get_connection()

        # Get job counts by status
        cursor = conn.execute("""
            SELECT status, COUNT(*) as count
            FROM jobs
            WHERE session_id = ?
            GROUP BY status
        """, (session_id,))

        status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

        # Get total tokens used
        cursor = conn.execute("""
            SELECT COALESCE(SUM(r.tokens_in + r.tokens_out), 0) as total_tokens
            FROM jobs j
            JOIN runs r ON j.id = r.job_id
            WHERE j.session_id = ? AND r.status = 'succeeded'
        """, (session_id,))

        total_tokens = cursor.fetchone()['total_tokens']

        # Get total duration
        cursor = conn.execute("""
            SELECT COALESCE(SUM(r.duration_seconds), 0) as total_duration
            FROM jobs j
            JOIN runs r ON j.id = r.job_id
            WHERE j.session_id = ? AND r.status = 'succeeded'
        """, (session_id,))

        total_duration = cursor.fetchone()['total_duration']

        return {
            'session_id': session_id,
            'job_counts': status_counts,
            'total_jobs': sum(status_counts.values()),
            'total_tokens': total_tokens,
            'total_duration': total_duration
        }

    def mark_completed(
        self,
        job_id: str,
        result: Dict[str, Any],
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration: Optional[float] = None,
        langchain_queries_count: int = 0,
        langchain_cost_estimate: float = 0.0,
        premium_agents_used: int = 0,
        estimated_premium_cost: float = 0.0,
    ) -> None:
        """Mark a job as successfully completed.

        Args:
            job_id: Job identifier
            result: Job output data
            tokens_in: Input tokens consumed
            tokens_out: Output tokens generated
            duration: Execution time in seconds
        """
        now = datetime.now()

        # Update job status
        self.update_status(job_id, "succeeded", ended_at=now)

        # Update job with LangChain and premium metrics
        conn = self._get_connection()
        conn.execute("""
            UPDATE jobs
            SET langchain_queries_count = ?, langchain_cost_estimate = ?,
                premium_agents_used = ?, estimated_premium_cost = ?
            WHERE id = ?
        """, (langchain_queries_count, langchain_cost_estimate,
              premium_agents_used, estimated_premium_cost, job_id))
        conn.commit()

        # Create run record
        run_id = str(uuid.uuid4())
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO runs (
                id, job_id, agent_name, status, tokens_in, tokens_out,
                duration_seconds, retry_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, job_id, result.get("agent_name", "unknown"),
            "succeeded", tokens_in, tokens_out, duration, 0
        ))
        conn.commit()

        # Store result as artifact
        self.store_artifact(job_id, "result", json.dumps(result, default=str))

        # Log event
        self.log_event(job_id, "succeeded", {
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "duration": duration,
        })

    def mark_failed(
        self,
        job_id: str,
        error: str,
        retry_count: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """Mark a job as failed.

        Args:
            job_id: Job identifier
            error: Error message
            retry_count: Number of retry attempts
            tokens_in: Input tokens consumed before failure
            tokens_out: Output tokens generated before failure
        """
        now = datetime.now()

        # Determine if this is a permanent failure
        status = "dead" if retry_count >= 3 else "failed"

        # Update job status
        self.update_status(job_id, status, ended_at=now)

        # Create run record
        run_id = str(uuid.uuid4())
        conn = self._get_connection()
        conn.execute("""
            INSERT INTO runs (
                id, job_id, agent_name, status, tokens_in, tokens_out,
                error_message, retry_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, job_id, "unknown", "failed",
            tokens_in, tokens_out, error, retry_count
        ))
        conn.commit()

        # Log event
        self.log_event(job_id, "failed", {
            "error": error,
            "retry_count": retry_count,
        })

    def store_artifact(
        self,
        job_id: str,
        kind: str,
        content: str | bytes,
    ) -> str:
        """Store an artifact for a job.

        Args:
            job_id: Job identifier
            kind: Artifact type (draft, outline, citation, etc.)
            content: Artifact content

        Returns:
            Artifact ID
        """
        artifact_id = str(uuid.uuid4())

        # Compute content hash
        if isinstance(content, str):
            content_bytes = content.encode()
        else:
            content_bytes = content

        content_hash = hashlib.sha256(content_bytes).hexdigest()

        conn = self._get_connection()
        conn.execute("""
            INSERT INTO artifacts (id, job_id, kind, uri_or_blob, hash)
            VALUES (?, ?, ?, ?, ?)
        """, (artifact_id, job_id, kind, content, content_hash))
        conn.commit()

        return artifact_id

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve an artifact by ID.

        Args:
            artifact_id: Artifact identifier

        Returns:
            Artifact object or None if not found
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM artifacts WHERE id = ?
        """, (artifact_id,))

        row = cursor.fetchone()
        if row:
            return Artifact(**dict(row))
        return None

    def get_job_artifacts(self, job_id: str) -> List[Artifact]:
        """Get all artifacts for a job.

        Args:
            job_id: Job identifier

        Returns:
            List of Artifact objects
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM artifacts WHERE job_id = ? ORDER BY created_at ASC
        """, (job_id,))

        return [Artifact(**dict(row)) for row in cursor.fetchall()]

    def get_job_events(self, job_id: str) -> List[Event]:
        """Get all events for a job.

        Args:
            job_id: Job identifier

        Returns:
            List of Event objects
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM events WHERE job_id = ? ORDER BY ts ASC
        """, (job_id,))

        return [Event(**dict(row)) for row in cursor.fetchall()]

    def get_phase_stats(self, phase: str) -> Dict[str, Any]:
        """Get statistics for a workflow phase.

        Args:
            phase: Phase name (research, drafting, citation, qa)

        Returns:
            Dictionary with counts by status, avg tokens, avg duration
        """
        conn = self._get_connection()

        # Count by status
        cursor = conn.execute("""
            SELECT status, COUNT(*) as count
            FROM jobs
            WHERE phase = ?
            GROUP BY status
        """, (phase,))

        status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

        # Average tokens and duration for completed jobs
        cursor = conn.execute("""
            SELECT
                AVG(r.tokens_in) as avg_tokens_in,
                AVG(r.tokens_out) as avg_tokens_out,
                AVG(r.duration_seconds) as avg_duration
            FROM jobs j
            JOIN runs r ON j.id = r.job_id
            WHERE j.phase = ? AND j.status = 'succeeded'
        """, (phase,))

        row = cursor.fetchone()

        return {
            "phase": phase,
            "status_counts": status_counts,
            "avg_tokens_in": row["avg_tokens_in"] or 0,
            "avg_tokens_out": row["avg_tokens_out"] or 0,
            "avg_duration": row["avg_duration"] or 0,
        }

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
