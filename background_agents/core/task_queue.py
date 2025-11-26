"""Task queue management for background agents."""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A task to be processed by an agent."""
    id: str
    agent_name: str
    task_type: str
    priority: TaskPriority
    status: TaskStatus
    data: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    @classmethod
    def create(
        cls,
        agent_name: str,
        task_type: str,
        data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        max_retries: int = 3
    ) -> "Task":
        """Create a new task."""
        return cls(
            id=str(uuid4()),
            agent_name=agent_name,
            task_type=task_type,
            priority=priority,
            status=TaskStatus.PENDING,
            data=data,
            created_at=datetime.now(),
            max_retries=max_retries
        )


class TaskQueue:
    """SQLite-backed task queue with priority support."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.logger = logging.getLogger("task_queue")
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_priority
                ON tasks(status, priority, created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent
                ON tasks(agent_name, status)
            """)

    def add_task(self, task: Task) -> None:
        """
        Add a task to the queue.

        Args:
            task: The task to add
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO tasks (
                    id, agent_name, task_type, priority, status, data,
                    created_at, retry_count, max_retries
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id,
                task.agent_name,
                task.task_type,
                task.priority.value,
                task.status.value,
                json.dumps(task.data),
                task.created_at.isoformat(),
                task.retry_count,
                task.max_retries
            ))

        self.logger.debug(f"Added task {task.id} for agent {task.agent_name}")

    def get_next_task(self, agent_name: str) -> Optional[Task]:
        """
        Get the next pending task for an agent, ordered by priority.

        Args:
            agent_name: The agent requesting a task

        Returns:
            The next task or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM tasks
                WHERE agent_name = ? AND status = 'pending'
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
            """, (agent_name,))

            row = cursor.fetchone()
            if not row:
                return None

            return self._row_to_task(row)

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Update task status.

        Args:
            task_id: Task ID
            status: New status
            result: Task result if completed
            error: Error message if failed
        """
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            if status == TaskStatus.IN_PROGRESS:
                conn.execute("""
                    UPDATE tasks
                    SET status = ?, started_at = ?
                    WHERE id = ?
                """, (status.value, now, task_id))

            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                conn.execute("""
                    UPDATE tasks
                    SET status = ?, completed_at = ?, result = ?, error = ?
                    WHERE id = ?
                """, (status.value, now, json.dumps(result) if result else None, error, task_id))

            else:
                conn.execute("""
                    UPDATE tasks
                    SET status = ?
                    WHERE id = ?
                """, (status.value, task_id))

        self.logger.debug(f"Updated task {task_id} status to {status.value}")

    def increment_retry(self, task_id: str) -> bool:
        """
        Increment retry count and reset to pending if under max retries.

        Args:
            task_id: Task ID

        Returns:
            True if task will be retried, False if max retries exceeded
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT retry_count, max_retries FROM tasks WHERE id = ?
            """, (task_id,))

            row = cursor.fetchone()
            if not row:
                return False

            retry_count, max_retries = row

            if retry_count < max_retries:
                conn.execute("""
                    UPDATE tasks
                    SET retry_count = retry_count + 1, status = 'pending'
                    WHERE id = ?
                """, (task_id,))
                self.logger.info(f"Task {task_id} will be retried ({retry_count + 1}/{max_retries})")
                return True
            else:
                self.logger.warning(f"Task {task_id} exceeded max retries")
                return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dictionary of statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    agent_name,
                    status,
                    COUNT(*) as count
                FROM tasks
                GROUP BY agent_name, status
            """)

            stats_by_agent: Dict[str, Dict[str, int]] = {}
            for row in cursor.fetchall():
                agent, status, count = row
                if agent not in stats_by_agent:
                    stats_by_agent[agent] = {}
                stats_by_agent[agent][status] = count

            return stats_by_agent

    def cleanup_old_tasks(self, days: int = 30) -> int:
        """
        Delete completed/failed tasks older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of tasks deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM tasks
                WHERE status IN ('completed', 'failed')
                AND datetime(completed_at) < datetime('now', '-' || ? || ' days')
            """, (days,))

            deleted = cursor.rowcount
            self.logger.info(f"Cleaned up {deleted} old tasks")
            return deleted

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """Convert database row to Task object."""
        return Task(
            id=row['id'],
            agent_name=row['agent_name'],
            task_type=row['task_type'],
            priority=TaskPriority(row['priority']),
            status=TaskStatus(row['status']),
            data=json.loads(row['data']),
            created_at=datetime.fromisoformat(row['created_at']),
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            result=json.loads(row['result']) if row['result'] else None,
            error=row['error'],
            retry_count=row['retry_count'],
            max_retries=row['max_retries']
        )

