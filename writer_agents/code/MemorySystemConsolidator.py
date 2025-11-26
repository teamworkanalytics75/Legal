"""Memory System Consolidator for migrating all memory systems into EpisodicMemoryBank.

Provides utilities to migrate data from separate memory systems into the unified EpisodicMemoryBank.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .EpisodicMemoryBank import EpisodicMemoryEntry, EpisodicMemoryBank
except ImportError:
    from EpisodicMemoryBank import EpisodicMemoryEntry, EpisodicMemoryBank

logger = logging.getLogger(__name__)


class MemorySystemConsolidator:
    """Consolidates all memory systems into EpisodicMemoryBank."""

    def __init__(self, memory_store: EpisodicMemoryBank):
        """Initialize consolidator with target memory store.

        Args:
            memory_store: EpisodicMemoryBank to consolidate memories into
        """
        self.store = memory_store
        logger.info("MemorySystemConsolidator initialized")

    def migrate_database_queries(
        self,
        db_path: str = "memory_store/langchain_meta_memory.sqlite"
    ) -> int:
        """Migrate DatabaseQueryRecorder query history.

        Args:
            db_path: Path to query database

        Returns:
            Number of memories migrated
        """
        db_file = Path(db_path)
        if not db_file.exists():
            logger.warning(f"Query database not found: {db_path}")
            return 0

        count = 0
        try:
            conn = sqlite3.connect(db_file)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT question, sql, result_summary, success, error,
                       created_at, token_cost, model
                FROM queries
                ORDER BY created_at ASC
            """)

            for row in cursor.fetchall():
                memory = EpisodicMemoryEntry(
                    agent_type="DatabaseQueryExecutor",
                    memory_id=str(uuid.uuid4()),
                    summary=f"Query: {row['question'][:100]}... Success: {bool(row['success'])}",
                    context={
                        "question": row["question"],
                        "sql": row["sql"],
                        "result_summary": row["result_summary"],
                        "success": bool(row["success"]),
                        "error": row["error"],
                        "token_cost": row["token_cost"],
                        "model": row["model"]
                    },
                    source="database_query_migration",
                    timestamp=datetime.fromisoformat(row["created_at"]),
                    memory_type="query"
                )
                self.store.add(memory)
                count += 1

            conn.close()
            logger.info(f"Migrated {count} database queries")
            return count

        except Exception as e:
            logger.error(f"Failed to migrate database queries: {e}")
            return count

    def migrate_conversations(
        self,
        db_path: str = "system_memory/data/lawsuit_memory.sqlite"
    ) -> int:
        """Migrate ConversationTranscriptRecorder conversation history.

        Args:
            db_path: Path to conversation database

        Returns:
            Number of memories migrated
        """
        db_file = Path(db_path)
        if not db_file.exists():
            logger.warning(f"Conversation database not found: {db_path}")
            return 0

        count = 0
        try:
            conn = sqlite3.connect(db_file)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timestamp, event_type, model_name, summary, metadata, artifacts
                FROM entries
                ORDER BY timestamp ASC
            """)

            for row in cursor.fetchall():
                metadata = json.loads(row["metadata"] or "{}")
                artifacts = json.loads(row["artifacts"] or "{}")

                memory = EpisodicMemoryEntry(
                    agent_type="ConversationDialogueCaptor",
                    memory_id=str(uuid.uuid4()),
                    summary=row["summary"] or f"{row['event_type']}: {row['model_name']}",
                    context={
                        "event_type": row["event_type"],
                        "model_name": row["model_name"],
                        "metadata": metadata,
                        "artifacts": artifacts
                    },
                    source="conversation_migration",
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    memory_type="conversation"
                )
                self.store.add(memory)
                count += 1

            conn.close()
            logger.info(f"Migrated {count} conversations")
            return count

        except Exception as e:
            logger.error(f"Failed to migrate conversations: {e}")
            return count

    def migrate_document_edits(
        self,
        db_path: str = "system_memory/version_history.db"
    ) -> int:
        """Migrate DocumentEditRecorder edit patterns.

        Args:
            db_path: Path to version history database

        Returns:
            Number of memories migrated
        """
        db_file = Path(db_path)
        if not db_file.exists():
            logger.warning(f"Version history database not found: {db_path}")
            return 0

        count = 0
        try:
            conn = sqlite3.connect(db_file)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT pattern_id, document_id, edit_type, before_text, after_text,
                       context, confidence_score, timestamp, user, section_id, metadata
                FROM edit_patterns
                ORDER BY timestamp ASC
            """)

            for row in cursor.fetchall():
                metadata = json.loads(row["metadata"] or "{}")

                memory = EpisodicMemoryEntry(
                    agent_type="DocumentEditRefiner",
                    memory_id=row["pattern_id"],
                    summary=f"{row['edit_type']}: {row['before_text'][:50]}... → {row['after_text'][:50]}...",
                    context={
                        "before": row["before_text"],
                        "after": row["after_text"],
                        "context": row["context"],
                        "section_id": row["section_id"],
                        "confidence": row["confidence_score"],
                        "user": row["user"],
                        "metadata": metadata
                    },
                    source="document_edit_migration",
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    memory_type="edit",
                    before_text=row["before_text"],
                    after_text=row["after_text"],
                    edit_type=row["edit_type"],
                    document_id=row["document_id"]
                )
                self.store.add(memory)
                count += 1

            conn.close()
            logger.info(f"Migrated {count} document edits")
            return count

        except Exception as e:
            logger.error(f"Failed to migrate document edits: {e}")
            return count

    def migrate_document_metadata(
        self,
        db_path: str = "system_memory/document_tracker.db"
    ) -> int:
        """Migrate DocumentMetadataRecorder document metadata.

        Args:
            db_path: Path to document tracker database

        Returns:
            Number of memories migrated
        """
        db_file = Path(db_path)
        if not db_file.exists():
            logger.warning(f"Document tracker database not found: {db_path}")
            return 0

        count = 0
        try:
            conn = sqlite3.connect(db_file)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT record_id, case_id, google_doc_id, doc_url, folder_id,
                       title, case_summary, status, created_at, metadata
                FROM documents
                ORDER BY created_at ASC
            """)

            for row in cursor.fetchall():
                metadata = json.loads(row["metadata"] or "{}")

                memory = EpisodicMemoryEntry(
                    agent_type="DocumentMetadataRecorder",
                    memory_id=str(uuid.uuid4()),
                    summary=f"Document: {row['title']} for case {row['case_id']}",
                    context={
                        "google_doc_id": row["google_doc_id"],
                        "doc_url": row["doc_url"],
                        "title": row["title"],
                        "case_summary": row["case_summary"],
                        "folder_id": row["folder_id"],
                        "status": row["status"],
                        "metadata": metadata
                    },
                    source="document_metadata_migration",
                    timestamp=datetime.fromisoformat(row["created_at"]),
                    memory_type="document",
                    document_id=row["google_doc_id"],
                    case_id=row["case_id"]
                )
                self.store.add(memory)
                count += 1

            conn.close()
            logger.info(f"Migrated {count} document metadata records")
            return count

        except Exception as e:
            logger.error(f"Failed to migrate document metadata: {e}")
            return count

    def migrate_all(self) -> Dict[str, int]:
        """Run all migrations and return counts.

        Returns:
            Dictionary with migration counts for each system
        """
        logger.info("Starting full memory system consolidation...")

        results = {
            "queries": self.migrate_database_queries(),
            "conversations": self.migrate_conversations(),
            "edits": self.migrate_document_edits(),
            "documents": self.migrate_document_metadata()
        }

        total = sum(results.values())
        logger.info(f"Consolidation complete: {total} total memories migrated")

        # Save the consolidated store
        self.store.save()
        logger.info("Consolidated memory store saved")

        return results


# Export main class
__all__ = ["MemorySystemConsolidator"]

