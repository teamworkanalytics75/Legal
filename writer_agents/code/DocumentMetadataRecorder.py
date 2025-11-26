"""Document Metadata Recorder for Google Docs Integration.

Records document metadata and integrates with EpisodicMemoryBank.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from .EpisodicMemoryBank import EpisodicMemoryEntry, EpisodicMemoryBank
except ImportError:
    from EpisodicMemoryBank import EpisodicMemoryEntry, EpisodicMemoryBank

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Represents a tracked Google Docs document."""

    record_id: str
    case_id: str
    google_doc_id: str
    doc_url: str
    folder_id: str
    title: str
    case_summary: str
    status: str = "active"  # active, archived, deleted
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    archived_reason: Optional[str] = None


@dataclass
class DocumentStatistics:
    """Statistics about tracked documents."""

    total_documents: int
    active_documents: int
    archived_documents: int
    deleted_documents: int
    total_cases: int
    documents_by_status: Dict[str, int]
    recent_documents: List[DocumentRecord]


class DocumentMetadataRecorder:
    """Records document metadata and integrates with EpisodicMemoryBank."""

    def __init__(
        self,
        db_path: str = "system_memory/document_tracker.db",
        memory_store: Optional[EpisodicMemoryBank] = None
    ):
        """Initialize the document metadata recorder.

        Args:
            db_path: Path to SQLite database file
            memory_store: Optional EpisodicMemoryBank for unified memory
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_store = memory_store

        # Initialize database
        self._init_database()

        logger.info(f"DocumentMetadataRecorder initialized with database: {self.db_path}")

    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    record_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    google_doc_id TEXT UNIQUE NOT NULL,
                    doc_url TEXT NOT NULL,
                    folder_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    case_summary TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    archived_reason TEXT,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_case_id ON documents(case_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_google_doc_id ON documents(google_doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)")

            conn.commit()

    def create_document_record(self,
                             case_id: str,
                             google_doc_id: str,
                             doc_url: str,
                             folder_id: str,
                             title: str,
                             case_summary: str,
                             metadata: Optional[Dict[str, Any]] = None) -> DocumentRecord:
        """Create a new document record.

        Args:
            case_id: Unique case identifier
            google_doc_id: Google Docs document ID
            doc_url: Google Docs document URL
            folder_id: Google Drive folder ID
            title: Document title
            case_summary: Summary of the case
            metadata: Additional metadata

        Returns:
            DocumentRecord object
        """
        record_id = f"{case_id}_{google_doc_id}"
        now = datetime.now().isoformat()

        record = DocumentRecord(
            record_id=record_id,
            case_id=case_id,
            google_doc_id=google_doc_id,
            doc_url=doc_url,
            folder_id=folder_id,
            title=title,
            case_summary=case_summary,
            metadata=metadata or {}
        )

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO documents
                    (record_id, case_id, google_doc_id, doc_url, folder_id,
                     title, case_summary, status, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.record_id,
                    record.case_id,
                    record.google_doc_id,
                    record.doc_url,
                    record.folder_id,
                    record.title,
                    record.case_summary,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    json.dumps(record.metadata)
                ))

                conn.commit()
                logger.info(f"Created document record: {record_id}")

                # NEW: Also store in EpisodicMemoryBank
                if self.memory_store:
                    memory = EpisodicMemoryEntry(
                        agent_type="DocumentMetadataRecorder",
                        memory_id=str(uuid.uuid4()),
                        summary=f"Document created: {title} for case {case_id}",
                        context={
                            "google_doc_id": google_doc_id,
                            "doc_url": doc_url,
                            "title": title,
                            "case_summary": case_summary,
                            "folder_id": folder_id,
                            "status": "active"
                        },
                        source="document_metadata",
                        timestamp=datetime.fromisoformat(record.created_at),
                        memory_type="document",
                        document_id=google_doc_id,
                        case_id=case_id
                    )
                    self.memory_store.add(memory)

        except Exception as e:
            logger.error(f"Failed to create document record {record_id}: {e}")
            raise

        return record

    def get_doc_for_case(self, case_id: str) -> Optional[DocumentRecord]:
        """Get the document record for a specific case.

        Args:
            case_id: Case identifier

        Returns:
            DocumentRecord if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM documents
                    WHERE case_id = ? AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (case_id,))

                row = cursor.fetchone()
                if row:
                    return self._row_to_document_record(row)

                return None

        except Exception as e:
            logger.error(f"Failed to get document for case {case_id}: {e}")
            return None

    def get_doc_by_google_id(self, google_doc_id: str) -> Optional[DocumentRecord]:
        """Get document record by Google Doc ID.

        Args:
            google_doc_id: Google Docs document ID

        Returns:
            DocumentRecord if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM documents
                    WHERE google_doc_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (google_doc_id,))

                row = cursor.fetchone()
                if row:
                    return self._row_to_document_record(row)

                return None

        except Exception as e:
            logger.error(f"Failed to get document by Google ID {google_doc_id}: {e}")
            return None

    def list_all_docs(self, status: Optional[str] = None, limit: int = 100) -> List[DocumentRecord]:
        """List all document records.

        Args:
            status: Filter by status (active, archived, deleted)
            limit: Maximum number of records to return

        Returns:
            List of DocumentRecord objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM documents"
                params = []

                if status:
                    query += " WHERE status = ?"
                    params.append(status)

                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [self._row_to_document_record(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def update_document(self,
                      google_doc_id: str,
                      title: Optional[str] = None,
                      case_summary: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update document information.

        Args:
            google_doc_id: Google Docs document ID
            title: New title
            case_summary: New case summary
            metadata: New metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Build update query dynamically
                updates = []
                params = []

                if title is not None:
                    updates.append("title = ?")
                    params.append(title)

                if case_summary is not None:
                    updates.append("case_summary = ?")
                    params.append(case_summary)

                if metadata is not None:
                    updates.append("metadata = ?")
                    params.append(json.dumps(metadata))

                if updates:
                    updates.append("updated_at = ?")
                    params.append(datetime.now().isoformat())

                    query = f"UPDATE documents SET {', '.join(updates)} WHERE google_doc_id = ?"
                    params.append(google_doc_id)

                    cursor.execute(query, params)
                    conn.commit()

                    logger.info(f"Updated document: {google_doc_id}")
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to update document {google_doc_id}: {e}")
            return False

    def archive_document(self, google_doc_id: str, reason: Optional[str] = None) -> bool:
        """Archive a document.

        Args:
            google_doc_id: Google Docs document ID
            reason: Reason for archiving

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE documents
                    SET status = 'archived', archived_reason = ?, updated_at = ?
                    WHERE google_doc_id = ?
                """, (reason, datetime.now().isoformat(), google_doc_id))

                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Archived document: {google_doc_id}")
                    return True
                else:
                    logger.warning(f"Document not found for archiving: {google_doc_id}")
                    return False

        except Exception as e:
            logger.error(f"Failed to archive document {google_doc_id}: {e}")
            return False

    def delete_document(self, google_doc_id: str, reason: Optional[str] = None) -> bool:
        """Mark a document as deleted.

        Args:
            google_doc_id: Google Docs document ID
            reason: Reason for deletion

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE documents
                    SET status = 'deleted', archived_reason = ?, updated_at = ?
                    WHERE google_doc_id = ?
                """, (reason, datetime.now().isoformat(), google_doc_id))

                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Deleted document: {google_doc_id}")
                    return True
                else:
                    logger.warning(f"Document not found for deletion: {google_doc_id}")
                    return False

        except Exception as e:
            logger.error(f"Failed to delete document {google_doc_id}: {e}")
            return False

    def get_doc_by_title(self, title: str, folder_id: Optional[str] = None) -> Optional[DocumentRecord]:
        """Get document by exact title match.

        Args:
            title: Exact document title
            folder_id: Optional folder ID to filter

        Returns:
            DocumentRecord if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if folder_id:
                    cursor.execute("""
                        SELECT * FROM documents
                        WHERE title = ? AND folder_id = ? AND status = 'active'
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (title, folder_id))
                else:
                    cursor.execute("""
                        SELECT * FROM documents
                        WHERE title = ? AND status = 'active'
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (title,))

                row = cursor.fetchone()
                if row:
                    return self._row_to_document_record(row)

                return None

        except Exception as e:
            logger.error(f"Failed to get document by title {title}: {e}")
            return None

    def search_documents(self, query: str, limit: int = 50) -> List[DocumentRecord]:
        """Search documents by title, case summary, or case ID.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching DocumentRecord objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                search_query = f"%{query}%"
                cursor.execute("""
                    SELECT * FROM documents
                    WHERE (title LIKE ? OR case_summary LIKE ? OR case_id LIKE ?)
                    AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (search_query, search_query, search_query, limit))

                rows = cursor.fetchall()
                return [self._row_to_document_record(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []

    def generate_statistics(self) -> DocumentStatistics:
        """Generate statistics about tracked documents.

        Returns:
            DocumentStatistics object
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get total counts
                cursor.execute("SELECT COUNT(*) FROM documents")
                total_documents = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'active'")
                active_documents = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'archived'")
                archived_documents = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'deleted'")
                deleted_documents = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(DISTINCT case_id) FROM documents")
                total_cases = cursor.fetchone()[0]

                # Get documents by status
                cursor.execute("""
                    SELECT status, COUNT(*)
                    FROM documents
                    GROUP BY status
                """)
                documents_by_status = dict(cursor.fetchall())

                # Get recent documents
                cursor.execute("""
                    SELECT * FROM documents
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                recent_rows = cursor.fetchall()
                recent_documents = [self._row_to_document_record(row) for row in recent_rows]

                return DocumentStatistics(
                    total_documents=total_documents,
                    active_documents=active_documents,
                    archived_documents=archived_documents,
                    deleted_documents=deleted_documents,
                    total_cases=total_cases,
                    documents_by_status=documents_by_status,
                    recent_documents=recent_documents
                )

        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
            return DocumentStatistics(0, 0, 0, 0, 0, {}, [])

    def get_document_history(self, google_doc_id: str) -> List[DocumentRecord]:
        """Get the history of a document (all versions).

        Args:
            google_doc_id: Google Docs document ID

        Returns:
            List of DocumentRecord objects in chronological order
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM documents
                    WHERE google_doc_id = ?
                    ORDER BY created_at ASC
                """, (google_doc_id,))

                rows = cursor.fetchall()
                return [self._row_to_document_record(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get document history for {google_doc_id}: {e}")
            return []

    def _row_to_document_record(self, row) -> DocumentRecord:
        """Convert database row to DocumentRecord object.

        Args:
            row: Database row tuple

        Returns:
            DocumentRecord object
        """
        return DocumentRecord(
            record_id=row[0],
            case_id=row[1],
            google_doc_id=row[2],
            doc_url=row[3],
            folder_id=row[4],
            title=row[5],
            case_summary=row[6],
            status=row[7],
            created_at=row[8],
            updated_at=row[9],
            metadata=json.loads(row[10]) if row[10] else {},
            archived_reason=row[11]
        )


def create_document_metadata_recorder(
    db_path: str = "system_memory/document_tracker.db",
    memory_store: Optional[EpisodicMemoryBank] = None
) -> DocumentMetadataRecorder:
    """Create a DocumentMetadataRecorder instance.

    Args:
        db_path: Path to SQLite database file
        memory_store: Optional EpisodicMemoryBank for unified memory

    Returns:
        DocumentMetadataRecorder instance
    """
    return DocumentMetadataRecorder(db_path, memory_store)


# Backward compatibility aliases
DocumentTracker = DocumentMetadataRecorder


def create_document_tracker(
    db_path: str = "system_memory/document_tracker.db"
) -> DocumentMetadataRecorder:
    """Create a DocumentMetadataRecorder instance (backward compatibility).

    Args:
        db_path: Path to SQLite database file

    Returns:
        DocumentMetadataRecorder instance
    """
    return DocumentMetadataRecorder(db_path)


# Export main classes and functions
__all__ = [
    "DocumentMetadataRecorder",
    "DocumentRecord",
    "DocumentStatistics",
    "create_document_metadata_recorder",
    "DocumentTracker",  # Backward compatibility
    "create_document_tracker"  # Backward compatibility
]
