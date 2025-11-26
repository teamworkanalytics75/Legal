"""Document Edit Recorder for Google Docs Integration.

Records document edit patterns and integrates with EpisodicMemoryBank.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    from .EpisodicMemoryBank import EpisodicMemoryEntry, EpisodicMemoryBank
except ImportError:
    from EpisodicMemoryBank import EpisodicMemoryEntry, EpisodicMemoryBank

logger = logging.getLogger(__name__)


@dataclass
class EditPattern:
    """Represents a pattern of edits made to a document."""

    pattern_id: str
    document_id: str
    edit_type: str  # citation_fix, argument_improvement, tone_adjustment, formatting
    before_text: str
    after_text: str
    context: str
    confidence_score: float
    timestamp: str
    user: str
    section_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VersionStatistics:
    """Statistics about document versions and edits."""

    total_edits: int
    document_count: int
    edit_types: Dict[str, int]
    most_common_patterns: List[Tuple[str, int]]
    average_confidence: float
    time_range: Tuple[str, str]


class DocumentEditRecorder:
    """Records document edit patterns and integrates with EpisodicMemoryBank."""

    def __init__(
        self,
        db_path: str = "system_memory/version_history.db",
        memory_store: Optional[EpisodicMemoryBank] = None
    ):
        """Initialize the document edit recorder.

        Args:
            db_path: Path to SQLite database file
            memory_store: Optional EpisodicMemoryBank for unified memory
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_store = memory_store

        # Initialize database
        self._init_database()

        logger.info(f"DocumentEditRecorder initialized with database: {self.db_path}")

    def _init_database(self) -> None:
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create edit_patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edit_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    edit_type TEXT NOT NULL,
                    before_text TEXT NOT NULL,
                    after_text TEXT NOT NULL,
                    context TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    user TEXT NOT NULL,
                    section_id TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create document_versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_versions (
                    version_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user TEXT NOT NULL,
                    change_summary TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edit_patterns_document_id ON edit_patterns(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edit_patterns_edit_type ON edit_patterns(edit_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edit_patterns_timestamp ON edit_patterns(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_versions_document_id ON document_versions(document_id)")

            conn.commit()

    def store_edit_pattern(self, pattern: EditPattern) -> bool:
        """Store an edit pattern in the database and EpisodicMemoryBank.

        Args:
            pattern: EditPattern to store

        Returns:
            True if successful, False otherwise
        """
        try:
            # Store in SQLite
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO edit_patterns
                    (pattern_id, document_id, edit_type, before_text, after_text,
                     context, confidence_score, timestamp, user, section_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.document_id,
                    pattern.edit_type,
                    pattern.before_text,
                    pattern.after_text,
                    pattern.context,
                    pattern.confidence_score,
                    pattern.timestamp,
                    pattern.user,
                    pattern.section_id,
                    json.dumps(pattern.metadata)
                ))

                conn.commit()
                logger.debug(f"Stored edit pattern: {pattern.pattern_id}")

            # NEW: Also store in EpisodicMemoryBank
            if self.memory_store:
                memory = EpisodicMemoryEntry(
                    agent_type="DocumentEditRefiner",
                    memory_id=pattern.pattern_id,
                    summary=f"{pattern.edit_type}: {pattern.before_text[:50]}... → {pattern.after_text[:50]}...",
                    context={
                        "before": pattern.before_text,
                        "after": pattern.after_text,
                        "context": pattern.context,
                        "section_id": pattern.section_id,
                        "confidence": pattern.confidence_score
                    },
                    source="document_edit",
                    timestamp=datetime.fromisoformat(pattern.timestamp),
                    memory_type="edit",
                    before_text=pattern.before_text,
                    after_text=pattern.after_text,
                    edit_type=pattern.edit_type,
                    document_id=pattern.document_id
                )
                self.memory_store.add(memory)

            return True

        except Exception as e:
            logger.error(f"Failed to store edit pattern {pattern.pattern_id}: {e}")
            return False

    def get_edit_patterns(self,
                         document_id: Optional[str] = None,
                         edit_type: Optional[str] = None,
                         limit: int = 100) -> List[EditPattern]:
        """Retrieve edit patterns from the database.

        Args:
            document_id: Filter by document ID
            edit_type: Filter by edit type
            limit: Maximum number of patterns to return

        Returns:
            List of EditPattern objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM edit_patterns WHERE 1=1"
                params = []

                if document_id:
                    query += " AND document_id = ?"
                    params.append(document_id)

                if edit_type:
                    query += " AND edit_type = ?"
                    params.append(edit_type)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                patterns = []
                for row in rows:
                    pattern = EditPattern(
                        pattern_id=row[0],
                        document_id=row[1],
                        edit_type=row[2],
                        before_text=row[3],
                        after_text=row[4],
                        context=row[5],
                        confidence_score=row[6],
                        timestamp=row[7],
                        user=row[8],
                        section_id=row[9],
                        metadata=json.loads(row[10]) if row[10] else {}
                    )
                    patterns.append(pattern)

                logger.debug(f"Retrieved {len(patterns)} edit patterns")
                return patterns

        except Exception as e:
            logger.error(f"Failed to retrieve edit patterns: {e}")
            return []

    def capture_version_history(self, doc_id: str, google_docs_bridge) -> Dict[str, Any]:
        """Capture version history from Google Docs.

        Args:
            doc_id: Google Doc ID
            google_docs_bridge: GoogleDocsBridge instance

        Returns:
            Dictionary with capture results
        """
        try:
            # Get document revisions from Google Drive
            revisions = google_docs_bridge.get_document_revisions(doc_id)

            if not revisions:
                return {"status": "no_revisions", "count": 0}

            # Store each revision
            stored_count = 0
            for i, revision in enumerate(revisions):
                version_id = f"{doc_id}_v{i+1}"

                # Store version record
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT OR REPLACE INTO document_versions
                        (version_id, document_id, version_number, content_hash,
                         timestamp, user, change_summary, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        version_id,
                        doc_id,
                        i + 1,
                        f"hash_{revision.get('id', 'unknown')}",  # Simplified hash
                        revision.get('modifiedTime', ''),
                        revision.get('lastModifyingUser', {}).get('displayName', 'unknown'),
                        f"Revision {i + 1}",
                        json.dumps({"revision_id": revision.get('id')})
                    ))

                    conn.commit()
                    stored_count += 1

            logger.info(f"Captured {stored_count} versions for document {doc_id}")
            return {"status": "success", "count": stored_count}

        except Exception as e:
            logger.error(f"Failed to capture version history for {doc_id}: {e}")
            return {"status": "error", "error": str(e)}

    def generate_statistics(self) -> VersionStatistics:
        """Generate statistics about stored patterns and versions.

        Returns:
            VersionStatistics object
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get edit pattern statistics
                cursor.execute("SELECT COUNT(*) FROM edit_patterns")
                total_edits = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(DISTINCT document_id) FROM edit_patterns")
                document_count = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT edit_type, COUNT(*)
                    FROM edit_patterns
                    GROUP BY edit_type
                    ORDER BY COUNT(*) DESC
                """)
                edit_types = dict(cursor.fetchall())

                cursor.execute("SELECT AVG(confidence_score) FROM edit_patterns")
                avg_confidence = cursor.fetchone()[0] or 0.0

                cursor.execute("""
                    SELECT before_text, COUNT(*)
                    FROM edit_patterns
                    GROUP BY before_text
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                """)
                most_common_patterns = cursor.fetchall()

                cursor.execute("""
                    SELECT MIN(timestamp), MAX(timestamp)
                    FROM edit_patterns
                """)
                time_range = cursor.fetchone()

                return VersionStatistics(
                    total_edits=total_edits,
                    document_count=document_count,
                    edit_types=edit_types,
                    most_common_patterns=most_common_patterns,
                    average_confidence=avg_confidence,
                    time_range=time_range or ("", "")
                )

        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
            return VersionStatistics(0, 0, {}, [], 0.0, ("", ""))

    def export_patterns_for_ml(self, output_path: str) -> Dict[str, Any]:
        """Export edit patterns for machine learning training.

        Args:
            output_path: Path to output JSON file

        Returns:
            Dictionary with export results
        """
        try:
            patterns = self.get_edit_patterns(limit=10000)  # Get all patterns

            # Convert to ML-friendly format
            ml_data = []
            for pattern in patterns:
                ml_data.append({
                    "input_text": pattern.before_text,
                    "output_text": pattern.after_text,
                    "edit_type": pattern.edit_type,
                    "context": pattern.context,
                    "confidence_score": pattern.confidence_score,
                    "section_id": pattern.section_id,
                    "metadata": pattern.metadata
                })

            # Write to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ml_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(ml_data)} patterns to {output_path}")
            return {
                "status": "success",
                "patterns_exported": len(ml_data),
                "output_path": str(output_file)
            }

        except Exception as e:
            logger.error(f"Failed to export patterns for ML: {e}")
            return {"status": "error", "error": str(e)}

    def _classify_edit_type(self, before_text: str, after_text: str) -> str:
        """Classify the type of edit based on before/after text.

        Args:
            before_text: Text before the edit
            after_text: Text after the edit

        Returns:
            Edit type classification
        """
        # Citation fix detection
        if "[Node:" in before_text and "[Node:" in after_text:
            return "citation_fix"

        # Argument improvement detection
        if len(after_text) > len(before_text) * 1.5:
            return "argument_improvement"

        # Tone adjustment detection
        informal_words = ["gonna", "wanna", "gotta", "ain't", "don't", "can't"]
        if any(word in before_text.lower() for word in informal_words):
            return "tone_adjustment"

        # Formatting detection
        if before_text.strip() != after_text.strip() and len(before_text) == len(after_text):
            return "formatting"

        # Default classification
        return "content_update"


# Backward compatibility aliases
VersionHistoryTracker = DocumentEditRecorder


def create_document_edit_recorder(
    db_path: str = "system_memory/version_history.db",
    memory_store: Optional[EpisodicMemoryBank] = None
) -> DocumentEditRecorder:
    """Create a DocumentEditRecorder instance.

    Args:
        db_path: Path to SQLite database file
        memory_store: Optional EpisodicMemoryBank for unified memory

    Returns:
        DocumentEditRecorder instance
    """
    return DocumentEditRecorder(db_path, memory_store)


# Also create backward compatibility function
def create_version_tracker(db_path: str = "system_memory/version_history.db") -> DocumentEditRecorder:
    """Create a DocumentEditRecorder instance (backward compatibility).

    Args:
        db_path: Path to SQLite database file

    Returns:
        DocumentEditRecorder instance
    """
    return DocumentEditRecorder(db_path)


# Export main classes and functions
__all__ = [
    "DocumentEditRecorder",
    "EditPattern",
    "VersionStatistics",
    "create_document_edit_recorder",
    "VersionHistoryTracker",  # Backward compatibility
    "create_version_tracker"   # Backward compatibility
]

