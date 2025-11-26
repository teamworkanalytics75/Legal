"""Meta-memory support for LangChain SQL agents.

Provides lightweight persistence for schema snapshots, sample rows,
and prior query attempts so agents can avoid repeating expensive
schema introspection and can learn from previous successes/failures.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from difflib import SequenceMatcher


DEFAULT_MEMORY_PATH = Path("memory_store") / "langchain_meta_memory.sqlite"


@dataclass
class LoggedQuery:
    """Container for previously executed queries."""

    question: str
    sql: Optional[str]
    result_summary: Optional[str]
    success: bool
    error: Optional[str]
    created_at: datetime
    token_cost: Optional[float]
    model: Optional[str]


class LangChainMetaMemory:
    """SQLite-backed memory for schemas and executed SQL queries."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or DEFAULT_MEMORY_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schemas (
                    table_name TEXT PRIMARY KEY,
                    schema_text TEXT NOT NULL,
                    sample_rows TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    sql TEXT,
                    result_summary TEXT,
                    success INTEGER NOT NULL,
                    error TEXT,
                    token_cost REAL,
                    model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    # ------------------------------------------------------------------ #
    # Schema caching
    # ------------------------------------------------------------------ #
    def upsert_schema(
        self,
        table_name: str,
        schema_text: str,
        sample_rows: Optional[str] = None
    ) -> None:
        """Store schema snapshot and optional sample rows."""
        if not table_name or not schema_text:
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO schemas (table_name, schema_text, sample_rows, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(table_name) DO UPDATE SET
                    schema_text = excluded.schema_text,
                    sample_rows = excluded.sample_rows,
                    last_updated = CURRENT_TIMESTAMP;
                """,
                (table_name, schema_text, sample_rows),
            )
            conn.commit()

    def get_schema_snippets(self, table_names: Sequence[str]) -> List[str]:
        """Return stored schema snippets for the provided tables."""
        if not table_names:
            return []

        placeholders = ",".join("?" for _ in table_names)
        query = f"SELECT table_name, schema_text, sample_rows FROM schemas WHERE table_name IN ({placeholders})"

        with self._connect() as conn:
            rows = conn.execute(query, tuple(table_names)).fetchall()

        snippets: List[str] = []
        for row in rows:
            snippet = [f"Table: {row['table_name']}"]
            snippet.append(row["schema_text"])
            if row["sample_rows"]:
                snippet.append("Sample rows:")
                snippet.append(row["sample_rows"])
            snippets.append("\n".join(snippet))
        return snippets

    def list_cached_tables(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT table_name FROM schemas ORDER BY table_name").fetchall()
        return [row["table_name"] for row in rows]

    # ------------------------------------------------------------------ #
    # Query logging + retrieval
    # ------------------------------------------------------------------ #
    def log_query(
        self,
        question: str,
        sql: Optional[str],
        result_summary: Optional[str],
        success: bool,
        error: Optional[str] = None,
        token_cost: Optional[float] = None,
        model: Optional[str] = None
    ) -> None:
        """Persist query attempt with outcome details."""
        if not question:
            return

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO queries (
                    question,
                    sql,
                    result_summary,
                    success,
                    error,
                    token_cost,
                    model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question,
                    sql,
                    result_summary,
                    1 if success else 0,
                    error,
                    token_cost,
                    model,
                ),
            )
            conn.commit()

    def _fetch_recent_queries(self) -> List[LoggedQuery]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, sql, result_summary, success, error, created_at, token_cost, model
                FROM queries
                ORDER BY created_at DESC
                LIMIT 200
                """
            ).fetchall()

        return [
            LoggedQuery(
                question=row["question"],
                sql=row["sql"],
                result_summary=row["result_summary"],
                success=bool(row["success"]),
                error=row["error"],
                created_at=datetime.fromisoformat(row["created_at"]),
                token_cost=row["token_cost"],
                model=row["model"],
            )
            for row in rows
        ]

    def find_similar_queries(self, question: str, limit: int = 3) -> List[LoggedQuery]:
        """Return up to `limit` prior queries most similar to the question."""
        if not question:
            return []

        recent = self._fetch_recent_queries()
        scored: List[Tuple[float, LoggedQuery]] = []
        for entry in recent:
            ratio = SequenceMatcher(
                None,
                question.lower(),
                entry.question.lower()
            ).ratio()
            if ratio > 0.35:  # basic relevance threshold
                scored.append((ratio, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    # ------------------------------------------------------------------ #
    # Context construction
    # ------------------------------------------------------------------ #
    def build_context_block(
        self,
        question: str,
        candidate_tables: Iterable[str],
        include_history: bool = True,
        history_limit: int = 3
    ) -> str:
        """Compose a context snippet combining schema info and related history."""
        sections: List[str] = []

        schema_snippets = self.get_schema_snippets(list(candidate_tables))
        if schema_snippets:
            sections.append("Cached schema snapshots:\n" + "\n\n".join(schema_snippets))

        if include_history and question:
            matches = self.find_similar_queries(question, limit=history_limit)
            if matches:
                lines: List[str] = []
                for match in matches:
                    status = "SUCCESS" if match.success else "FAILURE"
                    summary_bits = []
                    if match.sql:
                        summary_bits.append(f"SQL: {match.sql}")
                    if match.result_summary:
                        summary_bits.append(f"Summary: {match.result_summary[:200]}")
                    if match.error:
                        summary_bits.append(f"Error: {match.error}")
                    lines.append(f"[{status} at {match.created_at:%Y-%m-%d %H:%M}] " + " | ".join(summary_bits))
                sections.append("Recent similar attempts:\n" + "\n".join(lines))

        return "\n\n".join(sections)

