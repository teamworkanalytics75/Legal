"""Utilities to synchronize KnowledgeGraph facts with the SQLite fact registry."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph

from writer_agents.scripts.extract_fact_registry import FactEntry

logger = logging.getLogger(__name__)


class KGFactAdapter:
    """Adapter that converts KnowledgeGraph fact nodes into FactEntry rows."""

    FACT_PREFIXES = ("fact::", "fact:")

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[str, str]] = {}

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #
    def load_facts_from_graph(self, kg: KnowledgeGraph) -> List[FactEntry]:
        """Extract ``FactEntry`` objects from the provided graph."""
        facts: List[FactEntry] = []
        for node, attrs in kg.graph.nodes(data=True):
            parsed = self._parse_fact_node(node, attrs)
            if not parsed:
                continue
            fact_type, fact_value = parsed
            facts.append(
                FactEntry(
                    fact_type=fact_type,
                    fact_value=fact_value,
                    description=attrs.get("description"),
                    source_doc=attrs.get("source_doc", ""),
                    extraction_method=attrs.get("extraction_method", "knowledge_graph"),
                    confidence=float(attrs.get("confidence", 0.0) or 0.0),
                )
            )
        return facts

    def _parse_fact_node(self, node: str, attrs: Dict[str, object]) -> Optional[Tuple[str, str]]:
        """Detect if node represents a fact and return ``(type, value)``."""
        if node in self._cache:
            return self._cache[node]

        node_type = attrs.get("type")
        fact_value = attrs.get("fact_value")

        if isinstance(node_type, str) and node_type.startswith("fact"):
            _, _, remainder = node_type.partition(":")
            fact_type = remainder or node_type.replace("fact", "", 1).strip(" :_")
            value = str(fact_value or node).split("::")[-1].replace("_", " ")
            self._cache[node] = (fact_type, value)
            return self._cache[node]

        for prefix in self.FACT_PREFIXES:
            if node.startswith(prefix):
                parts = node.split("::" if prefix.endswith("::") else ":", 2)
                if len(parts) >= 3:
                    fact_type = parts[1]
                    value = parts[2].replace("_", " ")
                    self._cache[node] = (fact_type, value)
                    return self._cache[node]

        return None

    # ------------------------------------------------------------------ #
    # Synchronization
    # ------------------------------------------------------------------ #
    def sync_graph_to_registry(self, kg: KnowledgeGraph, database_path: Path) -> int:
        """Persist graph facts into the ``fact_registry`` table."""
        facts = self.load_facts_from_graph(kg)
        if not facts:
            logger.info("No facts found in KnowledgeGraph; skipping registry sync.")
            return 0

        database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(database_path)) as conn:
            self._ensure_table(conn)
            inserted = self._save_fact_entries(conn, facts)
        logger.info("Synced %d KnowledgeGraph facts into %s", inserted, database_path)
        return inserted

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        """Ensure the fact registry table exists."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_type TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                description TEXT,
                source_doc TEXT,
                extraction_method TEXT,
                confidence REAL DEFAULT 0.0
            )
            """
        )

    def _save_fact_entries(self, conn: sqlite3.Connection, entries: List[FactEntry]) -> int:
        """Insert entries, coalescing duplicates by (type,value,source_doc)."""
        inserted = 0
        for entry in entries:
            conn.execute(
                """
                INSERT INTO fact_registry (fact_type, fact_value, description, source_doc, extraction_method, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.fact_type,
                    entry.fact_value,
                    entry.description,
                    entry.source_doc,
                    entry.extraction_method,
                    entry.confidence,
                ),
            )
            inserted += 1
        conn.commit()
        return inserted

