"""
Case Facts Provider - Exposes structured case facts to SK plugins.

This module provides access to case-specific facts extracted from HK Statement
and OGC emails, allowing plugins to reference actual allegations instead of
generic templates. Now includes semantic retrieval capabilities for finding
relevant corpus paragraphs.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
import logging
import json
import sqlite3
import re
import textwrap
from collections import Counter
import time

logger = logging.getLogger(__name__)

# Simple in-memory caches to avoid repeated heavy operations
_JSON_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# Import EmbeddingRetriever
try:
    from .EmbeddingRetriever import EmbeddingRetriever, RetrievedChunk
    EMBEDDING_RETRIEVER_AVAILABLE = True
except ImportError:
    EMBEDDING_RETRIEVER_AVAILABLE = False
    logger.warning("EmbeddingRetriever not available, semantic search disabled")
    @dataclass
    class RetrievedChunk:  # type: ignore[redefinition]
        text: str
        score: float
        source_file: str
        chunk_index: int
        metadata: Dict[str, Any]

_EMBEDDING_CACHE: Dict[Tuple[Optional[str], Optional[str]], Optional[EmbeddingRetriever]] = {}


class CaseFactsProvider:
    """Provides structured case facts to SK plugins."""

    def __init__(
        self,
        case_insights: Optional[Any] = None,
        embedding_db_path: Optional[Path] = None,
        embedding_faiss_path: Optional[Path] = None,
        enable_factuality_filter: bool = True,
        lawsuit_facts_db_path: Optional[Path] = None,
        truths_db_path: Optional[Path] = None,  # DEPRECATED: use lawsuit_facts_db_path
        personal_corpus_dir: Optional[Path] = None,
        strict_filtering: bool = True,
    ):
        """
        Initialize CaseFactsProvider.

        Args:
            case_insights: CaseInsights object containing evidence and fact blocks
            embedding_db_path: Path to SQLite database with embeddings
            embedding_faiss_path: Path to FAISS index file
            enable_factuality_filter: Whether to remove hypothetical/speculative content
                from fact blocks using factuality_filter pipeline.
            lawsuit_facts_db_path: Optional path to lawsuit facts SQLite database.
            truths_db_path: [DEPRECATED] Use lawsuit_facts_db_path instead.
            personal_corpus_dir: Optional override for the personal corpus directory.
            strict_filtering: If True, never fall back to generic evidence when personal facts exist.
        """
        self.case_insights = case_insights
        self._fact_blocks: Dict[str, str] = {}
        self._structured_facts: Dict[str, Any] = {}
        self._fact_source: str = "none"
        self._registry_fact_details: Dict[str, Dict[str, Any]] = {}
        self._embedding_retriever: Optional[EmbeddingRetriever] = None
        self._factuality_filter = None
        self._factuality_filter_enabled = enable_factuality_filter
        self._factuality_stats = {
            "facts_filtered": 0,
            "hypothetical_sentences_removed": 0,
        }
        # Cache for filtered facts to prevent multiple validation passes
        self._filtered_facts_cache: Dict[str, str] = {}
        # Support both new and deprecated parameter names
        self._lawsuit_facts_db_path = Path(lawsuit_facts_db_path) if lawsuit_facts_db_path else (Path(truths_db_path) if truths_db_path else None)
        self._lawsuit_facts_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None
        # Keep old name for backward compatibility during transition
        self._truths_db_path = self._lawsuit_facts_db_path
        self._truths_cache = self._lawsuit_facts_cache
        self._retriever_warning_logged = False
        self._personal_corpus_dir: Optional[Path] = None
        self._personal_corpus_index: Dict[str, str] = {}
        self._known_evidence_nodes: Set[str] = set()
        self._last_filter_stats: Dict[str, Any] = {}
        self._personal_corpus_override = personal_corpus_dir
        self._strict_filtering = strict_filtering
        # Track fallback usage for summary logging
        self._fallback_log = {
            "primary_source": None,
            "primary_success": False,
            "json_fallback_attempted": False,
            "json_fallback_paths": [],
            "json_fallback_success": False,
            "json_facts_loaded": 0,
            "registry_facts_attempted": False,
            "registry_facts_loaded": 0,
            "registry_failure_reason": None,
            "embedding_retriever_available": False,
            "embedding_retriever_failure_reason": None,
            "fact_blocks_loaded": 0,
            "facts_filtered": 0,
            "hypothetical_sentences_removed": 0,
            "personal_corpus_attempted": False,
            "personal_corpus_loaded": 0,
            "personal_corpus_dir": None
        }

        load_start = time.perf_counter()

        if enable_factuality_filter:
            try:
                from factuality_filter.code.pipeline import FactualityFilter  # type: ignore(import-error)

                self._factuality_filter = FactualityFilter()
                logger.info("[FACTS] Factuality filter enabled for CaseFactsProvider")
            except Exception as exc:  # pragma: no cover - best effort guard
                logger.warning(f"[FACTS] Factuality filter unavailable: {exc}")
                self._factuality_filter = None

        if case_insights:
            self._extract_facts_from_insights(case_insights)
            if self._fact_blocks:
                self._fact_source = "case_insights"
            self._capture_case_evidence_nodes(case_insights)

        # Robust fallback: if no facts were extracted, attempt to load from JSON
        if not self._fact_blocks:
            try:
                fact_blocks = self._load_facts_from_case_insights_json()
                if fact_blocks:
                    self._fact_blocks.update(fact_blocks)
                    self._fact_source = "lawsuit_facts_extracted_json"
                    self._fallback_log["primary_source"] = "lawsuit_facts_extracted_json"
                    self._fallback_log["primary_success"] = True
                    self._fallback_log["fact_blocks_loaded"] = len(self._fact_blocks)
                    logger.info(
                        f"[CaseFactsProvider] Loaded {len(fact_blocks)} fact blocks from lawsuit facts JSON fallback"
                    )
                else:
                    logger.warning(
                        "[CaseFactsProvider] No fact blocks available after inspecting lawsuit facts JSON fallback paths"
                    )
            except Exception as exc:
                logger.debug(f"[CaseFactsProvider] Could not load lawsuit facts JSON fallback: {exc}")

        registry_blocks = self._load_fact_blocks_from_database()
        if registry_blocks:
            added_keys = [
                key for key in registry_blocks.keys() if key not in self._fact_blocks
            ]
            for key in added_keys:
                self._fact_blocks[key] = registry_blocks[key]
            if added_keys:
                logger.info(
                    "[FACTS] Loaded %d fact_registry entries from %s",
                    len(added_keys),
                    self._lawsuit_facts_db_path,
                )
                if self._fact_source in ("none", ""):
                    self._fact_source = "fact_registry"
            elif not self._fact_source or self._fact_source == "none":
                # Registry rows existed but were already merged earlier.
                self._fact_source = "fact_registry"

        personal_facts = self._load_personal_corpus_facts()
        if personal_facts:
            new_keys = [k for k in personal_facts.keys() if k not in self._fact_blocks]
            self._fact_blocks.update(personal_facts)
            if new_keys:
                logger.info(
                    "[FACTS] Loaded %d personal corpus fact entries (%s)",
                    len(new_keys),
                    ", ".join(new_keys[:5]) + ("..." if len(new_keys) > 5 else "")
                )
            self._fallback_log["personal_corpus_loaded"] = len(personal_facts)
            if self._fact_source == "none":
                self._fact_source = "personal_corpus"

        if self._fact_blocks:
            self._build_structured_facts()
        else:
            logger.error(
                "[FACTS] No case facts available. "
                "Ensure writer_agents/outputs/lawsuit_facts_extracted.json or case_law_data/lawsuit_source_documents/ contains your documents."
            )

        # Initialize embedding retriever if available
        if EMBEDDING_RETRIEVER_AVAILABLE:
            try:
                self._embedding_retriever = self._get_or_create_embedding_retriever(
                    embedding_db_path,
                    embedding_faiss_path
                )
                if self._embedding_retriever.is_available():
                    self._fallback_log["embedding_retriever_available"] = True
                    logger.info("[CaseFactsProvider] EmbeddingRetriever initialized successfully")
                else:
                    self._fallback_log["embedding_retriever_available"] = False
                    reason = "Missing DB/model files"
                    self._fallback_log["embedding_retriever_failure_reason"] = reason
                    logger.warning(f"[CaseFactsProvider] EmbeddingRetriever initialized but not available ({reason})")
                    self._embedding_retriever = None
            except Exception as e:
                self._fallback_log["embedding_retriever_available"] = False
                reason = str(e)
                self._fallback_log["embedding_retriever_failure_reason"] = reason
                logger.warning(f"[CaseFactsProvider] Failed to initialize EmbeddingRetriever: {reason}")
                self._embedding_retriever = None
        else:
            self._fallback_log["embedding_retriever_available"] = False
            self._fallback_log["embedding_retriever_failure_reason"] = "EMBEDDING_RETRIEVER_AVAILABLE = False"

        # Log comprehensive fallback summary
        if self._fact_blocks:
            self._fallback_log["fact_blocks_loaded"] = len(self._fact_blocks)
        self._log_fallback_summary()
        logger.info(
            "[FACTS] Total fact blocks available: %d (source=%s, load_time=%.2fs)",
            len(self._fact_blocks),
            self._fact_source,
            time.perf_counter() - load_start,
        )

    def _log_fallback_summary(self) -> None:
        """Log comprehensive summary of fallback usage and final state."""
        log = self._fallback_log
        summary_lines = [
            "[CaseFactsProvider] Fallback summary:",
            f"  - Primary source: {log['primary_source'] or 'none'} ({'success' if log['primary_success'] else 'failed'})",
        ]

        if log["json_fallback_attempted"]:
            summary_lines.append(
                f"  - JSON fallback: attempted {len(log['json_fallback_paths'])} paths, "
                f"{'success' if log['json_fallback_success'] else 'failed'}, "
                f"loaded {log['json_facts_loaded']} facts"
            )
            if not log["json_fallback_success"] and log["json_fallback_paths"]:
                summary_lines.append(f"    Paths attempted: {', '.join(log['json_fallback_paths'][:3])}")

        if log["registry_facts_attempted"]:
            if log["registry_facts_loaded"]:
                summary_lines.append(
                    f"  - Fact registry: loaded {log['registry_facts_loaded']} entries"
                )
            else:
                reason = log["registry_failure_reason"] or "no entries available"
                summary_lines.append(f"  - Fact registry: no entries loaded ({reason})")

        summary_lines.append(
            f"  - Embedding retriever: {'available' if log['embedding_retriever_available'] else 'unavailable'}"
        )
        if not log["embedding_retriever_available"] and log["embedding_retriever_failure_reason"]:
            summary_lines.append(f"    Reason: {log['embedding_retriever_failure_reason']}")

        summary_lines.append(f"  - Fact blocks: {log['fact_blocks_loaded']} loaded")
        if log["facts_filtered"] > 0:
            summary_lines.append(
                f"  - Facts filtered: {log['facts_filtered']} facts, "
                f"{log['hypothetical_sentences_removed']} hypothetical sentences removed"
            )
        if log["personal_corpus_attempted"]:
            summary_lines.append(
                f"  - Personal corpus: "
                f"{'loaded' if log['personal_corpus_loaded'] else 'not loaded'} "
                f"(dir={log['personal_corpus_dir'] or 'n/a'}, facts={log['personal_corpus_loaded']})"
            )

        final_fact_count = len(self._fact_blocks)
        summary_lines.append(f"  - Final state: {final_fact_count} structured facts available")

        logger.info("\n".join(summary_lines))
        logger.info("[FACTS] Total fact blocks available: %d (source=%s)", len(self._fact_blocks), self._fact_source)

    def _extract_facts_from_insights(self, insights: Any) -> None:
        """Extract fact blocks from CaseInsights evidence."""
        if not hasattr(insights, 'evidence'):
            return

        before_count = len(self._fact_blocks)
        # Extract fact blocks from evidence items
        for item in insights.evidence:
            node_id, state = self._extract_node_and_state(item)
            if not node_id or state is None:
                continue
            if node_id.startswith('fact_block_'):
                fact_key = node_id.replace('fact_block_', '')
                self._fact_blocks[fact_key] = self._sanitize_fact_text(state, fact_key)
            else:
                # Backward‑compat: accept raw fact keys if they match known pattern keys
                # This allows older pipelines to populate facts without the prefix
                # Keys are stored as-is; downstream get_fact_block uses the raw key
                self._fact_blocks.setdefault(node_id, self._sanitize_fact_text(state, node_id))

        # Build structured facts
        self._build_structured_facts()
        added = len(self._fact_blocks) - before_count
        if added > 0:
            logger.info("[FACTS] Extracted %d fact blocks from CaseInsights evidence", added)
        if self._fact_blocks:
            self._fallback_log["primary_source"] = "case_insights"
            self._fallback_log["primary_success"] = True
            self._fallback_log["fact_blocks_loaded"] = len(self._fact_blocks)
            logger.debug(f"[CaseFactsProvider] Primary source loaded {len(self._fact_blocks)} fact blocks from case_insights")

    def _capture_case_evidence_nodes(self, insights: Any) -> None:
        """Store known evidence node identifiers for filtering."""
        evidence_items = getattr(insights, "evidence", None)
        if not evidence_items:
            return
        for item in evidence_items:
            node_id = None
            if isinstance(item, dict):
                node_id = item.get("node_id")
            else:
                node_id = getattr(item, "node_id", None)
            if node_id:
                self._known_evidence_nodes.add(node_id.lower())

    def _load_facts_from_case_insights_json(self) -> Dict[str, str]:
        """Load fact blocks from lawsuit_facts_extracted.json or case_insights.json if present."""
        possible_paths = [
            # New name first
            Path(__file__).resolve().parents[4] / "outputs" / "lawsuit_facts_extracted.json",
            Path.cwd() / "writer_agents" / "outputs" / "lawsuit_facts_extracted.json",
            Path.cwd() / "outputs" / "lawsuit_facts_extracted.json",
            # Old name for backward compatibility
            Path(__file__).resolve().parents[4] / "outputs" / "case_insights.json",
            Path.cwd() / "writer_agents" / "outputs" / "case_insights.json",
            Path.cwd() / "outputs" / "case_insights.json",
        ]

        self._fallback_log["json_fallback_attempted"] = True
        self._fallback_log["json_fallback_paths"] = [str(p) for p in possible_paths]

        for p in possible_paths:
            if not p.exists():
                continue

            try:
                logger.debug(f"[CaseFactsProvider] JSON fallback: Attempting to load from {p}")
                data = self._read_json_cached(p)
            except json.JSONDecodeError as exc:
                logger.warning(f"[CaseFactsProvider] JSON fallback: Invalid JSON in {p}: {exc}")
                self._fallback_log["json_fallback_paths"].append(f"{p} (JSON decode error: {exc})")
                continue
            except Exception as exc:
                logger.warning(f"[CaseFactsProvider] JSON fallback: Failed to read {p}: {exc}")
                self._fallback_log["json_fallback_paths"].append(f"{p} (read error: {exc})")
                continue
            if data is None:
                continue

            raw_blocks = data.get("fact_blocks") or {}
            if raw_blocks:
                fact_blocks: Dict[str, str] = {}
                for key, value in raw_blocks.items():
                    text_value = None
                    if isinstance(value, str):
                        text_value = value
                    elif isinstance(value, dict):
                        text_value = value.get("text") or value.get("value") or json.dumps(value)
                    if text_value:
                        sanitized = self._sanitize_fact_text(text_value, key)
                        if sanitized:
                            fact_blocks[key] = sanitized

                self._fallback_log["json_fallback_success"] = True
                self._fallback_log["json_facts_loaded"] = len(fact_blocks)
                logger.info(f"[CaseFactsProvider] JSON fallback: Successfully loaded {len(fact_blocks)} fact blocks from {p}")
                return fact_blocks
            else:
                logger.debug(f"[CaseFactsProvider] JSON fallback: File {p} exists but contains no fact_blocks")
                self._fallback_log["json_fallback_paths"].append(f"{p} (no fact_blocks key)")

        logger.warning(
            f"[CaseFactsProvider] JSON fallback: No valid lawsuit_facts_extracted.json or case_insights.json found in attempted paths"
        )
        self._fallback_log["json_fallback_success"] = False
        return {}

    def _load_fact_blocks_from_database(self) -> Dict[str, str]:
        """Load fact blocks + metadata from the fact_registry table."""
        self._fallback_log["registry_facts_attempted"] = True
        facts: Dict[str, str] = {}
        self._registry_fact_details = {}

        if not self._lawsuit_facts_db_path:
            self._fallback_log["registry_failure_reason"] = "no database path configured"
            return facts
        if not self._lawsuit_facts_db_path.exists():
            self._fallback_log["registry_failure_reason"] = f"missing database at {self._lawsuit_facts_db_path}"
            return facts

        try:
            with sqlite3.connect(str(self._lawsuit_facts_db_path)) as conn:
                conn.row_factory = sqlite3.Row
                columns_cursor = conn.execute("PRAGMA table_info(fact_registry)")
                columns = {row[1] for row in columns_cursor.fetchall()}
                if not columns or "fact_value" not in columns:
                    self._fallback_log["registry_failure_reason"] = "fact_registry missing required columns"
                    return facts
                fact_id_column = "fact_id" if "fact_id" in columns else ("id" if "id" in columns else None)
                if not fact_id_column:
                    self._fallback_log["registry_failure_reason"] = "fact_registry missing fact_id"
                    return facts

                select_columns = [fact_id_column, "fact_value"]
                optional_columns = [
                    "description",
                    "fact_type",
                    "source_doc",
                    "extraction_method",
                    "confidence",
                    "metadata",
                    "created_at",
                ]
                for column in optional_columns:
                    if column in columns:
                        select_columns.append(column)

                query = f"SELECT {', '.join(select_columns)} FROM fact_registry"
                rows = conn.execute(query).fetchall()

                details_map: Dict[str, Dict[str, Any]] = {}
                for row in rows:
                    row_dict = {key: row[key] for key in row.keys()}
                    fact_id_raw = row_dict.get(fact_id_column)
                    if fact_id_raw is None:
                        continue
                    fact_id = str(fact_id_raw).strip()
                    if not fact_id:
                        continue
                    fact_text = row_dict.get("fact_value") or row_dict.get("description")
                    sanitized = self._sanitize_fact_text(fact_text, fact_id)
                    if not sanitized:
                        continue
                    metadata_json = row_dict.get("metadata")
                    metadata: Dict[str, Any] = {}
                    if metadata_json:
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            logger.debug("[FACTS] Invalid metadata JSON for fact %s", fact_id)
                            metadata = {}
                    details_map[fact_id] = {
                        "fact_id": fact_id,
                        "fact_value": sanitized,
                        "raw_fact_value": row_dict.get("fact_value"),
                        "description": row_dict.get("description"),
                        "fact_type": row_dict.get("fact_type"),
                        "source_doc": row_dict.get("source_doc"),
                        "extraction_method": row_dict.get("extraction_method"),
                        "confidence": row_dict.get("confidence"),
                        "created_at": row_dict.get("created_at"),
                        "metadata": metadata,
                    }
                    facts[fact_id] = sanitized

                self._registry_fact_details = details_map
                self._fallback_log["registry_failure_reason"] = None
                self._fallback_log["registry_facts_loaded"] = len(facts)
                return facts
        except sqlite3.Error as exc:  # pragma: no cover - defensive logging
            self._fallback_log["registry_failure_reason"] = str(exc)
            logger.debug(f"[FACTS] Failed to load fact_registry: {exc}")
        return facts

    def _read_json_cached(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON file with simple mtime-based caching."""
        try:
            resolved = str(path.resolve())
            mtime = path.stat().st_mtime
        except OSError:
            return None

        cached = _JSON_CACHE.get(resolved)
        if cached and cached[0] == mtime:
            return cached[1]

        data = json.loads(path.read_text(encoding="utf-8"))
        _JSON_CACHE[resolved] = (mtime, data)
        return data

    def _load_personal_corpus_facts(self) -> Dict[str, str]:
        """Load fact entries from the personal corpus directory if available."""
        try:
            repo_root = Path(__file__).resolve().parents[4]
        except Exception:
            repo_root = Path.cwd()

        # Fallback logic: try new name first, then legacy name
        if self._personal_corpus_override:
            corpus_dir = self._personal_corpus_override
        else:
            default_dir = repo_root / "case_law_data" / "lawsuit_source_documents"
            legacy_dir = repo_root / "case_law_data" / "tmp_corpus"
            if default_dir.exists():
                corpus_dir = default_dir
            elif legacy_dir.exists():
                corpus_dir = legacy_dir
                logger.info(f"[FACTS] Using legacy corpus directory: {legacy_dir}")
            else:
                corpus_dir = default_dir  # Will fail gracefully if doesn't exist
        
        if corpus_dir.exists():
            self._fallback_log["personal_corpus_attempted"] = True
            self._fallback_log["personal_corpus_dir"] = str(corpus_dir)
        else:
            self._fallback_log["personal_corpus_attempted"] = False
            return {}

        self._personal_corpus_dir = corpus_dir
        fact_entries: Dict[str, str] = {}
        for path in sorted(corpus_dir.glob("*.txt")):
            if len(fact_entries) >= 40:
                break
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as exc:  # pragma: no cover - best effort guard
                logger.debug(f"[FACTS] Could not read personal corpus file %s: %s", path.name, exc)
                continue

            if not text:
                continue

            slug = self._slugify_personal_key(path.stem)
            key = f"personal_corpus_{slug}"
            snippet = self._summarize_personal_text(text)
            fact_entries[key] = f"{path.stem}: {snippet}"
            self._personal_corpus_index[key] = str(path)

        return fact_entries

    def _build_structured_facts(self) -> None:
        """Build structured facts dictionary from fact blocks."""
        self._structured_facts = {
            'hk_allegation_ccp_family': self.get_fact_block('hk_allegation_ccp_family'),
            'harvard_retaliation_events': self.get_fact_block('harvard_retaliation_events'),
            'ogc_email_allegations': self._get_ogc_email_allegations(),
            'privacy_leak_events': self.get_fact_block('privacy_leak_events'),
            'safety_concerns': self.get_fact_block('safety_concerns'),
            'hk_retaliation_events': self.get_fact_block('hk_retaliation_events'),
            'hk_allegation_defamation': self.get_fact_block('hk_allegation_defamation'),
            'hk_allegation_competitor': self.get_fact_block('hk_allegation_competitor'),
        }

        # Overlay fact registry entries (extensible for any fact type)
        fact_registry = self.get_fact_registry()
        for fact_type, values in fact_registry.items():
            if not values:
                continue
            normalized_values = [value.strip() for value in values if value and value.strip()]
            if not normalized_values:
                continue
            # Deduplicate while preserving order
            deduped = list(dict.fromkeys(normalized_values))
            self._structured_facts[fact_type] = "; ".join(deduped)

        if 'citizenship' not in self._structured_facts:
            self._structured_facts['citizenship'] = self.get_fact_block('plaintiff_identity_citizenship') or "Not disclosed in source documents"

        personal_section = self._format_personal_corpus_section()
        if personal_section:
            self._structured_facts['personal_corpus_documents'] = personal_section

    def _get_ogc_email_allegations(self) -> str:
        """Get combined OGC email allegations."""
        parts = []
        for key in ['ogc_email_1_threat', 'ogc_email_2_non_response', 'ogc_email_3_meet_confer']:
            fact = self.get_fact_block(key)
            if fact:
                parts.append(fact)
        return ' '.join(parts) if parts else ''

    def get_fact_block(self, fact_key: str) -> Optional[str]:
        """
        Get a specific fact block by key.

        Args:
            fact_key: Key identifying the fact block (e.g., 'hk_allegation_ccp_family')

        Returns:
            Fact text or None if not found
        """
        return self._fact_blocks.get(fact_key)

    def get_all_facts(self) -> Dict[str, str]:
        """
        Get all fact blocks.

        Returns:
            Dictionary of fact_key -> fact_text
        """
        return self._fact_blocks.copy()

    def get_structured_fact(self, fact_type: str, use_filtered: bool = False) -> Optional[str]:
        """
        Get a structured fact by type.

        Args:
            fact_type: Type of fact ('hk_allegation_ccp_family', 'harvard_retaliation_events', etc.)
            use_filtered: If True, re-run factuality filter prior to returning.

        Returns:
            Fact text or None if not found
        """
        fact = self._structured_facts.get(fact_type)
        if not fact or not use_filtered:
            return fact
        return self._apply_factuality_filter(fact, fact_type)

    def get_all_structured_facts(self, use_filtered: bool = True) -> Dict[str, str]:
        """
        Get all structured facts.

        Args:
            use_filtered: If True, ensure no hypothetical content is returned.

        Returns:
            Dictionary of fact_type -> fact_text
        """
        result: Dict[str, str] = {}
        for key, value in self._structured_facts.items():
            if not value:
                continue
            result[key] = self._apply_factuality_filter(value, key) if use_filtered else value
        return result

    def search_facts(self, query: str) -> List[tuple[str, str]]:
        """
        Search facts by keyword (simple text matching).

        Args:
            query: Search query (keyword or phrase)

        Returns:
            List of (fact_key, fact_text) tuples matching the query
        """
        query_lower = query.lower()
        matches = []

        for fact_key, fact_text in self._fact_blocks.items():
            if query_lower in fact_text.lower():
                matches.append((fact_key, fact_text))

        return matches

    def get_fact_registry(self, fact_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Retrieve fact entries from the fact_registry table.

        Args:
            fact_type: Optional filter for a specific fact type.

        Returns:
            Dictionary mapping fact_type -> list of fact values.
        """
        results: Dict[str, List[str]] = {}
        if not self._lawsuit_facts_db_path or not self._lawsuit_facts_db_path.exists():
            return results

        try:
            with sqlite3.connect(str(self._lawsuit_facts_db_path)) as conn:
                cursor = conn.cursor()
                if fact_type:
                    cursor.execute(
                        "SELECT fact_type, fact_value FROM fact_registry WHERE fact_type = ?",
                        (fact_type,),
                    )
                else:
                    cursor.execute("SELECT fact_type, fact_value FROM fact_registry")
                for row in cursor.fetchall():
                    f_type, f_value = row
                    if not f_value:
                        continue
                    results.setdefault(f_type, []).append(str(f_value))
        except sqlite3.Error as exc:  # pragma: no cover - defensive logging
            logger.debug(f"[FACTS] Failed to load fact registry: {exc}")

        return results

    def get_fact_details(self, fact_key: str) -> Optional[Dict[str, Any]]:
        """Return the full fact_registry record (including metadata) for a given key."""
        if not fact_key:
            return None
        return self._registry_fact_details.get(fact_key)

    def get_fact_metadata(self, fact_key: str) -> Optional[Dict[str, Any]]:
        """Return parsed metadata for a fact key if the registry provided it."""
        details = self.get_fact_details(fact_key)
        if not details:
            return None
        metadata = details.get("metadata")
        return metadata if isinstance(metadata, dict) else None

    def has_facts(self) -> bool:
        """Check if any facts are available."""
        return len(self._fact_blocks) > 0

    def get_fact_summary(self) -> str:
        """
        Get a summary of all available facts.

        Returns:
            Formatted summary string
        """
        if not self.has_facts():
            return "No case facts available."

        lines = ["Available Case Facts:"]
        for fact_key, fact_text in self._fact_blocks.items():
            lines.append(f"\n{fact_key}:")
            lines.append(f"  {fact_text[:200]}..." if len(fact_text) > 200 else f"  {fact_text}")

        return "\n".join(lines)

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.3
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks from corpus using semantic search.

        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of RetrievedChunk objects, or empty list if retrieval unavailable
        """
        if self._embedding_retriever is None:
            if not self._retriever_warning_logged:
                logger.debug("[CaseFactsProvider] Embedding retriever unavailable; using fact block fallback")
                self._retriever_warning_logged = True
            return self._fact_blocks_to_chunks(top_k=top_k)

        try:
            chunks = self._embedding_retriever.retrieve_relevant_chunks(query, top_k, min_score)
            if chunks:
                return chunks
            logger.debug("[CaseFactsProvider] Embedding retriever returned no results, falling back to fact blocks")
            return self._fact_blocks_to_chunks(top_k=top_k)
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return self._fact_blocks_to_chunks(top_k=top_k)

    def get_facts_for_plugin(
        self,
        plugin_name: str,
        context: str = "",
        top_k: int = 3
    ) -> List[RetrievedChunk]:
        """
        Get relevant facts for a specific plugin using semantic search.

        Maps plugin names to relevant fact queries and retrieves corpus chunks.

        Args:
            plugin_name: Name of the plugin (e.g., "PrivacyPlugin", "SafetyPlugin")
            context: Additional context for the query
            top_k: Number of chunks to retrieve

        Returns:
            List of RetrievedChunk objects with relevant corpus text
        """
        # Map plugin names to fact queries
        plugin_queries = {
            "PrivacyPlugin": "privacy leak Xi Mingze photograph disclosure personal information",
            "SafetyPlugin": "safety concerns political persecution torture arrest PRC Xi Mingze",
            "RetaliationPlugin": "Harvard Clubs defamation retaliation OGC non-response statements",
            "mentions_privacy": "privacy violation personal information disclosure",
            "mentions_safety": "safety risk persecution torture arrest",
            "mentions_retaliation": "retaliation defamation Harvard Clubs OGC",
        }

        # Get base query for plugin
        base_query = plugin_queries.get(plugin_name, "")
        if not base_query:
            # Try to extract from plugin name
            if "privacy" in plugin_name.lower():
                base_query = "privacy leak disclosure personal information"
            elif "safety" in plugin_name.lower():
                base_query = "safety concerns persecution torture"
            elif "retaliation" in plugin_name.lower():
                base_query = "retaliation defamation Harvard"
            else:
                # Generic query
                base_query = "case facts allegations"

        # Combine with context
        query = f"{base_query} {context}".strip()

        # Retrieve chunks
        chunks = self.retrieve_relevant_chunks(query, top_k=top_k)

        # If no chunks found, fall back to fact blocks
        if not chunks:
            logger.debug(f"No chunks found for {plugin_name}, falling back to fact blocks")
            relevant_facts = self._get_relevant_fact_blocks_for_plugin(plugin_name)
            chunks = self._fact_blocks_to_chunks(facts=relevant_facts, top_k=top_k)

        return chunks

    # ---------------------------------------------------------------------
    # New: Lawsuit facts helpers for AutoGen agents
    # ---------------------------------------------------------------------

    def has_lawsuit_facts(self) -> bool:
        """Return True if lawsuit-specific fact blocks are available.

        This is a semantic alias for has_facts(), kept for clarity when
        integrating with AutoGen prompt builders.
        """
        return self.has_facts()

    def format_facts_for_autogen(self) -> str:
        """Format structured facts for AutoGen prompts.

        Returns a compact, markdown-style string with clear sections that
        AutoGen agents can drop into their system or task prompts. Includes
        ALL facts from the database, but prioritizes the most important ones
        at the top for better LLM attention.
        """
        # Ensure structured facts are available
        if not self._structured_facts:
            self._build_structured_facts()

        def _add_section(lines: List[str], title: str, entries: List[tuple[str, str]], priority: bool = False) -> None:
            entries = [(k, v) for k, v in entries if v]
            if not entries:
                return
            # Use emoji or marker for priority sections
            prefix = "⭐ " if priority else ""
            lines.append(f"## {prefix}{title}")
            for key, text in entries:
                # Keep each entry concise; AutoGen can ask for details later
                snippet = text.strip()
                # Truncate very long entries to prevent overwhelming the prompt
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                lines.append(f"- [{key}] {snippet}")
            lines.append("")

        # PRIORITY SECTION: Most important facts first (these are the core case facts)
        priority_lines: List[str] = []
        priority_lines.append("# ⭐ KEY FACTS (USE THESE IN YOUR MOTION)")
        priority_lines.append("")

        # Map internal keys to human-readable sections (backward compatibility)
        hk_allegations = [
            ("hk_allegation_defamation", self._structured_facts.get("hk_allegation_defamation")),
            ("hk_allegation_ccp_family", self._structured_facts.get("hk_allegation_ccp_family")),
            ("hk_allegation_competitor", self._structured_facts.get("hk_allegation_competitor")),
        ]
        _add_section(priority_lines, "HK Allegations", hk_allegations, priority=True)
        processed_keys = {"hk_allegation_defamation", "hk_allegation_ccp_family", "hk_allegation_competitor"}

        retaliation = [
            ("harvard_retaliation_events", self._structured_facts.get("harvard_retaliation_events")),
            ("hk_retaliation_events", self._structured_facts.get("hk_retaliation_events")),
        ]
        _add_section(priority_lines, "Harvard Retaliation Events", retaliation, priority=True)
        processed_keys.update(["harvard_retaliation_events", "hk_retaliation_events"])

        emails = [
            ("ogc_email_allegations", self._structured_facts.get("ogc_email_allegations")),
        ]
        _add_section(priority_lines, "OGC Email Allegations", emails, priority=True)
        processed_keys.add("ogc_email_allegations")

        privacy = [("privacy_leak_events", self._structured_facts.get("privacy_leak_events"))]
        _add_section(priority_lines, "Privacy Leak Events", privacy, priority=True)
        processed_keys.add("privacy_leak_events")

        safety = [("safety_concerns", self._structured_facts.get("safety_concerns"))]
        _add_section(priority_lines, "Safety Concerns", safety, priority=True)
        processed_keys.add("safety_concerns")

        # ALL FACTS SECTION: Complete database facts (organized but not prioritized)
        all_facts_lines: List[str] = []
        all_facts_lines.append("\n# 📋 ALL AVAILABLE FACTS (Reference as needed)")
        all_facts_lines.append("")

        # CRITICAL: Include ALL fact_type entries from database that weren't already processed
        # These come from fact_registry table and represent real case facts
        database_facts: List[tuple[str, str]] = []
        for key, value in self._structured_facts.items():
            if key not in processed_keys and value and value.strip():
                # Skip internal/metadata keys
                if key not in ["citizenship", "personal_corpus_documents"]:
                    database_facts.append((key, value))
        
        if database_facts:
            # Group by fact type for better organization
            fact_type_groups: Dict[str, List[tuple[str, str]]] = {}
            for key, value in database_facts:
                # Use the key as the fact type (it's already the fact_type from database)
                fact_type = key.replace("_", " ").title()
                fact_type_groups.setdefault(fact_type, []).append((key, value))
            
            # Add each fact type group as a section (limit per section to prevent overwhelming)
            for fact_type, entries in sorted(fact_type_groups.items()):
                # Limit to top 50 entries per fact type to keep it manageable
                limited_entries = entries[:50] if len(entries) > 50 else entries
                _add_section(all_facts_lines, f"Database Facts: {fact_type}", limited_entries, priority=False)
                if len(entries) > 50:
                    all_facts_lines.append(f"  * (showing {len(limited_entries)} of {len(entries)} entries for this type)")
                    all_facts_lines.append("")

        # Combine priority and all facts
        formatted = "\n".join(priority_lines) + "\n" + "\n".join(all_facts_lines)
        formatted = formatted.strip()
        if not formatted:
            return "(No lawsuit-specific facts available.)"
        return formatted

    def filter_evidence_for_lawsuit(self, evidence: List[Any]) -> List[Any]:
        """Filter evidence to lawsuit-specific items only.

        Includes items that are clearly marked as fact blocks (node_id starts
        with 'fact_block_') or whose node_id matches a known fact key derived
        from current case fact blocks.

        Args:
            evidence: List of evidence items (dicts or EvidenceItem-like)

        Returns:
            Filtered list of evidence preserving original item types.
        """
        if not evidence:
            self._last_filter_stats = {
                "provider": "CaseFactsProvider",
                "original_count": 0,
                "filtered_count": 0,
                "dropped_count": 0,
                "dropped_reasons": {},
                "used_personal_corpus": bool(self._personal_corpus_index),
            }
            return []

        total = len(evidence)
        kept: List[Any] = []
        dropped_reasons: Counter[str] = Counter()
        personal_corpus_loaded = bool(self._personal_corpus_index)

        for item in evidence:
            node_id = self._extract_node_id(item)
            if not node_id:
                dropped_reasons["missing_node_id"] += 1
                continue

            normalized = node_id.lower()
            raw_key = node_id.replace("fact_block_", "")
            source_hint = self._extract_source_hint(item)

            if node_id.startswith("fact_block_") or raw_key in self._fact_blocks:
                kept.append(item)
                continue
            if normalized in self._known_evidence_nodes:
                kept.append(item)
                continue
            if raw_key in self._personal_corpus_index or normalized.startswith("personal_corpus_"):
                kept.append(item)
                continue
            if source_hint and self._is_personal_corpus_source(source_hint):
                kept.append(item)
                continue

            dropped_reasons["unmatched_node"] += 1

        if not kept:
            if self._strict_filtering and personal_corpus_loaded:
                stats_payload = {
                    "provider": "CaseFactsProvider",
                    "original_count": total,
                    "filtered_count": 0,
                    "dropped_count": total,
                    "dropped_reasons": dict(dropped_reasons),
                    "used_personal_corpus": personal_corpus_loaded,
                    "strict_filtering": True,
                    "strict_filtering_failed": True,
                }
                self._last_filter_stats = stats_payload
                logger.error(
                    "[FACTS] Strict filtering removed all evidence despite personal corpus facts being available."
                )
                return []

            if self._strict_filtering:
                logger.warning(
                    "[FACTS] Strict filtering skipped because no personal corpus facts are loaded; "
                    "falling back to original evidence."
                )

            kept = evidence
            dropped_reasons.clear()

        self._last_filter_stats = {
            "provider": "CaseFactsProvider",
            "original_count": total,
            "filtered_count": len(kept),
            "dropped_count": total - len(kept),
            "dropped_reasons": dict(dropped_reasons),
            "used_personal_corpus": personal_corpus_loaded,
            "strict_filtering": self._strict_filtering,
            "strict_filtering_failed": False,
        }

        logger.info(
            "[FACTS] Evidence filtering kept %d/%d items (personal corpus=%s, strict_filtering=%s)",
            len(kept),
            total,
            "yes" if personal_corpus_loaded else "no",
            "on" if self._strict_filtering else "off",
        )
        if dropped_reasons:
            logger.debug("[FACTS] Evidence dropped reasons: %s", dict(dropped_reasons))

        return kept

    def _fact_blocks_to_chunks(
        self,
        facts: Optional[List[Tuple[str, str]]] = None,
        top_k: int = 3
    ) -> List[RetrievedChunk]:
        """Convert available fact blocks into RetrievedChunk-style records."""
        if not self._fact_blocks:
            return []

        fact_items = facts if facts is not None else list(self._fact_blocks.items())
        chunks: List[RetrievedChunk] = []
        for fact_key, fact_text in fact_items[:top_k]:
            chunks.append(self._build_chunk_from_fact(fact_key, fact_text))
        return chunks

    def _build_chunk_from_fact(self, fact_key: str, fact_text: str) -> RetrievedChunk:
        """Helper to wrap a single fact block in the retrieved chunk schema."""
        metadata: Dict[str, Any] = {"fact_key": fact_key}
        details = self._registry_fact_details.get(fact_key)
        if details:
            for field in ("fact_type", "source_doc", "extraction_method", "confidence", "created_at"):
                value = details.get(field)
                if value:
                    metadata[field] = value
            if details.get("metadata"):
                metadata["fact_metadata"] = details["metadata"]

        return RetrievedChunk(
            text=fact_text,
            score=0.8,
            source_file="fact_blocks",
            chunk_index=0,
            metadata=metadata
        )

    def _get_relevant_fact_blocks_for_plugin(self, plugin_name: str) -> List[Tuple[str, str]]:
        """Get relevant fact blocks for a plugin based on plugin name."""
        plugin_fact_mapping = {
            "PrivacyPlugin": ["privacy_leak_events", "ogc_email_allegations"],
            "SafetyPlugin": ["safety_concerns", "hk_allegation_ccp_family"],
            "RetaliationPlugin": ["harvard_retaliation_events", "hk_retaliation_events", "ogc_email_allegations"],
            "mentions_privacy": ["privacy_leak_events"],
            "mentions_safety": ["safety_concerns"],
            "mentions_retaliation": ["harvard_retaliation_events", "hk_retaliation_events"],
        }

        fact_keys = plugin_fact_mapping.get(plugin_name, [])
        if not fact_keys:
            # Try partial match
            plugin_lower = plugin_name.lower()
            if "privacy" in plugin_lower:
                fact_keys = ["privacy_leak_events"]
            elif "safety" in plugin_lower:
                fact_keys = ["safety_concerns"]
            elif "retaliation" in plugin_lower:
                fact_keys = ["harvard_retaliation_events"]

        results = []
        for fact_key in fact_keys:
            fact_text = self.get_fact_block(fact_key)
            if fact_text:
                results.append((fact_key, fact_text))

        return results

    def get_filtered_fact_block(self, fact_key: str) -> Optional[str]:
        """Return fact block content with hypothetical content removed if available."""
        fact = self.get_fact_block(fact_key)
        if not fact:
            return None
        return self._apply_factuality_filter(fact, fact_key)

    # ------------------------------------------------------------------
    # Lawsuit facts database helpers
    # ------------------------------------------------------------------

    def has_lawsuit_facts_database(self) -> bool:
        """Check if lawsuit facts database exists."""
        return bool(self._lawsuit_facts_db_path and self._lawsuit_facts_db_path.exists())

    def has_truth_database(self) -> bool:
        """[DEPRECATED] Use has_lawsuit_facts_database() instead."""
        return self.has_lawsuit_facts_database()

    def refresh_lawsuit_facts_cache(self) -> None:
        """Refresh the lawsuit facts cache."""
        self._lawsuit_facts_cache = None
        self._truths_cache = None  # Keep in sync for backward compatibility

    def refresh_truths_cache(self) -> None:
        """[DEPRECATED] Use refresh_lawsuit_facts_cache() instead."""
        self.refresh_lawsuit_facts_cache()

    def get_lawsuit_facts_records(
        self,
        include_implications: bool = False,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get lawsuit facts records (document facts or legal implications)."""
        cache = self._load_lawsuit_facts_records()
        key = "implications" if include_implications else "document_facts"
        records = cache.get(key, [])

        if topic:
            topic_lower = topic.lower()
            records = [
                row
                for row in records
                if topic_lower in " ".join(str(value).lower() for value in row.values())
            ]

        if limit is not None:
            return records[:limit]
        return records

    def get_truth_records(
        self,
        derived: bool = False,
        topic: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """[DEPRECATED] Use get_lawsuit_facts_records() instead."""
        return self.get_lawsuit_facts_records(include_implications=derived, topic=topic, limit=limit)

    def format_lawsuit_facts_for_prompt(
        self,
        include_implications: bool = True,
        limit: int = 10,
    ) -> str:
        """Format lawsuit facts for LLM prompts."""
        if not self.has_lawsuit_facts_database():
            return "(No structured lawsuit facts database detected.)"

        document_facts = self.get_lawsuit_facts_records(include_implications=False, limit=limit)
        legal_implications = (
            self.get_lawsuit_facts_records(include_implications=True, limit=limit) if include_implications else []
        )

        lines: List[str] = []
        if document_facts:
            lines.append("## Verified Lawsuit Facts")
            for row in document_facts:
                citation = row.get("citation") or row.get("fact_type")
                lines.append(
                    f"- **{row.get('fact_type')}**: {row.get('fact_text')} "
                    f"(Source: {row.get('source_doc')}; Ref: {citation})"
                )
            lines.append("")

        if include_implications and legal_implications:
            lines.append("## Legal Implications")
            for row in legal_implications:
                lines.append(
                    f"- **{row.get('implication_id')}**: {row.get('legal_implication')} "
                    f"(Supports: {row.get('supporting_fact_ids')}; "
                    f"Assumption: {row.get('jurisdiction_assumption')})"
                )
            lines.append("")

        formatted = "\n".join(lines).strip()
        return formatted or "(Lawsuit facts database is empty.)"

    def format_truths_for_prompt(
        self,
        include_derived: bool = True,
        limit: int = 10,
    ) -> str:
        """[DEPRECATED] Use format_lawsuit_facts_for_prompt() instead."""
        return self.format_lawsuit_facts_for_prompt(include_implications=include_derived, limit=limit)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_embedding_retriever(
        self,
        db_path: Optional[Path],
        faiss_path: Optional[Path]
    ) -> EmbeddingRetriever:
        key = (
            str(db_path.resolve()) if db_path else None,
            str(faiss_path.resolve()) if faiss_path else None,
        )
        cached = _EMBEDDING_CACHE.get(key)
        if cached is not None:
            return cached

        retriever = EmbeddingRetriever(
            db_path=db_path,
            faiss_path=faiss_path
        )
        _EMBEDDING_CACHE[key] = retriever
        return retriever

    def _load_lawsuit_facts_records(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load lawsuit facts records from database."""
        if self._lawsuit_facts_cache is not None:
            return self._lawsuit_facts_cache

        cache: Dict[str, List[Dict[str, Any]]] = {"document_facts": [], "implications": []}
        if not self._lawsuit_facts_db_path or not self._lawsuit_facts_db_path.exists():
            self._lawsuit_facts_cache = cache
            self._truths_cache = {"source": [], "derived": []}  # Keep old format for backward compatibility
            return cache

        attempts = 3
        for attempt in range(attempts):
            try:
                with sqlite3.connect(str(self._lawsuit_facts_db_path)) as conn:
                    conn.row_factory = sqlite3.Row
                    # Try new table names first
                    try:
                        document_facts_rows = conn.execute(
                            """
                            SELECT fact_id, source_doc, fact_type, fact_text, citation, confidence
                            FROM document_facts
                            ORDER BY created_at DESC
                            """
                        ).fetchall()
                        implications_rows = conn.execute(
                            """
                            SELECT implication_id, derived_from, legal_implication,
                                   supporting_fact_ids, jurisdiction_assumption, rationale
                            FROM legal_implications
                            ORDER BY created_at DESC
                            """
                        ).fetchall()
                        cache["document_facts"] = [dict(row) for row in document_facts_rows]
                        cache["implications"] = [dict(row) for row in implications_rows]
                        # Also populate backward-compatible format
                        self._truths_cache = {
                            "source": [dict(row) for row in document_facts_rows],
                            "derived": [dict(row) for row in implications_rows]
                        }
                    except sqlite3.OperationalError:
                        # Fallback to old table names for backward compatibility
                        document_facts_rows = conn.execute(
                            """
                            SELECT truth_id as fact_id, source_doc, fact_type, fact_text, citation, confidence
                            FROM source_facts
                            ORDER BY created_at DESC
                            """
                        ).fetchall()
                        implications_rows = conn.execute(
                            """
                            SELECT truth_id as implication_id, derived_from, legal_implication,
                                   supporting_fact_ids, jurisdiction_assumption, rationale
                            FROM derived_truths
                            ORDER BY created_at DESC
                            """
                        ).fetchall()
                        cache["document_facts"] = [dict(row) for row in document_facts_rows]
                        cache["implications"] = [dict(row) for row in implications_rows]
                        self._truths_cache = {
                            "source": [dict(row) for row in document_facts_rows],
                            "derived": [dict(row) for row in implications_rows]
                        }
                break
            except sqlite3.OperationalError as exc:
                if attempt == attempts - 1:
                    logger.debug(f"[FACTS] Failed to load lawsuit facts database: {exc}")
                else:
                    time.sleep(0.1 * (attempt + 1))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(f"[FACTS] Failed to load lawsuit facts database: {exc}")
                break

        self._lawsuit_facts_cache = cache
        return cache

    def _load_truth_records(self) -> Dict[str, List[Dict[str, Any]]]:
        """[DEPRECATED] Use _load_lawsuit_facts_records() instead."""
        lawsuit_facts = self._load_lawsuit_facts_records()
        # Return in old format for backward compatibility
        if self._truths_cache is not None:
            return self._truths_cache
        return {"source": lawsuit_facts.get("document_facts", []), "derived": lawsuit_facts.get("implications", [])}

    @staticmethod
    def _extract_node_and_state(item: Any) -> Tuple[Optional[str], Optional[Any]]:
        """Normalize evidence item into (node_id, state)."""
        if isinstance(item, dict):
            return item.get('node_id'), item.get('state')
        return getattr(item, 'node_id', None), getattr(item, 'state', None)

    @staticmethod
    def _extract_node_id(item: Any) -> Optional[str]:
        if isinstance(item, dict):
            return item.get("node_id")
        return getattr(item, "node_id", None)

    @staticmethod
    def _extract_source_hint(item: Any) -> Optional[str]:
        if isinstance(item, dict):
            return item.get("source_file") or item.get("source_path")
        return getattr(item, "source_file", None)

    def _is_personal_corpus_source(self, source_hint: str) -> bool:
        if not source_hint:
            return False
        if not self._personal_corpus_dir:
            return False
        try:
            path = Path(source_hint).resolve()
            return str(self._personal_corpus_dir.resolve()) in str(path)
        except Exception:
            return "case_law_data/lawsuit_source_documents" in source_hint.replace("\\", "/").lower() or "case_law_data/tmp_corpus" in source_hint.replace("\\", "/").lower()

    def get_last_filter_stats(self) -> Dict[str, Any]:
        """Return most recent filtering statistics."""
        return dict(self._last_filter_stats)

    @staticmethod
    def _slugify_personal_key(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
        slug = slug.strip("_")
        return slug or "document"

    @staticmethod
    def _summarize_personal_text(text: str, limit: int = 1200) -> str:
        normalized = " ".join(text.split())
        if not normalized:
            return ""
        if len(normalized) <= limit:
            return normalized
        return textwrap.shorten(normalized, width=limit, placeholder=" ...")

    def _sanitize_fact_text(self, text: Any, fact_key: Optional[str]) -> str:
        """Normalize fact text and remove hypothetical content if enabled."""
        cleaned = str(text).strip() if text is not None else ""
        if not cleaned:
            return ""
        return self._apply_factuality_filter(cleaned, fact_key)

    def _apply_factuality_filter(self, text: str, fact_key: Optional[str]) -> str:
        """Apply factuality filter if available, removing hypothetical content.
        
        CACHED: Results are cached to prevent multiple validation passes on the same fact.
        """
        if not text or not self._factuality_filter:
            return text
        
        # Check cache first to prevent multiple validation passes
        cache_key = f"{fact_key or 'unknown'}:{hash(text)}"
        if cache_key in self._filtered_facts_cache:
            return self._filtered_facts_cache[cache_key]
        
        try:
            result = self._factuality_filter.filter_text(
                text,
                remove_hypotheticals=True,
                extract_claims=False,
                verify_claims=False,
            )
            filtered = result.factual_text.strip()
            if filtered and filtered != text:
                removed = len(result.get_hypothetical_sentences())
                self._factuality_stats["facts_filtered"] += 1
                self._factuality_stats["hypothetical_sentences_removed"] += removed
                self._fallback_log["facts_filtered"] += 1
                self._fallback_log["hypothetical_sentences_removed"] += removed
                logger.info(
                    "[FACTS] Removed %d hypothetical sentences from %s",
                    removed,
                    fact_key or "fact block",
                )
                # Cache the result
                self._filtered_facts_cache[cache_key] = filtered
                return filtered
            # Cache even if no changes
            final_result = filtered or text
            self._filtered_facts_cache[cache_key] = final_result
            return final_result
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(f"[FACTS] Factuality filter failed for %s: %s", fact_key, exc)
            # Cache the original text on error
            self._filtered_facts_cache[cache_key] = text
            return text

    def _format_personal_corpus_section(self) -> Optional[str]:
        if not self._personal_corpus_index:
            return None
        lines: List[str] = []
        for key, path in sorted(self._personal_corpus_index.items()):
            text = self._fact_blocks.get(key)
            if not text:
                continue
            doc_label = Path(path).stem
            lines.append(f"{doc_label}\n{text}")
        if not lines:
            return None
        return "\n\n".join(lines)


# Global instance (will be set by WorkflowOrchestrator)
_global_provider: Optional[CaseFactsProvider] = None


def get_case_facts_provider() -> Optional[CaseFactsProvider]:
    """Get the global CaseFactsProvider instance."""
    return _global_provider


def set_case_facts_provider(provider: CaseFactsProvider) -> None:
    """Set the global CaseFactsProvider instance."""
    global _global_provider
    _global_provider = provider
    logger.info(f"CaseFactsProvider set with {len(provider._fact_blocks)} fact blocks")


def clear_case_facts_provider() -> None:
    """Clear the global CaseFactsProvider instance."""
    global _global_provider
    _global_provider = None
