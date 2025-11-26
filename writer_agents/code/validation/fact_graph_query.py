"""Query interface for interacting with the KnowledgeGraph-backed fact registry."""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph

try:
    from writer_agents.code.sk_plugins.FeaturePlugin.EmbeddingRetriever import EmbeddingRetriever
except Exception:  # pragma: no cover - optional dependency
    EmbeddingRetriever = None  # type: ignore

logger = logging.getLogger(__name__)

_DATE_FORMATS = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%y",
)

_FACT_SYNONYMS = {
    "ogc": {"office of general counsel", "harvard ogc"},
    "office of general counsel": {"ogc", "harvard ogc"},
}


@dataclass
class FactMatch:
    """Container describing a fact that matched a query."""

    entity_id: str
    fact_type: str
    fact_value: str
    score: float
    relationships: List[Tuple[str, str, str]]


class FactGraphQuery:
    """Query interface for fact validation using the KnowledgeGraph."""

    def __init__(
        self,
        kg: KnowledgeGraph,
        fact_registry_db: Optional[Path] = None,
        embedding_retriever: Optional["EmbeddingRetriever"] = None,
    ) -> None:
        self.kg = kg
        self.fact_registry_db = Path(fact_registry_db) if fact_registry_db else None
        self.embedding_retriever = embedding_retriever

        self._fact_nodes: Dict[str, Dict[str, Any]] = {}
        self._facts_by_type: Dict[str, Set[str]] = defaultdict(set)
        self._fact_lookup_cache: Dict[Tuple[str, str], Optional[str]] = {}
        self._relationships_cache: Dict[str, List[Tuple[str, str, str]]] = {}
        self._similarity_cache: Dict[str, np.ndarray] = {}
        self._db_lookup_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        self._result_cache: Dict[Tuple[str, Tuple[Any, ...], Tuple[Tuple[str, Any], ...]], Any] = {}
        self._metrics: Dict[str, float] = defaultdict(float)

        self._index_graph_facts()

    # ------------------------------------------------------------------ #
    # Fact discovery utilities
    # ------------------------------------------------------------------ #
    def _index_graph_facts(self) -> None:
        """Index fact nodes so queries are cheap."""
        if not self.kg or not getattr(self.kg, "graph", None):
            logger.debug("KnowledgeGraph missing or empty; FactGraphQuery will operate in fallback mode.")
            return
        for node, attrs in self.kg.graph.nodes(data=True):
            parsed = self._parse_fact_node(node, attrs)
            if not parsed:
                continue
            fact_type, fact_value = parsed
            normalized_type = fact_type.lower().strip()
            self._fact_nodes[node] = {
                "fact_type": fact_type,
                "fact_value": fact_value,
                "attributes": attrs,
            }
            self._facts_by_type[normalized_type].add(node)

    def _parse_fact_node(self, node: str, attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Return ``(fact_type, fact_value)`` if node represents a fact."""
        value = attrs.get("fact_value") or self._extract_value_from_entity(node)
        node_type = attrs.get("type") or self.kg.entity_types.get(node, "")

        if isinstance(node_type, str) and node_type.startswith("fact"):
            # e.g., "fact:citizenship"
            _, _, fact_type = node_type.partition(":")
            return fact_type or node_type.replace("fact", "", 1).strip(" :_"), value

        if node.startswith("fact::"):
            _, _, remainder = node.partition("::")
            fact_type, _, fact_value_token = remainder.partition("::")
            if fact_type:
                return fact_type, value or fact_value_token.replace("_", " ")

        if node.startswith("fact:"):
            parts = node.split(":", 2)
            if len(parts) >= 3:
                return parts[1], value or parts[2]

        return None

    def _extract_value_from_entity(self, entity_id: str) -> str:
        """Return a human-readable fact value from an entity id."""
        if "::" in entity_id:
            parts = entity_id.split("::")
            if len(parts) >= 3:
                return parts[2].replace("_", " ")
        if ":" in entity_id:
            parts = entity_id.split(":", 2)
            if len(parts) >= 3:
                return parts[2].replace("_", " ")
        return entity_id

    def _extract_type_from_entity(self, entity_id: str, attrs: Optional[Dict[str, Any]] = None) -> str:
        """Extract fact type from entity id or metadata."""
        if attrs:
            node_type = attrs.get("type") or ""
            if isinstance(node_type, str) and node_type.startswith("fact"):
                return node_type.split(":", 1)[-1]

        if "::" in entity_id:
            parts = entity_id.split("::")
            if len(parts) >= 3 and parts[0] == "fact":
                return parts[1]
        if entity_id.startswith("fact:"):
            parts = entity_id.split(":", 2)
            if len(parts) >= 3:
                return parts[1]
        return attrs.get("fact_type") if attrs else ""

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def find_fact_entity(self, fact_type: str, fact_value: str) -> Optional[str]:
        """Find the KnowledgeGraph entity id for a fact."""
        self._record_metric("fact_entity_lookups")
        if not fact_type or not fact_value or not self._fact_nodes:
            return None
        key = (fact_type.lower().strip(), self._normalize_text(fact_value))
        if key in self._fact_lookup_cache:
            return self._fact_lookup_cache[key]

        candidates = self._facts_by_type.get(key[0], set())
        for node in candidates:
            metadata = self._fact_nodes.get(node, {})
            stored_value = self._normalize_text(metadata.get("fact_value") or "")
            if stored_value == key[1]:
                self._fact_lookup_cache[key] = node
                return node

        # Fallback: check aliases/descriptions in attributes
        for node in candidates:
            attrs = self._fact_nodes.get(node, {}).get("attributes", {})
            aliases = attrs.get("aliases") or attrs.get("synonyms") or []
            for alias in aliases:
                if self._normalize_text(alias) == key[1]:
                    self._fact_lookup_cache[key] = node
                    return node

        # Optional fallback: look up in registry DB
        db_entry = self._lookup_fact_in_db(fact_type, fact_value)
        if db_entry:
            logger.debug("Fact not in KG, but present in DB: %s=%s", fact_type, fact_value)

        self._fact_lookup_cache[key] = None
        return None

    def get_fact_relationships(self, fact_type: str, fact_value: str) -> List[Tuple[str, str, str]]:
        """Return all relationships for the fact entity."""
        entity_id = self.find_fact_entity(fact_type, fact_value)
        if not entity_id:
            return []

        self._record_metric("relationship_queries")
        if entity_id not in self._relationships_cache:
            self._relationships_cache[entity_id] = self.kg.get_relations_for_entity(entity_id)
        return self._relationships_cache[entity_id]

    def verify_fact_exists(self, fact_type: str, fact_value: str) -> bool:
        """Check if the fact is represented in the graph."""
        return self.find_fact_entity(fact_type, fact_value) is not None

    def find_related_facts(self, fact_type: str, fact_value: str, max_depth: int = 2) -> Set[str]:
        """Return ids of facts related to the provided fact."""
        entity_id = self.find_fact_entity(fact_type, fact_value)
        if not entity_id:
            return set()

        neighbors = self.kg.get_entity_neighbors(entity_id, depth=max_depth)
        self._record_metric("related_fact_queries")
        return {neighbor for neighbor in neighbors if neighbor.startswith("fact")}

    # ------------------------------------------------------------------ #
    # Semantic search
    # ------------------------------------------------------------------ #
    def find_similar_facts(
        self,
        query_text: str,
        fact_type: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[FactMatch]:
        """Return facts semantically similar to ``query_text``."""
        if not query_text.strip():
            return []

        normalized_type = fact_type.lower().strip() if fact_type else None
        candidate_nodes = (
            self._facts_by_type.get(normalized_type, set()) if normalized_type else set(self._fact_nodes.keys())
        )
        start_time = perf_counter()
        self._record_metric("similarity_queries")

        embeddings_available = self._ensure_fact_embeddings()
        if embeddings_available:
            query_vector = self._encode_text(query_text)
            if query_vector is None:
                embeddings_available = False

        matches: List[FactMatch] = []
        if embeddings_available:
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
            for node in candidate_nodes:
                node_embedding = self._similarity_cache.get(node)
                if node_embedding is None:
                    continue
                score = float(np.dot(query_norm, node_embedding))
                if score < similarity_threshold:
                    continue
                metadata = self._fact_nodes[node]
                relationships = self.get_fact_relationships(metadata["fact_type"], metadata["fact_value"])
                matches.append(
                    FactMatch(
                        entity_id=node,
                        fact_type=metadata["fact_type"],
                        fact_value=metadata["fact_value"],
                        score=score,
                        relationships=relationships,
                    )
                )
        else:
            # Fallback to fuzzy ratio if embeddings unavailable
            text_lower = self._normalize_text(query_text)
            for node in candidate_nodes:
                metadata = self._fact_nodes[node]
                candidate_text = self._normalize_text(self._build_candidate_text(node))
                ratio = SequenceMatcher(None, text_lower, candidate_text).ratio()
                if ratio < similarity_threshold:
                    continue
                relationships = self.get_fact_relationships(metadata["fact_type"], metadata["fact_value"])
                matches.append(
                    FactMatch(
                        entity_id=node,
                        fact_type=metadata["fact_type"],
                        fact_value=metadata["fact_value"],
                        score=ratio,
                        relationships=relationships,
                    )
                )

        matches.sort(key=lambda item: item.score, reverse=True)
        limited = matches[:top_k]
        self._record_metric("similarity_results", len(limited))
        self._record_metric("similarity_duration_ms", (perf_counter() - start_time) * 1000)
        return limited

    def _ensure_fact_embeddings(self) -> bool:
        """Build embeddings for fact values when semantic search is enabled."""
        if not self.embedding_retriever:
            return False
        if not hasattr(self.embedding_retriever, "is_available"):
            return False
        if not self.embedding_retriever.is_available():
            return False
        if self._similarity_cache:
            return True

        for node in self._fact_nodes:
            candidate_text = self._build_candidate_text(node)
            if not candidate_text:
                continue
            embedding = self._encode_text(candidate_text)
            if embedding is None:
                continue
            norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            self._similarity_cache[node] = norm.astype(np.float32)
        return bool(self._similarity_cache)

    def _encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text using EmbeddingRetriever if available."""
        if not self.embedding_retriever:
            return None
        encode_func = getattr(self.embedding_retriever, "_encode_query", None)
        if not callable(encode_func):
            return None
        try:
            return encode_func(text)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Embedding encoding failed, falling back to fuzzy search: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Validation-focused queries
    # ------------------------------------------------------------------ #
    def validate_fact_claim(self, claim_text: str, expected_fact_type: Optional[str] = None) -> Dict[str, Any]:
        """Validate a factual claim against the KnowledgeGraph."""
        matches = self.find_similar_facts(
            claim_text,
            fact_type=expected_fact_type,
            top_k=3,
            similarity_threshold=0.2,
        )
        result: Dict[str, Any] = {
            "claim": claim_text,
            "expected_type": expected_fact_type,
            "matched_fact": None,
            "score": 0.0,
            "relationships": [],
        }
        if matches:
            best = matches[0]
            result.update(
                {
                    "matched_fact": best.fact_value,
                    "fact_type": best.fact_type,
                    "entity_id": best.entity_id,
                    "score": best.score,
                    "relationships": best.relationships,
                }
            )
            result["valid"] = best.score >= 0.25
        else:
            result["valid"] = False
        return result

    def detect_fact_contradictions(self, motion_text: str) -> List[Dict[str, Any]]:
        """Detect textual contradictions against graph facts."""
        text_lower = motion_text.lower()
        contradictions: List[Dict[str, Any]] = []

        for node, metadata in self._fact_nodes.items():
            fact_value = (metadata.get("fact_value") or "").strip()
            if not fact_value:
                continue
            value_lower = fact_value.lower()
            negation_phrase = f"not {value_lower}"
            if value_lower in text_lower and negation_phrase in text_lower:
                contradictions.append(
                    {
                        "fact_type": metadata["fact_type"],
                        "fact_value": fact_value,
                        "entity_id": node,
                        "evidence": f"Detected both '{fact_value}' and '{negation_phrase}' in motion text.",
                    }
                )
        contradictions.extend(self.find_contradictory_facts_in_graph())
        return contradictions

    def find_facts_by_date_range(
        self,
        start_date: Optional[str | datetime],
        end_date: Optional[str | datetime],
        fact_type: str = "date",
    ) -> List[Dict[str, Any]]:
        """Return facts whose values fall between ``start_date`` and ``end_date``."""
        cache_key_args = (
            str(start_date) if start_date is not None else None,
            str(end_date) if end_date is not None else None,
            fact_type,
        )
        cached = self._get_cached("find_facts_by_date_range", cache_key_args, {})
        if cached is not None:
            self._record_metric("date_range_cache_hits")
            return cached

        start = self._coerce_date(start_date)
        end = self._coerce_date(end_date)
        nodes = self._facts_by_type.get(fact_type.lower().strip(), set())
        results: List[Dict[str, Any]] = []
        for node in nodes:
            metadata = self._fact_nodes[node]
            fact_date = self._coerce_date(metadata.get("fact_value"))
            if fact_date is None:
                continue
            if start and fact_date < start:
                continue
            if end and fact_date > end:
                continue
            results.append(
                {
                    "entity": node,
                    "type": metadata["fact_type"],
                    "value": metadata["fact_value"],
                    "date": fact_date,
                }
            )
        results.sort(key=lambda item: item["date"])
        self._set_cached("find_facts_by_date_range", cache_key_args, {}, results)
        self._record_metric("date_range_queries")
        return results

    def find_facts_by_source_document(self, doc_path: str) -> List[Dict[str, Any]]:
        """Return facts linked to ``doc_path``."""
        cached = self._get_cached("find_facts_by_source_document", (doc_path,), {})
        if cached is not None:
            self._record_metric("source_doc_cache_hits")
            return cached

        normalized = doc_path.strip().lower()
        matches: List[Dict[str, Any]] = []
        for node, metadata in self._fact_nodes.items():
            attrs = metadata["attributes"]
            source = (attrs.get("source_doc") or "").strip().lower()
            if source == normalized:
                matches.append(
                    {
                        "entity": node,
                        "type": metadata["fact_type"],
                        "value": metadata["fact_value"],
                    }
                )
        self._set_cached("find_facts_by_source_document", (doc_path,), {}, matches)
        self._record_metric("source_doc_queries")
        return matches

    def get_fact_timeline(self, fact_type: str = "date") -> List[Dict[str, Any]]:
        """Return chronological ordering for ``fact_type``."""
        cached = self._get_cached("get_fact_timeline", (fact_type,), {})
        if cached is not None:
            self._record_metric("timeline_cache_hits")
            return cached
        results = self.find_facts_by_date_range(None, None, fact_type=fact_type)
        self._set_cached("get_fact_timeline", (fact_type,), {}, results)
        self._record_metric("timeline_queries")
        return results

    def find_contradictory_facts(
        self,
        fact_type: str,
        fact_value: str,
    ) -> List[Dict[str, Any]]:
        """Return conflicting facts for the provided value."""
        normalized_type = fact_type.lower().strip()
        normalized_value = self._normalize_text(fact_value)
        contradictions: List[Dict[str, Any]] = []
        for node in self._facts_by_type.get(normalized_type, set()):
            metadata = self._fact_nodes[node]
            candidate_value = self._normalize_text(metadata["fact_value"])
            if candidate_value == normalized_value:
                continue
            if self._values_conflict(normalized_value, candidate_value):
                contradictions.append(
                    {
                        "entity": node,
                        "fact_value": metadata["fact_value"],
                        "fact_type": metadata["fact_type"],
                    }
                )
        return contradictions

    def find_contradictory_facts_in_graph(self) -> List[Dict[str, Any]]:
        """Detect contradictions by scanning the fact set."""
        contradictions: List[Dict[str, Any]] = []
        for fact_type, nodes in self._facts_by_type.items():
            normalized_values: Dict[str, str] = {}
            for node in nodes:
                metadata = self._fact_nodes[node]
                normalized_value = self._normalize_text(metadata["fact_value"])
                for existing_value, existing_node in normalized_values.items():
                    if self._values_conflict(existing_value, normalized_value):
                        contradictions.append(
                            {
                                "fact_type": metadata["fact_type"],
                                "fact_value_a": metadata["fact_value"],
                                "fact_value_b": self._fact_nodes[existing_node]["fact_value"],
                                "entity_a": node,
                                "entity_b": existing_node,
                            }
                        )
                normalized_values[normalized_value] = node
        return contradictions

    # ------------------------------------------------------------------ #
    # Hierarchical queries and helpers
    # ------------------------------------------------------------------ #
    def get_all_facts_by_type(self, fact_type: str) -> List[Dict[str, Any]]:
        """Return all graph facts of ``fact_type``."""
        normalized_type = fact_type.lower().strip()
        results: List[Dict[str, Any]] = []
        for node in self._facts_by_type.get(normalized_type, set()):
            metadata = self._fact_nodes[node]
            results.append(
                {
                    "entity": node,
                    "type": metadata["fact_type"],
                    "value": metadata["fact_value"],
                    "relationships": self.get_fact_relationships(metadata["fact_type"], metadata["fact_value"]),
                }
            )
        return results

    def get_fact_hierarchy(self) -> Dict[str, List[str]]:
        """Return mapping of fact types to values."""
        hierarchy: Dict[str, List[str]] = {}
        for node, metadata in self._fact_nodes.items():
            fact_type = metadata["fact_type"]
            hierarchy.setdefault(fact_type, []).append(metadata["fact_value"])
        return hierarchy

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _lookup_fact_in_db(self, fact_type: str, fact_value: str) -> Optional[Dict[str, Any]]:
        """Lookup fact in backing SQLite registry as a fallback."""
        if not self.fact_registry_db:
            return None
        key = (fact_type.lower().strip(), self._normalize_text(fact_value))
        if key in self._db_lookup_cache:
            return self._db_lookup_cache[key]

        if not self.fact_registry_db.exists():
            self._db_lookup_cache[key] = None
            return None

        query = """
            SELECT fact_type, fact_value, description, source_doc, extraction_method, confidence
            FROM fact_registry
            WHERE LOWER(fact_type) = ? AND LOWER(fact_value) = ?
            LIMIT 1
        """
        try:
            with sqlite3.connect(str(self.fact_registry_db)) as conn:
                row = conn.execute(query, key).fetchone()
        except Exception as exc:  # pragma: no cover - IO failure
            logger.warning("Failed to query fact registry DB %s: %s", self.fact_registry_db, exc)
            row = None

        if row:
            result = {
                "fact_type": row[0],
                "fact_value": row[1],
                "description": row[2],
                "source_doc": row[3],
                "extraction_method": row[4],
                "confidence": row[5],
            }
        else:
            result = None
        self._db_lookup_cache[key] = result
        return result

    def _coerce_date(self, value: Optional[str | datetime]) -> Optional[datetime]:
        """Convert strings to ``datetime`` objects using a set of formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        text = value.strip()
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def _normalize_text(self, text: str) -> str:
        normalized = text.strip().lower()
        if normalized in _FACT_SYNONYMS:
            return normalized
        for canonical, synonyms in _FACT_SYNONYMS.items():
            if normalized in synonyms:
                return canonical
        return normalized

    def _build_candidate_text(self, node: str) -> str:
        """Compose semantic text for a fact node."""
        metadata = self._fact_nodes.get(node, {})
        attrs = metadata.get("attributes", {})
        components = [metadata.get("fact_value", "")]
        description = attrs.get("description")
        if description:
            components.append(str(description))

        try:
            relations = self.kg.get_relations_for_entity(node)
        except Exception:  # pragma: no cover - safety
            relations = []
        rel_text = " ".join(f"{subj} {rel} {obj}" for subj, rel, obj in relations)
        if rel_text:
            components.append(rel_text)

        return " ".join(part for part in components if part).strip()

    def _values_conflict(self, value_a: str, value_b: str) -> bool:
        tokens_a = set(value_a.split())
        tokens_b = set(value_b.split())
        if ("not" in tokens_a) ^ ("not" in tokens_b):
            base_a = tokens_a - {"not"}
            base_b = tokens_b - {"not"}
            if base_a == base_b:
                return True
        opposites = {
            ("us citizen", "prc citizen"),
            ("plaintiff", "defendant"),
        }
        return (value_a, value_b) in opposites or (value_b, value_a) in opposites

    def _record_metric(self, name: str, value: float = 1.0) -> None:
        self._metrics[name] += value

    def _get_cached(self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[Any]:
        key = (name, args, tuple(sorted(kwargs.items())))
        return self._result_cache.get(key)

    def _set_cached(self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any], value: Any) -> None:
        key = (name, args, tuple(sorted(kwargs.items())))
        self._result_cache[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Export cached facts for debugging/inspection."""
        return {
            "fact_count": len(self._fact_nodes),
            "fact_types": sorted(self._facts_by_type),
        }

    def get_query_metrics(self) -> Dict[str, float]:
        """Return a snapshot of collected query metrics."""
        return dict(self._metrics)
