"""
Lightweight contradiction detection utilities for generated motions.

The detector focuses on high-impact factual claims and reports
contradictions, unsupported inferences, or hallucinated facts by
cross-referencing source documents and an optional fact registry.

The module is designed to be extensible:
    * Built-in validators (e.g., citizenship) register automatically.
    * Additional validators can be registered via
      `ContradictionDetector.register_validator(...)`.
    * Each validator receives the motion text and returns a list of
      `Contradiction` objects for its fact type (dates, locations,
      relationships, etc.).

See `register_validator` docstring for guidance on adding new fact-type
validators.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    from writer_agents.code.validation.fact_graph_query import FactGraphQuery
except Exception:  # pragma: no cover - optional dependency
    FactGraphQuery = None  # type: ignore[misc]

logger = logging.getLogger(__name__)


_CITIZENSHIP_LABELS: Dict[str, Tuple[str, ...]] = {
    "us": ("united states", "u.s.", "us", "usa", "american", "america"),
    "prc": ("prc", "people's republic of china", "china", "chinese"),
    "hong_kong": ("hong kong", "hksar"),
}

_CITIZENSHIP_CANONICAL: Dict[str, str] = {
    "us": "US citizen",
    "prc": "PRC citizen",
    "hong_kong": "Hong Kong resident",
}


@dataclass
class Contradiction:
    claim: str
    contradiction_type: str
    severity: str
    location: str
    source_evidence: Optional[str] = None
    fact_type: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "contradiction_type": self.contradiction_type,
            "severity": self.severity,
            "location": self.location,
            "source_evidence": self.source_evidence,
            "fact_type": self.fact_type,
        }


class ContradictionDetector:
    """Detect contradictions between a motion and trusted sources."""

    def __init__(
        self,
        source_docs_dir: Optional[Path] = None,
        lawsuit_facts_db: Optional[Path] = None,
        fact_registry: Optional[Dict[str, Any]] = None,
        knowledge_graph: Optional[Any] = None,
        fact_graph_query: Optional["FactGraphQuery"] = None,
    ) -> None:
        self.source_docs_dir = Path(source_docs_dir).resolve() if source_docs_dir else None
        self.source_documents = self._load_source_documents()
        self.fact_registry = fact_registry or self._load_fact_registry(lawsuit_facts_db)
        self.fact_query = fact_graph_query
        if not self.fact_query and knowledge_graph and FactGraphQuery:
            db_path = Path(lawsuit_facts_db) if lawsuit_facts_db else None
            try:
                self.fact_query = FactGraphQuery(knowledge_graph, fact_registry_db=db_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to initialize FactGraphQuery: %s", exc)
                self.fact_query = None
        self._validators: Dict[str, List[Callable[[str], List[Contradiction]]]] = {}
        # Register built-in validators
        self.register_validator("citizenship", self._validate_citizenship_claims)
        if self.fact_query:
            self.register_validator("date", self._validate_date_claims)
            self.register_validator("allegation", self._validate_allegation_claims)
            self.register_validator("timeline_event", self._validate_timeline_claims)

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#
    def detect_contradictions(self, motion_text: str) -> List[Dict[str, Any]]:
        """Return structured contradiction reports for the motion text."""
        if not isinstance(motion_text, str):
            raise TypeError("motion_text must be a string")

        contradictions: List[Contradiction] = []
        for fact_type, validators in self._validators.items():
            for validator in validators:
                try:
                    violations = validator(motion_text)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[FACTS] %s validator failed: %s", fact_type, exc)
                    continue
                contradictions.extend(violations)

        return [item.to_dict() for item in contradictions]

    def register_validator(
        self,
        fact_type: str,
        validator: Callable[[str], List[Contradiction]],
    ) -> None:
        """
        Register a contradiction validator for a specific fact type.

        Args:
            fact_type: Logical group for the validator (e.g., "citizenship",
                       "dates", "locations").
            validator: Callable that accepts the motion text and returns a list
                       of `Contradiction` instances.

        Example:
            detector.register_validator(
                "dates",
                lambda text: _my_date_validator(detector, text)
            )
        """
        if not fact_type:
            raise ValueError("fact_type is required")
        if not callable(validator):
            raise TypeError("validator must be callable")

        self._validators.setdefault(fact_type, []).append(validator)

    # ------------------------------------------------------------------#
    # Loaders
    # ------------------------------------------------------------------#
    def _load_source_documents(self) -> Dict[str, str]:
        if not self.source_docs_dir or not self.source_docs_dir.exists():
            return {}

        docs: Dict[str, str] = {}
        for path in self.source_docs_dir.rglob("*.txt"):
            try:
                docs[str(path)] = path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.debug("Could not read %s: %s", path, exc)
        return docs

    def _load_fact_registry(self, db_path: Optional[Path]) -> Dict[str, Any]:
        if not db_path:
            return {}

        db_path = Path(db_path)
        if not db_path.exists():
            logger.debug("Fact registry %s does not exist", db_path)
            return {}

        if db_path.suffix.lower() in {".json", ".jsonl"}:
            try:
                return json.loads(db_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to load fact registry JSON %s: %s", db_path, exc)
                return {}

        # Best-effort sqlite loader
        registry: Dict[str, Any] = {}
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT fact_type, fact_value FROM fact_registry"
            )
            for fact_type, fact_value in cursor.fetchall():
                registry[str(fact_type)] = fact_value
        except sqlite3.Error as exc:
            logger.debug("Fact registry sqlite load failed: %s", exc)
            return {}
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return registry

    # ------------------------------------------------------------------#
    # Built-in validators
    # ------------------------------------------------------------------#
    def _validate_citizenship_claims(self, motion_text: str) -> List[Contradiction]:
        sentences = self._split_sentences(motion_text)
        canonical_fact = self._get_canonical_citizenship_fact()
        contradictions: List[Contradiction] = []

        for sentence in sentences:
            claim = self._extract_citizenship_claim(sentence)
            if not claim:
                continue

            location = sentence.strip()
            label = claim["label"]
            if canonical_fact:
                if label and label != canonical_fact:
                    contradictions.append(
                        Contradiction(
                            claim=location,
                            contradiction_type="DIRECT_CONTRADICTION",
                            severity="critical",
                            location=location,
                            source_evidence=_CITIZENSHIP_CANONICAL.get(canonical_fact, canonical_fact),
                            fact_type="citizenship",
                        )
                    )
                continue

            supported = self._citizenship_claim_supported(label)
            if supported:
                continue

            contradiction_type = "INFERENCE" if "home country" in location.lower() else "HALLUCINATION"
            severity = "warning" if contradiction_type == "INFERENCE" else "critical"
            contradictions.append(
                Contradiction(
                    claim=location,
                    contradiction_type=contradiction_type,
                    severity=severity,
                    location=location,
                    source_evidence=None,
                    fact_type="citizenship",
                )
            )

        return contradictions

    def _get_canonical_citizenship_fact(self) -> Optional[str]:
        raw_value: Optional[str] = None
        if self.fact_query:
            graph_values = self._graph_fact_values("citizenship")
            if graph_values:
                raw_value = graph_values[0]
        if raw_value is None:
            raw_value = self.fact_registry.get("citizenship")
        if not raw_value or not isinstance(raw_value, str):
            return None
        return self._normalize_citizenship_value(raw_value)

    def _citizenship_claim_supported(self, label: Optional[str]) -> Optional[str]:
        if not label or not self.source_documents:
            return None

        phrases = _CITIZENSHIP_LABELS.get(label, ())
        if not phrases:
            return None

        for doc_text in self.source_documents.values():
            lowered = doc_text.lower()
            for phrase in phrases:
                if phrase in lowered and "citizen" in lowered:
                    return phrase
        return None

    def _extract_citizenship_claim(self, sentence: str) -> Optional[Dict[str, Any]]:
        lowered = sentence.lower()
        if not any(keyword in lowered for keyword in ("citizen", "national", "home country")):
            return None

        for label, aliases in _CITIZENSHIP_LABELS.items():
            if any(alias in lowered for alias in aliases):
                return {"label": label, "sentence": sentence.strip()}

        if "home country" in lowered:
            return {"label": None, "sentence": sentence.strip()}
        return None

    def _graph_fact_values(self, fact_type: str) -> List[str]:
        if not self.fact_query:
            return []
        try:
            facts = self.fact_query.get_all_facts_by_type(fact_type)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to query KnowledgeGraph facts (%s): %s", fact_type, exc)
            return []
        values: List[str] = []
        for fact in facts:
            value = fact.get("value") or fact.get("fact_value")
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
        return values

    def _validate_graph_fact_negations(
        self,
        motion_text: str,
        fact_type: str,
        severity: str = "warning",
    ) -> List[Contradiction]:
        if not self.fact_query:
            return []
        contradictions: List[Contradiction] = []
        motion_lower = motion_text.lower()
        for fact_value in self._graph_fact_values(fact_type):
            normalized_value = fact_value.lower()
            if not normalized_value:
                continue
            negation_phrases = (
                f"not {normalized_value}",
                f"no {normalized_value}",
                f"never {normalized_value}",
            )
            if any(phrase in motion_lower for phrase in negation_phrases):
                contradictions.append(
                    Contradiction(
                        claim=fact_value,
                        contradiction_type="DIRECT_CONTRADICTION",
                        severity=severity,
                        location=fact_value,
                        source_evidence=fact_value,
                        fact_type=fact_type,
                    )
                )
        return contradictions

    def _validate_date_claims(self, motion_text: str) -> List[Contradiction]:
        return self._validate_graph_fact_negations(motion_text, "date")

    def _validate_allegation_claims(self, motion_text: str) -> List[Contradiction]:
        return self._validate_graph_fact_negations(motion_text, "allegation")

    def _validate_timeline_claims(self, motion_text: str) -> List[Contradiction]:
        return self._validate_graph_fact_negations(motion_text, "timeline_event")

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        return re.split(r'(?<=[.!?])\s+', text)

    @staticmethod
    def _normalize_citizenship_value(value: str) -> Optional[str]:
        lowered = value.lower()
        for label, aliases in _CITIZENSHIP_LABELS.items():
            if any(alias in lowered for alias in aliases):
                return label
        if "not a prc" in lowered or "not prc" in lowered:
            return "us"
        return None
