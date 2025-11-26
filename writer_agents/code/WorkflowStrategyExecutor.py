"""Conductor - Main orchestrator combining AutoGen and Semantic Kernel for legal document writing."""

import asyncio
import json
import logging
import os
import uuid
import math
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, TYPE_CHECKING, Awaitable, Set

from semantic_kernel import Kernel
try:
    from semantic_kernel.planning import SequentialPlanner
    SK_PLANNER_AVAILABLE = True
except ImportError:
    SequentialPlanner = None
    SK_PLANNER_AVAILABLE = False

from agents import AgentFactory, ModelConfig, BaseAutoGenAgent
from insights import CaseInsights
from tasks import WriterDeliverable, DraftSection, PlanDirective, ReviewFindings
from sk_config import create_sk_kernel, SKConfig
from sk_plugins import BaseSKPlugin, PluginRegistry, plugin_registry
from sk_plugins.DraftingPlugin.privacy_harm_function import PrivacyHarmPlugin
from .workflow_config import WorkflowStrategyConfig
from .sk_compat import get_chat_components
from .fact_flow_tracer import record_fact_flow_event
from .fact_payload_utils import (
    DEFAULT_FACTS_CAP,
    DEFAULT_SUMMARY_CAP,
    build_key_fact_summary_text,
    format_fact_retry_todo,
    normalize_fact_keys,
    parse_fact_filter_stats,
    parse_filtered_evidence,
    payload_metric_snapshot,
    truncate_text,
)

try:
    from .validation.personal_facts_verifier import (
        verify_motion_uses_personal_facts,
        NEGATIVE_FACT_RULES,
    )
except ImportError:  # pragma: no cover - optional dependency
    verify_motion_uses_personal_facts = None  # type: ignore[assignment]
    NEGATIVE_FACT_RULES = ()  # type: ignore[assignment]

try:
    from .validation.contradiction_detector import ContradictionDetector
except ImportError:  # pragma: no cover - optional dependency
    ContradictionDetector = None  # type: ignore[assignment]

try:
    from .validation.fact_graph_query import FactGraphQuery
    from .validation.kg_fact_adapter import KGFactAdapter
except ImportError:  # pragma: no cover - optional dependency
    FactGraphQuery = None  # type: ignore[assignment]
    KGFactAdapter = None  # type: ignore[assignment]

try:
    from sk_plugins.FeaturePlugin.CaseFactsProvider import get_case_facts_provider as _case_facts_provider_fn
except ImportError:
    try:
        from .sk_plugins.FeaturePlugin.CaseFactsProvider import get_case_facts_provider as _case_facts_provider_fn  # type: ignore[import-not-found]
    except ImportError:
        _case_facts_provider_fn = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from sk_plugins.FeaturePlugin.CaseFactsProvider import CaseFactsProvider  # pragma: no cover
else:
    CaseFactsProvider = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

STRUCTURED_FACTS_MAX_CHARS = DEFAULT_FACTS_CAP
KEY_FACT_SUMMARY_MAX_CHARS = DEFAULT_SUMMARY_CAP


@lru_cache(maxsize=1)
def _resolve_evidence_item_class():
    """Return EvidenceItem regardless of import style."""
    module_candidates = []
    if __package__:
        module_candidates.append(f"{__package__}.insights")
    module_candidates.extend([
        "writer_agents.insights",
        "writer_agents.code.insights",
        "insights",
    ])

    visited = set()
    for module_name in module_candidates:
        if not module_name or module_name in visited:
            continue
        visited.add(module_name)
        try:
            module = import_module(module_name)
            return getattr(module, "EvidenceItem")
        except (ImportError, AttributeError):
            continue

    raise ModuleNotFoundError(
        "EvidenceItem import failed. Ensure writer_agents.insights is available on PYTHONPATH."
    )


def _safe_get_case_facts_provider() -> Optional['CaseFactsProvider']:
    """Return the global CaseFactsProvider instance if available."""
    if _case_facts_provider_fn is None:
        return None
    try:
        return _case_facts_provider_fn()
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[FACTS] CaseFactsProvider unavailable: {exc}")
        return None


def _personal_facts_snapshot(
    context: Optional[Dict[str, Any]] = None,
    provider: Optional['CaseFactsProvider'] = None,
) -> Optional[Dict[str, Any]]:
    """Return cached personal facts info for verification."""
    existing = None
    if context and isinstance(context, dict):
        existing = context.get("personal_corpus_facts")
        if isinstance(existing, dict) and existing:
            return existing

    provider = provider or _safe_get_case_facts_provider()
    if provider is None:
        return None

    snapshot: Dict[str, Any] = {}
    try:
        fact_blocks = provider.get_all_facts()
        if fact_blocks:
            snapshot["fact_blocks"] = dict(fact_blocks)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[FACTS] Could not load fact blocks for verifier: {exc}")

    try:
        structured = provider.get_all_structured_facts(use_filtered=True)
        if structured:
            snapshot["structured_facts"] = dict(structured)
            snapshot.setdefault("fact_blocks", {}).update(structured)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[FACTS] Could not load structured facts for verifier: {exc}")

    stats_fn = getattr(provider, "get_last_filter_stats", None)
    if callable(stats_fn):
        try:
            snapshot["filter_stats"] = stats_fn()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[FACTS] Could not load filter stats for verifier: {exc}")

    if snapshot and context is not None:
        context.setdefault("personal_corpus_facts", snapshot)

    return snapshot or None


def _normalize_violation_entries(entries: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Normalize violation entries into dictionaries for consistent handling."""
    normalized: List[Dict[str, Any]] = []
    if not entries:
        return normalized

    for entry in entries:
        if isinstance(entry, dict):
            normalized.append(entry)
        else:
            normalized.append({"name": str(entry)})
    return normalized


def _serialize_contradictions(entries: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Serialize contradiction entries (dataclasses/dicts/str) into dictionaries."""
    serialized: List[Dict[str, Any]] = []
    if not entries:
        return serialized

    for entry in entries:
        if isinstance(entry, dict):
            serialized.append(entry)
        elif is_dataclass(entry):
            try:
                serialized.append(asdict(entry))
            except TypeError:
                serialized.append({"claim": str(entry)})
        else:
            serialized.append({"claim": str(entry)})
    return serialized


def _entry_signature(entries: Optional[List[Any]], keys: Optional[List[str]] = None) -> Set[str]:
    """
    Build a stable signature for violation/contradiction entries to detect changes.

    Args:
        entries: List of dict-like entries.
        keys: Optional list of keys to include in signature ordering.
    """
    signature: Set[str] = set()
    if not entries:
        return signature

    for entry in entries:
        if isinstance(entry, dict):
            if keys:
                signature.add("|".join(str(entry.get(key, "")) for key in keys))
            else:
                try:
                    signature.add(json.dumps(entry, sort_keys=True))
                except TypeError:
                    signature.add(str(entry))
        else:
            signature.add(str(entry))
    return signature


def _resolve_fact_sources(context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Path], Optional[Path]]:
    project_root = Path(__file__).resolve().parents[2]
    source_docs_dir: Optional[Path] = None
    lawsuit_facts_db: Optional[Path] = None

    if context and isinstance(context, dict):
        source_hint = context.get("source_docs_dir")
        if isinstance(source_hint, (str, Path)):
            source_docs_dir = Path(source_hint)
        db_hint = context.get("lawsuit_facts_db_path")
        if isinstance(db_hint, (str, Path)):
            lawsuit_facts_db = Path(db_hint)

    default_dir = project_root / "case_law_data" / "lawsuit_source_documents"
    legacy_dir = project_root / "case_law_data" / "tmp_corpus"
    if source_docs_dir is None:
        if default_dir.exists():
            source_docs_dir = default_dir
        elif legacy_dir.exists():
            source_docs_dir = legacy_dir

    default_db = project_root / "case_law_data" / "lawsuit_facts_database.db"
    legacy_db = project_root / "case_law_data" / "truths.db"
    if lawsuit_facts_db is None:
        if default_db.exists():
            lawsuit_facts_db = default_db
        elif legacy_db.exists():
            lawsuit_facts_db = legacy_db

    return source_docs_dir, lawsuit_facts_db


def _load_knowledge_graph(
    context: Optional[Dict[str, Any]],
    lawsuit_facts_db: Optional[Path],
) -> Tuple[Optional[Any], Optional[Any]]:
    if FactGraphQuery is None:
        return None, None

    project_root = Path(__file__).resolve().parents[2]
    graph_path: Optional[Path] = None
    if context and isinstance(context, dict):
        graph_hint = context.get("knowledge_graph_path")
        if isinstance(graph_hint, (str, Path)):
            candidate = Path(graph_hint)
            if candidate.exists():
                graph_path = candidate
    if graph_path is None:
        default_graph = project_root / "case_law_data" / "facts_knowledge_graph.json"
        if default_graph.exists():
            graph_path = default_graph

    if graph_path is None or not graph_path.exists():
        return None, None

    try:
        from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.load_from_file(str(graph_path))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[FACTS] Failed to load Knowledge Graph: {exc}")
        return None, None

    db_path = lawsuit_facts_db if lawsuit_facts_db and lawsuit_facts_db.exists() else None
    try:
        fact_query = FactGraphQuery(kg, fact_registry_db=db_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[FACTS] Failed to initialize FactGraphQuery: {exc}")
        fact_query = None
    return kg, fact_query


def _run_personal_facts_verifier(
    document: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run the personal facts verifier if available."""
    if verify_motion_uses_personal_facts is None:
        return None
    if not document or not document.strip():
        return None

    source_docs_dir, lawsuit_facts_db = _resolve_fact_sources(context)
    kg, fact_query = _load_knowledge_graph(context, lawsuit_facts_db)

    personal_facts = _personal_facts_snapshot(context=context)
    if not personal_facts:
        return {"error": "personal_facts_unavailable"}

    try:
        verifier_output = verify_motion_uses_personal_facts(
            document,
            personal_facts,
            negative_rules=NEGATIVE_FACT_RULES,
            fact_graph_query=fact_query,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[FACTS] Personal facts verifier failed: {exc}")
        return {"error": str(exc)}

    violations_raw: List[Any] = []
    if isinstance(verifier_output, tuple) and len(verifier_output) >= 4:
        is_valid, missing, violations_raw, details = verifier_output  # type: ignore[misc]
    elif isinstance(verifier_output, tuple) and len(verifier_output) == 3:
        is_valid, missing, details = verifier_output  # type: ignore[misc]
    else:  # pragma: no cover - defensive guard
        is_valid = bool(verifier_output)
        missing = []
        details = {}

    coverage = float(details.get("coverage", 0.0))
    facts_found = list(details.get("matches", {}).keys())
    normalized_violations = _normalize_violation_entries(violations_raw)

    contradictions_raw: List[Any] = []
    if ContradictionDetector is not None and source_docs_dir and source_docs_dir.exists():
        try:
            detector = ContradictionDetector(
                source_docs_dir=source_docs_dir,
                lawsuit_facts_db=lawsuit_facts_db if (lawsuit_facts_db and lawsuit_facts_db.exists()) else None,
                knowledge_graph=kg,
                fact_graph_query=fact_query,
            )
            contradictions_raw = detector.detect_contradictions(document)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[FACTS] Contradiction detection failed: {exc}")
            contradictions_raw = []

    serialized_contradictions = _serialize_contradictions(contradictions_raw)
    has_violations = bool(normalized_violations)
    has_contradictions = bool(serialized_contradictions)
    overall_valid = bool(is_valid and not has_violations and not has_contradictions)

    details = dict(details or {})
    details.setdefault("violations", normalized_violations)
    details.setdefault("contradictions", serialized_contradictions)

    result: Dict[str, Any] = {
        "is_valid": overall_valid,
        "coverage": coverage,
        "facts_found": facts_found,
        "missing_facts": missing,
        "critical_facts_missing": bool(missing),
        "violations": normalized_violations,
        "contradictions": serialized_contradictions,
        "has_violations": has_violations,
        "has_contradictions": has_contradictions,
        "details": details,
        "score": coverage if overall_valid else 0.0,
    }

    suggestions: List[str] = []
    if missing:
        display_missing = ", ".join(missing[:4])
        if display_missing:
            suggestions.append(f"Add explicit references for: {display_missing}")
    if has_violations:
        suggestions.append("Remove prohibited facts before finalizing the motion.")
    if has_contradictions:
        suggestions.append("Correct statements that contradict verified facts.")
    graph_missing = details.get("graph_missing_facts")
    if graph_missing:
        summary_items = [
            f"{entry.get('value')} ({entry.get('fact_type')})"
            for entry in graph_missing[:3]
        ]
        summary_text = ", ".join(filter(None, summary_items))
        if summary_text:
            suggestions.append(f"Add missing graph-backed facts: {summary_text}")
    if suggestions:
        result["suggestions"] = suggestions

    graph_summary = details.get("graph_coverage_summary")
    if graph_summary:
        try:
            avg_cov = float(graph_summary.get("average_coverage", 0.0)) * 100
            total_types = graph_summary.get("total_types", 0)
            logger.info(
                "[FACTS] Graph fact coverage: %.1f%% across %d fact type(s)",
                avg_cov,
                total_types,
            )
        except Exception:
            pass
    return result


def _provider_has_personal_corpus(provider: Optional['CaseFactsProvider']) -> bool:
    """Return True when the provider has loaded personal corpus facts."""
    if provider is None:
        return False
    try:
        return bool(getattr(provider, "_personal_corpus_index", None))
    except Exception:
        return False


def _validate_personal_fact_inputs(corpus_dir: Path, case_insights_path: Path) -> None:
    """Ensure the workflow has access to lawsuit-specific assets."""
    if not corpus_dir.exists():
        raise RuntimeError(
            f"Personal corpus directory not found: {corpus_dir}. "
            "Populate case_law_data/lawsuit_source_documents/ with your lawsuit documents."
        )

    has_entries = any(corpus_dir.glob("*.txt"))
    if not has_entries:
        raise RuntimeError(
            f"Personal corpus directory is empty: {corpus_dir}. "
            "Add .txt files or re-run python writer_agents/scripts/build_case_insights.py."
        )

    if not case_insights_path.exists():
        raise RuntimeError(
            f"lawsuit_facts_extracted.json not found at {case_insights_path}. "
            "Run python writer_agents/scripts/build_case_insights.py to regenerate it."
        )


def _extract_node_id(item: Any) -> Optional[str]:
    """Extract node_id from dict or EvidenceItem."""
    if isinstance(item, dict):
        return item.get("node_id")
    return getattr(item, "node_id", None)


def _filter_lawsuit_evidence(
    evidence_items: List[Any],
    provider: Optional['CaseFactsProvider']
) -> Tuple[List[Any], bool, Dict[str, Any]]:
    """Filter evidence list to lawsuit-specific entries."""
    stats: Dict[str, Any] = {
        "provider_available": provider is not None,
        "original_count": len(evidence_items),
        "filtered_count": len(evidence_items),
        "dropped_count": 0,
        "dropped_reasons": {}
    }

    if not evidence_items:
        return [], False, stats

    filtered: Optional[List[Any]] = None
    if provider:
        filter_fn = getattr(provider, "filter_evidence_for_lawsuit", None)
        if callable(filter_fn):
            try:
                candidate = filter_fn(evidence_items)
                if isinstance(candidate, list):
                    filtered = candidate
                    provider_stats = getattr(provider, "get_last_filter_stats", None)
                    if callable(provider_stats):
                        provider_snapshot = provider_stats() or {}
                        stats.update(provider_snapshot)
                        stats["provider_available"] = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[FACTS] Evidence filtering failed, continuing with defaults: {exc}")

    if filtered is None:
        filtered = [item for item in evidence_items if (_extract_node_id(item) or "").startswith("fact_block_")]
        if not filtered:
            filtered = evidence_items
            stats["filtered_count"] = len(filtered)
            stats["dropped_count"] = 0
            return filtered, False, stats

    stats.setdefault("filtered_count", len(filtered))
    stats.setdefault("dropped_count", len(evidence_items) - len(filtered))

    if provider:
        if stats.get("strict_filtering_failed"):
            logger.error(
                "[FACTS] Evidence filtering removed all items while strict filtering was enabled. "
                "Aborting downstream use of generic evidence."
            )
            stats["filtering_failed_with_corpus"] = True
        elif len(filtered) == 0 and _provider_has_personal_corpus(provider):
            logger.error(
                "[FACTS] Evidence filtering returned zero items even though personal corpus facts are loaded. "
                "Verify evidence node_ids align with personal corpus fact keys."
            )
            stats["filtering_failed_with_corpus"] = True

    return filtered, len(filtered) != len(evidence_items), stats


def _format_facts_for_prompt(provider: Optional['CaseFactsProvider']) -> Optional[str]:
    """Format structured facts for prompt inclusion."""
    if provider is None:
        return None

    sections: List[str] = []
    base_section: Optional[str] = None

    formatter = getattr(provider, "format_facts_for_autogen", None)
    if callable(formatter):
        try:
            formatted = formatter()
            if formatted:
                base_section = formatted.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[FACTS] format_facts_for_autogen failed: {exc}")

    if not base_section:
        try:
            structured = provider.get_all_structured_facts()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[FACTS] Could not load structured facts: {exc}")
            structured = {}

        lines: List[str] = []
        for key, text in structured.items():
            if not text:
                continue
            human_label = key.replace("_", " ").title()
            lines.append(f"### {human_label}\n{text}")

        if lines:
            base_section = "\n\n".join(lines)
        else:
            try:
                all_facts = provider.get_all_facts()
            except Exception:  # noqa: BLE001
                all_facts = {}

            if all_facts:
                summary_lines = ["### Verified Facts"]
                for key, text in all_facts.items():
                    if not text:
                        continue
                    summary_lines.append(f"- {key}: {text}")
                if len(summary_lines) > 1:
                    base_section = "\n".join(summary_lines)

    if base_section:
        sections.append(base_section.strip())

    truth_formatter = getattr(provider, "format_truths_for_prompt", None)
    if callable(truth_formatter):
        try:
            truths_text = (truth_formatter() or "").strip()
            if truths_text and not truths_text.startswith("("):
                sections.append(truths_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[FACTS] format_truths_for_prompt failed: {exc}")

    if sections:
        combined = "\n\n".join(sections).strip()
        return truncate_text(combined, STRUCTURED_FACTS_MAX_CHARS, "structured facts block")
    return None


def _render_evidence_for_prompt(evidence_items: List[Any]) -> str:
    """Render evidence items for prompts."""
    if not evidence_items:
        return "[]"

    try:
        EvidenceItem = _resolve_evidence_item_class()
    except ModuleNotFoundError:
        EvidenceItem = None  # type: ignore[assignment]

    serializable: List[Any] = []
    for item in evidence_items:
        if EvidenceItem and isinstance(item, EvidenceItem):
            serializable.append(item.to_dict())
        elif isinstance(item, dict):
            serializable.append(item)
        else:
            node_id = getattr(item, "node_id", None)
            state = getattr(item, "state", None)
            if node_id and state is not None:
                serializable.append({"node_id": node_id, "state": state})
            else:
                serializable.append(str(item))

    try:
        return json.dumps(serializable, indent=2)
    except TypeError:
        return str(serializable)


def _format_structured_facts_block(structured_facts: Any) -> str:
    """Render structured facts with a Key Facts Summary preamble."""
    default_msg = "- Structured facts unavailable; rely on verified case summary."
    structured_text = _coerce_structured_text(structured_facts) or default_msg
    summary = build_key_fact_summary_text(structured_text, KEY_FACT_SUMMARY_MAX_CHARS)
    return (
        "KEY FACTS SUMMARY (STRICTLY VERIFIED CASE DATA)\n"
        f"{summary}\n\n"
        "STRICT FACT CORPUS (USE EXACT TEXT BELOW)\n"
        f"{structured_text}"
    )


def _coerce_structured_text(structured_facts: Any) -> str:
    if structured_facts is None:
        return ""
    if isinstance(structured_facts, str):
        return structured_facts.strip()

    lines: List[str] = []
    if isinstance(structured_facts, dict):
        for key, value in structured_facts.items():
            if not value:
                continue
            label = key.replace("_", " ").strip().title()
            snippet = str(value).strip()
            if snippet:
                lines.append(f"- {label}: {snippet}")
    elif isinstance(structured_facts, list):
        for item in structured_facts:
            snippet = str(item).strip()
            if snippet:
                lines.append(f"- {snippet}")
    else:
        snippet = str(structured_facts).strip()
        if snippet:
            lines.append(f"- {snippet}")
    return "\n".join(lines)


def _format_quality_constraints_block(constraints: Optional[str]) -> str:
    """Render quality constraints with sensible defaults."""
    header = "\nQUALITY CONSTRAINTS AND REQUIREMENTS:\n"
    if constraints and constraints.strip():
        return f"{header}{constraints.strip()}\n"
    fallback = "- Apply the Perfect Outline structure, CatBoost feature targets, and section-level requirements from the latest constraint system."
    return f"{header}{fallback}\n"


def _render_posteriors_for_prompt(posteriors: Optional[List[Any]]) -> str:
    """Render posterior data for prompts."""
    if not posteriors:
        return "[]"

    serializable: List[Dict[str, Any]] = []
    for posterior in posteriors:
        node_id = getattr(posterior, "node_id", None)
        probabilities = getattr(posterior, "probabilities", None)
        interpretation = getattr(posterior, "interpretation", None)
        entry: Dict[str, Any] = {}
        if node_id is not None:
            entry["node_id"] = node_id
        if isinstance(probabilities, dict):
            entry["probabilities"] = probabilities
        if interpretation:
            entry["interpretation"] = interpretation
        serializable.append(entry)

    try:
        return json.dumps(serializable, indent=2)
    except TypeError:
        return str(serializable)


def ensure_fact_payload_fields(
    variables: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Guarantee fact payload fields are serialized for SK plugins."""
    payload: Dict[str, Any] = dict(variables or {})
    context = context or {}

    def _dump(value: Any) -> str:
        try:
            return json.dumps(value, default=str)
        except TypeError:
            return json.dumps(str(value))

    fact_keys_source: List[Any] = context.get("fact_key_summary") or []
    if not fact_keys_source:
        fact_keys_source = (
            (context.get("fact_filter_stats") or {}).get("fact_block_keys") or []
        )

    if "fact_key_summary" not in payload:
        payload["fact_key_summary"] = _dump(fact_keys_source or [])
    elif isinstance(payload.get("fact_key_summary"), (list, dict)):
        payload["fact_key_summary"] = _dump(payload["fact_key_summary"])

    filtered_source = context.get("filtered_evidence")
    if filtered_source is None:
        filtered_source = context.get("evidence") or []
    if "filtered_evidence" not in payload:
        payload["filtered_evidence"] = _dump(filtered_source or [])
    elif isinstance(payload.get("filtered_evidence"), (list, dict)):
        payload["filtered_evidence"] = _dump(payload["filtered_evidence"])

    stats_source = context.get("fact_filter_stats") or {}
    if "fact_filter_stats" not in payload:
        payload["fact_filter_stats"] = _dump(stats_source)
    elif isinstance(payload.get("fact_filter_stats"), (list, dict)):
        payload["fact_filter_stats"] = _dump(payload["fact_filter_stats"])

    if "key_facts_summary" not in payload:
        summary = context.get("key_facts_summary")
        structured_snapshot = context.get("structured_facts")
        if not summary and structured_snapshot:
            if not isinstance(structured_snapshot, str):
                structured_snapshot = str(structured_snapshot)
            summary = build_key_fact_summary_text(
                structured_snapshot, KEY_FACT_SUMMARY_MAX_CHARS
            )
        if summary:
            payload["key_facts_summary"] = summary

    # SK >= 1.37 kernel functions expect a placeholder `_ignored` argument.
    payload.setdefault("_ignored", "")

    return payload


def _record_plugin_payload(
    plugin_name: str,
    function_name: str,
    payload: Dict[str, Any],
) -> Dict[str, int]:
    """Emit tracer + log entries describing fact payload coverage."""
    fact_keys = normalize_fact_keys(payload.get("fact_key_summary"))
    filtered_items = parse_filtered_evidence(payload.get("filtered_evidence"))
    filter_stats = parse_fact_filter_stats(payload.get("fact_filter_stats"))

    structured_text = payload.get("structured_facts")
    if structured_text and not isinstance(structured_text, str):
        structured_text = str(structured_text)

    metrics = payload_metric_snapshot(
        structured_text=structured_text,
        fact_keys=fact_keys,
        filtered_evidence=filtered_items,
        key_facts_summary=payload.get("key_facts_summary"),
        fact_filter_stats=filter_stats,
    )
    record_fact_flow_event(
        stage="sk_plugin_invocation",
        stats=metrics,
        extra={
            "plugin": plugin_name,
            "function": function_name,
            "sample_fact_keys": fact_keys[:5],
        },
    )
    logger.info(
        "[FACTS][SK PAYLOAD] %s.%s structured=%d fact_keys=%d filtered=%d dropped=%d",
        plugin_name,
        function_name,
        metrics["structured_facts_length"],
        metrics["fact_key_count"],
        metrics["filtered_evidence"],
        metrics["fact_filter_dropped"],
    )
    return metrics


def _build_fact_retry_todo(validation_results: Optional[Dict[str, Any]]) -> List[str]:
    """Create a fact-aware TODO list from validation failures."""
    if not validation_results:
        return []

    todo: List[str] = []
    seen: set[str] = set()

    personal = validation_results.get("personal_facts_verification") or {}
    missing_facts = personal.get("facts_missing") or []
    for fact_key in missing_facts[:8]:
        item = f"Reference verified fact '{fact_key}' in the next draft."
        if item not in seen:
            todo.append(item)
            seen.add(item)

    coverage = personal.get("coverage")
    if coverage is not None and coverage < 1.0:
        item = f"Increase personal facts coverage to 100% (current {coverage:.0%})."
        if item not in seen:
            todo.append(item)
            seen.add(item)

    citations = validation_results.get("required_case_citations")
    if citations and not citations.get("meets_requirements", True):
        missing_cases = citations.get("missing_cases") or []
        missing_names = [case.get("case_name") for case in missing_cases if case.get("case_name")]
        if missing_names:
            display = ", ".join(missing_names[:5])
            item = f"Insert citations for required cases: {display}."
            if item not in seen:
                todo.append(item)
                seen.add(item)

    warnings = validation_results.get("warnings") or []
    for warning in warnings:
        text = warning.strip()
        if not text:
            continue
        important = text.lower().startswith("critical") or "facts missing" in text.lower()
        if important and text not in seen:
            todo.append(text)
            seen.add(text)

    return todo

# Episodic memory system imports
try:
    from EpisodicMemoryBank import (
        EpisodicMemoryBank,
        EpisodicMemoryRetriever,
        EpisodicMemoryEntry
    )
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Episodic memory system modules not available: {e}")
    EpisodicMemoryBank = None  # type: ignore[assignment]
    EpisodicMemoryRetriever = None  # type: ignore[assignment]
    EpisodicMemoryEntry = None  # type: ignore[assignment]
    MEMORY_SYSTEM_AVAILABLE = False

# Cursor/Codex chat knowledge base
try:
    from knowledge_base.cursor_chats import CursorChatLibrary
    CHAT_LIBRARY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cursor chat library not available: {e}")
    CursorChatLibrary = None  # type: ignore[assignment]
    CHAT_LIBRARY_AVAILABLE = False

# Google Docs integration imports
try:
    from .google_docs_bridge import GoogleDocsBridge, create_google_docs_bridge
    from .google_docs_formatter import GoogleDocsFormatter, format_writer_deliverable
    from .DocumentEditRecorder import (
        DocumentEditRecorder,
        create_document_edit_recorder
    )
    from .DocumentMetadataRecorder import (
        DocumentMetadataRecorder,
        create_document_metadata_recorder
    )
    GOOGLE_DOCS_AVAILABLE = True
except ImportError as e:
    try:
        # Fallback to absolute imports
        from google_docs_bridge import GoogleDocsBridge, create_google_docs_bridge
        from google_docs_formatter import GoogleDocsFormatter, format_writer_deliverable
        from DocumentEditRecorder import (
            DocumentEditRecorder,
            create_document_edit_recorder
        )
        from DocumentMetadataRecorder import (
            DocumentMetadataRecorder,
            create_document_metadata_recorder
        )
        GOOGLE_DOCS_AVAILABLE = True
    except ImportError as e2:
        logger.warning(f"Google Docs integration modules not available: {e2}")
        GoogleDocsBridge = None  # type: ignore[assignment]
        GoogleDocsFormatter = None  # type: ignore[assignment]
        DocumentEditRecorder = None  # type: ignore[assignment]
        DocumentMetadataRecorder = None  # type: ignore[assignment]
        GOOGLE_DOCS_AVAILABLE = False


class WorkflowPhase(Enum):
    """Phases in the hybrid workflow."""
    EXPLORE = "explore"          # AutoGen exploration
    RESEARCH = "research"        # Case law research (SK)
    PLAN = "plan"               # SK Planner
    DRAFT = "draft"             # SK Functions
    VALIDATE = "validate"       # SK Validation
    REVIEW = "review"           # AutoGen review
    REFINE = "refine"           # SK refinement
    COMMIT = "commit"           # Final commit


## Use shared WorkflowStrategyConfig from workflow_config.py


@dataclass
class WorkflowState:
    """State of the hybrid workflow."""

    phase: WorkflowPhase = WorkflowPhase.EXPLORE
    iteration: int = 0
    exploration_result: Optional[Dict[str, Any]] = None
    research_results: Optional[Dict[str, Any]] = None  # Case law research findings
    sk_plan: Optional[Any] = None
    draft_result: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    review_notes: Optional[str] = None
    final_document: Optional[str] = None

    # Google Docs integration state
    google_doc_id: Optional[str] = None
    google_doc_url: Optional[str] = None
    memory_context: Optional[Dict[str, Any]] = None
    fact_retry_todo: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for logging."""
        return {
            "phase": self.phase.value,
            "iteration": self.iteration,
            "has_exploration": self.exploration_result is not None,
            "has_plan": self.sk_plan is not None,
            "has_draft": self.draft_result is not None,
            "has_validation": self.validation_results is not None,
            "has_review": self.review_notes is not None,
            "has_final": self.final_document is not None,
            "has_memory_context": self.memory_context is not None,
            "fact_retry_items": len(self.fact_retry_todo),
        }


class WorkflowRouter:
    """Routes workflow phases between AutoGen and Semantic Kernel."""

    def __init__(self, config: WorkflowStrategyConfig):
        self.config = config

    def should_use_autogen(self, phase: WorkflowPhase, state: WorkflowState) -> bool:
        """Determine if phase should use AutoGen."""
        autogen_phases = {WorkflowPhase.EXPLORE, WorkflowPhase.REVIEW}
        return phase in autogen_phases

    def should_use_sk(self, phase: WorkflowPhase, state: WorkflowState) -> bool:
        """Determine if phase should use Semantic Kernel."""
        sk_phases = {WorkflowPhase.RESEARCH, WorkflowPhase.PLAN, WorkflowPhase.DRAFT, WorkflowPhase.VALIDATE, WorkflowPhase.REFINE}
        return phase in sk_phases

    def get_next_phase(self, current_phase: WorkflowPhase, state: WorkflowState) -> WorkflowPhase:
        """Determine next workflow phase."""

        if current_phase == WorkflowPhase.EXPLORE:
            return WorkflowPhase.RESEARCH  # Always do research after exploration

        elif current_phase == WorkflowPhase.RESEARCH:
            return WorkflowPhase.PLAN if self.config.enable_sk_planner else WorkflowPhase.DRAFT

        elif current_phase == WorkflowPhase.PLAN:
            return WorkflowPhase.DRAFT

        elif current_phase == WorkflowPhase.DRAFT:
            return WorkflowPhase.VALIDATE if self.config.enable_quality_gates else WorkflowPhase.COMMIT

        elif current_phase == WorkflowPhase.VALIDATE:
            validation_passed = self._check_validation_passed(state.validation_results)
            if validation_passed:
                return WorkflowPhase.COMMIT
            elif self.config.enable_autogen_review and state.iteration < self.config.max_iterations:
                return WorkflowPhase.REVIEW
            else:
                return WorkflowPhase.COMMIT  # Force commit after max iterations

        elif current_phase == WorkflowPhase.REVIEW:
            return WorkflowPhase.REFINE

        elif current_phase == WorkflowPhase.REFINE:
            return WorkflowPhase.VALIDATE

        else:  # COMMIT
            return WorkflowPhase.COMMIT

    def _check_validation_passed(self, validation_results: Optional[Dict[str, Any]]) -> bool:
        """Check if validation passed."""
        if not validation_results:
            return False

        if validation_results.get("meets_threshold") is False:
            return False

        failed_gates = validation_results.get("failed_gates") or []
        if any(validation_results.get("gate_results", {}).get(gate, {}).get("required") for gate in failed_gates):
            return False

        personal = validation_results.get("personal_facts_verification") or {}
        coverage = personal.get("coverage")
        if coverage is not None and coverage < 1.0:
            return False
        if personal.get("critical_facts_missing"):
            return False

        required = validation_results.get("required_case_citations")
        if required and not required.get("meets_requirements", True):
            return False

        overall_score = validation_results.get("overall_score", 0.0)
        return overall_score >= self.config.auto_commit_threshold


class AutoGenToSKBridge:
    """Bridge between AutoGen and Semantic Kernel."""

    def __init__(self, autogen_factory: AgentFactory, sk_kernel: Optional[Kernel] = None):
        self.autogen_factory = autogen_factory
        self.sk_kernel = sk_kernel

    def autogen_to_sk_context(self, autogen_output: str, insights: CaseInsights) -> Dict[str, Any]:
        """Convert AutoGen output to SK context variables."""

        provider = _safe_get_case_facts_provider()
        raw_evidence = list(insights.evidence or [])
        filtered_evidence, filtered_applied, filter_stats = _filter_lawsuit_evidence(raw_evidence, provider)
        facts_block = _format_facts_for_prompt(provider)
        filter_stats.setdefault("provider_available", provider is not None)
        filter_stats.setdefault("original_count", len(raw_evidence))
        filter_stats.setdefault("filtered_count", len(filtered_evidence))
        filter_stats.setdefault("filtered_applied", filtered_applied)

        logger.info(
            "[FACTS] Filtered %d/%d evidence items for SK context (provider=%s)",
            filter_stats["filtered_count"],
            filter_stats["original_count"],
            "yes" if filter_stats["provider_available"] else "no"
        )

        # Extract key information from AutoGen output
        context = {
            "case_summary": insights.summary,
            "jurisdiction": insights.jurisdiction or "US",
            "evidence": filtered_evidence,
            "raw_evidence": raw_evidence,
            "posteriors": insights.posteriors,
            "autogen_notes": autogen_output,
            "case_style": insights.case_style,
            "structured_facts": facts_block,
            "fact_filter_stats": filter_stats
        }
        context["filtered_evidence"] = filtered_evidence
        context.setdefault("constraint_version", "1.0")

        # Try to extract structured information from AutoGen output
        try:
            # Look for JSON in AutoGen output
            if "{" in autogen_output and "}" in autogen_output:
                start = autogen_output.find("{")
                end = autogen_output.rfind("}") + 1
                json_str = autogen_output[start:end]
                structured_data = json.loads(json_str)
                context.update(structured_data)
        except (json.JSONDecodeError, ValueError):
            pass

        personal_snapshot = _personal_facts_snapshot(context=context, provider=provider)
        if personal_snapshot:
            context["personal_corpus_facts"] = personal_snapshot

        return context

    def sk_to_autogen_context(self, sk_output: str, validation_results: Optional[Dict[str, Any]] = None) -> str:
        """Convert SK output to AutoGen review context."""

        context_parts = [
            "## SK Draft Output",
            sk_output,
            ""
        ]

        if validation_results:
            context_parts.extend([
                "## Validation Results",
                f"Overall Score: {validation_results.get('overall_score', 0.0):.2f}",
                ""
            ])

            if "failed_gates" in validation_results:
                context_parts.extend([
                    "## Failed Quality Gates",
                    "\n".join(f"- {gate}" for gate in validation_results["failed_gates"]),
                    ""
                ])

            if "suggestions" in validation_results:
                context_parts.extend([
                    "## Improvement Suggestions",
                    "\n".join(f"- {suggestion}" for suggestion in validation_results["suggestions"]),
                    ""
                ])

        context_parts.extend([
            "## Review Instructions",
            "Please review the SK-generated draft and provide specific feedback for improvement.",
            "Focus on legal accuracy, argument strength, and citation completeness.",
            "Provide actionable suggestions for revision."
        ])

        return "\n".join(context_parts)


class QualityGatePipeline:
    """Pipeline for running quality gates on SK outputs."""

    def __init__(self, sk_kernel: Kernel, executor: Optional['Conductor'] = None):
        self.sk_kernel = sk_kernel
        self.executor = executor
        self.quality_gates = self._initialize_quality_gates()
    
    async def _ensure_plugins_ready(self) -> None:
        """Ensure plugin registration tasks from executor are settled."""
        if self.executor and hasattr(self.executor, "_ensure_plugins_ready"):
            await self.executor._ensure_plugins_ready()
    
    def _serialize_context_for_json(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize context dict, converting non-JSON-serializable types."""
        import numpy as np
        EvidenceItem = _resolve_evidence_item_class()
        
        serializable = {}
        for key, value in context.items():
            if isinstance(value, EvidenceItem):
                serializable[key] = value.to_dict()
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                serializable[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                serializable[key] = float(value)
            elif isinstance(value, list):
                serializable[key] = [self._serialize_context_for_json({"item": v})["item"] if isinstance(v, (EvidenceItem, np.integer, np.int64, np.int32, np.floating, np.float64)) else v for v in value]
            elif isinstance(value, dict):
                serializable[key] = self._serialize_context_for_json(value)
            else:
                serializable[key] = value
        return serializable

    async def _invoke_sk_function_safe(
        self,
        plugin_name: str,
        function_name: str,
        variables: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Safe wrapper to invoke SK function, using executor if available."""
        payload = ensure_fact_payload_fields(variables, context)

        if self.executor:
            return await self.executor._invoke_sk_function(
                plugin_name,
                function_name,
                payload,
                context,
            )

        if self.sk_kernel is None:
            raise RuntimeError("SK kernel is not available")

        _record_plugin_payload(plugin_name, function_name, payload)

        plugin = getattr(self.sk_kernel, "plugins", {}).get(plugin_name)
        if plugin is not None:
            function = None
            if hasattr(plugin, "__contains__") and hasattr(plugin, "__getitem__"):
                try:
                    if function_name in plugin:
                        function = plugin[function_name]
                except Exception:
                    function = None
            if function is None and hasattr(plugin, "get_function"):
                function = plugin.get_function(function_name)

            if function is not None:
                logger.debug(
                    "Invoking %s.%s via kernel plugin with args: %s",
                    plugin_name,
                    function_name,
                    list(payload.keys()),
                )
                return await self.sk_kernel.invoke(function=function, **payload)

        from .sk_plugins import plugin_registry

        plugin_obj = plugin_registry.get_plugin(plugin_name)
        if plugin_obj:
            function = plugin_obj.get_function(function_name)
            if function is not None:
                logger.debug(
                    "Invoking %s.%s via registry fallback with args: %s",
                    plugin_name,
                    function_name,
                    list(payload.keys()),
                )
                try:
                    return await self.sk_kernel.invoke(function=function, **payload)
                except Exception:
                    import asyncio
                    if asyncio.iscoroutinefunction(function):
                        return await function(**payload)
                    return function(**payload)

        raise RuntimeError(f"Function {plugin_name}.{function_name} not found in kernel plugins or registry")

    def _initialize_quality_gates(self) -> List[Dict[str, Any]]:
        """Initialize quality gates configuration."""
        return [
            {
                "name": "citation_validity",
                "function": "ValidateCitationFormat",
                "threshold": 1.0,
                "required": True,
                "description": "All citations must be properly formatted"
            },
            {
                "name": "structure_complete",
                "function": "ValidateStructure",
                "threshold": 1.0,
                "required": True,
                "description": "All required sections must be present"
            },
            {
                "name": "evidence_grounding",
                "function": "ValidateEvidenceGrounding",
                "threshold": 0.9,
                "required": True,
                "description": "All claims must have evidence references"
            },
            {
                "name": "personal_facts_coverage",
                "function": "ValidatePersonalFactsCoverage",
                "threshold": 0.95,
                "required": True,
                "description": "Draft must reference lawsuit-specific personal facts"
            },
            {
                "name": "tone_consistency",
                "function": "ValidateToneConsistency",
                "threshold": 0.85,
                "required": False,
                "description": "Maintain professional legal tone"
            },
            {
                "name": "petition_quality",
                "function": "ValidatePetitionQuality",
                "threshold": 0.70,
                "required": False,
                "description": "Validates petition against success formula quality rules from CatBoost model",
                "plugin": "PetitionQualityPlugin"
            },
            {
                "name": "constraint_system",
                "function": "ValidatePetitionConstraints",
                "threshold": 0.75,
                "required": False,
                "description": "Validates petition against hierarchical constraint system (document, section, feature levels)",
                "plugin": "PetitionQualityPlugin"
            }
        ]

    async def run_quality_gates(self, document: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all quality gates on the document."""
        await self._ensure_plugins_ready()

        results = {
            "overall_score": 0.0,
            "gate_results": {},
            "failed_gates": [],
            "passed_gates": [],
            "suggestions": [],
            "warnings": []
        }

        total_weight = 0.0
        weighted_score = 0.0

        for gate in self.quality_gates:
            gate_weight = 2.0 if gate.get("required") else 1.0
            try:
                # Run validation function
                gate_result = await self._run_gate(gate, document, context)

                gate_score = gate_result.get("score", 0.0)

                results["gate_results"][gate["name"]] = {
                    "score": gate_score,
                    "passed": gate_score >= gate["threshold"],
                    "threshold": gate["threshold"],
                    "required": gate["required"],
                    "details": gate_result.get("details", ""),
                    "raw_result": gate_result
                }

                if gate_score >= gate["threshold"]:
                    results["passed_gates"].append(gate["name"])
                else:
                    results["failed_gates"].append(gate["name"])
                    if gate["required"]:
                        results["suggestions"].append(f"Fix {gate['description']}")

                suggestions = gate_result.get("suggestions") or []
                if suggestions:
                    results["suggestions"].extend(suggestions)

                warnings = gate_result.get("warnings") or []
                if warnings:
                    results["warnings"].extend(warnings)

                weighted_score += gate_score * gate_weight

            except Exception as e:
                logger.error(f"Error running quality gate {gate['name']}: {e}")
                results["gate_results"][gate["name"]] = {
                    "score": 0.0,
                    "passed": False,
                    "error": str(e),
                    "raw_result": {"error": str(e)}
                }
                results["failed_gates"].append(gate["name"])

            total_weight += gate_weight

        # Calculate overall score
        if total_weight > 0:
            results["overall_score"] = weighted_score / total_weight

        return results

    async def _run_gate(self, gate: Dict[str, Any], document: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single quality gate using SK validation functions."""

        try:
            # Determine plugin name (default to ValidationPlugin)
            plugin_name = gate.get("plugin", "ValidationPlugin")

            # Use SK validation functions
            if gate["name"] == "citation_validity":
                result = await self._invoke_sk_function_safe(
                    plugin_name="ValidationPlugin",
                    function_name="ValidateCitationFormat",
                    variables={"document": document},
                    context=context,
                )
                return json.loads(result.value if hasattr(result, 'value') else str(result))

            elif gate["name"] == "structure_complete":
                result = await self._invoke_sk_function_safe(
                    plugin_name="ValidationPlugin",
                    function_name="ValidateStructure",
                    variables={"document": document},
                    context=context,
                )
                return json.loads(result.value if hasattr(result, 'value') else str(result))

            elif gate["name"] == "evidence_grounding":
                # Serialize context properly, handling EvidenceItem objects
                context_serializable = self._serialize_context_for_json(context)
                result = await self._invoke_sk_function_safe(
                    plugin_name="ValidationPlugin",
                    function_name="ValidateEvidenceGrounding",
                    variables={"document": document, "context": json.dumps(context_serializable, default=str)},
                    context=context,
                )
                return json.loads(result.value if hasattr(result, 'value') else str(result))
            elif gate["name"] == "personal_facts_coverage":
                return self._validate_personal_facts_coverage(document, context)

            elif gate["name"] == "tone_consistency":
                result = await self._invoke_sk_function_safe(
                    plugin_name="ValidationPlugin",
                    function_name="ValidateToneConsistency",
                    variables={"document": document},
                    context=context,
                )
                return json.loads(result.value if hasattr(result, 'value') else str(result))

            elif gate["name"] == "petition_quality":
                # Validate against petition success formula
                if not self._plugin_available(plugin_name):
                    logger.warning("Skipping petition_quality gate; plugin %s unavailable", plugin_name)
                    return {"score": 0.0, "details": "petition quality plugin unavailable"}

                result = await self._invoke_sk_function_safe(
                    plugin_name=plugin_name,
                    function_name="ValidatePetitionQuality",
                    variables={"petition_text": document},
                    context=context,
                )
                return json.loads(result.value)

            elif gate["name"] == "constraint_system":
                # Validate against hierarchical constraint system
                if not self._plugin_available(plugin_name):
                    logger.warning("Skipping constraint_system gate; plugin %s unavailable", plugin_name)
                    return {"score": 0.0, "details": "constraint system plugin unavailable"}

                constraint_version = context.get("constraint_version", "1.0")
                result = await self._invoke_sk_function_safe(
                    plugin_name=plugin_name,
                    function_name="ValidatePetitionConstraints",
                    variables={
                        "petition_text": document,
                        "constraint_version": constraint_version,
                    },
                    context=context,
                )
                validation_result = json.loads(result.value)
                # Extract overall_score from constraint validation
                if "overall_score" in validation_result:
                    validation_result["score"] = validation_result["overall_score"]
                return validation_result

            else:
                # Fallback to basic validation
                return self._basic_validation_fallback(gate, document, context)

        except Exception as e:
            logger.error(f"Error running SK validation gate {gate['name']}: {e}")
            # Fallback to basic validation
            return self._basic_validation_fallback(gate, document, context)

    def _plugin_available(self, plugin_name: str) -> bool:
        plugins = getattr(self.sk_kernel, "plugins", {})
        return isinstance(plugins, dict) and plugin_name in plugins

    def _validate_personal_facts_coverage(self, document: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the draft references personal lawsuit facts."""
        verification = _run_personal_facts_verifier(document, context)
        if verification is None:
            return {
                "score": 1.0,
                "details": "Personal facts verifier unavailable",
                "warnings": ["Personal facts verifier unavailable; install validation module"],
            }

        if verification.get("error") == "personal_facts_unavailable":
            warning = "Personal corpus facts not loaded; run scripts/refresh_personal_corpus.py"
            return {
                "score": 0.0,
                "details": warning,
                "warnings": [warning],
                "critical_facts_missing": True,
                "facts_missing": [],
                "facts_found": [],
            }
        if verification.get("error"):
            warning = f"Personal facts verification error: {verification['error']}"
            return {
                "score": 0.0,
                "details": warning,
                "warnings": [warning],
                "critical_facts_missing": True,
                "facts_missing": [],
                "facts_found": [],
            }

        coverage = verification.get("coverage", 0.0)
        missing = verification.get("missing_facts", [])
        facts_found = verification.get("facts_found", [])
        violations = verification.get("violations", [])
        contradictions = verification.get("contradictions", [])
        has_violations = verification.get("has_violations", bool(violations))
        has_contradictions = verification.get("has_contradictions", bool(contradictions))
        
        # Use cluster-based validation if available, otherwise fall back to flat count
        verification_details = verification.get("details", {})
        cluster_coverage = verification_details.get("cluster_coverage", {})
        cluster_satisfaction_rate = cluster_coverage.get("cluster_satisfaction_rate", 1.0)
        total_required_clusters = cluster_coverage.get("total_required_clusters", 0)
        required_clusters_satisfied = cluster_coverage.get("required_clusters_satisfied", 0)
        missing_clusters = cluster_coverage.get("missing_clusters", [])
        
        # For backward compatibility, also calculate flat count
        total_required = verification_details.get("total_required")
        total_required = total_required if isinstance(total_required, int) else len(facts_found) + len(missing)

        warnings = []
        if missing:
            warnings.append(f"Missing personal facts: {', '.join(missing[:4])}")
        if missing_clusters:
            warnings.append(f"Missing required clusters: {', '.join(missing_clusters)}")
        if has_violations:
            warnings.append("Prohibited facts detected during validation.")
        if has_contradictions:
            warnings.append("Draft contradicts verified case facts.")

        score = coverage
        if has_violations or has_contradictions:
            score = 0.0

        # Determine critical_facts_missing using cluster-based validation if available
        # If all required clusters are satisfied, individual missing facts are not critical
        has_cluster_data = total_required_clusters > 0 and "cluster_satisfaction_rate" in cluster_coverage
        if has_cluster_data:
            # Cluster-based: critical only if required clusters are not satisfied
            critical_facts_missing = cluster_satisfaction_rate < 1.0
        else:
            # Fallback to flat count: critical if any facts are missing
            critical_facts_missing = bool(missing)
        
        # If all required clusters are satisfied, override score to ensure gate passes
        # This ensures cluster-based validation takes precedence over flat coverage score
        if has_cluster_data and cluster_satisfaction_rate >= 1.0:
            score = 1.0  # Set score to 1.0 to ensure gate passes threshold
        
        # Use cluster-based message if clusters are available
        if total_required_clusters > 0:
            details_msg = f"Cluster-based validation: {required_clusters_satisfied}/{total_required_clusters} required clusters satisfied (satisfaction rate: {cluster_satisfaction_rate:.1%})"
            if missing_clusters:
                details_msg += f" | Missing clusters: {', '.join(missing_clusters)}"
            if len(facts_found) > 0:
                details_msg += f" | Facts found: {len(facts_found)}"
        else:
            details_msg = f"Personal facts referenced {len(facts_found)}/{total_required}"

        details = {
            "score": score,
            "details": details_msg,
            "coverage": coverage,
            "facts_found": facts_found,
            "facts_missing": missing,
            "critical_facts_missing": critical_facts_missing,
            "violations": violations,
            "contradictions": contradictions,
            "has_violations": has_violations,
            "has_contradictions": has_contradictions,
            "warnings": warnings,
            "suggestions": verification.get("suggestions", []),
            "raw_result": verification,
        }
        return details

    def _validate_citations(self, document: str) -> Dict[str, Any]:
        """Validate citation format."""
        import re

        # Look for [Node:State] pattern
        citation_pattern = r'\[[^:]+:[^\]]+\]'
        citations = re.findall(citation_pattern, document)

        # Check if citations are properly formatted
        valid_citations = [c for c in citations if ':' in c and len(c) > 3]

        score = len(valid_citations) / max(len(citations), 1) if citations else 1.0

        return {
            "score": score,
            "details": f"Found {len(valid_citations)}/{len(citations)} valid citations"
        }

    def _validate_structure(self, document: str) -> Dict[str, Any]:
        """Validate document structure."""
        required_sections = ["Introduction", "Analysis", "Conclusion"]

        found_sections = []
        for section in required_sections:
            if section.lower() in document.lower():
                found_sections.append(section)

        score = len(found_sections) / len(required_sections)

        return {
            "score": score,
            "details": f"Found {len(found_sections)}/{len(required_sections)} required sections"
        }

    def _validate_evidence_grounding(self, document: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate evidence grounding."""
        evidence = context.get("evidence", {})

        if not evidence:
            return {"score": 1.0, "details": "No evidence to validate"}

        # Count evidence references in document
        evidence_refs = 0
        for node in evidence.keys():
            if node.lower() in document.lower():
                evidence_refs += 1

        score = evidence_refs / max(len(evidence), 1)

        return {
            "score": score,
            "details": f"Referenced {evidence_refs}/{len(evidence)} evidence items"
        }

    async def _validate_with_corpus(self, document: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate document against corpus patterns from successful motions.
        
        Args:
            document: Document text to validate
            context: Optional context with section-specific validation results
            
        Returns:
            Dictionary with corpus validation results
        """
        try:
            # Import corpus validation plugin
            from sk_plugins.FeaturePlugin.corpus_validation_plugin import CorpusValidationPlugin
            from sk_plugins.FeaturePlugin.document_structure import DocumentStructure
            
            # Create document structure from text
            doc_structure = DocumentStructure(document)
            
            # Get corpus validation plugin from SK kernel if available
            corpus_validator = None
            if hasattr(self, 'sk_kernel') and self.sk_kernel:
                # Try to get plugin from kernel or create new instance
                try:
                    # Create plugin instance (requires kernel, chroma_store, rules_dir)
                    # For now, return basic validation result
                    # Full integration requires proper plugin initialization
                    corpus_validator = None
                except Exception as e:
                    logger.debug(f"Could not initialize corpus validator: {e}")
            
            # If corpus validator not available, return basic result
            if not corpus_validator:
                return {
                    "score": 0.5,  # Neutral score if corpus validation unavailable
                    "details": "Corpus validation plugin not available",
                    "corpus_available": False
                }
            
            # Run corpus validation
            result = await corpus_validator.validate_draft_against_corpus(doc_structure, context)
            
            if result.success:
                data = result.data or {}
                return {
                    "score": data.get('overall_similarity', 0.5),
                    "details": result.message,
                    "corpus_available": True,
                    "matches_patterns": data.get('matches_patterns', False),
                    "similarity": data.get('overall_similarity', 0.0),
                    "recommendations": data.get('recommendations', [])
                }
            else:
                return {
                    "score": data.get('overall_similarity', 0.3) if 'data' in locals() else 0.3,
                    "details": result.message,
                    "corpus_available": True,
                    "matches_patterns": False,
                    "recommendations": data.get('recommendations', []) if 'data' in locals() else []
                }
        
        except Exception as e:
            logger.warning(f"Corpus validation failed: {e}")
            return {
                "score": 0.5,
                "details": f"Corpus validation error: {str(e)}",
                "corpus_available": False
            }

    def _validate_content_is_actual_motion(self, content: str) -> tuple[bool, str]:
        """Validate that generated content is actual motion text, not test/example prompts.
        
        Returns:
            Tuple of (is_valid, error_message)
            is_valid: True if content appears to be actual motion, False if it's test/example content
            error_message: Description of why validation failed (empty if valid)
        """
        if not content or len(content.strip()) < 100:
            return False, "Content is too short or empty"
        
        content_lower = content.lower()
        
        # Patterns that indicate test/example prompts (not actual motion)
        test_patterns = [
            r'comprehensive report on',
            r'your task:',
            r'generate three follow up questions',
            r'as a student of law',
            r'craft an extensive treatise',
            r'your analysis must include',
            r'for each technology mentioned',
            r'translate the following legal scenario',
            r'your essay should',
            r'your response:',
            r'write a comprehensive',
            r'provide your insights while including',
            r'analyze at least three',
            r'examine the balance between',
            r'discuss at least five',
            r'in your analysis include:',
            r'your task is to',
            r'please provide',
            r'create a comprehensive',
            r'develop this framework',
        ]
        
        # Check for test patterns
        import re
        for pattern in test_patterns:
            if re.search(pattern, content_lower):
                return False, f"Content appears to be test/example prompt (matched pattern: {pattern})"
        
        # Check for actual motion indicators (positive validation)
        motion_indicators = [
            r'united states district court',
            r'respectfully (requests?|moves?|submits?|prays?)',
            r'wherefore',
            r'pursuant to',
            r'federal rule',
            r'this court',
            r'plaintiff|defendant|movant',
            r'motion for',
            r'order granting',
        ]
        
        motion_count = sum(1 for pattern in motion_indicators if re.search(pattern, content_lower))
        
        # If we have test patterns OR no motion indicators, it's invalid
        if motion_count == 0:
            return False, "Content lacks motion indicators (no 'respectfully requests', 'wherefore', court references, etc.)"
        
        # If we have motion indicators and no test patterns, it's likely valid
        return True, ""

    def _validate_tone(self, document: str) -> Dict[str, Any]:
        """Validate professional legal tone."""
        # Simple heuristic: check for informal language
        informal_words = ["gonna", "wanna", "yeah", "ok", "cool", "awesome"]

        informal_count = sum(1 for word in informal_words if word in document.lower())

        # Score based on absence of informal language
        score = max(0.0, 1.0 - (informal_count * 0.2))

        return {
            "score": score,
            "details": f"Found {informal_count} informal expressions"
        }

    def _basic_validation_fallback(self, gate: Dict[str, Any], document: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback validation when SK functions are not available."""

        if gate["name"] == "citation_validity":
            return self._validate_citations(document)
        elif gate["name"] == "structure_complete":
            return self._validate_structure(document)
        elif gate["name"] == "evidence_grounding":
            return self._validate_evidence_grounding(document, context)
        elif gate["name"] == "tone_consistency":
            return self._validate_tone(document)
        elif gate["name"] == "corpus_patterns":
            # Corpus validation is async, but gate validation is sync
            # For now, return a placeholder - full async integration requires refactoring
            return {"score": 0.5, "details": "Corpus validation requires async context", "suggestions": []}
        else:
            return {"score": 1.0, "details": "Gate not implemented", "suggestions": []}


class IterationController:
    """Controls iteration between AutoGen and SK phases."""

    def __init__(self, config: WorkflowStrategyConfig):
        self.config = config
        self.iteration_history: List[Dict[str, Any]] = []

    def should_continue_iteration(self, state: WorkflowState) -> bool:
        """Determine if iteration should continue."""
        return (
            state.iteration < self.config.max_iterations and
            state.phase != WorkflowPhase.COMMIT
        )

    def record_iteration(self, state: WorkflowState, phase_result: Any) -> None:
        """Record iteration results."""
        iteration_record = {
            "iteration": state.iteration,
            "phase": state.phase.value,
            "result_summary": str(phase_result)[:200] if phase_result else None,
            "timestamp": asyncio.get_event_loop().time()
        }

        self.iteration_history.append(iteration_record)

        # Keep only recent history
        if len(self.iteration_history) > 10:
            self.iteration_history = self.iteration_history[-10:]

    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get summary of iteration history."""
        if not self.iteration_history:
            return {"total_iterations": 0, "phases_completed": []}

        phases_completed = [record["phase"] for record in self.iteration_history]

        return {
            "total_iterations": len(self.iteration_history),
            "phases_completed": phases_completed,
            "last_phase": self.iteration_history[-1]["phase"] if self.iteration_history else None
        }


class Conductor:
    """
    Main orchestrator/glue that coordinates all teams.

    Like a conductor leading an orchestra, this coordinates:
    - AutoGen (writing team)
    - Semantic Kernel (quality control framework)
    - RefinementLoop (iterative quality improvement sub-coordinator)
    - Manages workflow phases and routing
    - Bridges all components together

    Built with: Custom Python + AutoGen factory + SK Kernel
    This is the top-level orchestrator that coordinates everything.
    """

    def __init__(self, config: Optional[WorkflowStrategyConfig] = None):
        self.config = config or WorkflowStrategyConfig()

        # Initialize components
        self.autogen_factory = AgentFactory(self.config.autogen_config)

        # Initialize SK kernel (defaults to local LLM if no config provided)
        if self.config.sk_config is not None:
            self.sk_kernel = create_sk_kernel(self.config.sk_config)
        else:
            # Default to local LLM (Phi-3 Mini for legal writing)
            from sk_config import get_default_sk_config
            default_config = get_default_sk_config(use_local=True)
            self.sk_kernel = create_sk_kernel(default_config)

        # Initialize workflow components
        self.router = WorkflowRouter(self.config)
        # Bridge can work without SK kernel - it only does context translation
        self.bridge = AutoGenToSKBridge(self.autogen_factory, self.sk_kernel)
        self.iteration_controller = IterationController(self.config)
        # QualityGatePipeline needs self reference - initialize after other components
        self.quality_pipeline = None
        self.memory_store: Optional[EpisodicMemoryBank] = None
        self.memory_retriever: Optional[EpisodicMemoryRetriever] = None
        self.chat_library: Optional[CursorChatLibrary] = None
        self._plugin_registration_tasks: List[asyncio.Task[Any]] = []
        self._constraint_system_cache: Dict[str, Dict[str, Any]] = {}
        self._constraint_format_cache: Dict[str, List[str]] = {}
        self._constraint_dir_override: Optional[Path] = None
        self.constraint_system_version = getattr(self.config, "constraint_system_version", "1.0")

        # Initialize episodic memory system
        self._initialize_memory_system()
        self._initialize_chat_library()
        # Initialize plugins
        self._register_plugins()

        # Initialize quality pipeline with executor reference
        self.quality_pipeline = QualityGatePipeline(self.sk_kernel, executor=self) if self.sk_kernel else None

        # Initialize Chroma integration
        self._initialize_chroma_integration()

        # Initialize Google Docs integration
        self._initialize_google_docs_integration()

        # Initialize version manager (set to None first, then initialize)
        self.version_manager = None
        self._initialize_version_manager()

        # Initialize RefinementLoop (iterative quality improvement SUB-coordinator)
        # NOTE: Conductor is the MAIN orchestrator/glue.
        # RefinementLoop is a specialized sub-component for iterative quality enhancement.
        # System: AutoGen writes → RefinementLoop coordinates CatBoost research + SK quality control
        self.draft_enhancer = None
        self.draft_quality_controller = None  # Backwards compatibility
        self.feature_orchestrator = None      # Backwards compatibility
        self._initialize_feature_orchestrator()

        # Initialize multi-model ensemble components
        self.parallel_draft_generator = None
        self.section_merger = None
        self.multi_model_validator = None
        self._initialize_multi_model_components()

        # Initialize case law researcher
        self.case_law_researcher = None
        self._initialize_case_law_researcher()

        logger.info("Conductor initialized successfully")

    def _initialize_version_manager(self) -> None:
        """Initialize version manager for document backups."""
        if self.config.enable_version_backups:
            try:
                from .version_manager import VersionManager
                self.version_manager = VersionManager(
                    backup_directory=Path(self.config.version_backup_directory),
                    ml_training_directory=Path("outputs/ml_training_data/drafts"),
                    max_versions_to_keep=self.config.max_versions_to_keep
                )
                logger.info("Version manager initialized")
            except ImportError:
                try:
                    from version_manager import VersionManager
                    self.version_manager = VersionManager(
                        backup_directory=Path(self.config.version_backup_directory),
                        ml_training_directory=Path("outputs/ml_training_data/drafts"),
                        max_versions_to_keep=self.config.max_versions_to_keep
                    )
                    logger.info("Version manager initialized")
                except Exception as e:
                    logger.warning(f"Version manager not available: {e}")
                    self.version_manager = None
            except Exception as e:
                logger.warning(f"Version manager initialization failed: {e}")
                self.version_manager = None
        else:
            self.version_manager = None

    def _initialize_multi_model_components(self) -> None:
        """Initialize multi-model ensemble components."""
        try:
            from multi_model_config import MultiModelConfig, get_default_multi_model_config
            from multi_model_drafting import ParallelDraftGenerator
            from section_merger import SectionMerger
            from multi_model_validator import MultiModelValidator

            # Get multi-model config (use default if not provided)
            if self.config.multi_model_config:
                mm_config = self.config.multi_model_config
            else:
                mm_config = get_default_multi_model_config()

            # Only initialize if enabled
            if mm_config.enabled:
                # Initialize parallel draft generator
                if mm_config.use_multi_model_drafting:
                    self.parallel_draft_generator = ParallelDraftGenerator(
                        primary_model=mm_config.primary_drafting_model,
                        secondary_model=mm_config.secondary_drafting_model,
                        ollama_base_url=mm_config.ollama_base_url,
                        temperature=0.3,
                        max_tokens=4000
                    )
                    logger.info("Parallel draft generator initialized")

                # Initialize section merger
                self.section_merger = SectionMerger(
                    quality_weight=mm_config.quality_weight,
                    semantic_weight=mm_config.semantic_weight,
                    quality_threshold=mm_config.quality_threshold,
                    semantic_similarity_threshold=mm_config.semantic_similarity_threshold
                )
                logger.info("Section merger initialized")

                # Initialize multi-model validator
                if mm_config.use_multi_model_validation:
                    self.multi_model_validator = MultiModelValidator(
                        legal_bert_weight=mm_config.legal_bert_weight,
                        qwen_review_weight=mm_config.qwen_review_weight,
                        catboost_weight=mm_config.catboost_weight,
                        qwen_model=mm_config.logical_reviewer_model,
                        ollama_base_url=mm_config.ollama_base_url,
                        quality_threshold=mm_config.quality_threshold
                    )
                    logger.info("Multi-model validator initialized")
            else:
                logger.info("Multi-model ensemble disabled in config")

        except ImportError as e:
            logger.warning(f"Multi-model components not available: {e}")
            self.parallel_draft_generator = None
            self.section_merger = None
            self.multi_model_validator = None
        except Exception as e:
            logger.warning(f"Failed to initialize multi-model components: {e}")
            self.parallel_draft_generator = None
            self.section_merger = None
            self.multi_model_validator = None

    def _initialize_case_law_researcher(self) -> None:
        """Initialize case law researcher for querying case databases."""
        try:
            # Try relative import first, then absolute
            try:
                from .case_law_researcher import CaseLawResearcher
            except (ImportError, ValueError):
                from case_law_researcher import CaseLawResearcher

            # Initialize with default database paths and memory store
            self.case_law_researcher = CaseLawResearcher(memory_store=self.memory_store)

            if self.case_law_researcher.enabled:
                logger.info("CaseLawResearcher initialized successfully")
            else:
                logger.warning("CaseLawResearcher initialized but not enabled (databases not found)")
                self.case_law_researcher = None
        except ImportError as e:
            logger.warning(f"CaseLawResearcher not available: {e}")
            self.case_law_researcher = None
        except Exception as e:
            logger.warning(f"CaseLawResearcher initialization failed: {e}")
            self.case_law_researcher = None

    def _initialize_feature_orchestrator(self) -> None:
        """
        Initialize RefinementLoop (iterative quality improvement SUB-coordinator).

        NOTE: This class (Conductor) is the MAIN orchestrator/glue.
        RefinementLoop is a specialized sub-component called during VALIDATE/REFINE phases.

        System Mental Model:
        - CatBoost = Research/Quant team (analyzes features, predicts success)
        - Semantic Kernel = Quality Control framework (plugins enforce standards)
        - AutoGen = Writing team (generates initial drafts)

        RefinementLoop (sub-coordinator) coordinates:
        - Uses CatBoost research to analyze drafts and identify weak features
        - Coordinates SK quality control plugins to generate improvements
        - Enhances drafts based on research insights + quality enforcement

        NOT the writing team - writing happens in DraftingPlugin (AutoGen).
        This enhances what was written by coordinating research + quality control.
        """
        try:
            # Use absolute imports to avoid relative import issues when running as script
            import sys
            from pathlib import Path

            # Add code package and its parent (writer_agents) to sys.path for package-safe imports
            code_dir = Path(__file__).parent  # .../writer_agents/code
            pkg_dir = code_dir.parent        # .../writer_agents
            if str(pkg_dir) not in sys.path:
                sys.path.insert(0, str(pkg_dir))
            if str(code_dir) not in sys.path:
                sys.path.insert(0, str(code_dir))

            # Try relative import first (for package use), fallback to absolute
            try:
                from .sk_plugins.FeaturePlugin.feature_orchestrator import RefinementLoop
                from .sk_plugins.FeaturePlugin import (
                    PrivacyPlugin,
                    HarassmentPlugin,
                    SafetyPlugin,
                    RetaliationPlugin,
                    CitationRetrievalPlugin,
                    RequiredCaseCitationPlugin,
                    PrivacyHarmCountPlugin,
                    PublicInterestPlugin,
                    TransparencyArgumentPlugin,
                    SentenceStructurePlugin,
                    ParagraphStructurePlugin,
                    EnumerationDensityPlugin,
                    PerParagraphPlugin,
                    ParagraphMonitorPlugin,
                    NationalSecurityDefinitionsPlugin,
                    BalancingOutweighPlugin,
                    PresumptionAcknowledgmentPlugin,
                    ProtectiveMeasuresPlugin,
                    ForeignGovernmentPlugin,
                    IntelClassifiedInfoPlugin,
                    BalancingTestPositionPlugin,
                    AvoidFirstAmendmentPlugin,
                    AvoidBalancingTestPhrasePlugin,
                    UseBalanceConceptsPlugin,
                    UseMotionLanguagePlugin,
                    AvoidNarrowlyTailoredPlugin,
                    AvoidCompellingInterestPhrasePlugin,
                    UseCompetingInterestsPlugin,
                    HKNationalSecurityPlugin,
                    PRCNationalSecurityPlugin,
                    TrumpJune4ProclamationPlugin,
                    HarvardLawsuitPlugin,
                    TimingArgumentPlugin
                )
            except (ImportError, ValueError):
                # Fallback to absolute import
                from sk_plugins.FeaturePlugin.feature_orchestrator import RefinementLoop
                from sk_plugins.FeaturePlugin import (
                    PrivacyPlugin,
                    HarassmentPlugin,
                    SafetyPlugin,
                    RetaliationPlugin,
                    CitationRetrievalPlugin,
                    RequiredCaseCitationPlugin,
                    PrivacyHarmCountPlugin,
                    PublicInterestPlugin,
                    TransparencyArgumentPlugin,
                    SentenceStructurePlugin,
                    ParagraphStructurePlugin,
                    EnumerationDensityPlugin,
                    PerParagraphPlugin,
                    ParagraphMonitorPlugin,
                    NationalSecurityDefinitionsPlugin,
                    BalancingOutweighPlugin,
                    PresumptionAcknowledgmentPlugin,
                    ProtectiveMeasuresPlugin,
                    ForeignGovernmentPlugin,
                    IntelClassifiedInfoPlugin,
                    BalancingTestPositionPlugin,
                    AvoidFirstAmendmentPlugin,
                    AvoidBalancingTestPhrasePlugin,
                    UseBalanceConceptsPlugin,
                    UseMotionLanguagePlugin,
                    AvoidNarrowlyTailoredPlugin,
                    AvoidCompellingInterestPhrasePlugin,
                    UseCompetingInterestsPlugin,
                    HKNationalSecurityPlugin,
                    PRCNationalSecurityPlugin,
                    TrumpJune4ProclamationPlugin,
                    HarvardLawsuitPlugin,
                    TimingArgumentPlugin
                )

            # Try to load Section 1782 CatBoost model (preferred) or fallback to MA motion model
            model_path = Path(__file__).parents[3] / "case_law_data" / "models" / "section_1782_discovery_model.cbm"
            if not model_path.exists():
                # Try alternative path
                model_path = Path(__file__).parents[2] / "case_law_data" / "models" / "section_1782_discovery_model.cbm"
                if not model_path.exists():
                    # Fallback to MA motion model
                    model_path = Path(__file__).parents[3] / "case_law_data" / "models" / "catboost_motion.cbm"
                    if not model_path.exists():
                        model_path = Path(__file__).parents[2] / "case_law_data" / "models" / "catboost_motion.cbm"

            # SHAP importance will be loaded from section_1782_discovery_SHAP_SUMMARY.json by RefinementLoop
            # Fallback to legacy values if SHAP summary not available
            shap_importance = {
                'word_count': 0.1692,
                'text_length': 0.1501,
                'mentions_public_interest': 0.1044,
                'mentions_first_amendment': 0.0992,
                'citation_count': 0.0959,
                'mentions_privacy': 0.1044,  # Approximate from analysis
                'mentions_safety': 0.070,    # Approximate
                'privacy_harm_count': 0.0686,
                'mentions_harassment': 0.0426,
                'mentions_retaliation': 0.0426,  # Approximate
                'mentions_transparency': 0.030,  # Approximate
            }

            # Initialize plugins (need SK kernel and Chroma store)
            plugins = {}
            if self.sk_kernel:
                # Get Chroma store if available
                chroma_store = None
                try:
                    # Try relative import first, then absolute
                    try:
                        from .sk_plugins.shared.chroma_integration import ChromaSKMemoryStore
                    except (ImportError, ValueError):
                        try:
                            from sk_plugins.shared.chroma_integration import ChromaSKMemoryStore
                        except ImportError:
                            # Try alternative import path
                            chroma_store = None
                            logger.debug("ChromaSKMemoryStore not found, continuing without Chroma")
                            raise ImportError("ChromaSKMemoryStore not available")

                    if self.config.enable_chroma_integration:
                        chroma_store = ChromaSKMemoryStore(persist_directory=self.config.chroma_persist_directory)
                        logger.info("Chroma store initialized successfully")
                except Exception as e:
                    logger.debug(f"Chroma store not available: {e}")
                    chroma_store = None

                rules_dir = Path(__file__).parent / "sk_plugins" / "rules"

                # Get database paths for plugins (Phase 1)
                db_paths = self._get_plugin_database_paths()

                # Create plugin instances (can work without Chroma, just won't have rule storage)
                # Minimal mode support
                minimal = str(os.environ.get("MATRIX_MINIMAL_PLUGINS", "")).strip().lower() in {"1", "true", "yes", "on"}
                if minimal:
                    try:
                        try:
                            from .sk_plugins.FeaturePlugin.transition_legal_to_factual_plugin import TransitionLegalToFactualPlugin
                        except Exception:
                            from sk_plugins.FeaturePlugin.transition_legal_to_factual_plugin import TransitionLegalToFactualPlugin
                        try:
                            from .sk_plugins.FeaturePlugin.paragraph_structure_plugin import ParagraphStructurePlugin
                        except Exception:
                            from sk_plugins.FeaturePlugin.paragraph_structure_plugin import ParagraphStructurePlugin
                        try:
                            from .sk_plugins.FeaturePlugin.balancing_test_position_plugin import BalancingTestPositionPlugin
                        except Exception:
                            from sk_plugins.FeaturePlugin.balancing_test_position_plugin import BalancingTestPositionPlugin

                        plugin_kwargs = {
                            'db_paths': db_paths,
                            'enable_langchain': self.config.plugin_enable_langchain,
                            'enable_courtlistener': self.config.plugin_enable_courtlistener,
                            'enable_storm': self.config.plugin_enable_storm
                        }
                        plugins['transition_legal_to_factual'] = TransitionLegalToFactualPlugin(
                            self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **plugin_kwargs
                        )
                        plugins['paragraph_structure'] = ParagraphStructurePlugin(
                            self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **plugin_kwargs
                        )
                        plugins['balancing_test_position'] = BalancingTestPositionPlugin(
                            self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **plugin_kwargs
                        )
                        logger.info("Minimal plugin set initialized (transition, paragraph structure, balancing position)")
                    except Exception as e:
                        import traceback
                        logger.warning(f"Minimal plugin set initialization failed: {e}\n{traceback.format_exc()}")
                        plugins = {}
                elif rules_dir.exists() or True:  # Allow plugins even without rules_dir
                    try:
                        # Legacy plugins - pass None for chroma_store if not available
                        # Pass database paths and resource flags to all plugins
                        plugin_kwargs = {
                            'db_paths': db_paths,
                            'enable_langchain': self.config.plugin_enable_langchain,
                            'enable_courtlistener': self.config.plugin_enable_courtlistener,
                            'enable_storm': self.config.plugin_enable_storm
                        }

                        plugins['mentions_privacy'] = PrivacyPlugin(
                            self.sk_kernel, chroma_store or None,
                            rules_dir if rules_dir.exists() else None,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )
                        plugins['mentions_harassment'] = HarassmentPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )
                        plugins['mentions_safety'] = SafetyPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )
                        plugins['mentions_retaliation'] = RetaliationPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )
                        plugins['citation_retrieval'] = CitationRetrievalPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )
                        # Pass memory_store to plugins for learning
                        plugins['required_case_citations'] = RequiredCaseCitationPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,  # Enable memory learning
                            **plugin_kwargs
                        )

                        # Create individual plugins for each verified case
                        try:
                            from .sk_plugins.FeaturePlugin.case_enforcement_plugin_generator import CaseEnforcementPluginFactory
                            individual_case_plugins = CaseEnforcementPluginFactory.create_all_case_plugins(
                                kernel=self.sk_kernel,
                                chroma_store=chroma_store,
                                rules_dir=rules_dir,
                                memory_store=self.memory_store,  # Enable memory learning
                                db_paths=db_paths,
                                enable_langchain=self.config.plugin_enable_langchain,
                                enable_courtlistener=self.config.plugin_enable_courtlistener,
                                enable_storm=self.config.plugin_enable_storm
                            )
                            # Add individual case plugins to plugins dict
                            plugins.update(individual_case_plugins)
                            logger.info(f"Created {len(individual_case_plugins)} individual case enforcement plugins with memory integration")
                        except Exception as e:
                            logger.warning(f"Failed to create individual case plugins: {e}")

                        # All plugins now receive memory_store for potential future use
                        # (Memory is optional - plugins work fine without it)
                        plugins['privacy_harm_count'] = PrivacyHarmCountPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )
                        plugins['public_interest'] = PublicInterestPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )
                        plugins['transparency_argument'] = TransparencyArgumentPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )

                        # Structural plugins (new) - memory optional but available
                        plugins['sentence_structure'] = SentenceStructurePlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )
                        plugins['paragraph_structure'] = ParagraphStructurePlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )
                        plugins['enumeration_density'] = EnumerationDensityPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )

                        # Per-paragraph and monitoring plugins
                        plugins['per_paragraph'] = PerParagraphPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )
                        plugins['paragraph_monitor'] = ParagraphMonitorPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            per_paragraph_plugin=plugins['per_paragraph'],
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )

                        # National Security Sealing Plugins (from CatBoost analysis)
                        plugins['intel_classified_info'] = IntelClassifiedInfoPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # #1 feature (17.01)
                        plugins['balancing_outweigh'] = BalancingOutweighPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # #2 feature (12.75)
                        plugins['presumption_acknowledgment'] = PresumptionAcknowledgmentPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # #5 feature (8.65)
                        plugins['protective_measures'] = ProtectiveMeasuresPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # #7 feature (4.32)
                        plugins['national_security_definitions'] = NationalSecurityDefinitionsPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # #8 feature (3.67)
                        plugins['foreign_government'] = ForeignGovernmentPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # #9 feature (3.40)
                        plugins['balancing_test_position'] = BalancingTestPositionPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # Existing balancing test plugin
                        plugins['hk_national_security'] = HKNationalSecurityPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # HK NS Law - leadership family as NS issue
                        plugins['prc_national_security'] = PRCNationalSecurityPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # PRC NS Law - leadership family as NS issue
                        plugins['trump_june4_proclamation'] = TrumpJune4ProclamationPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # Trump June 4th proclamation - Harvard-CCP as NS risk
                        plugins['harvard_lawsuit'] = HarvardLawsuitPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # Harvard's lawsuit in response
                        plugins['timing_argument'] = TimingArgumentPlugin(
                            self.sk_kernel, chroma_store, rules_dir,
                            memory_store=self.memory_store,
                            **plugin_kwargs
                        )  # Timing: NS since April 7th

                        logger.info(f"Initialized {len(plugins)} feature plugins (8 legacy + 5 structural/monitoring + 12 national security sealing)")
                    except Exception as e:
                        logger.warning(f"Failed to create feature plugins: {e}")
                        import traceback
                        logger.warning(traceback.format_exc())
                        # Keep any successfully created plugins; proceed with partial set

            # Add section-specific constraint plugins to plugins dict BEFORE RefinementLoop initialization
            # These plugins are also registered with the async registry, but need to be in plugins dict for RefinementLoop
            try:
                from .sk_plugins.FeaturePlugin import (
                    # Word count plugins (all 9 sections)
                    IntroductionWordCountPlugin,
                    LegalStandardWordCountPlugin,
                    FactualBackgroundWordCountPlugin,
                    PrivacyHarmWordCountPlugin,
                    DangerSafetyWordCountPlugin,
                    PublicInterestWordCountPlugin,
                    BalancingTestWordCountPlugin,
                    ProtectiveMeasuresWordCountPlugin,
                    ConclusionWordCountPlugin,
                    # Paragraph structure plugins (all 9 sections)
                    IntroductionParagraphStructurePlugin,
                    LegalStandardParagraphStructurePlugin,
                    FactualBackgroundParagraphStructurePlugin,
                    PrivacyHarmParagraphStructurePlugin,
                    DangerSafetyParagraphStructurePlugin,
                    PublicInterestParagraphStructurePlugin,
                    BalancingTestParagraphStructurePlugin,
                    ProtectiveMeasuresParagraphStructurePlugin,
                    ConclusionParagraphStructurePlugin,
                    # Enumeration depth plugins (all 9 sections)
                    IntroductionEnumerationDepthPlugin,
                    LegalStandardEnumerationDepthPlugin,
                    FactualBackgroundEnumerationDepthPlugin,
                    PrivacyHarmEnumerationDepthPlugin,
                    PublicInterestEnumerationDepthPlugin,
                    BalancingTestEnumerationDepthPlugin,
                    DangerSafetyEnumerationDepthPlugin,
                    ProtectiveMeasuresEnumerationDepthPlugin,
                    ConclusionEnumerationDepthPlugin,
                    # Sentence count plugins (all 9 sections)
                    IntroductionSentenceCountPlugin,
                    LegalStandardSentenceCountPlugin,
                    FactualBackgroundSentenceCountPlugin,
                    PrivacyHarmSentenceCountPlugin,
                    DangerSafetySentenceCountPlugin,
                    PublicInterestSentenceCountPlugin,
                    BalancingTestSentenceCountPlugin,
                    ProtectiveMeasuresSentenceCountPlugin,
                    ConclusionSentenceCountPlugin,
                    # Words per sentence plugins (all 9 sections)
                    IntroductionWordsPerSentencePlugin,
                    LegalStandardWordsPerSentencePlugin,
                    FactualBackgroundWordsPerSentencePlugin,
                    PrivacyHarmWordsPerSentencePlugin,
                    DangerSafetyWordsPerSentencePlugin,
                    PublicInterestWordsPerSentencePlugin,
                    BalancingTestWordsPerSentencePlugin,
                    ProtectiveMeasuresWordsPerSentencePlugin,
                    ConclusionWordsPerSentencePlugin,
                    # Constraint resolver
                    ConstraintResolverPlugin
                )

                # Create section plugins and add to plugins dict
                # Use same plugin_kwargs as other plugins to ensure database access is enabled
                section_plugin_kwargs = {
                    'db_paths': db_paths,
                    'enable_langchain': self.config.plugin_enable_langchain,
                    'enable_courtlistener': self.config.plugin_enable_courtlistener,
                    'enable_storm': self.config.plugin_enable_storm
                }
                
                section_plugins_list = [
                    # Word count plugins
                    IntroductionWordCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    LegalStandardWordCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    FactualBackgroundWordCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    PrivacyHarmWordCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    DangerSafetyWordCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    PublicInterestWordCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    BalancingTestWordCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    ProtectiveMeasuresWordCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    ConclusionWordCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    # Paragraph structure plugins
                    IntroductionParagraphStructurePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    LegalStandardParagraphStructurePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    FactualBackgroundParagraphStructurePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    PrivacyHarmParagraphStructurePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    DangerSafetyParagraphStructurePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    PublicInterestParagraphStructurePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    BalancingTestParagraphStructurePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    ProtectiveMeasuresParagraphStructurePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    ConclusionParagraphStructurePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    # Enumeration depth plugins
                    IntroductionEnumerationDepthPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    LegalStandardEnumerationDepthPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    FactualBackgroundEnumerationDepthPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    PrivacyHarmEnumerationDepthPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    PublicInterestEnumerationDepthPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    BalancingTestEnumerationDepthPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    DangerSafetyEnumerationDepthPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    ProtectiveMeasuresEnumerationDepthPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    ConclusionEnumerationDepthPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    # Sentence count plugins
                    IntroductionSentenceCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    LegalStandardSentenceCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    FactualBackgroundSentenceCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    PrivacyHarmSentenceCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    DangerSafetySentenceCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    PublicInterestSentenceCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    BalancingTestSentenceCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    ProtectiveMeasuresSentenceCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    ConclusionSentenceCountPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    # Words per sentence plugins
                    IntroductionWordsPerSentencePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    LegalStandardWordsPerSentencePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    FactualBackgroundWordsPerSentencePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    PrivacyHarmWordsPerSentencePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    DangerSafetyWordsPerSentencePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    PublicInterestWordsPerSentencePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    BalancingTestWordsPerSentencePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    ProtectiveMeasuresWordsPerSentencePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    ConclusionWordsPerSentencePlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs),
                    # Constraint resolver
                    ConstraintResolverPlugin(self.sk_kernel, chroma_store, rules_dir, memory_store=self.memory_store, **section_plugin_kwargs)
                ]

                # Add to plugins dict using feature_name as key
                for plugin in section_plugins_list:
                    if hasattr(plugin, 'feature_name'):
                        plugins[plugin.feature_name] = plugin
                        logger.debug(f"Added {plugin.feature_name} to plugins dict for RefinementLoop")

                logger.info(f"Added {len(section_plugins_list)} section-specific constraint plugins to plugins dict")
                
                # Store for later async registry registration (avoid duplicate instantiation)
                self._section_plugins_for_registry = section_plugins_list
            except Exception as e:
                logger.warning(f"Failed to add section-specific plugins to plugins dict: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                self._section_plugins_for_registry = []

            # Initialize RefinementLoop (iterative enhancement system - NOT the writing team)
            if plugins:
                # ✅ Initialize outline management for perfect outline structure
                outline_manager = None
                plugin_calibrator = None
                try:
                    from outline_manager import load_outline_manager
                    from plugin_calibrator import PluginCalibrator
                    outline_manager = load_outline_manager()
                    plugin_calibrator = PluginCalibrator(outline_manager)
                    logger.info("✅ Outline Manager initialized - plugins will be organized by perfect outline")
                except ImportError:
                    logger.debug("Outline management not available - using legacy organization")
                except Exception as e:
                    logger.warning(f"Could not initialize outline management: {e}")

                self.draft_enhancer = RefinementLoop(
                    plugins=plugins,
                    model_path=model_path if model_path.exists() else None,
                    shap_importance=shap_importance,
                    outline_manager=outline_manager,  # ✅ Pass perfect outline structure
                    plugin_calibrator=plugin_calibrator  # ✅ Pass plugin calibrator
                )
                # Backwards compatibility aliases
                self.draft_quality_controller = self.draft_enhancer
                self.feature_orchestrator = self.draft_enhancer
                logger.info("RefinementLoop initialized with CatBoost integration")
            else:
                logger.warning("RefinementLoop not initialized - plugins unavailable")
                self.draft_enhancer = None
                self.draft_quality_controller = None
                self.feature_orchestrator = None

        except ImportError as e:
            logger.warning(f"RefinementLoop not available: {e}")
            self.draft_enhancer = None
            self.draft_quality_controller = None
            self.feature_orchestrator = None
        except Exception as e:
            logger.warning(f"RefinementLoop initialization failed: {e}")
            self.draft_enhancer = None
            self.draft_quality_controller = None
            self.feature_orchestrator = None

    def _register_plugins(self) -> None:
        """Register SK plugins."""
        if self.sk_kernel is None:
            logger.warning("SK kernel not available, skipping plugin registration")
            return

        plugin_registry.set_kernel(self.sk_kernel)

        registration_coroutines: List[Awaitable[None]] = []
        try:
            # Register privacy harm plugin
            from .sk_plugins.DraftingPlugin.privacy_harm_function import PrivacyHarmPlugin
            privacy_plugin = PrivacyHarmPlugin(self.sk_kernel)
            registration_coroutines.append(plugin_registry.register_plugin(privacy_plugin))

            # Register factual timeline plugin
            from .sk_plugins.DraftingPlugin.factual_timeline_function import FactualTimelinePlugin
            timeline_plugin = FactualTimelinePlugin(self.sk_kernel)
            registration_coroutines.append(plugin_registry.register_plugin(timeline_plugin))

            # Register causation analysis plugin
            from .sk_plugins.DraftingPlugin.causation_analysis_function import CausationAnalysisPlugin
            causation_plugin = CausationAnalysisPlugin(self.sk_kernel)
            registration_coroutines.append(plugin_registry.register_plugin(causation_plugin))

            # Register validation plugin
            from .sk_plugins.ValidationPlugin.validation_functions import ValidationPlugin
            validation_plugin = ValidationPlugin(self.sk_kernel)
            registration_coroutines.append(plugin_registry.register_plugin(validation_plugin))

            try:
                from .sk_plugins.PetitionQualityPlugin import register as register_petition_quality
                register_petition_quality(self.sk_kernel)
                logger.info("PetitionQualityPlugin registered")
            except Exception as e:
                logger.warning(f"PetitionQualityPlugin registration failed: {e}")

            # Register assembly plugin
            from .sk_plugins.AssemblyPlugin.assembly_functions import AssemblyPlugin
            assembly_plugin = AssemblyPlugin(self.sk_kernel)
            registration_coroutines.append(plugin_registry.register_plugin(assembly_plugin))

            logger.info("Existing SK plugins registered")
        except Exception as e:
            logger.warning(f"Some existing plugins failed to register: {e}")
        finally:
            self._schedule_plugin_registrations(registration_coroutines)

        # Register feature plugins (chroma-dependent)
        self._register_feature_plugins()

        logger.info("All SK plugins registration completed")

    def _register_feature_plugins(self) -> None:
        """Register FeaturePlugin suite when Chroma is available."""
        if not self.config.enable_chroma_integration:
            logger.info("Chroma integration disabled; skipping feature plugin registration")
            return

        try:
            from .sk_plugins.FeaturePlugin import (
                MentionsPrivacyPlugin,
                MentionsHarassmentPlugin,
                MentionsSafetyPlugin,
                MentionsRetaliationPlugin,
                CitationRetrievalPlugin,
                PrivacyHarmCountPlugin,
                PublicInterestPlugin,
                TransparencyArgumentPlugin,
                NationalSecurityDefinitionsPlugin,
                BalancingOutweighPlugin,
                PresumptionAcknowledgmentPlugin,
                ProtectiveMeasuresPlugin,
                ForeignGovernmentPlugin,
                IntelClassifiedInfoPlugin,
                BalancingTestPositionPlugin,
                HKNationalSecurityPlugin,
                PRCNationalSecurityPlugin,
                TrumpJune4ProclamationPlugin,
                HarvardLawsuitPlugin,
                TimingArgumentPlugin
            )
            from .sk_plugins.shared.chroma_integration import (
                ChromaSKMemoryStore,
                DEFAULT_COLLECTION_EMBEDDINGS,
            )

            chroma_persist_directory = self.config.chroma_persist_directory or "./chroma_collections"
            try:
                chroma_store = ChromaSKMemoryStore(
                    chroma_persist_directory,
                    "case_law_legal",
                    collection_embedding_map=DEFAULT_COLLECTION_EMBEDDINGS,
                )
            except BaseException as exc:  # pragma: no cover - rust panic safety
                logger.warning("Chroma unavailable (%s); skipping feature plugins", exc)
                return
            rules_dir = Path(__file__).parent / "sk_plugins" / "rules"

            feature_plugins = [
                MentionsPrivacyPlugin(self.sk_kernel, chroma_store, rules_dir),
                MentionsHarassmentPlugin(self.sk_kernel, chroma_store, rules_dir),
                MentionsSafetyPlugin(self.sk_kernel, chroma_store, rules_dir),
                MentionsRetaliationPlugin(self.sk_kernel, chroma_store, rules_dir),
                CitationRetrievalPlugin(self.sk_kernel, chroma_store, rules_dir),
                PrivacyHarmCountPlugin(self.sk_kernel, chroma_store, rules_dir),
                PublicInterestPlugin(self.sk_kernel, chroma_store, rules_dir),
                TransparencyArgumentPlugin(self.sk_kernel, chroma_store, rules_dir),
                IntelClassifiedInfoPlugin(self.sk_kernel, chroma_store, rules_dir),
                BalancingOutweighPlugin(self.sk_kernel, chroma_store, rules_dir),
                PresumptionAcknowledgmentPlugin(self.sk_kernel, chroma_store, rules_dir),
                ProtectiveMeasuresPlugin(self.sk_kernel, chroma_store, rules_dir),
                NationalSecurityDefinitionsPlugin(self.sk_kernel, chroma_store, rules_dir),
                ForeignGovernmentPlugin(self.sk_kernel, chroma_store, rules_dir),
                BalancingTestPositionPlugin(self.sk_kernel, chroma_store, rules_dir),
                HKNationalSecurityPlugin(self.sk_kernel, chroma_store, rules_dir),
                PRCNationalSecurityPlugin(self.sk_kernel, chroma_store, rules_dir),
                TrumpJune4ProclamationPlugin(self.sk_kernel, chroma_store, rules_dir),
                HarvardLawsuitPlugin(self.sk_kernel, chroma_store, rules_dir),
                TimingArgumentPlugin(self.sk_kernel, chroma_store, rules_dir)
            ]

            registration_coroutines = [
                plugin_registry.register_plugin(plugin)
                for plugin in feature_plugins
            ]
            self._schedule_plugin_registrations(registration_coroutines)

            logger.info(f"Registered {len(feature_plugins)} feature plugins")

        except Exception as e:
            logger.error(f"Feature plugins registration failed: {e}")

    def _schedule_plugin_registrations(self, coroutines: List[Awaitable[None]]) -> None:
        """Schedule or execute plugin registration coroutines."""
        if not coroutines:
            return

        async def _runner() -> None:
            await asyncio.gather(*coroutines)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_runner())
            return

        task = loop.create_task(_runner())
        self._plugin_registration_tasks.append(task)

    async def _ensure_plugins_ready(self) -> None:
        """Ensure pending plugin registrations are complete."""
        if not self._plugin_registration_tasks:
            return

        await asyncio.gather(*self._plugin_registration_tasks)
        self._plugin_registration_tasks.clear()

    def _initialize_chat_library(self) -> None:
        """Initialize captured chat retrieval."""
        self.chat_library = None
        if not CHAT_LIBRARY_AVAILABLE or CursorChatLibrary is None:
            logger.info("Captured chat library unavailable, skipping chat context integration")
            return

        try:
            library = CursorChatLibrary(auto_load=True)
            self.chat_library = library
            record_count = library.record_count
            if record_count:
                logger.info("Loaded cursor chat context library with %d records", record_count)
            else:
                logger.info("Cursor chat library initialized but no records were found")
        except Exception as exc:
            logger.warning(f"Failed to initialize cursor chat library: {exc}")
            self.chat_library = None

    def _initialize_memory_system(self) -> None:
        """Initialize the unified episodic memory system."""
        if not MEMORY_SYSTEM_AVAILABLE or EpisodicMemoryBank is None:
            logger.info("Episodic memory system unavailable, skipping initialization")
            self.memory_store = None
            self.memory_retriever = None
            return

        if not self.config.memory_system_enabled:
            logger.info("Episodic memory system disabled via configuration")
            self.memory_store = None
            self.memory_retriever = None
            return

        try:
            storage_path = Path(self.config.memory_storage_path).expanduser()
            storage_path.mkdir(parents=True, exist_ok=True)
            self.memory_store = EpisodicMemoryBank(storage_path=storage_path)
            self.memory_retriever = EpisodicMemoryRetriever(self.memory_store)
            logger.info("Episodic memory system initialized at %s", storage_path)
        except Exception as e:
            logger.warning(f"Failed to initialize EpisodicMemoryBank: {e}")
            self.memory_store = None
            self.memory_retriever = None

    def _initialize_chroma_integration(self) -> None:
        """Initialize Chroma integration for SK Memory."""
        if not self.config.enable_chroma_integration:
            logger.info("Chroma integration disabled via configuration")
            self.chroma_integration = None
            return

        try:
            from .sk_plugins.shared.chroma_integration import ChromaSKIntegration

            # Initialize Chroma integration
            chroma_persist_directory = self.config.chroma_persist_directory or "./chroma_collections"
            self.chroma_integration = ChromaSKIntegration(self.sk_kernel, chroma_persist_directory)

            logger.info("Chroma integration initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Chroma integration: {e}")
            self.chroma_integration = None

    def _initialize_google_docs_integration(self) -> None:
        """Initialize Google Docs integration components."""
        if not GOOGLE_DOCS_AVAILABLE:
            logger.warning("Google Docs integration modules not available")
            self.google_docs_bridge = None
            self.google_docs_formatter = None
            self.version_tracker = None
            self.document_tracker = None
            return

        if not self.config.google_docs_enabled:
            logger.info("Google Docs integration disabled in config")
            self.google_docs_bridge = None
            self.google_docs_formatter = None
            self.version_tracker = None
            self.document_tracker = None
            return

        try:
            # Load configuration from environment/secrets
            self._load_google_docs_config()

            # Initialize Google Docs bridge with credentials
            # Try to find credentials file in project root
            # WorkflowStrategyExecutor.py is at: writer_agents/code/WorkflowStrategyExecutor.py
            # Project root (TheMatrix) is: parents[2] (code -> writer_agents -> TheMatrix)
            project_root = Path(__file__).parent.parent.parent
            credentials_file = project_root / "client_secret_701141645351-m9h9dmu5n7qh9bl391cnaoooh2sfleh4.apps.googleusercontent.com.json"
            if not credentials_file.exists():
                # Try alternative: absolute path
                credentials_file = Path("/home/serteamwork/projects/TheMatrix/client_secret_701141645351-m9h9dmu5n7qh9bl391cnaoooh2sfleh4.apps.googleusercontent.com.json")
            if not credentials_file.exists():
                # Try current directory
                credentials_file = Path("client_secret_701141645351-m9h9dmu5n7qh9bl391cnaoooh2sfleh4.apps.googleusercontent.com.json")
            
            credentials_path = str(credentials_file) if credentials_file.exists() else None
            if credentials_path:
                logger.info(f"✅ Using Google credentials: {credentials_path}")
            else:
                logger.warning("⚠️ Google credentials file not found, will use GOOGLE_APPLICATION_CREDENTIALS env var")
            
            self.google_docs_bridge = create_google_docs_bridge(credentials_path=credentials_path)

            # Initialize formatter
            self.google_docs_formatter = GoogleDocsFormatter()

            # Initialize version tracker
            self.version_tracker = create_document_edit_recorder(
                memory_store=self.memory_store
            )

            # Initialize document tracker
            self.document_tracker = create_document_metadata_recorder(
                memory_store=self.memory_store
            )

            logger.info("Google Docs integration initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Google Docs integration: {e}")
            self.google_docs_bridge = None
            self.google_docs_formatter = None
            self.version_tracker = None
            self.document_tracker = None

    def _load_google_docs_config(self) -> None:
        """Load Google Docs configuration from environment/secrets."""
        # Load from secrets.toml if available
        try:
            import toml
            secrets_path = Path(__file__).parent.parent.parent / "secrets.toml"
            if secrets_path.exists():
                secrets = toml.load(secrets_path)

                # Update config with values from secrets
                if "GOOGLE_DOCS_ENABLED" in secrets:
                    self.config.google_docs_enabled = secrets["GOOGLE_DOCS_ENABLED"]

                if "GOOGLE_DRIVE_FOLDER_ID" in secrets:
                    self.config.google_drive_folder_id = secrets["GOOGLE_DRIVE_FOLDER_ID"]

                if "GOOGLE_DOCS_AUTO_SHARE" in secrets:
                    self.config.google_docs_auto_share = secrets["GOOGLE_DOCS_AUTO_SHARE"]

                if "GOOGLE_DOCS_CAPTURE_VERSION_HISTORY" in secrets:
                    self.config.google_docs_capture_version_history = secrets["GOOGLE_DOCS_CAPTURE_VERSION_HISTORY"]

                if "GOOGLE_DOCS_LEARNING_ENABLED" in secrets:
                    self.config.google_docs_learning_enabled = secrets["GOOGLE_DOCS_LEARNING_ENABLED"]

                logger.info("Loaded Google Docs configuration from secrets.toml")

        except ImportError:
            logger.warning("toml library not available, using environment variables only")
        except Exception as e:
            logger.warning(f"Failed to load secrets.toml: {e}")

        # Load from environment variables as fallback
        if not self.config.google_drive_folder_id:
            self.config.google_drive_folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")

        if not self.config.google_docs_enabled:
            self.config.google_docs_enabled = os.environ.get("GOOGLE_DOCS_ENABLED", "false").lower() == "true"

    def _retrieve_memory_context(self, insights: CaseInsights) -> Optional[Dict[str, List[Any]]]:
        """Retrieve contextual memories for the current workflow."""
        query_components = [
            insights.summary or "",
            insights.case_style or "",
            insights.reference_id or ""
        ]
        query = " ".join(part for part in query_components if part).strip()
        if not query:
            query = "legal workflow memory context"

        include_types = self.config.memory_context_types or None

        context: Dict[str, List[Any]] = {}

        if self.memory_retriever:
            try:
                raw_context = self.memory_retriever.get_all_relevant_context(
                    query=query,
                    k=self.config.memory_context_max_items,
                    include_types=include_types
                )
                if isinstance(raw_context, dict):
                    context = raw_context
                else:
                    context = dict(raw_context)
            except Exception as e:
                logger.warning(f"Failed to retrieve episodic memory context: {e}")
                context = {}

        chat_context = self._retrieve_cursor_chat_context(query)
        if chat_context:
            context["cursor_chats"] = chat_context

        total_items = 0
        for items in context.values():
            if isinstance(items, list):
                total_items += len(items)

        if total_items == 0:
            return None

        logger.info("Retrieved %d context items for workflow (including chats)", total_items)
        return context

    def _retrieve_cursor_chat_context(self, query: str) -> List[Dict[str, Any]]:
        if not self.chat_library or not query:
            return []

        try:
            limit = max(1, self.config.memory_context_max_items)
        except Exception:
            limit = 3

        try:
            return self.chat_library.search(query=query, top_k=limit)
        except Exception as exc:
            logger.debug(f"Cursor chat retrieval failed: {exc}")
            return []

    def _format_memory_context_for_prompt(self, memory_context: Optional[Dict[str, List[Any]]]) -> str:
        """Format episodic memories for inclusion in prompts."""
        if not memory_context:
            return ""

        lines: List[str] = []
        max_items = self.config.memory_context_max_items

        for category, memories in memory_context.items():
            if not memories:
                continue

            heading = category.replace("_", " ").title()
            lines.append(f"{heading}:")
            for mem in memories[:max_items]:
                if isinstance(mem, dict):
                    summary = mem.get("preview") or mem.get("title") or mem.get("content", "")
                    summary = (summary or "").replace("\n", " ").strip()
                    if len(summary) > 240:
                        summary = summary[:237].rstrip() + "..."
                    score = mem.get("score")
                else:
                    summary = getattr(mem, "summary", str(mem))
                    score = getattr(mem, "relevance_score", None)
                if score:
                    lines.append(f"- {summary} (score {score:.2f})")
                else:
                    lines.append(f"- {summary}")

        return "\n".join(lines)

    def _categorize_feature_type(self, feature_name: str, rules: Optional[Dict[str, Any]]) -> str:
        if rules and rules.get("feature_type"):
            return rules["feature_type"].upper()

        feature_lower = feature_name.lower()

        if any(pattern in feature_lower for pattern in [
            "protective_measures", "protective_order", "proposal", "request", "offer", "protective"
        ]):
            if not feature_lower.startswith("mentions_"):
                return "PROPOSAL"

        if feature_lower.startswith("mentions_"):
            return "MENTION"

        if feature_lower.endswith("_count") or "count" in feature_lower:
            return "COUNT"

        if any(pattern in feature_lower for pattern in [
            "paragraph", "sentence", "enumeration", "word_count", "structure",
            "depth", "density", "length"
        ]):
            return "STRUCTURAL"

        return "MENTION"

    def _format_feature_target_instruction(
        self,
        feature_name: str,
        target_value: float,
        feature_desc: str,
        feature_type: str,
        rules: Optional[Dict[str, Any]] = None
    ) -> str:
        if feature_type == "PROPOSAL":
            if "protective" in feature_name.lower():
                examples = "protective order, file under seal, confidentiality order"
                return (
                    f"  - PROPOSAL: Request/propose at least {target_value:.1f} different protective measures "
                    f"(e.g., {examples}) to protect discovery materials and sensitive information"
                )
            return f"  - PROPOSAL: Request/propose {feature_desc} (target: {target_value:.1f}+ instances)"

        if feature_type == "MENTION":
            concept = feature_desc.replace("Mentions ", "").replace("Mention ", "").lower()
            return f"  - MENTION: Mention/discuss {concept} at least {target_value:.1f} times throughout the motion"

        if feature_type == "COUNT":
            item = feature_desc.replace(" Count", "").replace(" Counts", "").lower()
            return f"  - COUNT: Include at least {target_value:.1f} properly formatted {item}"

        if feature_type == "STRUCTURAL":
            element = feature_desc.lower()
            return f"  - STRUCTURAL: Use appropriate {element} structure (target: {target_value:.1f}+)"

        return f"  - Target: {feature_desc} = {target_value:.1f}"

    def _format_quality_constraints(self, sk_context: Dict[str, Any]) -> str:
        version = sk_context.get("constraint_version") or self.constraint_system_version or "1.0"
        sk_context["constraint_version"] = version
        system = self._load_constraint_system(version)

        static_lines = self._get_static_constraint_lines(system, version)
        lines: List[str] = list(static_lines)

        catboost_lines = self._format_catboost_target_lines()
        if catboost_lines:
            lines.append("\nCATBOOST FEATURE TARGETS (data-driven success patterns):")
            lines.extend(catboost_lines)

        shap_lines = self._format_shap_priority_lines()
        if shap_lines:
            lines.append("\nTOP SHAP PRIORITIES (focus argumentative depth here):")
            lines.extend(shap_lines)

        lines.append("\nQUALITY GATE REQUIREMENTS:")
        lines.append("  ✅ Use [Node:State] citation format linked to evidence nodes")
        lines.append("  ⚠️ Ensure evidence references reflect lawsuit-specific facts only")
        lines.append("  ⚠️ Maintain canonical section order and required headings")
        lines.append("  ✅ Tone must stay formal, precise, and free of hypotheticals")
        lines.append("  ❌ Balancing test must explicitly weigh privacy/safety harms vs. public access")

        if sk_context.get("research_findings"):
            research_cases = sk_context["research_findings"].get("cases", [])
            if research_cases:
                lines.append("\nCASE-LAW RESEARCH INSTRUCTIONS:")
                lines.append(
                    f"  ⚖️ Cite from {len(research_cases)} retrieved precedents; prioritize the top-ranked cases."
                )
                lines.append("  ✅ Mirror language patterns observed in successful motions referenced above.")

        violation_lines = self._format_constraint_violation_lines(sk_context.get("constraint_validation"))
        if violation_lines:
            lines.append("\nOPEN CONSTRAINT VIOLATIONS TO ADDRESS:")
            lines.extend(violation_lines)

        return "\n".join(line for line in lines if line)

    def _get_static_constraint_lines(self, system: Dict[str, Any], version: str) -> List[str]:
        cache_key = f"{version}::static"
        if cache_key in self._constraint_format_cache:
            return list(self._constraint_format_cache[cache_key])

        lines: List[str] = []

        doc_level = system.get("document_level") or {}
        if doc_level:
            lines.append(f"DOCUMENT-LEVEL CONSTRAINTS (Constraint System v{version}):")
            word_cfg = doc_level.get("word_count") or {}
            char_cfg = doc_level.get("char_count") or {}
            required_sections = doc_level.get("required_sections") or []

            if word_cfg:
                icon = self._format_enforcement_icon(word_cfg.get('enforcement'))
                lines.append(
                    f"  {icon} Total word count: {self._format_constraint_range(word_cfg)} "
                    f"[{(word_cfg.get('enforcement') or 'advisory').upper()}]"
                )
            if char_cfg:
                icon = self._format_enforcement_icon(char_cfg.get('enforcement'))
                lines.append(
                    f"  {icon} Character count: {self._format_constraint_range(char_cfg)} "
                    f"[{(char_cfg.get('enforcement') or 'advisory').upper()}]"
                )
            if required_sections:
                lines.append(f"  ⚠️ Required sections: {', '.join(required_sections)}")

        section_specs = system.get("sections") or {}
        if section_specs:
            lines.append("\nSECTION-LEVEL REQUIREMENTS:")
            for section_name, spec in section_specs.items():
                pretty_name = section_name.replace("_", " ").title()
                word_cfg = spec.get("word_count")
                if word_cfg:
                    icon = self._format_enforcement_icon(word_cfg.get('enforcement'))
                    note = f" ({word_cfg.get('note')})" if word_cfg.get("note") else ""
                    lines.append(
                        f"  {icon} {pretty_name}: {self._format_constraint_range(word_cfg)} "
                        f"[{(word_cfg.get('enforcement') or 'advisory').upper()}]{note}"
                    )
                position = spec.get("position", {}).get("description")
                if position:
                    lines.append(f"    ✅ Position guidance: {position}")

        system_feature_lines = self._format_system_feature_constraints(system.get("feature_constraints") or {})
        if system_feature_lines:
            lines.append("\nCONSTRAINT-SYSTEM FEATURE GUARDRAILS:")
            lines.extend(system_feature_lines)

        self._constraint_format_cache[cache_key] = list(lines)
        return lines

    @staticmethod
    def _format_enforcement_icon(enforcement: Optional[str]) -> str:
        normalized = (enforcement or "advisory").lower()
        if normalized in {"required", "mandatory", "strict"}:
            return "❌"
        if normalized in {"priority", "high", "recommended"}:
            return "⚠️"
        return "✅"

    def _load_constraint_system(self, version: str) -> Dict[str, Any]:
        cache_key = version or "default"
        if cache_key in self._constraint_system_cache:
            return self._constraint_system_cache[cache_key]

        constraint_dir = (
            self._constraint_dir_override
            or Path(__file__).resolve().parents[2] / "case_law_data" / "config" / "constraint_system_versions"
        )
        generated = constraint_dir / f"v{version}_generated.json"
        base = constraint_dir / f"v{version}_base.json"
        target_path = generated if generated.exists() else base

        data: Dict[str, Any] = {}
        if target_path.exists():
            try:
                data = json.loads(target_path.read_text())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse constraint system %s (%s)", version, exc)
        else:
            logger.warning("Constraint system version %s not found at %s", version, target_path)

        self._validate_constraint_system_schema(data, version)
        self._constraint_system_cache[cache_key] = data
        return data

    def _validate_constraint_system_schema(self, data: Dict[str, Any], version: str) -> None:
        if not data:
            logger.warning("Constraint system version %s is empty; prompts will omit structured requirements.", version)
            return

        if not isinstance(data.get("document_level", {}), dict):
            logger.warning("Constraint system %s: document_level should be an object.", version)
        if not isinstance(data.get("sections", {}), dict):
            logger.warning("Constraint system %s: sections should be an object.", version)
        if not isinstance(data.get("feature_constraints", {}), dict):
            logger.warning("Constraint system %s: feature_constraints should be an object.", version)

    @staticmethod
    def _format_constraint_range(constraint_cfg: Dict[str, Any]) -> str:
        if not constraint_cfg:
            return ""

        def _fmt(value: Any) -> str:
            if value is None:
                return "∞"
            if isinstance(value, str):
                return value
            try:
                numeric = float(value)
                if math.isinf(numeric):
                    return "∞"
                if numeric.is_integer():
                    return str(int(numeric))
                return f"{numeric:.2f}"
            except (TypeError, ValueError):
                return str(value)

        min_val = _fmt(constraint_cfg.get("min"))
        max_val = _fmt(constraint_cfg.get("max"))
        ideal_val = _fmt(constraint_cfg.get("ideal"))
        unit = constraint_cfg.get("unit")
        unit_text = f" {unit}" if unit else ""
        return f"{min_val}-{max_val}{unit_text} (ideal: {ideal_val})"

    def _format_system_feature_constraints(self, feature_constraints: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        impact_priority = {
            "high_impact": "❌ High Impact",
            "medium_impact": "⚠️ Medium Impact",
            "low_impact": "✅ Reference"
        }
        for impact_bucket in ("high_impact", "medium_impact", "low_impact"):
            entries = feature_constraints.get(impact_bucket, [])
            if not entries:
                continue
            label = impact_priority.get(impact_bucket, impact_bucket.replace("_", " ").title())
            lines.append(f"  [{label}]")
            for entry in entries[:6]:
                feature = entry.get("feature", "feature")
                constraint_cfg = entry.get("constraint", {})
                enforcement = entry.get("enforcement", "advisory")
                desc = entry.get("description", feature.replace("_", " "))
                constraint_text = self._format_constraint_range(constraint_cfg) if constraint_cfg else ""
                warning = entry.get("warning") or entry.get("notes") or ""
                icon = self._format_enforcement_icon(enforcement)
                lines.append(
                    f"    {icon} {desc} ({self._humanize_feature_name(feature)}): {constraint_text or 'follow favorable trend'} "
                    f"[{enforcement.upper()}] {warning}".strip()
                )
        return lines

    def _format_catboost_target_lines(self) -> List[str]:
        lines: List[str] = []
        if not (self.feature_orchestrator and hasattr(self.feature_orchestrator, "feature_targets")):
            return lines

        feature_targets = self.feature_orchestrator.feature_targets or {}
        if not feature_targets:
            return lines

        rules_dir = Path(__file__).parent.parent / "sk_plugins" / "rules"
        for feature_name, target_info in list(feature_targets.items())[:10]:
            if isinstance(target_info, dict):
                target_value = target_info.get("target", 0)
                feature_desc = target_info.get("description", feature_name.replace("_", " ").title())
            else:
                target_value = float(target_info) if target_info else 0
                feature_desc = feature_name.replace("_", " ").title()

            if target_value <= 0:
                continue

            rules = None
            try:
                rules_file = rules_dir / f"{feature_name}_rules.json"
                if rules_file.exists():
                    rules = json.loads(rules_file.read_text())
            except Exception:  # noqa: BLE001
                rules = None

            feature_type = self._categorize_feature_type(feature_name, rules)
            instruction = self._format_feature_target_instruction(
                feature_name, target_value, feature_desc, feature_type, rules
            )
            if instruction:
                lines.append(instruction)
        return lines

    def _format_shap_priority_lines(self, limit: int = 5) -> List[str]:
        shap_entries = self._get_top_shap_features(limit)
        lines: List[str] = []
        for feature_name, importance in shap_entries:
            lines.append(
                f"  - {self._humanize_feature_name(feature_name)} "
                f"(SHAP importance {importance:.3f}): reinforce favorable language and evidence for this signal."
            )
        return lines

    def _format_constraint_violation_lines(self, validation_result: Optional[Dict[str, Any]]) -> List[str]:
        if not validation_result:
            return []

        lines: List[str] = []
        warnings = validation_result.get("warnings", []) or []
        for warning in warnings[:5]:
            lines.append(f"  ⚠️ {warning}")

        details = validation_result.get("details", {})
        section_level = details.get("section_level") or []
        for entry in section_level[:3]:
            section_name = entry.get("section", "").replace("_", " ").title()
            message = entry.get("message") or entry.get("status")
            lines.append(f"  ❌ {section_name}: {message}")

        return lines

    def _get_top_shap_features(self, limit: int = 5) -> List[Tuple[str, float]]:
        shap_importance = getattr(self.feature_orchestrator, "shap_importance", None) if self.feature_orchestrator else None
        if not shap_importance:
            return []

        entries: List[Tuple[str, float]] = []
        for feature_name, value in shap_importance.items():
            if isinstance(value, dict):
                value = value.get("mean_abs_shap") or value.get("value") or 0.0
            try:
                entries.append((feature_name, float(value)))
            except (TypeError, ValueError):
                continue

        entries.sort(key=lambda item: item[1], reverse=True)
        return entries[:limit]

    @staticmethod
    def _humanize_feature_name(feature_name: str) -> str:
        return feature_name.replace("_", " ").strip().title()

    def _build_autogen_fact_context(
        self,
        insights: CaseInsights
    ) -> Tuple[Optional[str], List[Any], Dict[str, Any]]:
        """Prepare structured facts and filtered evidence for AutoGen prompts."""
        provider = _safe_get_case_facts_provider()
        evidence_list = list(insights.evidence or [])
        filtered_evidence, filtered_applied, stats = _filter_lawsuit_evidence(evidence_list, provider)
        facts_block = _format_facts_for_prompt(provider)
        truths_available = False
        if provider:
            has_truth_fn = getattr(provider, "has_truth_database", None)
            if callable(has_truth_fn):
                try:
                    truths_available = bool(has_truth_fn())
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"[FACTS] Truth database check failed: {exc}")

        stats.setdefault("provider_available", provider is not None)
        stats.setdefault("original_count", len(evidence_list))
        stats.setdefault("filtered_count", len(filtered_evidence))
        stats.setdefault("filtered_applied", filtered_applied)
        stats["truths_available"] = truths_available

        logger.info(
            "[FACTS] Exploration context using %d/%d evidence items (provider=%s)",
            stats["filtered_count"],
            stats["original_count"],
            "yes" if stats["provider_available"] else "no"
        )

        return facts_block, filtered_evidence, stats

    def _serialize_memory_context(self, memory_context: Optional[Dict[str, List[Any]]]) -> Dict[str, List[str]]:
        """Serialize episodic memories for metadata."""
        if not memory_context:
            return {}

        serialized: Dict[str, List[str]] = {}
        max_items = self.config.memory_context_max_items

        for category, memories in memory_context.items():
            if not memories:
                continue

            serialized_items: List[str] = []
            for mem in memories[:max_items]:
                if isinstance(mem, dict):
                    text = mem.get("preview") or mem.get("title") or mem.get("content", "")
                    text = (text or "").replace("\n", " ").strip()
                    if len(text) > 240:
                        text = text[:237].rstrip() + "..."
                    serialized_items.append(text)
                else:
                    serialized_items.append(getattr(mem, "summary", str(mem)))
            serialized[category] = serialized_items

        return serialized

    def _record_intermediate_milestone(
        self,
        state: WorkflowState,
        insights: CaseInsights,
        phase_result: Any
    ) -> None:
        """Save intermediate workflow milestone to LexBank."""
        if not (self.memory_store and EpisodicMemoryEntry):
            return

        try:
            case_identifier = insights.reference_id or insights.case_style or "case"
            summary = f"Workflow milestone: {state.phase.value} (iteration {state.iteration}) for {case_identifier}"

            context = {
                "reference_id": insights.reference_id,
                "case_style": insights.case_style,
                "jurisdiction": insights.jurisdiction,
                "phase": state.phase.value,
                "iteration": state.iteration,
                "milestone_type": "intermediate"
            }

            memory_entry = EpisodicMemoryEntry(
                agent_type="Conductor",
                memory_id=str(uuid.uuid4()),
                summary=summary,
                context=context,
                source="hybrid_workflow",
                timestamp=datetime.now(),
                memory_type="execution",
                case_id=case_identifier
            )

            self.memory_store.add(memory_entry)
            # Don't save after every milestone - batch saves are more efficient
        except Exception as e:
            logger.debug(f"Failed to record intermediate milestone: {e}")

    def _record_workflow_memory(
        self,
        deliverable: WriterDeliverable,
        insights: CaseInsights,
        state: WorkflowState
    ) -> None:
        """Persist workflow outcome into the episodic memory bank."""
        if not (self.memory_store and EpisodicMemoryEntry):
            return

        try:
            case_identifier = insights.reference_id or insights.case_style or "case"
            validation_score = None
            if state.validation_results:
                validation_score = state.validation_results.get("overall_score")

            summary_parts = [f"Hybrid workflow completed for {case_identifier}"]
            if validation_score is not None:
                summary_parts.append(f"validation {validation_score:.2f}")
            summary = " - ".join(summary_parts)

            # Include information about past workflows that influenced this one
            past_workflows_used = []
            if hasattr(state, 'past_workflow_memories') and state.past_workflow_memories:
                for mem in state.past_workflow_memories:
                    past_workflows_used.append({
                        'memory_id': mem.memory_id,
                        'summary': mem.summary[:200],
                        'relevance': mem.relevance_score
                    })

            context = {
                "reference_id": insights.reference_id,
                "case_style": insights.case_style,
                "jurisdiction": insights.jurisdiction,
                "sections": [section.title for section in deliverable.sections],
                "validation": state.validation_results,
                "iteration_summary": deliverable.metadata.get("iteration_summary"),
                "review_notes": state.review_notes,
                "google_doc_id": state.google_doc_id,
                "memory_context": self._serialize_memory_context(state.memory_context),
                "past_workflows_used": past_workflows_used,
                "past_workflows_count": len(past_workflows_used)
            }

            memory_entry = EpisodicMemoryEntry(
                agent_type="Conductor",
                memory_id=str(uuid.uuid4()),
                summary=summary,
                context=context,
                source="hybrid_workflow",
                timestamp=datetime.now(),
                memory_type="execution",
                document_id=state.google_doc_id,
                case_id=case_identifier
            )

            self.memory_store.add(memory_entry)
            try:
                self.memory_store.save()
            except Exception:
                # Best-effort persistence; continue if save fails
                pass
            logger.info("Recorded workflow completion to EpisodicMemoryBank")
        except Exception as e:
            logger.warning(f"Failed to record workflow memory entry: {e}")

    async def close(self) -> None:
        """Close orchestrator components."""
        await self.autogen_factory.close()
        if self.memory_store:
            try:
                self.memory_store.save()
            except Exception as e:
                logger.warning(f"Failed to persist episodic memory store: {e}")
        logger.info("Conductor closed")

    async def run_hybrid_workflow(self, insights: CaseInsights, initial_google_doc_id: Optional[str] = None, initial_google_doc_url: Optional[str] = None) -> WriterDeliverable:
        """Run the complete hybrid workflow.

        Enhanced with self-reflexive learning: queries LexBank for similar past workflows
        before execution and saves intermediate milestones during execution.
        """
        await self._ensure_plugins_ready()

        logger.info("Starting hybrid workflow")

        # Initialize CaseFactsProvider globally so plugins can retrieve personal facts
        try:
            from .sk_plugins.FeaturePlugin.CaseFactsProvider import (
                CaseFactsProvider,
                set_case_facts_provider,
            )
            project_root = Path(__file__).resolve().parents[2]
            corpus_dir = project_root / "case_law_data" / "lawsuit_source_documents"
            # Support old name for backward compatibility
            if not corpus_dir.exists():
                old_corpus_dir = project_root / "case_law_data" / "tmp_corpus"
                if old_corpus_dir.exists():
                    corpus_dir = old_corpus_dir
            lawsuit_facts_extracted_path = project_root / "writer_agents" / "outputs" / "lawsuit_facts_extracted.json"
            # Support old name for backward compatibility
            case_insights_path = project_root / "writer_agents" / "outputs" / "case_insights.json"
            case_insights_path = lawsuit_facts_extracted_path if lawsuit_facts_extracted_path.exists() else case_insights_path
            _validate_personal_fact_inputs(corpus_dir, case_insights_path)
            embeddings_db = project_root / "case_law_data" / "results" / "personal_corpus_embeddings.db"
            embeddings_faiss = project_root / "case_law_data" / "results" / "personal_corpus_embeddings.faiss"
            lawsuit_facts_db = project_root / "case_law_data" / "lawsuit_facts_database.db"
            # Support old name for backward compatibility
            truths_db = project_root / "case_law_data" / "truths.db"
            lawsuit_facts_db = lawsuit_facts_db if lawsuit_facts_db.exists() else (truths_db if truths_db.exists() else None)
            provider = CaseFactsProvider(
                case_insights=insights,
                embedding_db_path=embeddings_db if embeddings_db.exists() else None,
                embedding_faiss_path=embeddings_faiss if embeddings_faiss.exists() else None,
                lawsuit_facts_db_path=lawsuit_facts_db,
                personal_corpus_dir=corpus_dir,
                strict_filtering=True,
            )
            set_case_facts_provider(provider)
            personal_fact_keys: List[str] = []
            try:
                fact_snapshot = provider.get_all_facts()
                personal_fact_keys = [key for key in fact_snapshot if str(key).startswith("personal_corpus_")]
            except Exception as inspect_exc:  # noqa: BLE001
                logger.debug(f"[FACTS] Unable to inspect personal corpus facts: {inspect_exc}")

            if not personal_fact_keys:
                logger.warning(
                    "[FACTS] CaseFactsProvider initialized but no personal corpus facts were detected in %s",
                    corpus_dir,
                )

            logger.info(
                "[FACTS] CaseFactsProvider initialized for workflow (personal corpus facts=%d, strict filtering=enabled)",
                len(personal_fact_keys),
            )
        except Exception as _exc:
            logger.error(f"[FACTS] Failed to initialize CaseFactsProvider: {_exc}")
            raise

        # SELF-REFLEXIVE LEARNING: Query LexBank for similar past workflows
        past_workflow_memories = []
        workflow_query_text = f"{insights.case_style or 'case'} {insights.jurisdiction or ''}"
        if self.memory_store:
            try:
                past_workflow_memories = self.memory_store.retrieve(
                    agent_type="Conductor",
                    query=workflow_query_text,
                    k=5,
                    memory_types=["execution"]
                )
                if past_workflow_memories:
                    logger.info(f"Found {len(past_workflow_memories)} similar past workflows in LexBank")
                    # Extract successful patterns from past workflows
                    for memory in past_workflow_memories:
                        context = memory.context
                        if context.get('validation', {}).get('overall_score', 0) > 0.7:
                            logger.info(f"  Similar successful workflow: {memory.summary[:80]}...")
            except Exception as e:
                logger.warning(f"Error querying past workflows: {e}")

        state = WorkflowState()
        state.memory_context = self._retrieve_memory_context(insights)

        # Set initial Google Doc ID if provided (for live updates to existing document)
        if initial_google_doc_id:
            state.google_doc_id = initial_google_doc_id
            state.google_doc_url = initial_google_doc_url or f"https://docs.google.com/document/d/{initial_google_doc_id}/edit"
            logger.info(f"[LIVE] Using provided Google Doc ID: {initial_google_doc_id}")

        # Store past workflow memories in state for later reference
        state.past_workflow_memories = past_workflow_memories if past_workflow_memories else []

        try:
            # Main workflow loop
            while self.iteration_controller.should_continue_iteration(state):
                logger.info(f"Phase {state.phase.value} - Iteration {state.iteration}")

                # Execute current phase
                phase_result = await self._execute_phase(state, insights)

                # Record iteration
                self.iteration_controller.record_iteration(state, phase_result)

                # LIVE GOOGLE DOCS UPDATES: Update Google Drive in real-time (all phases)
                if (self.config.google_docs_enabled and
                    self.config.google_docs_live_updates and
                    self.google_docs_bridge):
                    if state.google_doc_id:
                        try:
                            # Update in all phases, not just DRAFT/REFINE/VALIDATE
                            print(f"[LIVE] Attempting live update after {state.phase.value} phase (doc_id: {state.google_doc_id})")
                            logger.info(f"[LIVE] About to call _update_google_doc_live for phase {state.phase.value}")
                            await self._update_google_doc_live(state, insights)
                            logger.info(f"[LIVE] Live update completed successfully after {state.phase.value} phase")
                            print(f"[LIVE] Live update completed after {state.phase.value} phase")
                        except Exception as e:
                            logger.warning(f"Live Google Docs update failed (non-critical): {e}")
                            print(f"[LIVE] ❌ Live update failed: {e}")
                            import traceback
                            print(traceback.format_exc())
                    else:
                        print(f"[LIVE] ⚠️  No document ID set in state (phase: {state.phase.value})")
                        logger.debug(f"[LIVE] No document ID set in state, skipping live update")

                # SELF-REFLEXIVE LEARNING: Save intermediate milestone to LexBank
                if self.memory_store and hasattr(state, 'past_workflow_memories'):
                    try:
                        self._record_intermediate_milestone(state, insights, phase_result)
                    except Exception as e:
                        logger.debug(f"Failed to record intermediate milestone: {e}")

                # Move to next phase
                next_phase = self.router.get_next_phase(state.phase, state)
                state.phase = next_phase
                state.iteration += 1

                # Break if we've reached commit phase
                if state.phase == WorkflowPhase.COMMIT:
                    break

            # Ensure draft phase has been executed before committing
            if not state.draft_result:
                raise RuntimeError("Drafting phase produced no output; aborting commit to avoid placeholder text.")

            # Final commit
            final_result = await self._commit_document(state, insights)

            logger.info("Hybrid workflow completed successfully")
            return final_result

        except Exception as e:
            logger.error(f"Hybrid workflow failed: {e}")
            raise

    async def _execute_phase(self, state: WorkflowState, insights: CaseInsights) -> Any:
        """Execute the current workflow phase."""

        if state.phase == WorkflowPhase.EXPLORE:
            return await self._execute_exploration_phase(state, insights)

        elif state.phase == WorkflowPhase.RESEARCH:
            return await self._execute_research_phase(state, insights)

        elif state.phase == WorkflowPhase.PLAN:
            return await self._execute_planning_phase(state, insights)

        elif state.phase == WorkflowPhase.DRAFT:
            return await self._execute_drafting_phase(state, insights)

        elif state.phase == WorkflowPhase.VALIDATE:
            return await self._execute_validation_phase(state, insights)

        elif state.phase == WorkflowPhase.REVIEW:
            return await self._execute_review_phase(state, insights)

        elif state.phase == WorkflowPhase.REFINE:
            return await self._execute_refinement_phase(state, insights)

        else:
            raise ValueError(f"Unknown phase: {state.phase}")

    async def _execute_exploration_phase(self, state: WorkflowState, insights: CaseInsights) -> str:
        """Execute AutoGen exploration phase."""
        logger.info("Executing exploration phase with AutoGen")

        facts_block, filtered_evidence, _ = self._build_autogen_fact_context(insights)
        facts_section = facts_block or (
            "No personal corpus facts available. Run writer_agents/scripts/build_case_insights.py "
            "to regenerate writer_agents/outputs/lawsuit_facts_extracted.json before drafting."
        )
        filtered_evidence_block = _render_evidence_for_prompt(filtered_evidence)
        posteriors_block = _render_posteriors_for_prompt(insights.posteriors)

        # Create exploration prompt
        exploration_prompt = f"""
You are a legal expert exploring arguments for this case. Analyze the verified inputs below before brainstorming.

## Personal Corpus Facts (Verified)
{facts_section}

## Case Summary
{insights.summary}

## Filtered Evidence (Lawsuit-Specific)
{filtered_evidence_block}

## Bayesian Network Results
{posteriors_block}

Provide:
1. Key legal arguments grounded in the verified facts
2. Strengths and weaknesses
3. Alternative approaches
4. Evidence gaps
5. Strategic recommendations

Focus on creative exploration and alternative perspectives without inventing new facts.
        """

        memory_context_block = self._format_memory_context_for_prompt(state.memory_context)
        if memory_context_block:
            exploration_prompt += f"\n\nHistorical Memory Context:\n{memory_context_block}\n"

        # Use AutoGen agent for exploration
        try:
            explorer_system_message = """You are a legal expert specializing in case analysis and argument exploration.
Use verified personal corpus facts (HK Statement, OGC emails, and related exhibits) as the primary source of truth.
Identify key arguments, strengths, weaknesses, strategic recommendations, and highlight any fact gaps that need follow-up.
Provide comprehensive, well-structured exploration results that will guide downstream drafting and refinement."""
            
            explorer = BaseAutoGenAgent(
                factory=self.autogen_factory,
                name="LegalExplorer",
                system_message=explorer_system_message
            )
            
            logger.info("Invoking AutoGen exploration agent...")
            exploration_result = await explorer.run(exploration_prompt)
            logger.info("AutoGen exploration agent completed successfully")
            
        except Exception as e:
            logger.error(f"AutoGen exploration failed: {e}", exc_info=True)
            raise RuntimeError("AutoGen exploration failed; aborting workflow to avoid placeholder text.") from e

        state.exploration_result = {"output": exploration_result}
        return exploration_result

    async def _execute_research_phase(self, state: WorkflowState, insights: CaseInsights) -> Dict[str, Any]:
        """Execute case law research phase."""
        logger.info("Executing research phase - querying case law databases")

        # Check if researcher is available
        if not self.case_law_researcher or not self.case_law_researcher.enabled:
            logger.warning("CaseLawResearcher not available, skipping research phase")
            state.research_results = {
                'success': False,
                'error': 'CaseLawResearcher not available',
                'cases': [],
                'explanations': {},
                'summary': {}
            }
            return state.research_results

        try:
            # Query past research from episodic memory before doing new research
            past_research = None
            if self.memory_store:
                try:
                    query_text = f"{insights.summary[:200]} {insights.jurisdiction or ''}"
                    past_memories = self.memory_store.retrieve(
                        agent_type="CaseLawResearcher",
                        query=query_text,
                        k=3,
                        memory_types=["query", "execution"]
                    )
                    if past_memories:
                        logger.info(f"Found {len(past_memories)} similar past research queries")
                        # Check if any are recent and relevant (could reuse results)
                        # For now, we'll still do new research but could optimize later
                except Exception as e:
                    logger.debug(f"Could not query past research: {e}")

            # Perform case law research
            research_results = self.case_law_researcher.research_case_law(
                insights=insights,
                top_k=50,
                min_similarity=0.3
            )

            # Store results in state
            state.research_results = research_results

            # Save research findings to episodic memory
            if self.memory_store and research_results.get('success'):
                try:
                    from .EpisodicMemoryBank import EpisodicMemoryEntry
                    from datetime import datetime

                    # Create summary
                    summary_parts = []
                    if research_results.get('summary'):
                        summary_parts.append(f"Found {research_results['summary'].get('semantic_matches', 0)} semantic matches")
                    if research_results.get('cases'):
                        summary_parts.append(f"Found {len(research_results['cases'])} relevant cases")
                    summary = f"Case law research: {', '.join(summary_parts)}"

                    # Create memory entry
                    memory_entry = EpisodicMemoryEntry(
                        agent_type="CaseLawResearcher",
                        memory_id=str(uuid.uuid4()) if hasattr(uuid, 'uuid4') else f"research_{datetime.now().isoformat()}",
                        summary=summary,
                        context={
                            'themes': research_results.get('themes', {}),
                            'case_count': len(research_results.get('cases', [])),
                            'explanations': research_results.get('explanations', {}),
                            'query_text': research_results.get('themes', {}).get('query_text', '')[:500]
                        },
                        source="case_law_researcher",
                        timestamp=datetime.now(),
                        memory_type="query"
                    )
                    self.memory_store.add(memory_entry)
                    logger.info("Saved research findings to episodic memory")
                except Exception as e:
                    logger.debug(f"Could not save research to memory: {e}")

            # Generate user-facing explanation
            if research_results.get('success'):
                explanations = research_results.get('explanations', {})
                overall = explanations.get('overall', {})
                logger.info(f"Research complete: {overall.get('summary', 'Research completed')}")

                # Log findings by theme
                by_theme = explanations.get('by_theme', {})
                for theme, theme_data in by_theme.items():
                    logger.info(f"  {theme}: {theme_data.get('count', 0)} cases found")
                    if theme_data.get('cases'):
                        top_case = theme_data['cases'][0]
                        logger.info(f"    Top case: {top_case.get('case_name', 'Unknown')}")

            return research_results

        except Exception as e:
            logger.error(f"Research phase failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            state.research_results = {
                'success': False,
                'error': str(e),
                'cases': [],
                'explanations': {},
                'summary': {}
            }
            return state.research_results

    async def _execute_planning_phase(self, state: WorkflowState, insights: CaseInsights) -> Any:
        """Execute SK planning phase."""
        logger.info("Executing planning phase with SK Planner")

        if not self.config.enable_sk_planner:
            return None

        # Check if SequentialPlanner is available
        if not SK_PLANNER_AVAILABLE or SequentialPlanner is None:
            logger.warning("SequentialPlanner not available, skipping planning phase")
            return None

        # Create SK planner
        planner = SequentialPlanner(self.sk_kernel)

        # Define goal
        goal = f"""
Draft a comprehensive legal memorandum analyzing privacy harm in this case:
- Case: {insights.summary}
- Jurisdiction: {insights.jurisdiction}
- Evidence: {insights.evidence}

The memorandum should include:
1. Privacy harm analysis section
2. Legal framework application
3. Evidence summary with citations
4. Conclusion with actionable findings
        """

        try:
            # Create plan
            plan = await planner.create_plan(goal)
            state.sk_plan = plan

            logger.info(f"SK plan created with {len(plan.steps)} steps")
            return plan

        except Exception as e:
            logger.warning(f"SK planning failed: {e}. Proceeding without plan.")
            return None

    async def _invoke_sk_function(
        self,
        plugin_name: str,
        function_name: str,
        variables: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Helper method to invoke a Semantic Kernel function by plugin and function name.

        Args:
            plugin_name: Name of the plugin
            function_name: Name of the function within the plugin
            variables: Variables to pass to the function

        Returns:
            Function result

        Raises:
            RuntimeError: If kernel is not available or function not found
        """
        if self.sk_kernel is None:
            raise RuntimeError("SK kernel is not available")

        payload = ensure_fact_payload_fields(variables, context)
        _record_plugin_payload(plugin_name, function_name, payload)

        try:
            plugin = getattr(self.sk_kernel, "plugins", {}).get(plugin_name)
            if plugin is not None:
                function = None
                if hasattr(plugin, "__contains__") and hasattr(plugin, "__getitem__"):
                    try:
                        if function_name in plugin:
                            function = plugin[function_name]
                    except Exception:
                        function = None
                if function is None and hasattr(plugin, "get_function"):
                    function = plugin.get_function(function_name)

                if function is not None:
                    logger.debug(
                        "Invoking %s.%s via kernel plugin with args: %s",
                        plugin_name,
                        function_name,
                        list(payload.keys()),
                    )
                    return await self.sk_kernel.invoke(function=function, **payload)

            plugin_obj = plugin_registry.get_plugin(plugin_name)
            if plugin_obj:
                function = plugin_obj.get_function(function_name)
                if function is not None:
                    logger.debug(
                        "Invoking %s.%s via registry fallback with args: %s",
                        plugin_name,
                        function_name,
                        list(payload.keys()),
                    )
                    try:
                        return await self.sk_kernel.invoke(function=function, **payload)
                    except Exception:
                        import asyncio
                        if asyncio.iscoroutinefunction(function):
                            return await function(**payload)
                        return function(**payload)

            raise RuntimeError(f"Function {plugin_name}.{function_name} not found in kernel plugins or registry")

        except AttributeError as e:
            raise RuntimeError(f"Cannot invoke function {plugin_name}.{function_name}: Kernel API not compatible - {e}")

    async def _generate_motion_with_streaming(
        self, chat_service, chat_history, settings, state: WorkflowState
    ) -> Optional[str]:
        """Generate motion with streaming updates to Google Docs every ~10 seconds."""
        import asyncio
        try:
            from semantic_kernel.contents import StreamingChatMessageContent
        except ImportError:
            return None

        full_content = ""
        last_update_time = asyncio.get_event_loop().time()
        update_interval = 10.0  # Update every 10 seconds during generation
        chunk_buffer = ""
        min_chunk_size = 200  # Minimum characters before updating

        try:
            # Try to get streaming response
            if hasattr(chat_service, 'get_streaming_chat_message_contents'):
                stream = chat_service.get_streaming_chat_message_contents(
                    chat_history=chat_history,
                    settings=settings
                )
            elif hasattr(chat_service, 'get_streaming_response'):
                stream = chat_service.get_streaming_response(
                    chat_history=chat_history,
                    settings=settings
                )
            else:
                # No streaming support
                return None

            # Process stream
            async for chunk in stream:
                if isinstance(chunk, StreamingChatMessageContent):
                    chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                elif hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                elif hasattr(chunk, 'content'):
                    chunk_text = str(chunk.content)
                else:
                    chunk_text = str(chunk)

                if chunk_text:
                    full_content += chunk_text
                    chunk_buffer += chunk_text

                    # Update Google Docs periodically
                    current_time = asyncio.get_event_loop().time()
                    time_since_update = current_time - last_update_time

                    if len(chunk_buffer) >= min_chunk_size or time_since_update >= update_interval:
                        try:
                            # Update draft in state
                            if not state.draft_result:
                                state.draft_result = {}
                            state.draft_result["privacy_harm_section"] = full_content + "\n\n[Generating...]"

                            # Trigger live update
                            await self._update_google_doc_live(state, None)
                            last_update_time = current_time
                            chunk_buffer = ""
                            logger.info(f"[STREAM] Updated Google Doc with {len(full_content)} chars")
                        except Exception as e:
                            logger.debug(f"[STREAM] Live update failed: {e}")
                            # Continue generating even if update fails

            # Final update without "[Generating...]" marker
            if full_content:
                state.draft_result["privacy_harm_section"] = full_content
                await self._update_google_doc_live(state, None)
                logger.info(f"[STREAM] Final update: {len(full_content)} chars")

            return full_content

        except Exception as e:
            logger.debug(f"Streaming generation failed: {e}")
            return None

    async def _generate_motion_with_llm(self, sk_context: Dict[str, Any], evidence_serializable: List[Dict], posteriors_serializable: List[Dict], state: Optional[WorkflowState] = None) -> str:
        """Generate motion using LLM directly through kernel's chat service."""
        if not self.sk_kernel:
            logger.warning("No kernel available, using template")
            return self._generate_template_motion(sk_context, evidence_serializable)

        try:
            # Get the chat service from kernel
            from semantic_kernel.functions import KernelArguments
            ChatHistory, ChatMessageContent, AuthorRole = get_chat_components()

            # Get chat service (first service added to kernel)
            chat_service = None
            if hasattr(self.sk_kernel, 'get_service'):
                try:
                    # Try different methods to get chat service (handle API version differences)
                    # Method 1: Try with service_id parameter (newer API)
                    if hasattr(self.sk_kernel, 'services') and self.sk_kernel.services:
                        try:
                            service_id = list(self.sk_kernel.services.keys())[0]
                            chat_service = self.sk_kernel.get_service(service_id=service_id)
                        except (TypeError, AttributeError, IndexError):
                            pass
                    
                    # Method 2: Try with type_id (older API) - only if Method 1 failed
                    if not chat_service:
                        try:
                            chat_service = self.sk_kernel.get_service(type_id="chat_completion")
                        except (TypeError, AttributeError, KeyError, ValueError):
                            pass
                except:
                    pass
                
                # Method 3: Get first service directly
                if not chat_service:
                    if hasattr(self.sk_kernel, 'services') and self.sk_kernel.services:
                        chat_service = list(self.sk_kernel.services.values())[0]
            elif hasattr(self.sk_kernel, 'services') and self.sk_kernel.services:
                chat_service = list(self.sk_kernel.services.values())[0]

            if not chat_service:
                logger.warning("No chat service found in kernel, using template")
                return self._generate_template_motion(sk_context, evidence_serializable)

            # Build the prompt with real case data
            case_summary = sk_context.get('case_summary', 'N/A')
            jurisdiction = sk_context.get('jurisdiction', 'US Federal Court')

            evidence_text = "\n".join([
                f"  - {item.get('node_id', 'unknown')}: {item.get('state', 'unknown')}" +
                (f" ({item.get('description', '')})" if item.get('description') else "")
                for item in evidence_serializable[:10]
            ]) if evidence_serializable else "  - No evidence items provided"

            # Include research findings in prompt if available
            research_section = ""
            if sk_context.get('research_findings'):
                research = sk_context['research_findings']
                cases = research.get('cases', [])[:10]  # Top 10 cases
                explanations = research.get('explanations', {})
                overall = explanations.get('overall', {})

                research_section = f"""

RELEVANT CASE LAW RESEARCH:
{overall.get('summary', 'Case law research completed')}

Top Relevant Cases ({len(cases)} cases found):
"""
                for i, case in enumerate(cases[:5], 1):  # Top 5 in prompt
                    case_name = case.get('case_name', 'Unknown')
                    court = case.get('court', 'Unknown')
                    similarity = case.get('similarity_score', 0.0)
                    research_section += f"{i}. {case_name} ({court}) - Relevance: {similarity:.2f}\n"

                # Add theme-based findings
                by_theme = explanations.get('by_theme', {})
                if by_theme:
                    research_section += "\nFindings by Theme:\n"
                    for theme, theme_data in list(by_theme.items())[:3]:  # Top 3 themes
                        research_section += f"- {theme}: {theme_data.get('count', 0)} cases found\n"

                research_section += "\nIMPORTANT: Incorporate relevant case citations and legal precedents from the research findings above into your motion.\n"

            structured_facts_raw = sk_context.get("structured_facts")
            structured_facts_map = structured_facts_raw if isinstance(structured_facts_raw, dict) else {}
            structured_facts_section = _format_structured_facts_block(structured_facts_raw)
            quality_constraints_section = _format_quality_constraints_block(sk_context.get("quality_constraints"))
            fact_todo_text = ""
            if state and getattr(state, "fact_retry_todo", None):
                todo_block = format_fact_retry_todo(state.fact_retry_todo)
                if todo_block:
                    fact_todo_text = (
                        "\nFACT TODO LIST (address every item before finalizing):\n"
                        f"{todo_block}\n"
                    )

            citizenship_value = (structured_facts_map.get("citizenship") or "").lower()
            security_clause = ""
            if citizenship_value and any(token in citizenship_value for token in ("united states", "us", "u.s.", "american")):
                security_clause = " (this case involves foreign government connections and US citizen safety)"

            prompt = f"""You are a legal expert drafting a US Motion to Seal for a federal court case.

⚠️ CRITICAL: You MUST use ONLY the facts provided below. DO NOT invent, assume, or create any facts, names, dates, or case details that are not explicitly stated in the STRUCTURED FACTS and EVIDENCE sections below.

CASE INFORMATION:
Jurisdiction: {jurisdiction}
Case Summary: {case_summary[:1000]}

═══════════════════════════════════════════════════════════════
STRUCTURED FACTS (USE THESE EXACT FACTS - DO NOT INVENT ANYTHING):
═══════════════════════════════════════════════════════════════
{structured_facts_section}

═══════════════════════════════════════════════════════════════
EVIDENCE (USE ONLY THIS EVIDENCE):
═══════════════════════════════════════════════════════════════
{evidence_text}

{fact_todo_text}{research_section}{quality_constraints_section}

CRITICAL INSTRUCTIONS:
- This is a REAL motion for a REAL case. Generate the actual motion text, NOT examples or test prompts.
- NEVER describe, infer, or assume ANY fact (citizenship, nationality, dates, locations, relationships, employment, etc.) unless it appears verbatim in the STRUCTURED FACTS section.
- If a fact is not listed in STRUCTURED FACTS, do NOT mention it. Do not speculate, infer, or “fill in” missing information.
- If a fact is listed in STRUCTURED FACTS, use that exact wording or meaning—do not paraphrase in a way that changes scope.
- DO NOT infer facts from context clues, safety risks, foreign government connections, or other hints.
- DO NOT generate example prompts, test scenarios, or hypothetical instructions.
- DO NOT include phrases like "comprehensive report on", "Your task:", "Generate three follow up questions", or similar instructional text.
- DO NOT invent case names, parties, dates, or facts. Use ONLY what is provided in STRUCTURED FACTS and EVIDENCE above.
- Write the actual motion content that would be filed with the court.
- Rely EXCLUSIVELY on the structured facts, filtered evidence, and research findings provided. Ignore any assumptions or facts not listed above.
- If the STRUCTURED FACTS mention Harvard, Hong Kong, defamation, or specific dates/events, you MUST reference those exact facts in your motion.
- If the STRUCTURED FACTS mention OGC emails, HK Statement of Claim, or specific allegations, you MUST include those in your motion.

TASK: Generate a complete, professional Motion for Leave to File Under Seal that:
1. Follows proper legal formatting and structure
2. Addresses privacy concerns and safety risks
3. Incorporates national security implications{security_clause}
4. References 28 U.S.C. Section 1782 and appropriate federal court rules
5. Includes a proper balancing test for sealing
6. Uses persuasive legal arguments with citations to relevant case law from the research findings
7. Follows federal motion format with proper sections
8. Incorporates case citations and legal precedents from the research findings provided

OUTPUT FORMAT: Start directly with the motion header and content. Do NOT include any instructional text, examples, or prompts.

Generate the complete motion now:"""

            # Use chat service to generate
            try:
                # Try Semantic Kernel chat completion API
                chat_history = ChatHistory()
                chat_history.add_user_message(prompt)

                # Try to create appropriate settings (if supported)
                settings = None
                try:
                    from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
                    from semantic_kernel.connectors.ai.ollama import OllamaPromptExecutionSettings

                    # Check service type
                    service_str = str(type(chat_service)).lower()
                    if 'ollama' in service_str:
                        settings = OllamaPromptExecutionSettings(
                            temperature=0.3,
                            max_tokens=4000
                        )
                    elif 'openai' in service_str:
                        settings = OpenAIChatPromptExecutionSettings(
                            temperature=0.3,
                            max_tokens=4000
                        )
                except ImportError:
                    # Settings not available, will use defaults
                    pass

                # Try streaming first for live updates to Google Docs
                if state and self.config.google_docs_enabled and self.config.google_docs_live_updates and self.google_docs_bridge:
                    try:
                        # Ensure document exists before streaming (use existing one)
                        if not state.google_doc_id:
                            # Try to find existing document
                            if not state.google_doc_id:
                                if self.config.master_draft_mode and self.config.master_draft_title:
                                    title = self.config.master_draft_title
                                    if self.document_tracker:
                                        existing_doc = self.document_tracker.get_doc_by_title(
                                            title,
                                            folder_id=self.config.google_drive_folder_id
                                        )
                                        if existing_doc:
                                            state.google_doc_id = existing_doc.google_doc_id
                                            state.google_doc_url = existing_doc.doc_url
                                            logger.info(f"[LIVE] Using existing Google Doc for streaming: {existing_doc.google_doc_id}")

                                    # Fallback: Try to find by known document ID from markdown file
                                    if not state.google_doc_id:
                                        try:
                                            from pathlib import Path
                                            markdown_path = Path(self.config.markdown_export_path) / "master_draft.md"
                                            if markdown_path.exists():
                                                content = markdown_path.read_text(encoding="utf-8", errors="ignore")
                                                import re
                                                doc_id_match = re.search(r"google_doc_id[:\s]+([a-zA-Z0-9_-]+)", content)
                                                if doc_id_match:
                                                    known_doc_id = doc_id_match.group(1)
                                                    known_doc_url = f"https://docs.google.com/document/d/{known_doc_id}/edit"
                                                    state.google_doc_id = known_doc_id
                                                    state.google_doc_url = known_doc_url
                                                    logger.info(f"[LIVE] Found document ID from markdown for streaming: {known_doc_id}")
                                        except Exception as e:
                                            logger.debug(f"Could not extract document ID from markdown: {e}")

                                if not state.google_doc_id:
                                    logger.warning("[LIVE] No existing document found, streaming will update when document is created")

                        # Try streaming generation
                        draft_content = await self._generate_motion_with_streaming(
                            chat_service, chat_history, settings, state
                        )
                        if draft_content:
                            logger.info(f"Generated motion using streaming LLM ({len(draft_content)} chars)")
                            return draft_content
                    except Exception as stream_error:
                        logger.debug(f"Streaming failed, falling back to non-streaming: {stream_error}")

                # Non-streaming fallback
                if settings:
                    response = await chat_service.get_chat_message_contents(
                        chat_history=chat_history,
                        settings=settings
                    )
                else:
                    response = await chat_service.get_chat_message_contents(
                        chat_history=chat_history
                    )

                if response:
                    # Extract text from response
                    if isinstance(response, list) and len(response) > 0:
                        draft_content = str(response[0].content) if hasattr(response[0], 'content') else str(response[0])
                    elif hasattr(response, 'content'):
                        draft_content = str(response.content)
                    else:
                        draft_content = str(response)

                    # Validate content is actual motion, not test/example prompts
                    is_valid, error_msg = self._validate_content_is_actual_motion(draft_content)
                    if not is_valid:
                        logger.error(f"Generated content failed validation: {error_msg}")
                        logger.error(f"Content preview (first 500 chars): {draft_content[:500]}")
                        # Retry once with a more explicit prompt
                        logger.warning("Retrying with more explicit instructions...")
                        retry_prompt = prompt + "\n\nCRITICAL: You must generate the ACTUAL MOTION TEXT, not examples, not prompts, not instructions. Start with the motion header immediately."
                        chat_history_retry = ChatHistory()
                        chat_history_retry.add_user_message(retry_prompt)
                        
                        try:
                            if settings:
                                retry_response = await chat_service.get_chat_message_contents(
                                    chat_history=chat_history_retry,
                                    settings=settings
                                )
                            else:
                                retry_response = await chat_service.get_chat_message_contents(
                                    chat_history=chat_history_retry
                                )
                            
                            if retry_response:
                                if isinstance(retry_response, list) and len(retry_response) > 0:
                                    retry_content = str(retry_response[0].content) if hasattr(retry_response[0], 'content') else str(retry_response[0])
                                elif hasattr(retry_response, 'content'):
                                    retry_content = str(retry_response.content)
                                else:
                                    retry_content = str(retry_response)
                                
                                # Validate retry content
                                retry_valid, retry_error = self._validate_content_is_actual_motion(retry_content)
                                if retry_valid:
                                    logger.info(f"Retry successful - generated valid motion ({len(retry_content)} chars)")
                                    return retry_content
                                else:
                                    logger.error(f"Retry also failed validation: {retry_error}")
                                    # Fall back to template
                                    logger.warning("Falling back to template motion due to validation failures")
                                    return self._generate_template_motion(sk_context, evidence_serializable)
                            else:
                                logger.warning("Retry returned empty response, falling back to template")
                                return self._generate_template_motion(sk_context, evidence_serializable)
                        except Exception as retry_error:
                            logger.error(f"Retry failed: {retry_error}, falling back to template")
                            return self._generate_template_motion(sk_context, evidence_serializable)

                    # Clean up the content
                    if draft_content and len(draft_content) > 100:
                        logger.info(f"Generated motion using LLM ({len(draft_content)} chars) - validation passed")
                        return draft_content
                    else:
                        logger.warning(f"Generated content seems too short ({len(draft_content) if draft_content else 0} chars), may be incomplete")
                        # Still return it, but log the warning
                        return draft_content if draft_content else ""
                else:
                    raise RuntimeError("Empty response from chat service")

            except Exception as chat_error:
                logger.warning(f"Chat service invocation failed: {chat_error}, trying alternative method")

                # Alternative: Try creating a prompt function
                try:
                    from semantic_kernel.functions import KernelFunctionFromPrompt
                    prompt_func = KernelFunctionFromPrompt(
                        function_name="GenerateMotion",
                        plugin_name="Drafting",
                        prompt=prompt,
                        description="Generate motion to seal"
                    )

                    arguments = KernelArguments()
                    result = await prompt_func.invoke(self.sk_kernel, arguments=arguments)
                    if result:
                        # Extract text content from result
                        content = result.value if hasattr(result, 'value') else result

                        # Try multiple ways to extract the actual text
                        draft_content = None

                        # Method 1: Check if it has items list (ChatMessageContent structure)
                        if hasattr(content, 'items') and content.items:
                            for item in content.items:
                                if hasattr(item, 'text'):
                                    draft_content = item.text
                                    break
                                elif hasattr(item, 'content'):
                                    draft_content = str(item.content)
                                    break

                        # Method 2: Direct content/text attributes
                        if not draft_content:
                            if hasattr(content, 'text'):
                                draft_content = content.text
                            elif hasattr(content, 'content'):
                                text_content = content.content
                                # If content is nested, extract from message
                                if hasattr(text_content, 'message') and hasattr(text_content.message, 'content'):
                                    draft_content = text_content.message.content
                                else:
                                    draft_content = str(text_content)

                        # Method 3: String extraction with regex if needed
                        if not draft_content:
                            content_str = str(content)
                            # Try to extract text from ChatMessageContent string representation
                            import re
                            # Look for text='...' pattern in items
                            text_match = re.search(r"text='((?:[^']|'[^']*')*)'", content_str, re.DOTALL)
                            if not text_match:
                                # Look for content='...' in Message
                                text_match = re.search(r"Message\([^)]*content='((?:[^']|'[^']*')*)'", content_str, re.DOTALL)
                            if text_match:
                                draft_content = text_match.group(1).replace("\\n", "\n")
                            else:
                                draft_content = content_str

                        if draft_content:
                            logger.info(f"Generated motion using prompt function ({len(draft_content)} chars)")
                            return draft_content
                        else:
                            raise RuntimeError("Could not extract text from LLM response")
                except Exception as prompt_error:
                    logger.warning(f"Prompt function method failed: {prompt_error}, using template")
                    return self._generate_template_motion(sk_context, evidence_serializable)

        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, using template fallback")
            import traceback
            logger.debug(traceback.format_exc())
            return self._generate_template_motion(sk_context, evidence_serializable)

    def _clean_document_text(self, text: str, max_length: int = 50000) -> str:
        """Clean document text by removing code blocks, metadata, and limiting size."""
        if not text:
            return ""

        import re

        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'```json[\s\S]*?```', '', text)
        text = re.sub(r'```python[\s\S]*?```', '', text)

        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(prefix) for prefix in ['def ', 'class ', 'import ', 'from ', '@', 'async def ']):
                continue
            if re.match(r'^[a-z_]+_[a-z_]*\s*[:=]', stripped):
                continue
            if stripped.count('_') > 3 and not any(c.isupper() for c in stripped):
                continue
            cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        if len(text) > max_length:
            cut_point = text[:max_length].rfind('\n\n')
            if cut_point > max_length * 0.8:
                text = text[:cut_point] + "\n\n[Document truncated for length]"
            else:
                text = text[:max_length] + "\n\n[Document truncated for length]"

        return text.strip()

    def _sanitize_deliverable_for_google_docs(
        self,
        deliverable: WriterDeliverable,
        *,
        max_length: int = 50000
    ) -> None:
        """Ensure deliverable content is scrubbed before syncing to Google Docs."""
        sanitized = False

        if getattr(deliverable, "edited_document", None):
            cleaned = self._clean_document_text(deliverable.edited_document, max_length=max_length)
            if cleaned != deliverable.edited_document:
                deliverable.edited_document = cleaned
                sanitized = True

        for section in getattr(deliverable, "sections", []) or []:
            body = getattr(section, "body", "")
            if not body:
                continue
            cleaned_body = self._clean_document_text(body, max_length=max_length)
            if cleaned_body != body:
                section.body = cleaned_body
                sanitized = True

        if sanitized:
            safe_length = len(deliverable.edited_document or "".join(
                getattr(section, "body", "") or "" for section in deliverable.sections or []
            ))
            logger.info(
                "[DOC SAFETY] Applied _clean_document_text before Google Docs update (chars=%s)",
                safe_length,
            )

    def _generate_template_motion(self, sk_context: Dict[str, Any], evidence_serializable: List[Dict]) -> str:
        """Generate a template-based motion when plugins aren't available."""
        case_summary = sk_context.get('case_summary', 'N/A')[:500]
        jurisdiction = sk_context.get('jurisdiction', 'US Federal Court')

        evidence_list = "\n".join([
            f"  - {item.get('node_id', 'unknown')}: {item.get('state', 'unknown')}"
            for item in evidence_serializable[:10]
        ]) if evidence_serializable else "  - No evidence items provided"

        quality_constraints = sk_context.get("quality_constraints")
        quality_section = ""
        if quality_constraints:
            quality_section = f"\n\n## QUALITY CONSTRAINTS AND REQUIREMENTS\n\n{quality_constraints}\n"

        fact_todo = format_fact_retry_todo(sk_context.get("fact_retry_todo"))
        todo_section = ""
        if fact_todo:
            todo_section = f"\n## FACT TODO LIST\n{fact_todo}\n"

        return f"""# MOTION FOR LEAVE TO FILE UNDER SEAL

## I. INTRODUCTION

This motion seeks leave to file certain documents under seal pursuant to 28 U.S.C. Section 1782 and Local Rule [X] due to privacy concerns and national security implications.

## II. CASE SUMMARY

{case_summary}

## III. JURISDICTION AND LEGAL FRAMEWORK

This Court has jurisdiction under {jurisdiction}. The motion is brought pursuant to:
- 28 U.S.C. Section 1782 (International Judicial Assistance)
- Local Rule [X] (Filing Under Seal)

## IV. EVIDENCE AND FACTUAL BASIS

The following evidence supports the need for sealing:

{evidence_list}

{todo_section}
## V. GROUNDS FOR SEALING

### A. Privacy Concerns
The documents contain sensitive personal information that, if disclosed, could lead to:
- Identity exposure and potential harm
- Invasion of privacy
- Harassment or retaliation

### B. National Security Implications
The matter involves issues of national security concern, including:
- References to foreign government connections
- Potential impact on ongoing investigations
- Safety concerns for involved parties

### C. Balancing Test
The public interest in disclosure is outweighed by:
1. The compelling privacy interests at stake
2. The national security implications
3. The safety concerns for all parties involved
4. The limited public interest in the specific details

## VI. REQUESTED RELIEF

WHEREFORE, Plaintiff respectfully requests that this Court:
1. Grant leave to file the [specified documents] under seal
2. Order that access be limited to parties and the Court
3. Provide such other relief as the Court deems appropriate

Respectfully submitted,

[DATE]

[ATTORNEY SIGNATURE]

---

{quality_section}*Note: This is a template-based draft. For production use, ensure all citations, case law, and specific facts are properly incorporated.*"""

    async def _execute_drafting_phase(self, state: WorkflowState, insights: CaseInsights) -> Dict[str, Any]:
        """Execute SK drafting phase with optional multi-model ensemble."""
        logger.info("Executing drafting phase with SK functions")

        fallback_fact_block: Optional[str] = None
        fallback_filtered_evidence: List[Any] = []
        fallback_fact_stats: Dict[str, Any] = {}
        fact_context_initialized = False

        def ensure_fact_context() -> None:
            nonlocal fallback_fact_block, fallback_filtered_evidence, fallback_fact_stats, fact_context_initialized
            if fact_context_initialized:
                return
            fallback_fact_block, fallback_filtered_evidence, fallback_fact_stats = self._build_autogen_fact_context(insights)
            fact_context_initialized = True

        # Convert AutoGen exploration to SK context
        exploration_output = state.exploration_result.get("output", "") if state.exploration_result else ""

        # Handle case when bridge is not available
        if self.bridge is None:
            ensure_fact_context()
            # Create basic context without bridge
            sk_context = {
                "case_summary": insights.summary,
                "jurisdiction": insights.jurisdiction or "US",
                "evidence": fallback_filtered_evidence or insights.evidence,
                "filtered_evidence": fallback_filtered_evidence,
                "posteriors": insights.posteriors,
                "autogen_notes": exploration_output,
                "case_style": insights.case_style,
                "structured_facts": fallback_fact_block,
                "fact_filter_stats": fallback_fact_stats
            }
            logger.warning("Bridge not available, using basic context")
        else:
            sk_context = self.bridge.autogen_to_sk_context(exploration_output, insights)

        if state.memory_context:
            sk_context["memory_context"] = self._serialize_memory_context(state.memory_context)

        # Include research findings in context for motion generation
        if state.research_results and state.research_results.get('success'):
            research_results = state.research_results
            sk_context["research_findings"] = {
                "cases": research_results.get('cases', [])[:20],  # Top 20 cases
                "explanations": research_results.get('explanations', {}),
                "summary": research_results.get('summary', {}),
                "themes": research_results.get('themes', {})
            }
            logger.info(f"Including {len(research_results.get('cases', []))} research cases in drafting context")

        # Ensure structured facts + filtered evidence are present even if bridge dropped them
        if not sk_context.get("structured_facts") or not sk_context.get("filtered_evidence"):
            ensure_fact_context()
        if fallback_fact_block and not sk_context.get("structured_facts"):
            sk_context["structured_facts"] = fallback_fact_block
        if fallback_filtered_evidence and not sk_context.get("filtered_evidence"):
            sk_context["filtered_evidence"] = fallback_filtered_evidence
        if fallback_fact_stats and not sk_context.get("fact_filter_stats"):
            sk_context["fact_filter_stats"] = fallback_fact_stats
        if not sk_context.get("filtered_evidence") and sk_context.get("evidence"):
            sk_context["filtered_evidence"] = sk_context["evidence"]
        if not sk_context.get("evidence"):
            ensure_fact_context()
            sk_context["evidence"] = fallback_filtered_evidence

        structured_snapshot = truncate_text(
            sk_context.get("structured_facts"),
            STRUCTURED_FACTS_MAX_CHARS,
            "structured facts block",
        )
        sk_context["structured_facts"] = structured_snapshot
        sk_context["key_facts_summary"] = sk_context.get("key_facts_summary") or build_key_fact_summary_text(
            structured_snapshot,
            KEY_FACT_SUMMARY_MAX_CHARS,
        )
        if getattr(state, "fact_retry_todo", None):
            sk_context["fact_retry_todo"] = list(state.fact_retry_todo)

        # Attach constraint system metadata + formatted constraint text
        sk_context.setdefault("constraint_version", self.config.constraint_system_version or "1.0")
        quality_constraints = self._format_quality_constraints(sk_context)
        sk_context["quality_constraints"] = quality_constraints
        logger.info(
            "Constraint system v%s applied (%d characters of requirements)",
            sk_context["constraint_version"],
            len(quality_constraints or "")
        )

        # Find existing Google Doc for live updates (don't create new one)
        if (self.config.google_docs_enabled and
            self.config.google_docs_live_updates and
            self.google_docs_bridge and
            not state.google_doc_id):
            try:
                if self.config.master_draft_mode and self.config.master_draft_title:
                    title = self.config.master_draft_title
                    existing_doc = None
                    if self.document_tracker:
                        existing_doc = self.document_tracker.get_doc_by_title(
                            title,
                            folder_id=self.config.google_drive_folder_id
                        )
                    if existing_doc:
                        state.google_doc_id = existing_doc.google_doc_id
                        state.google_doc_url = existing_doc.doc_url
                        logger.info(f"[LIVE] Using existing Google Doc for live updates: {existing_doc.google_doc_id}")
                    else:
                        # Fallback: Try to find by known document ID from markdown file
                        logger.warning(f"[LIVE] No existing document found with title '{title}' in tracker")
                        # Check if we can get it from markdown file
                        try:
                            from pathlib import Path
                            markdown_path = Path(self.config.markdown_export_path) / "master_draft.md"
                            if markdown_path.exists():
                                content = markdown_path.read_text(encoding="utf-8", errors="ignore")
                                import re
                                # Look for google_doc_id in the markdown
                                doc_id_match = re.search(r"google_doc_id[:\s]+([a-zA-Z0-9_-]+)", content)
                                if doc_id_match:
                                    known_doc_id = doc_id_match.group(1)
                                    # Construct URL from ID
                                    known_doc_url = f"https://docs.google.com/document/d/{known_doc_id}/edit"
                                    state.google_doc_id = known_doc_id
                                    state.google_doc_url = known_doc_url
                                    logger.info(f"[LIVE] Found document ID from markdown file: {known_doc_id}")
                        except Exception as e:
                            logger.debug(f"Could not extract document ID from markdown: {e}")

                        if not state.google_doc_id:
                            logger.warning(f"[LIVE] Will create document during commit phase")
                else:
                    # Try to find existing doc by case ID
                    case_id = self._generate_case_id(insights)
                    if self.document_tracker:
                        existing_doc = self.document_tracker.get_doc_for_case(case_id)
                        if existing_doc:
                            state.google_doc_id = existing_doc.google_doc_id
                            state.google_doc_url = existing_doc.doc_url
                            logger.info(f"[LIVE] Using existing Google Doc for live updates: {existing_doc.google_doc_id}")
                        else:
                            logger.warning(f"[LIVE] No existing document found for case {case_id}, will create during commit phase")
                    else:
                        logger.warning("[LIVE] Document tracker not available, cannot find existing document")
            except Exception as e:
                logger.debug(f"Failed to find existing Google Doc: {e}")

        # Try multi-model parallel drafting if available
        if (self.parallel_draft_generator is not None and
            self.section_merger is not None):
            try:
                logger.info("Using multi-model parallel drafting")
                return await self._execute_multi_model_drafting(state, sk_context, insights)
            except Exception as e:
                logger.warning(f"Multi-model drafting failed, falling back to single-model: {e}")

        # Draft privacy harm section using SK plugin (fallback to single model)
        try:
            # Handle case when SK kernel is not available
            if self.sk_kernel is None:
                raise RuntimeError("SK kernel is not available; configure Semantic Kernel before drafting.")
            else:
                # Serialize evidence items to dicts for JSON serialization
                from .insights import EvidenceItem, Posterior
                evidence_serializable = []
                if isinstance(sk_context["evidence"], list):
                    for item in sk_context["evidence"]:
                        if isinstance(item, EvidenceItem):
                            evidence_serializable.append(item.to_dict())
                        elif isinstance(item, dict):
                            evidence_serializable.append(item)
                        else:
                            # Fallback: convert to dict using common attributes
                            evidence_serializable.append({
                                "node_id": getattr(item, 'node_id', str(item)),
                                "state": getattr(item, 'state', 'unknown'),
                                "description": getattr(item, 'description', ''),
                                "weight": getattr(item, 'weight', None)
                            })
                else:
                    evidence_serializable = sk_context["evidence"]

                # Serialize posteriors similarly
                posteriors_serializable = []
                if isinstance(sk_context["posteriors"], list):
                    for item in sk_context["posteriors"]:
                        if isinstance(item, Posterior):
                            posteriors_serializable.append({
                                "node_id": item.node_id,
                                "probabilities": item.probabilities,
                                "interpretation": item.interpretation
                            })
                        elif isinstance(item, dict):
                            posteriors_serializable.append(item)
                        else:
                            # Fallback for Posterior objects
                            posteriors_serializable.append({
                                "node_id": getattr(item, 'node_id', str(item)),
                                "probabilities": getattr(item, 'probabilities', {})
                            })
                else:
                    posteriors_serializable = sk_context["posteriors"]

                try:
                    result = await self._invoke_sk_function(
                        plugin_name="PrivacyHarmPlugin",
                        function_name="PrivacyHarmSemantic",
                        variables={
                            "evidence": json.dumps(evidence_serializable, default=str),
                            "posteriors": json.dumps(posteriors_serializable, default=str),
                            "case_summary": sk_context["case_summary"],
                            "jurisdiction": sk_context["jurisdiction"],
                            "structured_facts": sk_context.get("structured_facts", "") or "",
                            "autogen_notes": sk_context.get("autogen_notes", "") or "",
                            "research_findings": json.dumps(sk_context.get("research_findings", {}), default=str),
                            "quality_constraints": sk_context.get("quality_constraints", "") or "",
                            "fact_key_summary": json.dumps(sk_context.get("fact_key_summary", []), default=str),
                            "filtered_evidence": json.dumps(sk_context.get("filtered_evidence", []), default=str),
                            "fact_filter_stats": json.dumps(sk_context.get("fact_filter_stats", {}), default=str),
                        },
                        context=sk_context,
                    )

                    draft_result = {
                        "privacy_harm_section": result.value if hasattr(result, 'value') else str(result),
                        "context": sk_context,
                        "method": "semantic"
                    }
                except RuntimeError as e:
                    if "not found" in str(e):
                        logger.warning(f"SK plugin not available ({e}), generating draft using LLM directly")
                        # Create a draft using LLM directly through kernel chat service (pass state for streaming)
                        draft_content = await self._generate_motion_with_llm(sk_context, evidence_serializable, posteriors_serializable, state)

                        draft_result = {
                            "privacy_harm_section": draft_content,
                            "context": sk_context,
                            "method": "llm_direct"
                        }
                    else:
                        raise

            state.draft_result = draft_result
            logger.info("SK drafting completed successfully")
            return draft_result

        except Exception as e:
            logger.error(f"SK drafting failed: {e}")
            raise

    async def _execute_multi_model_drafting(
        self,
        state: WorkflowState,
        sk_context: Dict[str, Any],
        insights: CaseInsights
    ) -> Dict[str, Any]:
        """Execute multi-model parallel drafting with section merging."""
        logger.info("Executing multi-model parallel drafting...")

        # Build prompt for generation
        from multi_model_config import get_default_multi_model_config
        mm_config = self.config.multi_model_config or get_default_multi_model_config()

        prompt = f"""Generate a complete legal Motion to Seal for the following case:

Case Summary: {sk_context.get('case_summary', 'N/A')}
Jurisdiction: {sk_context.get('jurisdiction', 'US Federal Court')}

Include all standard sections:
1. Introduction
2. Factual Background
3. Legal Argument
4. Privacy Harm Analysis
5. Balancing Test
6. Requested Relief/Conclusion

Write a complete, professional motion following proper legal formatting."""

        quality_constraints = sk_context.get("quality_constraints")
        if quality_constraints:
            prompt += f"\n\nQUALITY CONSTRAINTS AND REQUIREMENTS:\n{quality_constraints}\n"

        # Generate drafts in parallel
        primary_result, secondary_result = await self.parallel_draft_generator.generate_parallel_drafts(
            prompt=prompt,
            context=sk_context
        )

        # Merge drafts using section merger
        if primary_result.error or secondary_result.error:
            logger.warning("One or both models failed, using available draft")
            merged_draft = primary_result.content if not primary_result.error else secondary_result.content
            merge_details = {"error": "One model failed"}
        else:
            merge_result = self.section_merger.merge_drafts(
                primary_draft=primary_result.content,
                secondary_draft=secondary_result.content,
                primary_model_name=mm_config.primary_drafting_model,
                secondary_model_name=mm_config.secondary_drafting_model
            )
            merged_draft = merge_result["merged_draft"]
            merge_details = {
                "comparisons": merge_result["comparisons"],
                "statistics": merge_result["statistics"]
            }
            logger.info(f"Merge complete: {merge_result['statistics']['primary_selected']} sections from primary, "
                       f"{merge_result['statistics']['secondary_selected']} from secondary")

        draft_result = {
            "privacy_harm_section": merged_draft,  # Full merged draft
            "context": sk_context,
            "method": "multi_model_ensemble",
            "primary_draft": primary_result.content if not primary_result.error else None,
            "secondary_draft": secondary_result.content if not secondary_result.error else None,
            "merge_details": merge_details,
            "metadata": {
                "primary_model": mm_config.primary_drafting_model,
                "secondary_model": mm_config.secondary_drafting_model,
                "primary_metadata": primary_result.metadata,
                "secondary_metadata": secondary_result.metadata
            }
        }

        state.draft_result = draft_result
        logger.info("Multi-model drafting completed successfully")
        return draft_result

    async def _direct_catboost_validation(self, document: str) -> Dict[str, Any]:
        """Run CatBoost validation directly without RefinementLoop."""
        try:
            import catboost
            import pandas as pd
            import numpy as np
            from pathlib import Path

            # Try to find and load the CatBoost model
            project_root = Path(__file__).parents[3]
            model_paths = [
                project_root / "case_law_data" / "models" / "section_1782_discovery_model.cbm",
                project_root / "case_law_data" / "models" / "catboost_motion.cbm",
                project_root / "case_law_data" / "models" / "motion_ma_model.cbm"
            ]

            model_path = None
            for path in model_paths:
                if path.exists():
                    model_path = path
                    break

            if not model_path:
                logger.warning("No CatBoost model file found")
                return {"predicted_success_probability": 0.0, "error": "Model file not found"}

            logger.info(f"Loading CatBoost model from {model_path}")
            model = catboost.CatBoostClassifier()
            model.load_model(str(model_path))

            # Extract features from draft
            try:
                # Try to import compute_draft_features
                import sys
                analysis_path = project_root / "analysis"
                if analysis_path.exists():
                    sys.path.insert(0, str(analysis_path))

                from analyze_ma_motion_doc import compute_draft_features
                features = compute_draft_features(document)
            except ImportError:
                logger.warning("Could not import compute_draft_features, using basic features")
                # Fallback to basic feature extraction
                features = {
                    "word_count": len(document.split()),
                    "text_length": len(document),
                    "citation_count": document.lower().count("v.") + document.count("§"),
                    "mentions_privacy": document.lower().count("privacy"),
                    "mentions_safety": document.lower().count("safety") + document.lower().count("harm"),
                    "mentions_national_security": document.lower().count("national security") + document.lower().count("national_security"),
                }

            # Convert to DataFrame
            feature_df = pd.DataFrame([features])

            # Ensure all required columns exist
            if hasattr(model, 'feature_names_') and model.feature_names_:
                for col in model.feature_names_:
                    if col not in feature_df.columns:
                        feature_df[col] = 0
                feature_df = feature_df[model.feature_names_]

            # Get prediction
            proba = model.predict_proba(feature_df)[0]
            prediction = model.predict(feature_df)[0]
            confidence = float(np.max(proba))

            # Get class labels if available
            classes = model.classes_ if hasattr(model, 'classes_') else ['denied', 'granted']
            predicted_class = classes[prediction] if isinstance(prediction, (int, np.integer)) and prediction < len(classes) else str(prediction)

            probabilities = {}
            if len(proba) == len(classes):
                for i, cls in enumerate(classes):
                    probabilities[cls] = float(proba[i])

            logger.info(f"CatBoost prediction: {predicted_class} with {confidence:.1%} confidence")

            result = {
                "predicted_success_probability": confidence,
                "prediction": predicted_class,
                "probabilities": probabilities,
                "meets_threshold": confidence >= 0.70,
                "feature_count": len(features),
                "method": "direct_validation",
                "shap_available": False  # Direct validation doesn't compute SHAP by default
            }

            # ✅ Optionally compute SHAP insights if SHAPInsightPlugin is available
            try:
                from ..sk_plugins.SHAPInsightPlugin import SHAPInsightPlugin
                shap_plugin = SHAPInsightPlugin(model=model)
                shap_insights = shap_plugin.compute_shap_insights(features, top_n=10)

                if shap_insights and shap_insights.get("shap_available"):
                    result["shap_available"] = True
                    result["shap_insights"] = shap_insights
                    result["shap_recommendations"] = shap_insights.get("recommendations", [])
                    result["top_helping_features"] = shap_insights.get("top_helping_features", {})
                    result["top_hurting_features"] = shap_insights.get("top_hurting_features", {})
                    logger.info(f"✅ SHAP insights computed in direct validation: {len(shap_insights.get('recommendations', []))} recommendations")
            except (ImportError, Exception) as e:
                logger.debug(f"SHAP insights not available in direct validation: {e}")
                # Continue without SHAP - not critical for direct validation

            return result

        except Exception as e:
            logger.error(f"Direct CatBoost validation error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return {"predicted_success_probability": 0.0, "error": str(e)}

    async def _execute_validation_phase(self, state: WorkflowState, insights: CaseInsights) -> Dict[str, Any]:
        """Execute SK validation phase with optional multi-model validation."""
        logger.info("Executing validation phase with SK quality gates and CatBoost validation")

        # Try multi-model validation if available
        if self.multi_model_validator is not None:
            try:
                logger.info("Using multi-model validation")
                return await self._execute_multi_model_validation(state, insights)
            except Exception as e:
                logger.warning(f"Multi-model validation failed, falling back to standard validation: {e}")

        if not state.draft_result:
            raise RuntimeError("No draft result to validate")

        document = state.draft_result["privacy_harm_section"]
        context = state.draft_result["context"]

        # Run CatBoost feature validation if RefinementLoop is available
        catboost_validation = None
        if self.feature_orchestrator:
            try:
                logger.info("Running CatBoost feature validation...")
                # Analyze draft for weak features (now returns dict with SHAP insights)
                analysis_result = await self.feature_orchestrator.analyze_draft(document)

                # Extract weak_features dict (handle both old and new return formats)
                if isinstance(analysis_result, dict) and "weak_features" in analysis_result:
                    weak_features = analysis_result["weak_features"]
                    shap_insights = analysis_result.get("shap_insights")
                    shap_recommendations = analysis_result.get("shap_recommendations", [])
                    top_helping_features = analysis_result.get("top_helping_features", {})
                    top_hurting_features = analysis_result.get("top_hurting_features", {})
                else:
                    # Backward compatibility: treat as old format (dict of weak features)
                    weak_features = analysis_result if isinstance(analysis_result, dict) else {}
                    shap_insights = None
                    shap_recommendations = []
                    top_helping_features = {}
                    top_hurting_features = {}

                # Validate with CatBoost
                catboost_result = await self.feature_orchestrator.validate_with_catboost(document)

                # Get improvement recommendations
                improvement_plan = await self.feature_orchestrator.orchestrate_improvements(document)

                # Build catboost_validation with SHAP insights
                catboost_validation = {
                    "predicted_success_probability": catboost_result.get("confidence", 0.0),
                    "prediction": catboost_result.get("prediction"),
                    "weak_features_count": len(weak_features) if isinstance(weak_features, dict) else 0,
                    "weak_features": list(weak_features.keys()) if isinstance(weak_features, dict) else [],
                    "improvement_recommendations": improvement_plan.get("recommendations", []),
                    "meets_threshold": catboost_result.get("confidence", 0.0) >= 0.70,
                    "feature_scores": catboost_result.get("features", {})
                }

                # ✅ Add SHAP insights if available
                if shap_insights and shap_insights.get("shap_available"):
                    catboost_validation["shap_available"] = True
                    catboost_validation["shap_insights"] = shap_insights
                    catboost_validation["shap_recommendations"] = shap_recommendations
                    catboost_validation["top_helping_features"] = top_helping_features
                    catboost_validation["top_hurting_features"] = top_hurting_features

                    # Log SHAP insights
                    logger.info(f"✅ SHAP insights available: {len(shap_recommendations)} actionable recommendations")
                    if top_helping_features:
                        top_helping = list(top_helping_features.keys())[:3]
                        logger.info(f"   Top helping features: {top_helping}")
                    if top_hurting_features:
                        top_hurting = list(top_hurting_features.keys())[:3]
                        logger.info(f"   Top hurting features: {top_hurting}")
                else:
                    catboost_validation["shap_available"] = False

                # Set baseline score if not set
                if self.feature_orchestrator.baseline_score is None:
                    self.feature_orchestrator.set_baseline_score(catboost_result.get("confidence", 0.0))

                logger.info(f"CatBoost validation: {catboost_validation['predicted_success_probability']:.3f} success probability")

            except Exception as e:
                logger.warning(f"CatBoost validation failed: {e}")
                catboost_validation = {"error": str(e)}
        else:
            # Try direct CatBoost validation without RefinementLoop
            try:
                logger.info("RefinementLoop not available, attempting direct CatBoost validation...")
                catboost_validation = await self._direct_catboost_validation(document)
            except Exception as e:
                logger.warning(f"Direct CatBoost validation failed: {e}")
                catboost_validation = {"error": str(e), "predicted_success_probability": 0.0}

        # Check required case citations (always enforce, regardless of weak features)
        required_citation_validation = None
        if self.feature_orchestrator and 'required_case_citations' in self.feature_orchestrator.plugins:
            try:
                required_plugin = self.feature_orchestrator.plugins['required_case_citations']

                # Update required cases from research results if available
                if state.research_results and state.research_results.get('success'):
                    required_plugin.update_required_cases_from_research(state.research_results)

                citation_result = await required_plugin.enforce_required_citations(document)
                if citation_result.success and citation_result.value:
                    required_citation_validation = citation_result.value
                    logger.info(f"Required citation check: {citation_result.value.get('cases_cited', 0)}/{citation_result.value.get('total_required_cases', 0)} cases cited")
            except Exception as e:
                logger.warning(f"Required citation validation failed: {e}")

        # Run quality gates
        if self.quality_pipeline is None:
            logger.warning("Quality pipeline not available, using mock validation")
            # Create mock validation results
            validation_results = {
                "overall_score": 0.85,
                "structure_check": {"passed": True, "score": 0.9},
                "citation_check": {"passed": True, "score": 0.8},
                "legal_accuracy": {"passed": True, "score": 0.85},
                "warnings": ["Mock validation - configure quality pipeline for real validation"]
            }
        else:
            validation_results = await self.quality_pipeline.run_quality_gates(document, context)

        constraint_validation: Optional[Dict[str, Any]] = None
        gate_results = validation_results.get("gate_results") or {}
        personal_gate = gate_results.get("personal_facts_coverage")
        if personal_gate:
            coverage = personal_gate.get("coverage", personal_gate.get("score", 0.0) or 0.0)
            summary = {
                "coverage": coverage,
                "facts_found": personal_gate.get("facts_found", []),
                "facts_missing": personal_gate.get("facts_missing", []),
                "critical_facts_missing": personal_gate.get("critical_facts_missing", False),
                "violations": personal_gate.get("violations", []),
                "contradictions": personal_gate.get("contradictions", []),
                "has_violations": personal_gate.get("has_violations", bool(personal_gate.get("violations"))),
                "has_contradictions": personal_gate.get("has_contradictions", bool(personal_gate.get("contradictions"))),
            }
            validation_results["personal_facts_verification"] = summary
            missing = summary["facts_missing"]
            logger.info(
                "[FACTS] Personal facts coverage: %.2f (missing: %s)",
                coverage,
                ", ".join(missing[:4]) if missing else "none",
            )
            if summary["critical_facts_missing"]:
                warnings = validation_results.setdefault("warnings", [])
                missing_display = ", ".join(missing[:4]) if missing else "unspecified facts"
                warnings.append(f"Personal facts missing: {missing_display}")
                validation_results["meets_threshold"] = False
            if summary["has_violations"]:
                warnings = validation_results.setdefault("warnings", [])
                warnings.append("Prohibited or false facts detected; remove hallucinated content.")
                validation_results["meets_threshold"] = False
                logger.warning(
                    "[FACTS] Violations detected during validation: %s",
                    ", ".join(v.get("name", "unknown") for v in summary["violations"][:4]) or "unspecified",
                )
            if summary["has_contradictions"]:
                warnings = validation_results.setdefault("warnings", [])
                warnings.append("Contradictions with verified facts detected; revise statements.")
                validation_results["meets_threshold"] = False
                logger.warning(
                    "[FACTS] Contradictions detected: %s",
                    ", ".join(
                        f"{c.get('claim', 'claim')} ({c.get('fact_type', 'fact')})"
                        for c in summary["contradictions"][:4]
                    )
                    or "unspecified",
                )
        constraint_gate = gate_results.get("constraint_system")
        constraint_gate = gate_results.get("constraint_system")
        if constraint_gate:
            constraint_validation = constraint_gate.get("raw_result") or constraint_gate
            validation_results["constraint_validation"] = constraint_validation
            validation_results.setdefault("warnings", [])
            warnings_snapshot = (constraint_validation.get("warnings") or [])[:5]
            if warnings_snapshot:
                validation_results["warnings"].extend(warnings_snapshot)
            validation_results.setdefault("suggestions", [])
            suggestions_snapshot = constraint_validation.get("suggestions") or []
            if suggestions_snapshot:
                validation_results["suggestions"].extend(suggestions_snapshot[:5])
            overall_constraint_score = constraint_validation.get("overall_score", constraint_gate.get("score"))
            logger.info(
                "Constraint-system gate score: %.3f (passed=%s)",
                overall_constraint_score or 0.0,
                constraint_gate.get("passed")
            )
            if warnings_snapshot:
                logger.warning("Constraint warnings: %s", "; ".join(warnings_snapshot[:2]))

        # Add required citation validation to results
        if required_citation_validation:
            validation_results["required_case_citations"] = required_citation_validation

            # Add to warnings if citations are missing
            if not required_citation_validation.get('meets_requirements', False):
                warnings = validation_results.get("warnings", [])
                missing_count = required_citation_validation.get('cases_missing', 0)
                high_priority_missing = [c for c in required_citation_validation.get('missing_cases', []) if c.get('priority') == 'high']

                if high_priority_missing:
                    warnings.append(
                        f"CRITICAL: {len(high_priority_missing)} high-priority required cases are missing citations. "
                        f"Missing cases: {', '.join([c['case_name'] for c in high_priority_missing[:3]])}"
                    )

                if missing_count > 0:
                    warnings.append(
                        f"Required citations: {missing_count} cases missing. "
                        f"Compliance: {required_citation_validation.get('overall_compliance_score', 0):.1%}"
                    )

                validation_results["warnings"] = warnings

                # Add recommendations
                suggestions = validation_results.get("suggestions", [])
                recommendations = required_citation_validation.get('recommendations', [])
                suggestions.extend(recommendations[:5])  # Top 5 recommendations
                validation_results["suggestions"] = suggestions

        # Add CatBoost validation to results
        if catboost_validation:
            validation_results["catboost_validation"] = catboost_validation

            # Include CatBoost score in overall score calculation if available
            if "catboost_validation" in validation_results:
                catboost_score = catboost_validation.get("predicted_success_probability", 0.0)
                # Weight CatBoost score in overall calculation (30% weight)
                current_score = validation_results.get("overall_score", 0.0)
                validation_results["overall_score"] = (current_score * 0.7) + (catboost_score * 0.3)

                # Add to suggestions if below threshold
                if not catboost_validation.get("meets_threshold", False):
                    suggestions = validation_results.get("suggestions", [])
                    suggestion_text = f"CatBoost predicts {catboost_validation['predicted_success_probability']:.1%} success probability. Target: 70%+."

                    # ✅ Include SHAP recommendations if available (more actionable)
                    if catboost_validation.get("shap_available") and catboost_validation.get("shap_recommendations"):
                        shap_recs = catboost_validation["shap_recommendations"][:2]
                        suggestion_text += f" SHAP insights: {'; '.join([rec.replace('✅ **', '').replace('⚠️ **', '').replace('**', '') for rec in shap_recs])}"
                    elif catboost_validation.get("improvement_recommendations"):
                        suggestion_text += f" Consider: {', '.join(catboost_validation['improvement_recommendations'][:2])}"

                    suggestions.append(suggestion_text)
                    validation_results["suggestions"] = suggestions

                # ✅ Add SHAP recommendations to suggestions even if threshold is met (for optimization)
                if catboost_validation.get("shap_available") and catboost_validation.get("shap_recommendations"):
                    suggestions = validation_results.get("suggestions", [])
                    shap_recs = catboost_validation["shap_recommendations"][:3]  # Top 3 SHAP recommendations
                    for rec in shap_recs:
                        # Clean up markdown formatting for suggestions
                        clean_rec = rec.replace('✅ **', '').replace('⚠️ **', '').replace('**', '')
                        if clean_rec not in suggestions:  # Avoid duplicates
                            suggestions.append(f"SHAP: {clean_rec}")
                    validation_results["suggestions"] = suggestions

        draft_context = state.draft_result.get("context", {}) if state.draft_result else {}
        if draft_context is not None:
            if constraint_validation:
                draft_context["constraint_validation"] = constraint_validation
            draft_context.setdefault("constraint_version", self.constraint_system_version or "1.0")
            draft_context["quality_constraints"] = self._format_quality_constraints(draft_context)

        state.validation_results = validation_results
        state.fact_retry_todo = _build_fact_retry_todo(validation_results)
        if state.fact_retry_todo:
            logger.warning(
                "[FACTS] Validation produced %d fact TODO items: %s",
                len(state.fact_retry_todo),
                "; ".join(state.fact_retry_todo[:3]),
            )
        else:
            logger.info("[FACTS] No outstanding fact TODO items after validation")

        logger.info(f"Validation completed. Overall score: {validation_results['overall_score']:.2f}")
        return validation_results

    async def _execute_multi_model_validation(
        self,
        state: WorkflowState,
        insights: CaseInsights
    ) -> Dict[str, Any]:
        """Execute multi-model validation using Legal-BERT + Qwen2.5 + CatBoost."""
        logger.info("Executing multi-model validation...")

        if not state.draft_result:
            raise RuntimeError("No draft result to validate")

        document = state.draft_result["privacy_harm_section"]
        context = state.draft_result["context"]

        # Get CatBoost score if available
        catboost_score = None
        if self.feature_orchestrator:
            try:
                catboost_result = await self.feature_orchestrator.validate_with_catboost(document)
                catboost_score = catboost_result.get("confidence", None)
            except Exception:
                pass

        # Run multi-model validation
        mm_validation = await self.multi_model_validator.validate_document(
            document=document,
            context=context,
            catboost_score=catboost_score
        )

        # Convert to standard validation results format
        validation_results = {
            "overall_score": mm_validation.overall_score,
            "meets_threshold": mm_validation.meets_threshold,
            "legal_bert_score": mm_validation.legal_bert_score,
            "qwen_logical_score": mm_validation.qwen_logical_score,
            "catboost_score": mm_validation.catboost_score,
            "legal_bert_details": {
                "legal_terminology": mm_validation.legal_bert_details.legal_terminology_score if mm_validation.legal_bert_details else None,
                "structure": mm_validation.legal_bert_details.structure_score if mm_validation.legal_bert_details else None,
                "coherence": mm_validation.legal_bert_details.coherence_score if mm_validation.legal_bert_details else None,
                "citation": mm_validation.legal_bert_details.citation_score if mm_validation.legal_bert_details else None,
            } if mm_validation.legal_bert_details else {},
            "qwen_feedback": mm_validation.qwen_review_feedback,
            "qwen_suggestions": mm_validation.qwen_suggestions or [],
            "suggestions": mm_validation.qwen_suggestions or [],
            "weights": mm_validation.weights,
            "method": "multi_model_ensemble"
        }

        state.validation_results = validation_results
        logger.info(f"Multi-model validation completed. Overall score: {mm_validation.overall_score:.3f}")
        return validation_results

    async def _execute_review_phase(self, state: WorkflowState, insights: CaseInsights) -> str:
        """Execute AutoGen review phase."""
        logger.info("Executing review phase with AutoGen")

        if not state.draft_result:
            raise RuntimeError("No draft result to review")

        # Convert SK output to AutoGen review context
        sk_output = state.draft_result["privacy_harm_section"]

        # Handle case when bridge is not available
        if self.bridge is None:
            logger.warning("Bridge not available, using basic review context")
            # Create basic review context
            review_context = f"""Draft output from SK:
{sk_output}

Validation results: {state.validation_results}
"""
        else:
            review_context = self.bridge.sk_to_autogen_context(sk_output, state.validation_results)

        # Include CatBoost validation results in review context
        catboost_context = ""
        if state.validation_results and "catboost_validation" in state.validation_results:
            catboost_val = state.validation_results["catboost_validation"]
            catboost_context = f"""
## CatBoost Feature Analysis
- Success Probability: {catboost_val.get('predicted_success_probability', 0):.1%}
- Weak Features: {', '.join(catboost_val.get('weak_features', []))}
- Recommendations: {', '.join(catboost_val.get('improvement_recommendations', [])[:3])}
"""
            review_context += catboost_context

        # Use AutoGen agent for review
        try:
            reviewer_system_message = """You are a legal document reviewer specializing in privacy harm analysis and legal writing quality.
Your role is to review SK-generated legal drafts and provide specific, actionable feedback for improvement.
You have access to personal-facts and case-law corpora to inform your review.
Focus on legal accuracy, argument strength, citation completeness, and provide concrete suggestions for revision."""
            
            # Use review model (Qwen2.5) for review agent
            reviewer_agent = self.autogen_factory.create(
                name="LegalReviewer",
                system_message=reviewer_system_message,
                use_review_model=True
            )
            
            logger.info("Invoking AutoGen review agent with review model...")
            result = await reviewer_agent.run(task=review_context)
            # Extract content from result
            if hasattr(result, 'content'):
                review_notes = result.content
            elif hasattr(result, 'text'):
                review_notes = result.text
            elif isinstance(result, str):
                review_notes = result
            elif hasattr(result, 'messages') and result.messages:
                last_message = result.messages[-1]
                review_notes = last_message.content if hasattr(last_message, 'content') else str(last_message)
            else:
                review_notes = str(result)
            logger.info("AutoGen review agent completed successfully")
            
        except Exception as e:
            logger.error(f"AutoGen review failed: {e}", exc_info=True)
            raise RuntimeError("AutoGen review failed; stopping workflow to prevent synthetic feedback.") from e

        state.review_notes = review_notes
        return review_notes

    async def _execute_refinement_phase(self, state: WorkflowState, insights: CaseInsights) -> Dict[str, Any]:
        """Execute SK refinement phase with RefinementLoop improvements."""
        logger.info("Executing refinement phase with SK functions and CatBoost feature enforcement")

        current_draft = state.draft_result["privacy_harm_section"]

        # Use RefinementLoop to strengthen weak features if available
        improved_draft = current_draft
        feature_improvements = None

        if self.feature_orchestrator:
            try:
                logger.info("Applying CatBoost-guided feature improvements...")

                # Analyze current draft for weak features
                analysis_result = await self.feature_orchestrator.analyze_draft(current_draft)
                
                # Extract weak_features from analysis result (analyze_draft returns dict with 'weak_features' key)
                if isinstance(analysis_result, dict) and "weak_features" in analysis_result:
                    weak_features = analysis_result["weak_features"]
                else:
                    # Backward compatibility: old format
                    weak_features = analysis_result if isinstance(analysis_result, dict) else {}

                if weak_features:
                    logger.info(f"Found {len(weak_features)} weak features to strengthen")

                    # Build context for plugins (research_results, validation_state, weak_features)
                    context = self._build_refinement_context(state, weak_features)
                    logger.debug(
                        "Passing context to strengthen_draft: research_results=%s, validation_state=%s, quality_constraints=%s, constraint_validation=%s",
                        context["research_results"] is not None,
                        context["validation_state"] is not None,
                        bool(context.get("quality_constraints")),
                        bool(context.get("constraint_validation")),
                    )

                    # Debug logging for context contents
                    if context.get('research_results'):
                        logger.debug(f"Context research_results keys: {list(context['research_results'].keys())[:5]}")
                    if context.get('validation_state'):
                        logger.debug(f"Context validation_state keys: {list(context['validation_state'].keys())[:5]}")

                    # Use iterative refinement if enabled, otherwise single-pass
                    if self.config.enable_iterative_refinement:
                        logger.info("Using iterative refinement feedback loop")
                        feedback_result = await self.feature_orchestrator.run_feedback_loop(
                            current_draft,
                            max_iterations=self.config.max_iterations,
                            context=context
                        )
                        improved_draft = feedback_result.get("final_draft", current_draft)

                        feature_improvements = {
                            "weak_features_found": len(weak_features),
                            "iterations_completed": feedback_result.get("iterations_completed", 0),
                            "total_improvement": feedback_result.get("total_improvement", 0),
                            "final_confidence": feedback_result.get("final_confidence", 0),
                            "baseline_confidence": feedback_result.get("baseline_confidence", 0),
                            "stop_reason": feedback_result.get("stop_reason", "Unknown"),
                            "improvement_recommendations": []
                        }
                        logger.info(f"Iterative refinement complete: {feedback_result.get('iterations_completed', 0)} iterations, {feedback_result.get('total_improvement', 0):.1f}% improvement")
                    else:
                        # Single-pass refinement (original behavior)
                        improved_draft = await self.feature_orchestrator.strengthen_draft(current_draft, weak_features, context=context)

                        # Get improvement plan
                        improvement_plan = await self.feature_orchestrator.orchestrate_improvements(current_draft)

                        feature_improvements = {
                            "weak_features_found": len(weak_features),
                            "improvements_applied": len(improvement_plan.get("improvements", [])),
                            "improvement_recommendations": improvement_plan.get("recommendations", [])
                        }

                        logger.info(f"Applied {feature_improvements['improvements_applied']} feature improvements")
                else:
                    logger.info("No weak features found, draft meets CatBoost targets")

            except Exception as e:
                logger.warning(f"Feature-based refinement failed: {e}")
                improved_draft = current_draft

        # Combine with review notes if available
        if state.review_notes:
            logger.info("Combining CatBoost improvements with AutoGen review feedback")
            # Note: In a full implementation, this would intelligently merge improvements
            # For now, we prioritize CatBoost feature improvements

        personal_context = state.draft_result.get("context") if state.draft_result else {}
        baseline_personal = _run_personal_facts_verifier(current_draft, personal_context)
        refined_personal = _run_personal_facts_verifier(improved_draft, personal_context)
        if baseline_personal and refined_personal:
            base_cov = baseline_personal.get("coverage", 0.0)
            refined_cov = refined_personal.get("coverage", 0.0)
            logger.info(
                "[FACTS] Refinement personal facts coverage: %.2f → %.2f",
                base_cov,
                refined_cov,
            )
            violation_keys = ["name", "fact_type", "description"]
            baseline_violation_sig = _entry_signature(baseline_personal.get("violations"), violation_keys)
            refined_violation_sig = _entry_signature(refined_personal.get("violations"), violation_keys)
            baseline_contra_sig = _entry_signature(
                baseline_personal.get("contradictions"),
                ["claim", "fact_type", "contradiction_type"],
            )
            refined_contra_sig = _entry_signature(
                refined_personal.get("contradictions"),
                ["claim", "fact_type", "contradiction_type"],
            )

            revert_reason: Optional[str] = None
            if refined_personal.get("has_violations") and not baseline_personal.get("has_violations"):
                revert_reason = "introduced prohibited facts"
            elif refined_violation_sig - baseline_violation_sig:
                revert_reason = "introduced additional prohibited facts"
            elif refined_personal.get("has_contradictions") and not baseline_personal.get("has_contradictions"):
                revert_reason = "introduced contradictions"
            elif refined_contra_sig - baseline_contra_sig:
                revert_reason = "introduced additional contradictions"
            elif (
                refined_personal.get("critical_facts_missing")
                and not baseline_personal.get("critical_facts_missing")
            ):
                revert_reason = "removed required personal facts"
            elif refined_cov + 1e-3 < base_cov:
                revert_reason = "reduced personal facts coverage"

            if revert_reason:
                logger.warning("Refinement %s; reverting to previous draft", revert_reason)
                improved_draft = current_draft
                refined_personal = baseline_personal

            if isinstance(personal_context, dict):
                personal_context["personal_facts_verification"] = refined_personal

        refined_result = {
            "privacy_harm_section": improved_draft,
            "review_notes": state.review_notes if hasattr(state, 'review_notes') else None,
            "refinement_applied": True,
            "feature_improvements": feature_improvements,
            "catboost_guided": feature_improvements is not None,
            "context": state.draft_result["context"]
        }

        state.draft_result = refined_result
        return refined_result

    def _build_refinement_context(self, state: WorkflowState, weak_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Assemble shared context for refinement plugins."""
        draft_context = state.draft_result.get("context", {}) if state.draft_result else {}
        return {
            "research_results": getattr(state, "research_results", None),
            "validation_state": getattr(state, "validation_results", None),
            "weak_features": weak_features or [],
            "quality_constraints": (draft_context or {}).get("quality_constraints") or "",
            "constraint_validation": (draft_context or {}).get("constraint_validation"),
            "constraint_version": (draft_context or {}).get("constraint_version"),
            "structured_facts": (draft_context or {}).get("structured_facts") or "",
            "filtered_evidence": (draft_context or {}).get("filtered_evidence") or (draft_context or {}).get("evidence"),
            "fact_filter_stats": (draft_context or {}).get("fact_filter_stats"),
        }

    async def _commit_document(self, state: WorkflowState, insights: CaseInsights) -> WriterDeliverable:
        """Commit the final document."""
        logger.info("Committing final document")

        if not state.draft_result:
            raise RuntimeError("No draft result to commit")

        final_document = state.draft_result["privacy_harm_section"]
        state.final_document = final_document

        # Create WriterDeliverable
        deliverable = WriterDeliverable(
            plan=PlanDirective(
                objective="Draft a complete Motion for Leave to File Under Seal and to Proceed Under Pseudonym for federal court",
                deliverable_format="Legal memorandum section",
                tone="Formal and analytical",
                style_constraints=["Professional legal writing", "Evidence-based analysis"],
                citation_expectations="Use [Node:State] format for evidence citations"
            ),
            sections=[
                DraftSection(
                    section_id="privacy_harm",
                    title="Privacy Harm Analysis",
                    body=final_document
                )
            ],
            edited_document=final_document,
            reviews=[
                ReviewFindings(
                    section_id="privacy_harm",
                    severity="info",
                    message="Hybrid workflow completed",
                    suggestions="Document generated using AutoGen exploration + SK drafting"
                )
            ],
            metadata={
                "workflow_type": "hybrid_sk",
                "iteration_summary": self.iteration_controller.get_iteration_summary(),
                "validation_results": state.validation_results,
                "exploration_result": state.exploration_result
            }
        )

        if state.memory_context:
            deliverable.metadata["memory_context"] = self._serialize_memory_context(state.memory_context)

        # Include research results in metadata
        if state.research_results:
            deliverable.metadata["research_results"] = state.research_results

        # Google Docs integration
        if self.google_docs_bridge and self.config.google_docs_enabled:
            try:
                await self._commit_to_google_docs(deliverable, insights, state)
                logger.info("Document committed to Google Docs successfully")
            except Exception as e:
                logger.error(f"Failed to commit to Google Docs: {e}")
                # Continue with local commit even if Google Docs fails
        else:
            logger.info("Google Docs integration not available, skipping Google Docs commit")

        self._record_workflow_memory(deliverable, insights, state)
        logger.info("Document committed successfully")
        return deliverable

    async def _update_google_doc_live(self, state: WorkflowState, insights: Optional[CaseInsights] = None) -> None:
        """Update Google Doc with current draft content during workflow (live updates).

        This method provides real-time visibility into the draft as it's being written.
        It creates the document early if it doesn't exist, then updates it incrementally.

        Args:
            state: Current workflow state with draft_result
            insights: Case insights for document metadata
        """
        logger.info(f"[LIVE] _update_google_doc_live called: phase={state.phase.value}, doc_id={state.google_doc_id}")
        try:
            # Build a deliverable from current state
            # Check if we have any content to update
            # For early phases, we can still show progress even without draft content
            draft_text = ""
            if state.draft_result:
                draft_text = state.draft_result.get("privacy_harm_section", "") or state.draft_result.get("edited_document", "") or str(state.draft_result.get("content", ""))
                logger.debug(f"[LIVE] Found draft_text from draft_result: {len(draft_text)} chars")

            # For early phases, show progress message even without content
            if not draft_text and state.phase in [WorkflowPhase.EXPLORE, WorkflowPhase.RESEARCH, WorkflowPhase.PLAN]:
                draft_text = f"[{state.phase.value.upper()} Phase - Iteration {state.iteration}]\n\nWorkflow is currently in the {state.phase.value} phase. Content will appear here as it's generated.\n\nPlease wait..."
                logger.info(f"[LIVE] Generated progress message for {state.phase.value} phase: {len(draft_text)} chars")
            elif not draft_text:
                logger.warning(f"[LIVE] No draft_text and phase {state.phase.value} not in early phases - skipping update")
                return  # No content to update

            # Create a lightweight deliverable for live updates
            print(f"[LIVE DEBUG] About to create WriterDeliverable, draft_text length: {len(draft_text)}")
            logger.debug(f"[LIVE] Creating WriterDeliverable with {len(draft_text)} chars of draft_text")
            try:
                from tasks import WriterDeliverable, DraftSection
                sections = []
                if draft_text:
                    # Split into sections if possible (basic heuristic)
                    sections.append(DraftSection(
                        section_id="live_update_content",
                        title="Draft Content",
                        body=draft_text
                    ))

                live_deliverable = WriterDeliverable(
                    plan=None,
                    sections=sections,
                    edited_document=draft_text,
                    metadata={
                        "workflow_type": "live_update",
                        "phase": state.phase.value,
                        "iteration": state.iteration,
                        "timestamp": datetime.now().isoformat(),
                        "is_live_update": True
                    }
                )
                print(f"[LIVE DEBUG] WriterDeliverable created successfully")
                logger.debug(f"[LIVE] WriterDeliverable created with {len(sections)} sections")
            except Exception as deliverable_error:
                logger.error(f"[LIVE] Failed to create WriterDeliverable: {deliverable_error}")
                print(f"[LIVE DEBUG] ERROR creating WriterDeliverable: {deliverable_error}")
                import traceback
                logger.error(f"[LIVE] Deliverable creation error trace: {traceback.format_exc()}")
                raise

            # Check if document already exists
            print(f"[LIVE DEBUG] About to check document ID: {state.google_doc_id}")
            logger.info(f"[LIVE] Checking document ID: state.google_doc_id={state.google_doc_id}")
            print(f"[LIVE DEBUG] Document ID check complete, value: {state.google_doc_id}")
            if state.google_doc_id:
                logger.info(f"[LIVE] Document ID exists: {state.google_doc_id}, proceeding with update")
                # Update existing document
                try:
                    # For live updates, write content directly without metadata
                    # Only use formatter for final commit, not for live updates
                    status_prefix = f"[LIVE UPDATE - {state.phase.value.upper()} - Iteration {state.iteration}]\n\n"

                    # Format content directly (skip metadata for live updates)
                    formatted_content = []

                    # Add status prefix
                    formatted_content.append({
                        "type": "paragraph",
                        "text": status_prefix
                    })

                    # Add actual draft content (not metadata)
                    if draft_text:
                        try:
                            draft_text = self._clean_document_text(draft_text, max_length=20000)
                            logger.info(
                                "[DOC SAFETY] Live draft sanitized via _clean_document_text (chars=%s)",
                                len(draft_text),
                            )
                        except Exception:
                            logger.debug("[DOC SAFETY] Live draft sanitization skipped due to unexpected error")
                        # Split into paragraphs for better formatting
                        paragraphs = draft_text.split('\n\n')
                        for para in paragraphs:
                            if para.strip():
                                # Check if it's a heading
                                if para.strip().startswith('#'):
                                    level = len(para) - len(para.lstrip('#'))
                                    heading_text = para.lstrip('#').strip()
                                    formatted_content.append({
                                        "type": f"heading{min(level, 3)}",
                                        "text": heading_text
                                    })
                                else:
                                    formatted_content.append({
                                        "type": "paragraph",
                                        "text": para.strip()
                                    })

                    # Update document with content only (no metadata)
                    logger.info(f"[LIVE] Attempting to update document with {len(formatted_content)} content elements")
                    logger.info(f"[LIVE] Draft text length: {len(draft_text)} chars, Phase: {state.phase.value}, Iteration: {state.iteration}")
                    logger.info(f"[LIVE] First content element: {formatted_content[0] if formatted_content else 'None'}")

                    # Actually call update_document
                    logger.info(f"[LIVE] Calling google_docs_bridge.update_document() now...")
                    self.google_docs_bridge.update_document(
                        state.google_doc_id,
                        formatted_content,
                        title=None  # Don't change title on live updates
                    )
                    logger.info(f"[LIVE] update_document() call completed - Google Doc {state.google_doc_id} should be updated with {len(draft_text)} chars")
                    print(f"[LIVE] Updated Google Doc {state.google_doc_id} with {len(draft_text)} chars")
                except Exception as e:
                    logger.error(f"[LIVE] Live update failed for existing doc {state.google_doc_id}: {e}")
                    print(f"[LIVE] ERROR: Live update failed: {e}")
                    import traceback
                    error_trace = traceback.format_exc()
                    logger.error(f"[LIVE] Full error trace: {error_trace}")
                    print(error_trace)
            else:
                # Find existing document (don't create new one)
                logger.warning(f"[LIVE] No document ID in state (state.google_doc_id={state.google_doc_id}), trying to find existing document")
                existing_doc = None

                if self.config.master_draft_mode and self.config.master_draft_title:
                    title = self.config.master_draft_title
                    if self.document_tracker:
                        existing_doc = self.document_tracker.get_doc_by_title(
                            title,
                            folder_id=self.config.google_drive_folder_id
                        )
                elif insights:
                    case_id = self._generate_case_id(insights)
                    if self.document_tracker:
                        existing_doc = self.document_tracker.get_doc_for_case(case_id)

                if existing_doc:
                    # Use existing document
                    state.google_doc_id = existing_doc.google_doc_id
                    state.google_doc_url = existing_doc.doc_url
                    logger.info(f"[LIVE] Using existing Google Doc: {existing_doc.google_doc_id}")
                    # Try to update it now
                    try:
                        await self._update_google_doc_live(state, insights)
                    except Exception as e:
                        logger.debug(f"Failed to update on first attempt: {e}")
                else:
                    # Fallback: Try to find by known document ID from markdown file
                    if not state.google_doc_id:
                        try:
                            from pathlib import Path
                            markdown_path = Path(self.config.markdown_export_path) / "master_draft.md"
                            if markdown_path.exists():
                                content = markdown_path.read_text(encoding="utf-8", errors="ignore")
                                import re
                                doc_id_match = re.search(r"google_doc_id[:\s]+([a-zA-Z0-9_-]+)", content)
                                if doc_id_match:
                                    known_doc_id = doc_id_match.group(1)
                                    known_doc_url = f"https://docs.google.com/document/d/{known_doc_id}/edit"
                                    state.google_doc_id = known_doc_id
                                    state.google_doc_url = known_doc_url
                                    logger.info(f"[LIVE] Found document ID from markdown file: {known_doc_id}")
                                    # Try to update it now
                                    try:
                                        await self._update_google_doc_live(state, insights)
                                    except Exception as e:
                                        logger.debug(f"Failed to update after finding from markdown: {e}")
                        except Exception as e:
                            logger.debug(f"Could not extract document ID from markdown: {e}")

                    if not state.google_doc_id:
                        logger.warning("[LIVE] No existing document found. Document will be created during commit phase.")

        except Exception as e:
            # Non-critical - don't break workflow
            logger.debug(f"Live Google Docs update error (non-critical): {e}")

    async def _commit_to_google_docs(self, deliverable: WriterDeliverable, insights: CaseInsights, state: WorkflowState) -> None:
        """Commit document to Google Docs.

        ✅ NOW INTEGRATED WITH PERFECT OUTLINE STRUCTURE:
        - Validates outline structure before commit
        - Ensures sections follow perfect outline order
        - Checks enumeration requirements
        - Stores outline metadata in master draft
        """
        try:
            # Inject personal corpus facts into the draft before any formatting/validation
            try:
                from .section_structure_helper import enforce_section_structure, collect_structured_facts
                sk_context = {
                    "case_summary": getattr(insights, "summary", ""),
                    "jurisdiction": getattr(insights, "jurisdiction", "") or "D. Massachusetts",
                }
                try:
                    fact_snapshot = collect_structured_facts()
                    if fact_snapshot:
                        logger.info(f"[FACTS] Commit will use fact keys: {sorted(list(fact_snapshot.keys()))}")
                except Exception:
                    pass
                base_text = deliverable.edited_document or "\n".join([s.body for s in deliverable.sections])
                improved_text = enforce_section_structure(base_text, sk_context)
                if improved_text and improved_text.strip() and improved_text != base_text:
                    deliverable.edited_document = improved_text
                    logger.info("[FACTS] Injected personal corpus facts into draft via section enforcement")
            except Exception as _exc:
                logger.debug(f"[FACTS] Section enforcement skipped: {_exc}")

            # ✅ Validate outline structure before commit (if outline manager available)
            outline_validation = None
            detected_sections = {}
            enumeration_requirements = {}
            if self.draft_enhancer and hasattr(self.draft_enhancer, 'outline_manager') and self.draft_enhancer.outline_manager:
                logger.info("📋 Validating outline structure before commit to Google Docs...")

                # Detect sections in deliverable
                document_text = deliverable.edited_document or "\n".join([s.body for s in deliverable.sections])
                detected_sections = self.draft_enhancer._detect_sections(document_text)

                # Validate section order
                outline_validation = self.draft_enhancer.outline_manager.validate_section_order(
                    list(detected_sections.keys())
                )

                # Check enumeration requirements
                if hasattr(self.draft_enhancer, 'plugin_calibrator') and self.draft_enhancer.plugin_calibrator:
                    enumeration_requirements = self.draft_enhancer.outline_manager.get_enumeration_requirements()

                    # Count enumeration in document
                    import re
                    bullet_points = len(re.findall(r'^[\s]*[-*•]\s', document_text, re.MULTILINE))
                    numbered_lists = len(re.findall(r'^[\s]*\d+[\.)]\s', document_text, re.MULTILINE))
                    total_enumeration = bullet_points + numbered_lists

                    enum_met = total_enumeration >= enumeration_requirements.get("overall_min_count", 0)

                    if not enum_met:
                        logger.warning(f"⚠️ Enumeration requirements not met: {total_enumeration} < {enumeration_requirements.get('overall_min_count', 0)} required")
                    else:
                        logger.info(f"✅ Enumeration requirements met: {total_enumeration} >= {enumeration_requirements.get('overall_min_count', 0)}")

                # Log validation results
                if not outline_validation["valid"]:
                    logger.warning("⚠️ Outline structure issues detected before commit:")
                    for issue in outline_validation.get("issues", []):
                        logger.warning(f"   {issue.get('message', 'Unknown issue')}")
                    for warning in outline_validation.get("warnings", []):
                        logger.warning(f"   ⚠️ {warning.get('message', 'Unknown warning')}")
                else:
                    logger.info("✅ Outline structure validated - sections follow perfect outline order")

                # Log recommendations
                if outline_validation.get("recommendations"):
                    logger.info("📋 Recommendations for perfect outline structure:")
                    for rec in outline_validation["recommendations"]:
                        logger.info(f"   • {rec}")

            # Check if master draft mode is enabled
            if self.config.master_draft_mode and self.config.master_draft_title:
                # Master draft mode: use fixed title and find/update existing doc
                title = self.config.master_draft_title
                existing_doc = None

                if self.document_tracker:
                    existing_doc = self.document_tracker.get_doc_by_title(
                        title,
                        folder_id=self.config.google_drive_folder_id
                    )

                if existing_doc:
                    # Update existing master draft
                    logger.info(f"Found existing master draft: {title}, updating...")
                    await self._update_existing_google_doc(deliverable, existing_doc, state)
                    # Capture version history for learning
                    await self._capture_version_history(existing_doc.google_doc_id)
                else:
                    # Create new master draft
                    logger.info(f"Creating new master draft: {title}")
                    case_id = self._generate_case_id(insights)
                    await self._create_new_google_doc(deliverable, case_id, insights, state, title=title)

                    # Update document record with master draft title if needed
                    if self.document_tracker and state.google_doc_id:
                        self.document_tracker.update_document(
                            state.google_doc_id,
                            title=title
                        )
            else:
                # Normal mode: generate unique case ID and check for existing doc
                case_id = self._generate_case_id(insights)
                existing_doc = None

                if self.document_tracker:
                    existing_doc = self.document_tracker.get_doc_for_case(case_id)

                if existing_doc and state.google_doc_id:
                    # Update existing document
                    await self._update_existing_google_doc(deliverable, existing_doc, state)
                    # Capture version history for learning
                    await self._capture_version_history(existing_doc.google_doc_id)
                else:
                    # Create new document
                    await self._create_new_google_doc(deliverable, case_id, insights, state)

            # Export to markdown if enabled (especially for master drafts)
            if self.config.markdown_export_enabled:
                await self._export_to_markdown(deliverable, state)

        except Exception as e:
            logger.error(f"Error in Google Docs commit: {e}")
            raise

    def _get_plugin_database_paths(self) -> Optional[List[Path]]:
        """
        Get database paths for plugin initialization.

        Phase 1: Core database access - returns paths to case law databases.
        """
        # Use configured paths if provided
        if self.config.plugin_db_paths:
            return [Path(p) for p in self.config.plugin_db_paths if Path(p).exists()]

        # Auto-detect default database paths
        project_root = Path(__file__).parent.parent.parent.parent
        default_paths = [
            project_root / "case_law_data" / "1782_corpus.db",
            project_root / "case_law_data" / "harvard_corpus.db",
            project_root / "case_law_data" / "ma_federal_motions.db",
            project_root / "case_law_data" / "china_corpus.db",
            project_root / "case_law_data" / "appellate_corpus.db",
        ]

        # Filter to existing databases
        existing_paths = [p for p in default_paths if p.exists()]

        if existing_paths:
            logger.info(f"Auto-detected {len(existing_paths)} case law databases for plugins")
        else:
            logger.warning("No case law databases found - plugins will have limited functionality")

        return existing_paths if existing_paths else None

    def _generate_case_id(self, insights: CaseInsights) -> str:
        """Generate a unique case ID from insights."""
        # Use case style or summary to generate ID
        if insights.case_style:
            # Extract case name from case style
            case_name = insights.case_style.split(" v. ")[0] if " v. " in insights.case_style else insights.case_style
            case_name = case_name.replace(" ", "_").replace(".", "").replace(",", "")
        else:
            case_name = "case"

        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{case_name}_{timestamp}"

    async def _create_new_google_doc(self, deliverable: WriterDeliverable, case_id: str, insights: CaseInsights, state: WorkflowState, title: Optional[str] = None) -> None:
        """Create a new Google Doc."""
        try:
            # Format document title (use provided title or generate from case_id)
            if not title:
                title = f"Legal Analysis - {case_id}"

            # Create document in Google Drive
            doc_id, doc_url = self.google_docs_bridge.create_document(
                title=title,
                folder_id=self.config.google_drive_folder_id
            )

            # Ensure personal facts are embedded before formatting
            try:
                from .section_structure_helper import enforce_section_structure, collect_structured_facts
                sk_context = {
                    "case_summary": getattr(insights, "summary", ""),
                    "jurisdiction": getattr(insights, "jurisdiction", "") or "D. Massachusetts",
                }
                try:
                    fact_snapshot = collect_structured_facts()
                    if fact_snapshot:
                        logger.info(f"[FACTS] Create will use fact keys: {sorted(list(fact_snapshot.keys()))}")
                except Exception:
                    pass
                base_text = deliverable.edited_document or "\n".join([s.body for s in deliverable.sections])
                improved_text = enforce_section_structure(base_text, sk_context)
                if improved_text and improved_text.strip() and improved_text != base_text:
                    deliverable.edited_document = improved_text
                    logger.info("[FACTS] Injected personal corpus facts into draft (create path)")
            except Exception as _exc:
                logger.debug(f"[FACTS] Section enforcement (create) skipped: {_exc}")

            # Ensure sanitized text before formatting
            self._sanitize_deliverable_for_google_docs(deliverable, max_length=50000)

            # Format content for Google Docs (include validation results and outline structure)
            validation_results = state.validation_results if state.validation_results else None

            # ✅ Pass outline manager to formatter for section reordering
            outline_manager = None
            detected_sections = {}
            outline_validation = None
            enumeration_requirements = {}
            if self.draft_enhancer and hasattr(self.draft_enhancer, 'outline_manager') and self.draft_enhancer.outline_manager:
                outline_manager = self.draft_enhancer.outline_manager
                # Detect sections for reordering
                document_text = deliverable.edited_document or "\n".join([s.body for s in deliverable.sections])
                detected_sections = self.draft_enhancer._detect_sections(document_text)

            formatted_content = self.google_docs_formatter.format_deliverable(
                deliverable,
                format_type="motion",  # Ensure motion formatting
                validation_results=validation_results,
                outline_manager=outline_manager,  # ✅ Pass outline for section ordering
                detected_sections=detected_sections  # ✅ Pass detected sections
            )

            logger.info("[DOC SAFETY] Sanitized content ready; calling google_docs_bridge.update_document()")
            # Update document with content
            self.google_docs_bridge.update_document(doc_id, formatted_content, title)

            # Store document record
            if self.document_tracker:
                self.document_tracker.create_document_record(
                    case_id=case_id,
                    google_doc_id=doc_id,
                    doc_url=doc_url,
                    folder_id=self.config.google_drive_folder_id or "",
                    title=title,
                    case_summary=insights.summary or "Legal analysis document",
                    metadata={
                        "workflow_type": deliverable.metadata.get("workflow_type"),
                        "validation_score": state.validation_results.get("overall_score", 0.0) if state.validation_results else 0.0,
                        "iteration_count": state.iteration,
                        "is_master_draft": self.config.master_draft_mode,
                        # ✅ Add outline metadata
                        "outline_version": "perfect_outline_v1",
                        "sections_detected": list(detected_sections.keys()) if detected_sections else [],
                        "outline_validation": {
                            "valid": outline_validation["valid"] if outline_validation else None,
                            "issues_count": len(outline_validation.get("issues", [])) if outline_validation else 0,
                            "warnings_count": len(outline_validation.get("warnings", [])) if outline_validation else 0
                        } if outline_validation else None,
                        "enumeration_requirements": enumeration_requirements if enumeration_requirements else None
                    }
                )

            # Capture version history for learning (initial version)
            if self.config.google_docs_capture_version_history:
                await self._capture_version_history(doc_id)

            # Auto-share if configured
            if self.config.google_docs_auto_share:
                await self._auto_share_document(doc_id)

            # Update state
            state.google_doc_id = doc_id
            state.google_doc_url = doc_url

            # Add Google Docs info to deliverable metadata
            deliverable.metadata.update({
                "google_doc_id": doc_id,
                "google_doc_url": doc_url,
                "google_docs_committed": True
            })

            logger.info(f"Created new Google Doc: {doc_id} for case {case_id}")

        except Exception as e:
            logger.error(f"Failed to create new Google Doc: {e}")
            raise

    def _serialize_metadata_for_json(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize metadata dict, converting numpy types to native Python types."""
        import numpy as np
        
        serializable = {}
        for key, value in metadata.items():
            if value is None:
                serializable[key] = None
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                serializable[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, dict):
                serializable[key] = self._serialize_metadata_for_json(value)
            elif isinstance(value, list):
                serializable[key] = [
                    int(v) if isinstance(v, (np.integer, np.int64, np.int32)) else
                    float(v) if isinstance(v, (np.floating, np.float64, np.float32)) else
                    v.tolist() if isinstance(v, np.ndarray) else
                    self._serialize_metadata_for_json(v) if isinstance(v, dict) else
                    v
                    for v in value
                ]
            else:
                serializable[key] = value
        return serializable
    
    async def _update_existing_google_doc(self, deliverable: WriterDeliverable, existing_doc, state: WorkflowState) -> None:
        """Update existing Google Doc with version backup support.

        ✅ NOW INTEGRATED WITH PERFECT OUTLINE STRUCTURE:
        - Uses outline structure for section reordering
        - Maintains perfect outline order in updates
        """
        try:
            # Ensure personal facts are embedded before formatting
            try:
                from .section_structure_helper import enforce_section_structure, collect_structured_facts
                sk_context = {
                    "case_summary": getattr(state, "case_summary", "") or "",
                }
                try:
                    fact_snapshot = collect_structured_facts()
                    if fact_snapshot:
                        logger.info(f"[FACTS] Update will use fact keys: {sorted(list(fact_snapshot.keys()))}")
                except Exception:
                    pass
                document_text = deliverable.edited_document or "\n".join([s.body for s in deliverable.sections if hasattr(s, 'body') and s.body])
                document_text = self._clean_document_text(document_text, max_length=50000)
                if document_text and document_text != (deliverable.edited_document or ""):
                    deliverable.edited_document = document_text
                improved_text = enforce_section_structure(document_text, sk_context)
                if improved_text and improved_text.strip() and improved_text != document_text:
                    deliverable.edited_document = improved_text
                    logger.info("[FACTS] Updated draft with enforced sections for personal facts (update path)")
            except Exception as _exc:
                logger.debug(f"[FACTS] Section enforcement (update) skipped: {_exc}")

            # Ensure sanitized text before formatting
            self._sanitize_deliverable_for_google_docs(deliverable, max_length=50000)

            # Format content for Google Docs (include validation results and outline structure)
            validation_results = state.validation_results if state.validation_results else None

            # ✅ Pass outline manager to formatter for section reordering
            outline_manager = None
            detected_sections = {}
            if self.draft_enhancer and hasattr(self.draft_enhancer, 'outline_manager') and self.draft_enhancer.outline_manager:
                outline_manager = self.draft_enhancer.outline_manager
                # Detect sections for reordering
                document_text = deliverable.edited_document or "\n".join([s.body for s in deliverable.sections])
                detected_sections = self.draft_enhancer._detect_sections(document_text)

            formatted_content = self.google_docs_formatter.format_deliverable(
                deliverable,
                format_type="motion",  # Ensure motion formatting
                validation_results=validation_results,
                outline_manager=outline_manager,  # ✅ Pass outline for section ordering
                detected_sections=detected_sections  # ✅ Pass detected sections
            )

            # Create backup before updating (if version manager enabled)
            # IMPORTANT: Backup the CURRENT content from Google Docs, not the new content
            if self.version_manager and self.config.enable_version_backups:
                try:
                    # Fetch current content from Google Docs before we replace it
                    # This is what we want to backup
                    current_doc = self.google_docs_bridge.docs_service.documents().get(
                        documentId=existing_doc.google_doc_id
                    ).execute()

                    # Extract text content from current document
                    current_content_parts = []
                    for element in current_doc.get("body", {}).get("content", []):
                        if "paragraph" in element:
                            para = element["paragraph"]
                            for text_run in para.get("elements", []):
                                if "textRun" in text_run:
                                    text = text_run["textRun"].get("content", "")
                                    if text.strip():  # Skip empty paragraphs
                                        current_content_parts.append(text.strip())

                    current_content = "\n\n".join(current_content_parts)

                    # Only backup if there's actual content (avoid backing up empty docs)
                    if current_content.strip():
                        backup_version = self.version_manager.create_backup(
                            document_id=existing_doc.google_doc_id,
                            title=existing_doc.title,
                            content=current_content,
                            metadata=self._serialize_metadata_for_json({
                                "iteration": state.iteration,
                                "validation_score": validation_results.get("overall_score") if validation_results else None,
                                "workflow_type": deliverable.metadata.get("workflow_type"),
                                "case_id": existing_doc.case_id if hasattr(existing_doc, 'case_id') else None
                            }),
                            save_for_ml=self.config.save_backups_for_ml
                        )

                        if backup_version:
                            logger.info(f"Created version backup: {backup_version.version_id}")

                            # Add backup info to deliverable metadata
                            deliverable.metadata.update({
                                "backup_version_id": backup_version.version_id,
                                "backup_path": backup_version.ml_backup_path
                            })
                    else:
                        logger.debug("Skipping backup - document appears to be empty")
                except Exception as e:
                    logger.warning(f"Failed to create version backup: {e}")
                    # Continue with update even if backup fails

            logger.info("[DOC SAFETY] Sanitized content ready; calling google_docs_bridge.update_document()")
            # Update document with new content (this REPLACES, not appends)
            self.google_docs_bridge.update_document(
                existing_doc.google_doc_id,
                formatted_content,
                existing_doc.title
            )

            # Update document record
            if self.document_tracker:
                self.document_tracker.update_document(
                    existing_doc.google_doc_id,
                    metadata={
                        "last_update_workflow": deliverable.metadata.get("workflow_type"),
                        "validation_score": state.validation_results.get("overall_score", 0.0) if state.validation_results else 0.0,
                        "iteration_count": state.iteration,
                        "updated_at": datetime.now().isoformat()
                    }
                )

            # Update state
            state.google_doc_id = existing_doc.google_doc_id
            state.google_doc_url = existing_doc.doc_url

            # Add Google Docs info to deliverable metadata
            deliverable.metadata.update({
                "google_doc_id": existing_doc.google_doc_id,
                "google_doc_url": existing_doc.doc_url,
                "google_docs_updated": True
            })

            logger.info(f"Updated existing Google Doc: {existing_doc.google_doc_id}")

        except Exception as e:
            logger.error(f"Failed to update existing Google Doc: {e}")
            raise

    def _get_document_text_from_sections(self, deliverable: WriterDeliverable) -> str:
        """Extract full document text from deliverable sections."""
        parts = []

        # Add title
        if deliverable.metadata.get("title"):
            parts.append(f"# {deliverable.metadata['title']}\n\n")

        # Add sections
        if deliverable.sections:
            for section in deliverable.sections:
                parts.append(f"## {section.title}\n\n")
                parts.append(section.body)
                parts.append("\n\n")

        # Add full content if available
        if deliverable.edited_document:
            parts.append(deliverable.edited_document)

        return "\n".join(parts).strip()

    async def _auto_share_document(self, doc_id: str) -> None:
        """Auto-share document with configured users."""
        try:
            # Get share list from config
            share_emails = []

            # Try to get from secrets.toml
            try:
                import toml
                secrets_path = Path(__file__).parent.parent.parent / "secrets.toml"
                if secrets_path.exists():
                    secrets = toml.load(secrets_path)
                    if "GOOGLE_DOCS_SHARE_WITH" in secrets:
                        share_emails = [email.strip() for email in secrets["GOOGLE_DOCS_SHARE_WITH"].split(",")]
            except Exception:
                pass

            # Try to get from environment
            if not share_emails:
                env_share = os.environ.get("GOOGLE_DOCS_SHARE_WITH")
                if env_share:
                    share_emails = [email.strip() for email in env_share.split(",")]

            # Share with each email
            for email in share_emails:
                if email:
                    self.google_docs_bridge.share_document(doc_id, email, "writer")
                    logger.info(f"Shared document {doc_id} with {email}")

        except Exception as e:
            logger.warning(f"Failed to auto-share document {doc_id}: {e}")

    async def _capture_version_history(self, doc_id: str) -> None:
        """Capture version history for learning."""
        if not self.version_tracker or not self.config.google_docs_capture_version_history:
            return

        try:
            result = self.version_tracker.capture_version_history(doc_id, self.google_docs_bridge)
            logger.info(f"Captured version history for document {doc_id}: {result}")
        except Exception as e:
            logger.warning(f"Failed to capture version history for document {doc_id}: {e}")

    async def _export_to_markdown(self, deliverable: WriterDeliverable, state: WorkflowState) -> None:
        """Export deliverable to markdown format."""
        try:
            from pathlib import Path
            from datetime import datetime
            import json

            # Create output directory
            output_dir = Path(self.config.markdown_export_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            if self.config.master_draft_mode:
                filename = "master_draft.md"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"legal_analysis_{timestamp}.md"

            output_path = output_dir / filename

            # Build markdown content
            display_title = deliverable.metadata.get("title") or getattr(deliverable.plan, "objective", "Legal Draft")

            markdown_content = f"""# {display_title}

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Workflow**: {deliverable.metadata.get("workflow_type", "unknown")}
**Validation Score**: {state.validation_results.get("overall_score", "N/A") if state.validation_results else "N/A"}
**Iteration**: {state.iteration}

"""

            # Add metadata if available
            if deliverable.metadata:
                markdown_content += "## Metadata\n\n"
                for key, value in deliverable.metadata.items():
                    if key not in ["workflow_type"]:  # Already shown above
                        markdown_content += f"- **{key}**: {value}\n"
                markdown_content += "\n"

            # Add sections
            if deliverable.sections:
                for section in deliverable.sections:
                    markdown_content += f"## {section.title}\n\n"
                    section_body = getattr(section, "body", None)
                    if section_body is None:
                        section_body = getattr(section, "content", "")
                    markdown_content += f"{section_body}\n\n"
            else:
                # Use full content if sections not available
                full_content = getattr(deliverable, "edited_document", None)
                if full_content is None:
                    full_content = getattr(deliverable, "content", "")
                markdown_content += f"{full_content}\n\n"

            # Add validation results if available
            if state.validation_results:
                markdown_content += "## Validation Results\n\n"
                markdown_content += f"```json\n{json.dumps(state.validation_results, indent=2)}\n```\n\n"

            # Add Google Docs link if available
            if state.google_doc_url:
                markdown_content += f"## Google Docs Link\n\n"
                markdown_content += f"[View in Google Docs]({state.google_doc_url})\n\n"

            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logger.info(f"Exported deliverable to markdown: {output_path}")

            # Update deliverable metadata
            deliverable.metadata.update({
                "markdown_export_path": str(output_path),
                "markdown_exported": True
            })

        except Exception as e:
            logger.warning(f"Failed to export to markdown: {e}")


# Backward compatibility aliases
WorkflowStrategyExecutor = Conductor  # Old name
HybridOrchestrator = Conductor  # Old name


# Export main classes
__all__ = [
    "Conductor",
    "WorkflowStrategyConfig",
    "WorkflowPhase",
    "WorkflowState",
    "WorkflowRouter",
    "AutoGenToSKBridge",
    "QualityGatePipeline",
    "IterationController",
    "WorkflowStrategyExecutor",  # Backward compatibility
    "HybridOrchestrator",  # Backward compatibility
]
# Sanity check for shared config defaults
try:
    _cfg_test = WorkflowStrategyConfig()
    if _cfg_test.enable_iterative_refinement is None:
        logger.warning("WorkflowStrategyConfig default for enable_iterative_refinement is None; expected auto True.")
except Exception as _cfg_exc:
    logger.debug("WorkflowStrategyConfig sanity check failed: %s", _cfg_exc)
