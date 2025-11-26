"""Integration helpers for running pgmpy (primary) and PySMILE (optional fallback) for building Writer insights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

# Import pgmpy as primary BN engine.
try:
    from .pgmpy_inference import HAVE_PGMPY, run_pgmpy_inference
except ImportError:
    HAVE_PGMPY = False
    run_pgmpy_inference = None # type: ignore

# Optional: Bayes Fusion support for legacy .xdsl models
try: # pragma: no cover - optional dependency
    import pysmile
except ImportError: # pragma: no cover - optional dependency
    pysmile = None # type: ignore

from .bn_bridge import build_insights
from .insights import CaseInsights, EvidenceItem

# Import always-on evidence system.
try:
    from .always_on_evidence import merge_case_evidence
    HAVE_ALWAYS_ON = True
except ImportError:
    HAVE_ALWAYS_ON = False
    merge_case_evidence = None # type: ignore


logger = logging.getLogger(__name__)


class PgmpyUnavailableError(RuntimeError):
    """Raised when pgmpy is not installed for Bayesian inference (primary engine)."""


class PySmileUnavailableError(RuntimeError):
    """Raised when PySMILE is not installed for Bayesian inference (optional fallback)."""


def _ensure_pgmpy() -> None:
    """Ensure pgmpy is available (primary BN engine)."""
    if not HAVE_PGMPY:
        raise PgmpyUnavailableError(
            "pgmpy is required for BN integration (primary engine). "
            "Install with: pip install pgmpy networkx"
        )


def _ensure_pysmile() -> None:
    """Ensure pysmile is available (optional fallback)."""
    if pysmile is None: # pragma: no cover - optional dependency
        raise PySmileUnavailableError(
            "PySMILE is required for legacy BN integration. "
            "Consider using pgmpy instead (pip install pgmpy networkx)"
        )


def run_bn_inference(
    model_path: Path,
    evidence: Dict[str, str],
    summary: str,
    *,
    reference_id: str = "case",
    apply_always_on: bool = False,
    prefer_pysmile: bool = False,
) -> Tuple[CaseInsights, Dict[str, Dict[str, float]]]:
    """Run Bayesian network inference using pgmpy (primary) or pysmile (optional).

    Args:
        model_path: Path to XDSL model file
        evidence: Evidence dictionary
        summary: Case summary text
        reference_id: Case identifier
        apply_always_on: If True, merge with always-on case evidence
        prefer_pysmile: If True, use pysmile instead of pgmpy (legacy mode)

    Returns:
        Tuple of (CaseInsights, posterior_data)
    """
    # Apply always-on evidence if requested
    if apply_always_on and HAVE_ALWAYS_ON and merge_case_evidence is not None:
        evidence = merge_case_evidence(evidence)
        logger.info(f"Applied always-on evidence. Final evidence: {evidence}")

    # Try pgmpy first (primary engine) unless explicitly preferring pysmile
    if not prefer_pysmile and HAVE_PGMPY and run_pgmpy_inference is not None:
        try:
            logger.info("Running BN inference with pgmpy (primary engine)")
            return run_pgmpy_inference(model_path, evidence, summary, reference_id=reference_id)
        except Exception as exc:
            logger.warning(f"pgmpy inference failed: {exc}")
            if pysmile is not None:
                logger.info("Falling back to pysmile")
            else:
                raise PgmpyUnavailableError(
                    f"pgmpy inference failed and pysmile not available. "
                    f"Error: {exc}"
                ) from exc

    # Fallback to pysmile (optional legacy support)
    if pysmile is not None:
        logger.info("Running BN inference with pysmile (legacy fallback)")
        _ensure_pysmile()

        network = pysmile.Network()
        network.read_file(str(model_path))
        for node_id, outcome in evidence.items():
            network.set_evidence(node_id, outcome)
        network.update_beliefs()

        posterior_data: Dict[str, Dict[str, float]] = {}

        for index in range(network.get_node_count()):
            handle = network.get_node_handle_by_index(index)
            node_id = network.get_node_id(handle)
            outcome_count = network.get_outcome_count(handle)
            node_map: Dict[str, float] = {}
            for outcome_index in range(outcome_count):
                outcome_id = network.get_outcome_id(handle, outcome_index)
                probability = network.get_node_value(handle)[outcome_index]
                node_map[outcome_id] = probability
            posterior_data[node_id] = node_map

        evidence_items = [EvidenceItem(node_id=node, state=state) for node, state in evidence.items()]
        insights = build_insights(
            reference_id=reference_id,
            summary=summary,
            posterior_data=posterior_data,
            evidence=evidence_items,
        )
        return insights, posterior_data
    else:
        raise PgmpyUnavailableError(
            "No Bayesian inference engine available. "
            "Install pgmpy (primary): pip install pgmpy networkx"
        )


def run_bn_inference_with_fallback(
    model_path: Path,
    evidence: Dict[str, str],
    summary: str,
    *,
    reference_id: str = "case",
    apply_always_on: bool = False,
) -> Tuple[CaseInsights, Dict[str, Dict[str, float]]]:
    """Run Bayesian network inference with automatic fallback.

    This function attempts to use pgmpy first (primary), then falls back to pysmile
    if pgmpy is unavailable or fails.

    Args:
        model_path: Path to the .xdsl file
        evidence: Dictionary mapping node IDs to observed states
        summary: Case summary text
        reference_id: Unique identifier for this case
        apply_always_on: If True, merge with always-on case evidence (OGC, Email, Statements)

    Returns:
        Tuple of (CaseInsights, posterior_data)

    Raises:
        RuntimeError: If both pgmpy and pysmile are unavailable or fail
    """
    # Apply always-on evidence if requested
    if apply_always_on and HAVE_ALWAYS_ON and merge_case_evidence is not None:
        original_evidence = evidence.copy()
        evidence = merge_case_evidence(evidence)
        logger.info(f"Applied always-on evidence: {original_evidence} -> {evidence}")

    # Try pgmpy first (primary engine).
    if HAVE_PGMPY and run_pgmpy_inference is not None:
        try:
            logger.info("Attempting inference with pgmpy (primary engine)")
            return run_pgmpy_inference(model_path, evidence, summary, reference_id=reference_id)
        except Exception as exc:
            logger.warning(f"pgmpy inference failed: {exc}")
            logger.info("Falling back to pysmile")
    else:
        logger.info("pgmpy not available, using pysmile fallback")

    # Fallback to pysmile (optional legacy support).
    if pysmile is not None:
        try:
            return run_bn_inference(model_path, evidence, summary, reference_id=reference_id, apply_always_on=False, prefer_pysmile=True) # Already applied
        except Exception as exc:
            logger.error(f"pysmile inference also failed: {exc}")
            raise RuntimeError(
                f"Both pgmpy and pysmile inference failed. "
                f"pgmpy available: {HAVE_PGMPY}, "
                f"pysmile available: {pysmile is not None}. "
                f"Last error: {exc}"
            ) from exc
    else:
        raise RuntimeError(
            "No Bayesian inference engine available. "
            "Install pgmpy (primary): pip install pgmpy networkx"
        )


__all__ = [
    "run_bn_inference",
    "run_bn_inference_with_fallback",
    "PgmpyUnavailableError",
    "PySmileUnavailableError",
]


