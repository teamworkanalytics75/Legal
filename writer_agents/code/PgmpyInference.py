"""Bayesian network inference using pgmpy library.

This module builds pgmpy BayesianNetwork objects from parsed XDSL data
and performs inference using Variable Elimination.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .bn_bridge import build_insights
from .insights import CaseInsights, EvidenceItem
from .xdsl_parser import XDSLNetwork, parse_xdsl, validate_network

# Optional pgmpy dependency.
try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    try:
        # Try new name first (pgmpy >= 0.1.25)
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        # Fall back to old name
        from pgmpy.models import BayesianNetwork
    
    HAVE_PGMPY = True
except ImportError:
    HAVE_PGMPY = False
    # Define placeholder types for type hints.
    BayesianNetwork = None # type: ignore
    TabularCPD = None # type: ignore
    VariableElimination = None # type: ignore


logger = logging.getLogger(__name__)


class PgmpyUnavailableError(RuntimeError):
    """Raised when pgmpy is not installed."""


class InferenceError(RuntimeError):
    """Raised when inference fails."""


def _ensure_pgmpy() -> None:
    """Check that pgmpy is available."""
    if not HAVE_PGMPY:
        raise PgmpyUnavailableError(
            "pgmpy is required for Bayesian inference. "
            "Install with: pip install pgmpy networkx"
        )


def build_bn_from_xdsl(xdsl_network: XDSLNetwork) -> BayesianNetwork:
    """Build a pgmpy BayesianNetwork from parsed XDSL data.
    
    Args:
        xdsl_network: Parsed XDSL network structure
        
    Returns:
        pgmpy BayesianNetwork ready for inference
        
    Raises:
        PgmpyUnavailableError: If pgmpy is not installed
        InferenceError: If network construction fails
    """
    _ensure_pgmpy()
    
    # Create the network structure.
    model = BayesianNetwork(xdsl_network.edges)
    
    # Add CPDs for each node.
    cpds: List[TabularCPD] = []
    
    for node_id, node_info in xdsl_network.nodes.items():
        try:
            cpd = _create_cpd(node_id, node_info, xdsl_network)
            cpds.append(cpd)
        except Exception as exc:
            raise InferenceError(f"Failed to create CPD for node '{node_id}': {exc}") from exc
    
    # Add all CPDs to the model.
    try:
        model.add_cpds(*cpds)
    except Exception as exc:
        raise InferenceError(f"Failed to add CPDs to model: {exc}") from exc
    
    # Verify that the model is valid.
    try:
        if not model.check_model():
            raise InferenceError("Model validation failed")
    except Exception as exc:
        raise InferenceError(f"Model validation error: {exc}") from exc
    
    logger.info(f"Built Bayesian network with {len(xdsl_network.nodes)} nodes")
    return model


def _create_cpd(node_id: str, node_info, xdsl_network: XDSLNetwork) -> TabularCPD:
    """Create a TabularCPD for a single node.
    
    Args:
        node_id: The node identifier
        node_info: NodeInfo with states and probabilities
        xdsl_network: Full network for looking up parent info
        
    Returns:
        TabularCPD ready to add to the network
    """
    num_states = len(node_info.states)
    state_names = list(node_info.states)
    
    if not node_info.parents:
        # Root node: simple prior probability distribution.
        # Shape: [num_states, 1]
        values = [[p] for p in node_info.probabilities]
        
        return TabularCPD(
            variable=node_id,
            variable_card=num_states,
            values=values,
            state_names={node_id: state_names},
        )
    else:
        # Child node: conditional probability table.
        parent_ids = list(node_info.parents)
        parent_cards = [len(xdsl_network.nodes[p].states) for p in parent_ids]
        
        # Build state names dict for all variables.
        state_names_dict = {node_id: state_names}
        for parent_id in parent_ids:
            state_names_dict[parent_id] = list(xdsl_network.nodes[parent_id].states)
        
        # Calculate total parent configurations.
        total_parent_configs = 1
        for card in parent_cards:
            total_parent_configs *= card
        
        # Reshape probabilities into pgmpy format.
        # XDSL format: flat list with child states cycling fastest.
        # pgmpy format: 2D array [num_states, total_parent_configs].
        probs = list(node_info.probabilities)
        
        # Group probabilities by parent configuration.
        values = []
        for state_idx in range(num_states):
            row = []
            for config_idx in range(total_parent_configs):
                prob_idx = config_idx * num_states + state_idx
                row.append(probs[prob_idx])
            values.append(row)
        
        return TabularCPD(
            variable=node_id,
            variable_card=num_states,
            values=values,
            evidence=parent_ids,
            evidence_card=parent_cards,
            state_names=state_names_dict,
        )


def run_inference(
    model: BayesianNetwork,
    evidence: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Run inference on the Bayesian network with given evidence.
    
    Args:
        model: The pgmpy BayesianNetwork
        evidence: Dictionary mapping node IDs to state names
        
    Returns:
        Dictionary mapping node IDs to posterior probability distributions
        
    Raises:
        InferenceError: If inference fails
    """
    _ensure_pgmpy()
    
    # Create inference engine.
    try:
        inference = VariableElimination(model)
    except Exception as exc:
        raise InferenceError(f"Failed to create inference engine: {exc}") from exc
    
    # Get all nodes in the network.
    all_nodes = list(model.nodes())
    
    # Run inference for each node.
    posterior_data: Dict[str, Dict[str, float]] = {}
    
    for node_id in all_nodes:
        try:
            # Query the posterior distribution for this node.
            result = inference.query(
                variables=[node_id],
                evidence=evidence,
                show_progress=False,
            )
            
            # Extract probabilities.
            prob_dict: Dict[str, float] = {}
            for state_name, prob in zip(result.state_names[node_id], result.values):
                prob_dict[state_name] = float(prob)
            
            posterior_data[node_id] = prob_dict
            
        except Exception as exc:
            logger.warning(f"Inference failed for node '{node_id}': {exc}")
            # Continue with other nodes.
            continue
    
    if not posterior_data:
        raise InferenceError("Inference produced no results")
    
    logger.info(f"Inference complete: computed posteriors for {len(posterior_data)} nodes")
    return posterior_data


def run_pgmpy_inference(
    model_path: Path,
    evidence: Dict[str, str],
    summary: str,
    *,
    reference_id: str = "case",
) -> Tuple[CaseInsights, Dict[str, Dict[str, float]]]:
    """Run pgmpy-based inference and produce case insights.
    
    This is the main entry point that mirrors the PySMILE interface.
    
    Args:
        model_path: Path to the .xdsl file
        evidence: Dictionary mapping node IDs to observed states
        summary: Case summary text
        reference_id: Unique identifier for this case
        
    Returns:
        Tuple of (CaseInsights, posterior_data)
        
    Raises:
        PgmpyUnavailableError: If pgmpy is not installed
        InferenceError: If inference fails
    """
    _ensure_pgmpy()
    
    logger.info(f"Running pgmpy inference on {model_path}")
    logger.info(f"Evidence: {evidence}")
    
    # Parse the XDSL file.
    try:
        xdsl_network = parse_xdsl(model_path)
        validate_network(xdsl_network)
    except Exception as exc:
        raise InferenceError(f"Failed to parse XDSL file: {exc}") from exc
    
    # Build the Bayesian network.
    model = build_bn_from_xdsl(xdsl_network)
    
    # Run inference.
    posterior_data = run_inference(model, evidence)
    
    # Build case insights.
    evidence_items = [
        EvidenceItem(node_id=node, state=state)
        for node, state in evidence.items()
    ]
    
    insights = build_insights(
        reference_id=reference_id,
        summary=summary,
        posterior_data=posterior_data,
        evidence=evidence_items,
    )
    
    logger.info("pgmpy inference completed successfully")
    return insights, posterior_data


__all__ = [
    "build_bn_from_xdsl",
    "run_inference",
    "run_pgmpy_inference",
    "PgmpyUnavailableError",
    "InferenceError",
    "HAVE_PGMPY",
]

