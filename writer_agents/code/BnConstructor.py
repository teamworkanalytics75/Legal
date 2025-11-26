"""Bayesian Network construction from selected structure.

This module builds pgmpy BayesianNetwork models from the selected KG structure
with exact inference using Variable Elimination.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork
    
    HAVE_PGMPY = True
except ImportError:
    HAVE_PGMPY = False
    BayesianNetwork = None # type: ignore
    TabularCPD = None # type: ignore
    VariableElimination = None # type: ignore

try:
    from .build_bn_structure_from_kg import BNStructure, SelectedNode
except ImportError:
    from build_bn_structure_from_kg import BNStructure, SelectedNode


logger = logging.getLogger(__name__)


class BNConstructionError(RuntimeError):
    """Raised when BN construction fails."""


def _ensure_pgmpy() -> None:
    """Check that pgmpy is available."""
    if not HAVE_PGMPY:
        raise BNConstructionError(
            "pgmpy is required. Install with: pip install pgmpy networkx"
        )


def initialize_uniform_cpds(
    structure: BNStructure,
) -> List[TabularCPD]:
    """Initialize CPDs with uniform priors.
    
    Args:
        structure: BN structure with nodes and edges
        
    Returns:
        List of TabularCPD objects
    """
    logger.info("Initializing uniform CPDs...")
    
    cpds: List[TabularCPD] = []
    node_dict = {node.node_id: node for node in structure.nodes}
    
    for node in structure.nodes:
        num_states = len(node.states)
        parents = structure.node_parents.get(node.node_id, [])
        
        if not parents:
            # Root node: uniform prior.
            values = [[1.0 / num_states] for _ in range(num_states)]
            
            cpd = TabularCPD(
                variable=node.node_id,
                variable_card=num_states,
                values=values,
                state_names={node.node_id: node.states},
            )
        else:
            # Child node: uniform conditional probabilities.
            parent_cards = [len(node_dict[p].states) for p in parents]
            total_parent_configs = 1
            for card in parent_cards:
                total_parent_configs *= card
            
            # Uniform distribution for each parent configuration.
            values = []
            for state_idx in range(num_states):
                row = [1.0 / num_states] * total_parent_configs
                values.append(row)
            
            # Build state names dict.
            state_names_dict = {node.node_id: node.states}
            for parent_id in parents:
                state_names_dict[parent_id] = node_dict[parent_id].states
            
            cpd = TabularCPD(
                variable=node.node_id,
                variable_card=num_states,
                values=values,
                evidence=parents,
                evidence_card=parent_cards,
                state_names=state_names_dict,
            )
        
        cpds.append(cpd)
    
    logger.info(f"Created {len(cpds)} uniform CPDs")
    return cpds


def initialize_expert_priors(
    structure: BNStructure,
    prior_knowledge: Optional[Dict[str, Dict]] = None,
) -> List[TabularCPD]:
    """Initialize CPDs with expert priors where available.
    
    Args:
        structure: BN structure
        prior_knowledge: Optional dict mapping node_id to prior distributions
        
    Returns:
        List of TabularCPD objects
    """
    logger.info("Initializing CPDs with expert priors...")
    
    if prior_knowledge is None:
        prior_knowledge = _get_default_priors()
    
    cpds: List[TabularCPD] = []
    node_dict = {node.node_id: node for node in structure.nodes}
    
    for node in structure.nodes:
        num_states = len(node.states)
        parents = structure.node_parents.get(node.node_id, [])
        
        # Check if we have expert priors for this node.
        if node.node_id in prior_knowledge:
            expert_priors = prior_knowledge[node.node_id]
            # Use expert priors (simplified - needs proper implementation).
            values = _apply_expert_priors(node, parents, expert_priors, node_dict)
        else:
            # Default to uniform.
            if not parents:
                values = [[1.0 / num_states] for _ in range(num_states)]
            else:
                parent_cards = [len(node_dict[p].states) for p in parents]
                total_configs = 1
                for card in parent_cards:
                    total_configs *= card
                values = [[1.0 / num_states] * total_configs for _ in range(num_states)]
        
        # Create CPD.
        if not parents:
            cpd = TabularCPD(
                variable=node.node_id,
                variable_card=num_states,
                values=values,
                state_names={node.node_id: node.states},
            )
        else:
            parent_cards = [len(node_dict[p].states) for p in parents]
            state_names_dict = {node.node_id: node.states}
            for parent_id in parents:
                state_names_dict[parent_id] = node_dict[parent_id].states
            
            cpd = TabularCPD(
                variable=node.node_id,
                variable_card=num_states,
                values=values,
                evidence=parents,
                evidence_card=parent_cards,
                state_names=state_names_dict,
            )
        
        cpds.append(cpd)
    
    logger.info(f"Created {len(cpds)} CPDs with expert priors")
    return cpds


def _get_default_priors() -> Dict[str, Dict]:
    """Get default expert priors for common legal variables."""
    return {
        'LegalSuccess_US': {'True': 0.3, 'False': 0.7},
        'LegalSuccess_HK': {'True': 0.25, 'False': 0.75},
        'FinancialDamage': {'Low': 0.4, 'Medium': 0.3, 'High': 0.2, 'VeryHigh': 0.1},
        'ReputationalDamage': {'Low': 0.3, 'Medium': 0.3, 'High': 0.25, 'VeryHigh': 0.15},
        'CriminalLiability': {'True': 0.1, 'False': 0.9},
    }


def _apply_expert_priors(
    node: SelectedNode,
    parents: List[str],
    priors: Dict,
    node_dict: Dict[str, SelectedNode],
) -> List[List[float]]:
    """Apply expert priors to CPD values.
    
    Simplified implementation - returns uniform if parents present.
    """
    num_states = len(node.states)
    
    if not parents:
        # Root node: use expert priors directly.
        values = []
        for state in node.states:
            prob = priors.get(state, 1.0 / num_states)
            values.append([prob])
        
        # Normalize.
        total = sum(v[0] for v in values)
        if total > 0:
            values = [[v[0] / total] for v in values]
        
        return values
    else:
        # With parents, use uniform (simplified).
        parent_cards = [len(node_dict[p].states) for p in parents]
        total_configs = 1
        for card in parent_cards:
            total_configs *= card
        return [[1.0 / num_states] * total_configs for _ in range(num_states)]


def build_bn_model(
    structure: BNStructure,
    use_expert_priors: bool = True,
    prior_knowledge: Optional[Dict[str, Dict]] = None,
) -> BayesianNetwork:
    """Build a pgmpy BayesianNetwork from structure.
    
    Args:
        structure: BN structure from build_bn_structure_from_kg
        use_expert_priors: Whether to use expert priors (vs uniform)
        prior_knowledge: Optional expert prior distributions
        
    Returns:
        Constructed and validated BayesianNetwork
        
    Raises:
        BNConstructionError: If construction or validation fails
    """
    _ensure_pgmpy()
    
    logger.info("Building Bayesian Network model...")
    logger.info(f"Nodes: {len(structure.nodes)}, Edges: {len(structure.edges)}")
    
    try:
        # Create network structure.
        model = BayesianNetwork(structure.edges)
        logger.info("Network structure created")
        
        # Initialize CPDs.
        if use_expert_priors:
            cpds = initialize_expert_priors(structure, prior_knowledge)
        else:
            cpds = initialize_uniform_cpds(structure)
        
        # Add CPDs to model.
        model.add_cpds(*cpds)
        logger.info("CPDs added to model")
        
        # Validate model.
        if not model.check_model():
            raise BNConstructionError("Model validation failed")
        
        logger.info("Model validation passed")
        
        return model
        
    except Exception as exc:
        raise BNConstructionError(f"Failed to build BN model: {exc}") from exc


def create_inference_engine(
    model: BayesianNetwork,
    elimination_order: Optional[List[str]] = None,
) -> VariableElimination:
    """Create Variable Elimination inference engine.
    
    Args:
        model: Bayesian Network model
        elimination_order: Optional custom elimination order for performance.
                          If None, pgmpy auto-selects using min-fill heuristic.
        
    Returns:
        VariableElimination inference engine
        
    Note:
        pgmpy automatically chooses elimination order using min-fill heuristic.
        For custom order, pass it to individual query() calls:
        inference.query(variables=[node], elimination_order=custom_order)
    """
    _ensure_pgmpy()
    
    logger.info("Creating Variable Elimination inference engine...")
    
    # pgmpy auto-chooses elimination order using min-fill heuristic by default.
    # Custom order can be provided at query time if needed.
    inference = VariableElimination(model)
    
    if elimination_order:
        logger.info(f"Custom elimination order provided: {elimination_order[:5]}...")
        logger.info("(Will be applied at query time)")
    else:
        logger.info("Using automatic elimination order (min-fill heuristic)")
    
    logger.info("Inference engine ready")
    
    return inference


def save_model(
    model: BayesianNetwork,
    output_path: Path,
) -> None:
    """Save BN model to file.
    
    Args:
        model: Bayesian Network to save
        output_path: Output file path (.pkl)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {output_path}")


def load_model(model_path: Path) -> BayesianNetwork:
    """Load BN model from file.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded BayesianNetwork
    """
    _ensure_pgmpy()
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {model_path}")
    return model


def main():
    """Command-line interface for BN construction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build BN model from structure"
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path("analysis_results/bn_structure/bn_structure.pkl"),
        help="Path to bn_structure.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results/bn_model_initial.pkl"),
        help="Output path for model",
    )
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="Use uniform priors instead of expert priors",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Load structure.
    logger.info(f"Loading structure from {args.structure}")
    with open(args.structure, 'rb') as f:
        structure = pickle.load(f)
    
    # Build model.
    model = build_bn_model(
        structure=structure,
        use_expert_priors=not args.uniform,
    )
    
    # Save model.
    save_model(model, args.output)
    
    # Also export to XDSL if possible.
    xdsl_path = args.output.with_suffix('.xdsl')
    try:
        from .xdsl_exporter import export_to_xdsl
        export_to_xdsl(model, structure, xdsl_path)
        logger.info(f"XDSL exported to {xdsl_path}")
    except ImportError:
        logger.warning("XDSL export not available (xdsl_exporter not found)")
    
    print(f"\n{'='*60}")
    print("BN Model Construction Complete")
    print(f"{'='*60}")
    print(f"Nodes: {len(model.nodes())}")
    print(f"Edges: {len(model.edges())}")
    print(f"Model saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

