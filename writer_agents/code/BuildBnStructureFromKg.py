"""Build Bayesian Network structure from Knowledge Graph.

This module selects and structures nodes from the full knowledge graph to create
a scalable, interpretable BN with <= 200 nodes and <= 3 parents each.

Usage:
    python -m writer_agents.build_bn_structure_from_kg --output output_dir/
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd

from writer_agents.code.bn_structural_constraints import (
    DEFAULT_CONSTRAINTS,
    apply_structural_constraints,
)

logger = logging.getLogger(__name__)


@dataclass
class NodeSelectionConfig:
    """Configuration for node selection from knowledge graph."""
    
    max_nodes: int = 150
    max_parents_per_node: int = 3
    max_states_per_variable: int = 4
    centrality_percentile: float = 0.25 # Top 25% centrality nodes.
    min_edge_weight: float = 0.1
    outcome_variables: List[str] = field(default_factory=lambda: [
        "LegalSuccess_US",
        "LegalSuccess_HK", 
        "FinancialDamage",
        "ReputationalDamage",
        "CriminalLiability"
    ])


@dataclass
class SelectedNode:
    """Information about a selected node for the BN."""
    
    node_id: str
    label: str
    states: List[str]
    importance_score: float
    centrality_metrics: Dict[str, float]
    has_evidence: bool = False


@dataclass
class BNStructure:
    """Complete BN structure ready for model creation."""
    
    nodes: List[SelectedNode]
    edges: List[Tuple[str, str]] # (parent, child)
    node_parents: Dict[str, List[str]]
    metadata: Dict[str, any]


def load_knowledge_graph(
    entities_path: Path,
    graph_path: Path,
) -> Tuple[List[Dict], nx.Graph]:
    """Load knowledge graph entities and co-occurrence graph.
    
    Args:
        entities_path: Path to entities_all.json
        graph_path: Path to cooccurrence_graph_pruned.gpickle
        
    Returns:
        Tuple of (entities list, networkx graph)
    """
    logger.info(f"Loading entities from {entities_path}")
    with open(entities_path, "r", encoding="utf-8") as f:
        entities = json.load(f)
    
    logger.info(f"Loading co-occurrence graph from {graph_path}")
    # Use pickle.load for NetworkX 3.0+ compatibility
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    logger.info(f"Loaded {len(entities)} entities and graph with {len(graph.nodes)} nodes")
    return entities, graph


def compute_centrality_metrics(graph: nx.Graph) -> pd.DataFrame:
    """Compute various centrality metrics for all nodes.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        DataFrame with centrality metrics for each node
    """
    logger.info("Computing centrality metrics...")
    
    metrics = {}
    
    # Degree centrality.
    metrics['degree'] = nx.degree_centrality(graph)
    
    # Betweenness centrality (bridge detection).
    try:
        metrics['betweenness'] = nx.betweenness_centrality(graph, k=min(100, len(graph.nodes)))
    except Exception as exc:
        logger.warning(f"Betweenness centrality failed: {exc}")
        metrics['betweenness'] = {node: 0.0 for node in graph.nodes}
    
    # PageRank (importance).
    try:
        metrics['pagerank'] = nx.pagerank(graph, max_iter=100)
    except Exception as exc:
        logger.warning(f"PageRank failed: {exc}")
        metrics['pagerank'] = {node: 0.0 for node in graph.nodes}
    
    # Eigenvector centrality.
    try:
        metrics['eigenvector'] = nx.eigenvector_centrality(graph, max_iter=100)
    except Exception as exc:
        logger.warning(f"Eigenvector centrality failed: {exc}")
        metrics['eigenvector'] = {node: 0.0 for node in graph.nodes}
    
    # Convert to DataFrame.
    df = pd.DataFrame(metrics).fillna(0)
    
    # Add composite importance score.
    df['importance'] = (
        df['degree'] * 0.3 +
        df['betweenness'] * 0.3 +
        df['pagerank'] * 0.2 +
        df['eigenvector'] * 0.2
    )
    
    logger.info("Centrality metrics computed")
    return df


def detect_communities(graph: nx.Graph) -> Dict[str, int]:
    """Detect communities in the graph.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Dictionary mapping node IDs to community IDs
    """
    logger.info("Detecting communities...")
    
    try:
        from networkx.algorithms import community
        communities = community.greedy_modularity_communities(graph)
        
        node_to_community = {}
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                node_to_community[node] = comm_id
        
        logger.info(f"Detected {len(communities)} communities")
        return node_to_community
        
    except Exception as exc:
        logger.warning(f"Community detection failed: {exc}")
        return {node: 0 for node in graph.nodes}


def select_nodes(
    entities: List[Dict],
    graph: nx.Graph,
    config: NodeSelectionConfig,
) -> List[SelectedNode]:
    """Select nodes from knowledge graph based on importance and structure.
    
    Args:
        entities: List of entity dictionaries
        graph: Co-occurrence graph
        config: Selection configuration
        
    Returns:
        List of selected nodes
    """
    logger.info("Selecting nodes from knowledge graph...")
    
    # Compute centrality metrics.
    centrality_df = compute_centrality_metrics(graph)
    
    # Detect communities.
    communities = detect_communities(graph)
    
    # Calculate thresholds for top centrality.
    thresholds = {
        metric: centrality_df[metric].quantile(1 - config.centrality_percentile)
        for metric in ['degree', 'betweenness', 'pagerank', 'eigenvector']
    }
    
    selected: List[SelectedNode] = []
    entity_map = {e['text']: e for e in entities}
    
    for node_id in graph.nodes:
        # Skip if not in entities list.
        if node_id not in entity_map:
            continue
        
        entity = entity_map[node_id]
        
        # Get centrality metrics for this node.
        if node_id not in centrality_df.index:
            continue
            
        node_metrics = centrality_df.loc[node_id]
        
        # Check selection criteria.
        should_select = False
        
        # Criterion 1: High centrality in any metric.
        for metric, threshold in thresholds.items():
            if node_metrics[metric] >= threshold:
                should_select = True
                break
        
        # Criterion 2: Outcome/decision variable.
        if node_id in config.outcome_variables:
            should_select = True
        
        # Criterion 3: Strong connections to outcome variables.
        for outcome in config.outcome_variables:
            if graph.has_edge(node_id, outcome):
                edge_data = graph.get_edge_data(node_id, outcome)
                weight = edge_data.get('weight', 0) if edge_data else 0
                if weight >= config.min_edge_weight:
                    should_select = True
                    break
        
        if should_select:
            # Create states for this node (simplified - needs domain knowledge).
            states = _generate_states_for_node(entity, config.max_states_per_variable)
            
            selected_node = SelectedNode(
                node_id=node_id,
                label=entity.get('label', 'UNKNOWN'),
                states=states,
                importance_score=float(node_metrics['importance']),
                centrality_metrics={
                    'degree': float(node_metrics['degree']),
                    'betweenness': float(node_metrics['betweenness']),
                    'pagerank': float(node_metrics['pagerank']),
                    'eigenvector': float(node_metrics['eigenvector']),
                },
                has_evidence=entity.get('count', 0) > 0,
            )
            selected.append(selected_node)
    
    # Sort by importance and limit to max_nodes.
    selected.sort(key=lambda x: x.importance_score, reverse=True)
    selected = selected[:config.max_nodes]
    
    logger.info(f"Selected {len(selected)} nodes")
    return selected


def _generate_states_for_node(entity: Dict, max_states: int) -> List[str]:
    """Generate discrete states for a node based on entity type.
    
    **ENFORCES**: <=4 states per node to prevent CPT explosion.
    
    Args:
        entity: Entity dictionary
        max_states: Maximum number of states allowed (hard limit: 4)
        
    Returns:
        List of state names (length <= max_states)
        
    Raises:
        ValueError: If generated states exceed max_states or max_states > 4
    """
    if max_states > 4:
        raise ValueError(f"max_states must be <=4, got {max_states}")
    
    label = entity.get('label', 'UNKNOWN')
    text = entity.get('text', '')
    
    # Generate states based on entity type.
    if label in ['PERSON', 'ORG', 'GPE']:
        states = ['Involved', 'NotInvolved']
    elif label in ['MONEY', 'QUANTITY']:
        states = ['Low', 'Medium', 'High', 'VeryHigh'][:max_states]
    elif label in ['DATE', 'TIME']:
        states = ['Past', 'Recent', 'Current'][:max_states]
    elif label in ['EVENT', 'LAW']:
        states = ['Occurred', 'NotOccurred']
    elif label == 'CARDINAL':
        states = ['Low', 'Medium', 'High'][:max_states]
    else:
        # Generic binary.
        states = ['True', 'False']
    
    # **HARD CHECK**: Enforce state limit.
    if len(states) > max_states:
        logger.warning(
            f"Generated {len(states)} states for '{text}' (label: {label}), "
            f"truncating to {max_states}"
        )
        states = states[:max_states]
    
    return states


def merge_redundant_nodes(selected: List[SelectedNode]) -> List[SelectedNode]:
    """Merge nodes that represent the same concept.
    
    Args:
        selected: List of selected nodes
        
    Returns:
        List with redundant nodes merged
    """
    logger.info("Checking for redundant nodes to merge...")
    
    # Simple heuristic: merge nodes with very similar IDs.
    # In production, use NLP similarity or domain ontology.
    
    merged_map: Dict[str, str] = {}
    representative_nodes: Dict[str, SelectedNode] = {}
    
    for node in selected:
        # Normalize node ID for comparison.
        normalized = node.node_id.lower().replace('_', '').replace('-', '')
        
        # Find if we have a similar node already.
        found_match = False
        for rep_id, rep_node in representative_nodes.items():
            rep_normalized = rep_id.lower().replace('_', '').replace('-', '')
            
            # Simple substring match.
            if normalized in rep_normalized or rep_normalized in normalized:
                merged_map[node.node_id] = rep_node.node_id
                # Update representative's importance if current is higher.
                if node.importance_score > rep_node.importance_score:
                    representative_nodes[rep_id] = node
                found_match = True
                break
        
        if not found_match:
            representative_nodes[node.node_id] = node
    
    result = list(representative_nodes.values())
    
    if len(result) < len(selected):
        logger.info(f"Merged {len(selected) - len(result)} redundant nodes")
    
    return result


def build_bn_edges(
    selected: List[SelectedNode],
    graph: nx.Graph,
    config: NodeSelectionConfig,
) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]]]:
    """Build parent-child relationships for the BN.
    
    Args:
        selected: Selected nodes
        graph: Original co-occurrence graph
        config: Selection configuration
        
    Returns:
        Tuple of (edge list, node_parents dict)
    """
    logger.info("Building BN edges...")
    
    node_ids = {node.node_id for node in selected}
    edges: List[Tuple[str, str]] = []
    node_parents: Dict[str, List[str]] = {node.node_id: [] for node in selected}
    
    # For each node, find its strongest connections as parents.
    for node in selected:
        if node.node_id not in graph:
            continue
        
        # Get all neighbors with weights.
        neighbors = []
        for neighbor in graph.neighbors(node.node_id):
            if neighbor in node_ids and neighbor != node.node_id:
                edge_data = graph.get_edge_data(node.node_id, neighbor)
                weight = edge_data.get('weight', 0) if edge_data else 0
                neighbors.append((neighbor, weight))
        
        # Sort by weight and take top N as parents.
        neighbors.sort(key=lambda x: x[1], reverse=True)
        parents = [n[0] for n in neighbors[:config.max_parents_per_node]]
        
        for parent in parents:
            edges.append((parent, node.node_id))
            node_parents[node.node_id].append(parent)
    
    logger.info(f"Created {len(edges)} edges")
    return edges, node_parents


def ensure_acyclic(
    nodes: List[SelectedNode],
    edges: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Ensure the BN structure is acyclic (DAG).
    
    Args:
        nodes: List of nodes
        edges: List of edges
        
    Returns:
        Filtered edge list that forms a DAG
    """
    logger.info("Ensuring acyclic structure...")
    
    # Build graph and detect cycles.
    G = nx.DiGraph()
    G.add_nodes_from([n.node_id for n in nodes])
    G.add_edges_from(edges)
    
    # Remove edges to break cycles.
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
            # Remove the last edge in the cycle.
            edge_to_remove = cycle[-1][:2]
            G.remove_edge(*edge_to_remove)
            logger.info(f"Removed edge {edge_to_remove} to break cycle")
        except nx.NetworkXNoCycle:
            break
    
    result_edges = list(G.edges())
    logger.info(f"Resulting DAG has {len(result_edges)} edges")
    return result_edges


def build_structure_from_kg(
    entities_path: Path,
    graph_path: Path,
    config: NodeSelectionConfig | None = None,
) -> BNStructure:
    """Main function to build BN structure from knowledge graph.
    
    Args:
        entities_path: Path to entities_all.json
        graph_path: Path to cooccurrence graph
        config: Selection configuration (uses defaults if None)
        
    Returns:
        Complete BN structure
    """
    if config is None:
        config = NodeSelectionConfig()
    
    logger.info("Starting BN structure construction from knowledge graph")
    
    # Load data.
    entities, graph = load_knowledge_graph(entities_path, graph_path)
    
    # Select nodes.
    selected = select_nodes(entities, graph, config)
    
    # Merge redundant nodes.
    selected = merge_redundant_nodes(selected)
    
    # Build edges.
    edges, node_parents = build_bn_edges(selected, graph, config)

    # Enforce structural constraints before cycle checks.
    edges = apply_structural_constraints(
        selected,
        edges,
        config=DEFAULT_CONSTRAINTS,
        max_parents=config.max_parents_per_node,
    )
    
    # Ensure DAG structure.
    edges = ensure_acyclic(selected, edges)
    
    # Rebuild node_parents after cycle removal.
    node_parents = {node.node_id: [] for node in selected}
    for parent, child in edges:
        node_parents[child].append(parent)
    
    # Verify constraints.
    avg_parents = sum(len(parents) for parents in node_parents.values()) / len(node_parents)
    logger.info(f"Average parents per node: {avg_parents:.2f}")
    
    structure = BNStructure(
        nodes=selected,
        edges=edges,
        node_parents=node_parents,
        metadata={
            'num_nodes': len(selected),
            'num_edges': len(edges),
            'avg_parents': avg_parents,
            'max_parents': max(len(p) for p in node_parents.values()) if node_parents else 0,
            'config': {
                'max_nodes': config.max_nodes,
                'max_parents_per_node': config.max_parents_per_node,
                'max_states_per_variable': config.max_states_per_variable,
            }
        }
    )
    
    logger.info("BN structure construction complete")
    return structure


def save_structure(structure: BNStructure, output_dir: Path) -> None:
    """Save BN structure to files.
    
    Args:
        structure: BN structure to save
        output_dir: Directory to save files to
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save selected nodes as JSON.
    nodes_data = [
        {
            'node_id': node.node_id,
            'label': node.label,
            'states': node.states,
            'importance_score': node.importance_score,
            'centrality_metrics': node.centrality_metrics,
            'has_evidence': node.has_evidence,
        }
        for node in structure.nodes
    ]
    
    nodes_path = output_dir / 'selected_nodes.json'
    with open(nodes_path, 'w', encoding='utf-8') as f:
        json.dump(nodes_data, f, indent=2)
    logger.info(f"Saved nodes to {nodes_path}")
    
    # Save edges as JSON.
    edges_path = output_dir / 'edges.json'
    with open(edges_path, 'w', encoding='utf-8') as f:
        json.dump({
            'edges': structure.edges,
            'node_parents': structure.node_parents,
        }, f, indent=2)
    logger.info(f"Saved edges to {edges_path}")
    
    # Save complete structure as pickle.
    structure_path = output_dir / 'bn_structure.pkl'
    with open(structure_path, 'wb') as f:
        pickle.dump(structure, f)
    logger.info(f"Saved structure to {structure_path}")
    
    # Save metadata.
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(structure.metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build BN structure from knowledge graph"
    )
    parser.add_argument(
        "--entities",
        type=Path,
        default=Path("complete_analysis_fast/entities_all.json"),
        help="Path to entities_all.json",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("analysis_results/cooccurrence_graph_pruned.gpickle"),
        help="Path to co-occurrence graph",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results/bn_structure"),
        help="Output directory for BN structure files",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=150,
        help="Maximum number of nodes",
    )
    parser.add_argument(
        "--max-parents",
        type=int,
        default=3,
        help="Maximum parents per node",
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
    
    # Build structure.
    config = NodeSelectionConfig(
        max_nodes=args.max_nodes,
        max_parents_per_node=args.max_parents,
    )
    
    structure = build_structure_from_kg(
        entities_path=args.entities,
        graph_path=args.graph,
        config=config,
    )
    
    # Save results.
    save_structure(structure, args.output)
    
    print(f"\n{'='*60}")
    print("BN Structure Construction Complete")
    print(f"{'='*60}")
    print(f"Nodes: {structure.metadata['num_nodes']}")
    print(f"Edges: {structure.metadata['num_edges']}")
    print(f"Average parents: {structure.metadata['avg_parents']:.2f}")
    print(f"Max parents: {structure.metadata['max_parents']}")
    print(f"\nOutput saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
