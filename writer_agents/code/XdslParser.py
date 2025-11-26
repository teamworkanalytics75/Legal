"""XDSL file parser for Bayesian network models.

This module parses GeNIe/SMILE .xdsl files and extracts the network structure,
including nodes, states, parent-child relationships, and probability tables.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class NodeInfo:
    """Information about a single node in the Bayesian network."""
    
    node_id: str
    states: Sequence[str]
    parents: Sequence[str] = field(default_factory=tuple)
    probabilities: Sequence[float] = field(default_factory=tuple)


@dataclass(frozen=True)
class XDSLNetwork:
    """Complete representation of a parsed XDSL Bayesian network."""
    
    network_id: str
    nodes: Dict[str, NodeInfo]
    edges: List[tuple[str, str]] # (parent, child) pairs


class XDSLParseError(RuntimeError):
    """Raised when XDSL file cannot be parsed."""


def parse_xdsl(file_path: Path) -> XDSLNetwork:
    """Parse a .xdsl file and return the network structure.
    
    Args:
        file_path: Path to the .xdsl file
        
    Returns:
        XDSLNetwork containing all nodes, edges, and probability tables
        
    Raises:
        XDSLParseError: If the file cannot be parsed or is malformed
    """
    if not file_path.exists():
        raise XDSLParseError(f"XDSL file not found: {file_path}")
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        raise XDSLParseError(f"XML parsing failed: {exc}") from exc
    
    # Extract network ID.
    network_id = root.get("id", "unknown_network")
    
    # Find the nodes container.
    nodes_elem = root.find("nodes")
    if nodes_elem is None:
        raise XDSLParseError("No <nodes> element found in XDSL file")
    
    # Extract all nodes and their CPTs.
    nodes_dict: Dict[str, NodeInfo] = {}
    edges: List[tuple[str, str]] = []
    
    for cpt_elem in nodes_elem.findall("cpt"):
        node_info = _extract_node(cpt_elem)
        nodes_dict[node_info.node_id] = node_info
        
        # Build edges from parent relationships.
        for parent_id in node_info.parents:
            edges.append((parent_id, node_info.node_id))
    
    if not nodes_dict:
        raise XDSLParseError("No nodes found in XDSL file")
    
    return XDSLNetwork(
        network_id=network_id,
        nodes=nodes_dict,
        edges=edges,
    )


def _extract_node(cpt_elem: ET.Element) -> NodeInfo:
    """Extract node information from a <cpt> element.
    
    Args:
        cpt_elem: XML element representing a CPT node
        
    Returns:
        NodeInfo with all node data
        
    Raises:
        XDSLParseError: If node is malformed
    """
    node_id = cpt_elem.get("id")
    if not node_id:
        raise XDSLParseError("CPT node missing 'id' attribute")
    
    # Extract states.
    state_elems = cpt_elem.findall("state")
    if not state_elems:
        raise XDSLParseError(f"Node '{node_id}' has no states")
    
    states = []
    for state_elem in state_elems:
        state_id = state_elem.get("id")
        if not state_id:
            raise XDSLParseError(f"Node '{node_id}' has state without 'id' attribute")
        states.append(state_id)
    
    # Extract parents (if any).
    parents_elem = cpt_elem.find("parents")
    parents: List[str] = []
    if parents_elem is not None and parents_elem.text:
        # Parents are space-separated in the text content.
        parents = parents_elem.text.strip().split()
    
    # Extract probabilities.
    probs_elem = cpt_elem.find("probabilities")
    if probs_elem is None or not probs_elem.text:
        raise XDSLParseError(f"Node '{node_id}' missing probabilities")
    
    try:
        # Probabilities are space-separated floats.
        prob_strings = probs_elem.text.strip().split()
        probabilities = [float(p) for p in prob_strings]
    except ValueError as exc:
        raise XDSLParseError(f"Node '{node_id}' has invalid probability values") from exc
    
    # Validate probability table size.
    expected_size = len(states)
    for parent in parents:
        # We'll validate this later when we have all nodes loaded.
        # For now, just accept any size.
        pass
    
    if len(probabilities) == 0:
        raise XDSLParseError(f"Node '{node_id}' has empty probability table")
    
    return NodeInfo(
        node_id=node_id,
        states=tuple(states),
        parents=tuple(parents),
        probabilities=tuple(probabilities),
    )


def get_node_cardinality(node_info: NodeInfo) -> int:
    """Return the number of states for a node."""
    return len(node_info.states)


def validate_network(network: XDSLNetwork) -> None:
    """Validate that the network structure is consistent.
    
    Args:
        network: The parsed network to validate
        
    Raises:
        XDSLParseError: If validation fails
    """
    # Check that all parent references are valid.
    for node_id, node_info in network.nodes.items():
        for parent_id in node_info.parents:
            if parent_id not in network.nodes:
                raise XDSLParseError(
                    f"Node '{node_id}' references unknown parent '{parent_id}'"
                )
    
    # Check that probability tables have correct size.
    for node_id, node_info in network.nodes.items():
        num_states = len(node_info.states)
        
        if not node_info.parents:
            # Root node: should have exactly num_states probabilities.
            if len(node_info.probabilities) != num_states:
                raise XDSLParseError(
                    f"Root node '{node_id}' has {len(node_info.probabilities)} "
                    f"probabilities but {num_states} states"
                )
        else:
            # Child node: probability table size depends on parent cardinalities.
            parent_sizes = [len(network.nodes[p].states) for p in node_info.parents]
            total_parent_configs = 1
            for size in parent_sizes:
                total_parent_configs *= size
            
            expected_size = num_states * total_parent_configs
            if len(node_info.probabilities) != expected_size:
                raise XDSLParseError(
                    f"Node '{node_id}' has {len(node_info.probabilities)} "
                    f"probabilities but expected {expected_size} "
                    f"(states={num_states}, parent configs={total_parent_configs})"
                )


__all__ = [
    "XDSLNetwork",
    "NodeInfo",
    "XDSLParseError",
    "parse_xdsl",
    "validate_network",
    "get_node_cardinality",
]

