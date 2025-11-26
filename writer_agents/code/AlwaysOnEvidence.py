"""Always-on evidence helper for case-specific inference.

This module provides functionality to add certain facts (OGC emails, Statements 1 & 2, etc.)
automatically whenever inference is run, without modifying the network's CPTs.

This keeps the model reusable for other cases and for parameter-learning while
locking specific evidence for the current case.
"""

from typing import Dict, Optional


def get_always_on_evidence(case_id: Optional[str] = None) -> Dict[str, str]:
    """Returns the evidence dictionary that should always be applied for the current case.
    
    These are facts known to be true for your specific case:
    - OGC received 3 emails (definitely aware)
    - Statements 1 and 2 definitely happened
    - Email communication occurred
    
    Args:
        case_id: Optional case identifier for different always-on configs
        
    Returns:
        Dictionary of node_id -> state mappings to always include as evidence
    """
    # Default always-on evidence for your current case
    return {
        "OGC": "Involved", # You sent 3 emails to OGC
        "Email": "Involved", # Email communication occurred
        "Statements": "Involved", # Statements 1 & 2 definitely happened
    }


def merge_case_evidence(
    user_evidence: Dict[str, str],
    case_id: Optional[str] = None,
    allow_override: bool = False
) -> Dict[str, str]:
    """Combine user-supplied evidence with always-on evidence.
    
    Args:
        user_evidence: Evidence provided by the user
        case_id: Optional case identifier for different always-on configs
        allow_override: If False (default), always-on evidence takes precedence.
                       If True, user evidence can override always-on.
    
    Returns:
        Merged evidence dictionary with always-on facts included
        
    Example:
        >>> user_ev = {"Harvard": "True", "Court": "Involved"}
        >>> merged = merge_case_evidence(user_ev)
        >>> print(merged)
        {'Harvard': 'True', 'Court': 'Involved', 'OGC': 'Involved', 
         'Email': 'Involved', 'Statements': 'Involved'}
    """
    always_on = get_always_on_evidence(case_id)
    
    if allow_override:
        # User evidence can override always-on
        merged = dict(always_on)
        merged.update(user_evidence)
    else:
        # Always-on evidence takes precedence
        merged = dict(user_evidence)
        merged.update(always_on)
    
    return merged


def get_always_on_nodes(case_id: Optional[str] = None) -> list:
    """Get list of node IDs that are always-on for this case.
    
    Useful for displaying which facts are locked in.
    
    Args:
        case_id: Optional case identifier
        
    Returns:
        List of node IDs that are always set as evidence
    """
    return list(get_always_on_evidence(case_id).keys())


def is_always_on_node(node_id: str, case_id: Optional[str] = None) -> bool:
    """Check if a node is configured as always-on evidence.
    
    Args:
        node_id: Node identifier to check
        case_id: Optional case identifier
        
    Returns:
        True if this node is in the always-on evidence list
    """
    return node_id in get_always_on_evidence(case_id)


def get_always_on_summary(case_id: Optional[str] = None) -> str:
    """Get human-readable summary of always-on evidence.
    
    Args:
        case_id: Optional case identifier
        
    Returns:
        Formatted string describing the always-on evidence
    """
    evidence = get_always_on_evidence(case_id)
    
    lines = ["Always-On Evidence (Case-Locked Facts):"]
    lines.append("-" * 50)
    for node, state in evidence.items():
        lines.append(f" {node} = {state}")
    lines.append("")
    lines.append("These facts are automatically included in all queries")
    lines.append("for this case without modifying the model's CPTs.")
    
    return "\n".join(lines)


# For backwards compatibility and convenience
def apply_case_facts(user_evidence: Dict[str, str]) -> Dict[str, str]:
    """Convenience alias for merge_case_evidence.
    
    Args:
        user_evidence: Evidence provided by the user
        
    Returns:
        Merged evidence with case facts included
    """
    return merge_case_evidence(user_evidence)

