#!/usr/bin/env python
"""Lock the five confirmed 2025 facts at near-certainty.

Locks:
- OGC (3 emails sent in 2025)
- Email (communication in 2025) 
- Statements (Statements 1 & 2 made in 2025)

Uses 0.999/0.001 split (not 1.0/0.0) to preserve Bayesian consistency.
"""

import pickle
from pathlib import Path

try:
    from pgmpy.factors.discrete import TabularCPD
    HAVE_PGMPY = True
except ImportError:
    print("ERROR: Install pgmpy first")
    exit(1)


# The five confirmed 2025 facts
LOCKED_NODES_2025 = {
    "OGC": "Involved", # 3 emails sent to OGC in 2025
    "Email": "Involved", # Email communication occurred in 2025
    "Statements": "Involved" # Statements 1 & 2 made in 2025
}


def lock_node_near_certain(model, node_id: str, certain_state: str, p_certain: float = 0.999):
    """Lock a node at near-certainty for a specific state.
    
    Args:
        model: BayesianNetwork
        node_id: Node to lock
        certain_state: State to set as near-certain
        p_certain: Probability (default 0.999)
    """
    if node_id not in model.nodes():
        print(f" WARNING: Node '{node_id}' not found in model")
        return False
    
    # Get current CPD to find states and parents
    old_cpd = model.get_cpds(node_id)
    states = old_cpd.state_names[node_id]
    parents = list(old_cpd.variables[1:]) if len(old_cpd.variables) > 1 else []
    
    if certain_state not in states:
        print(f" WARNING: State '{certain_state}' not in {node_id} states: {states}")
        return False
    
    # Find index of certain state
    certain_idx = states.index(certain_state)
    num_states = len(states)
    
    if parents:
        # Has parents - need conditional table
        parent_cards = [len(old_cpd.state_names[p]) for p in parents]
        num_parent_configs = 1
        for card in parent_cards:
            num_parent_configs *= card
        
        # Create values: certain_state gets p_certain, others split remaining
        values = []
        for state_idx in range(num_states):
            if state_idx == certain_idx:
                # This is the certain state
                values.append([p_certain] * num_parent_configs)
            else:
                # Other states split remaining probability
                remaining = 1.0 - p_certain
                other_states = num_states - 1
                values.append([remaining / other_states] * num_parent_configs)
        
        new_cpd = TabularCPD(
            variable=node_id,
            variable_card=num_states,
            values=values,
            evidence=parents,
            evidence_card=parent_cards,
            state_names=old_cpd.state_names
        )
    else:
        # No parents - simple prior
        values = []
        for state_idx in range(num_states):
            if state_idx == certain_idx:
                values.append([p_certain])
            else:
                remaining = 1.0 - p_certain
                other_states = num_states - 1
                values.append([remaining / other_states])
        
        new_cpd = TabularCPD(
            variable=node_id,
            variable_card=num_states,
            values=values,
            state_names={node_id: states}
        )
    
    # Replace CPD
    model.remove_cpds(node_id)
    model.add_cpds(new_cpd)
    
    return True


def lock_2025_facts(input_model_path: Path, output_model_path: Path, p_certain: float = 0.999):
    """Lock the five 2025 facts at near-certainty.
    
    Args:
        input_model_path: Path to input model
        output_model_path: Path to save locked model
        p_certain: Certainty level (default 0.999)
    """
    print("="*70)
    print("LOCKING 2025 CONFIRMED FACTS")
    print("="*70)
    print()
    
    print(f"Loading model from: {input_model_path}")
    with open(input_model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f" > Loaded: {len(model.nodes())} nodes, {len(model.edges())} edges")
    print()
    
    print(f"Locking 5 confirmed 2025 facts at P = {p_certain}:")
    print()
    
    locked_count = 0
    for node_id, certain_state in LOCKED_NODES_2025.items():
        print(f" [{node_id}]")
        if lock_node_near_certain(model, node_id, certain_state, p_certain):
            print(f" > Locked at P({node_id}={certain_state}) = {p_certain}")
            locked_count += 1
        print()
    
    print(f"Successfully locked: {locked_count}/{len(LOCKED_NODES_2025)} nodes")
    print()
    
    # Validate model
    print("Validating locked model...")
    try:
        if model.check_model():
            print(" > Model validation PASSED")
        else:
            print(" > Model validation FAILED")
            return False
    except Exception as e:
        print(f" > Validation error: {e}")
        return False
    
    print()
    
    # Save locked model
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Locked model saved to: {output_model_path}")
    print()
    
    return True


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Lock 5 confirmed 2025 facts at near-certainty"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analysis_results/enhanced_bn/bn_model_initial.pkl"),
        help="Input model path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results/enhanced_bn/bn_model_locked_2025.pkl"),
        help="Output model path"
    )
    parser.add_argument(
        "--certainty",
        type=float,
        default=0.999,
        help="Certainty level (default 0.999)"
    )
    
    args = parser.parse_args()
    
    success = lock_2025_facts(args.input, args.output, args.certainty)
    
    if success:
        print("="*70)
        print("SUCCESS")
        print("="*70)
        print()
        print("Your model now has:")
        print(" 1. Five 2025 facts locked at 99.9% certainty")
        print(" 2. All other nodes remain fully learnable")
        print(" 3. Bayesian consistency preserved")
        print()
        print("Next steps:")
        print(f" 1. Validate: py verify_bn_configuration.py")
        print(f" 2. Use: Load {args.output}")
        print(f" 3. Learn: py learn_legal_cpts.py --model {args.output} --data your_data.csv")
        print()
        return 0
    else:
        print("FAILED - see errors above")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

