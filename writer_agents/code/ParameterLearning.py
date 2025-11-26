"""Automatic parameter learning for Bayesian Networks.

This module implements CPT learning from data using MLE, Bayesian estimation,
and Expectation-Maximization for incomplete data.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from pgmpy.estimators import (
        BayesianEstimator,
        ExpectationMaximization,
        MaximumLikelihoodEstimator,
    )
    from pgmpy.factors.discrete import TabularCPD
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork
    
    HAVE_PGMPY = True
except ImportError:
    HAVE_PGMPY = False
    BayesianNetwork = None # type: ignore
    BayesianEstimator = None # type: ignore
    MaximumLikelihoodEstimator = None # type: ignore
    ExpectationMaximization = None # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """Configuration for parameter learning."""
    
    method: str = "bayesian" # "mle", "bayesian", or "em"
    prior_type: str = "BDeu" # For Bayesian estimator
    equivalent_sample_size: int = 5 # For BDeu prior
    max_em_iterations: int = 100 # For EM
    em_tolerance: float = 1e-4 # For EM convergence
    min_data_size: int = 10 # Minimum rows required


class ParameterLearningError(RuntimeError):
    """Raised when parameter learning fails."""


def _ensure_pgmpy() -> None:
    """Check that pgmpy is available."""
    if not HAVE_PGMPY:
        raise ParameterLearningError(
            "pgmpy is required. Install with: pip install pgmpy networkx"
        )


def extract_evidence_from_corpus(
    corpus_path: Path,
    model: BayesianNetwork,
) -> pd.DataFrame:
    """Extract structured evidence data from corpus.
    
    Args:
        corpus_path: Path to JSON corpus or SQL database
        model: BN model (for node list)
        
    Returns:
        DataFrame with columns for each BN variable
    """
    logger.info(f"Extracting evidence from {corpus_path}")
    
    # Get all node IDs from model.
    all_nodes = list(model.nodes())
    
    # Load corpus data.
    if corpus_path.suffix == '.json':
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
    elif corpus_path.suffix in ['.db', '.sqlite', '.sqlite3']:
        corpus_data = _extract_from_database(corpus_path)
    else:
        raise ParameterLearningError(f"Unsupported corpus format: {corpus_path.suffix}")
    
    # Convert to DataFrame structure.
    # Each row = one case/document, each column = one BN variable.
    records = []
    
    if isinstance(corpus_data, list):
        for item in corpus_data:
            record = {}
            for node in all_nodes:
                # Extract value for this node from item.
                value = _extract_node_value(node, item)
                record[node] = value
            records.append(record)
    
    df = pd.DataFrame(records)
    
    logger.info(f"Extracted {len(df)} records with {len(df.columns)} variables")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    
    return df


def _extract_from_database(db_path: Path) -> List[Dict]:
    """Extract data from SQL database."""
    import sqlite3
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Try to find evidence or cases table.
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    data = []
    
    # Look for relevant tables.
    for table_name in ['cases', 'evidence', 'documents']:
        if table_name in tables:
            cursor.execute(f"SELECT * FROM {table_name}")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            for row in rows:
                item = dict(zip(columns, row))
                data.append(item)
    
    conn.close()
    
    logger.info(f"Extracted {len(data)} records from database")
    return data


def _extract_node_value(node_id: str, item: Dict) -> Optional[str]:
    """Extract value for a specific node from a data item.
    
    Args:
        node_id: Node identifier
        item: Data item (dict)
        
    Returns:
        Value for the node, or None if not present
    """
    # Direct match.
    if node_id in item:
        return str(item[node_id])
    
    # Try lowercase/normalized versions.
    normalized = node_id.lower().replace('_', '').replace('-', '')
    for key, value in item.items():
        key_normalized = key.lower().replace('_', '').replace('-', '')
        if key_normalized == normalized:
            return str(value)
    
    # Not found.
    return None


def preprocess_training_data(
    data: pd.DataFrame,
    model: BayesianNetwork,
    max_states: int = 4,
) -> pd.DataFrame:
    """Preprocess training data before parameter learning.
    
    Ensures data hygiene:
    - Filters to only model variables
    - Converts numeric to discrete/categorical
    - Enforces state-space limits (<=4 states)
    - Validates state names against model CPDs
    
    Args:
        data: Raw training data
        model: BN model
        max_states: Maximum states per variable (default: 4)
        
    Returns:
        Preprocessed DataFrame ready for learning
        
    Raises:
        ParameterLearningError: If data validation fails
    """
    logger.info("Preprocessing training data...")
    
    # 1. Filter to only model variables.
    model_vars = list(model.nodes())
    available_vars = [v for v in model_vars if v in data.columns]
    
    logger.info(f"Model has {len(model_vars)} variables")
    logger.info(f"Data has {len(available_vars)} matching variables")
    
    if len(available_vars) < len(model_vars) * 0.5:
        logger.warning(
            f"Only {len(available_vars)}/{len(model_vars)} variables present in data"
        )
    
    # Create filtered dataframe with only model variables.
    df = pd.DataFrame()
    for var in model_vars:
        if var in data.columns:
            df[var] = data[var]
        else:
            # Add missing columns with NaN.
            df[var] = pd.NA
            logger.debug(f"Variable '{var}' not in data, added as NaN")
    
    # 2. Convert numeric columns to categorical (discrete).
    for col in df.columns:
        if df[col].dtype.kind in ('i', 'f'): # integer or float
            logger.warning(
                f"Variable '{col}' is numeric ({df[col].dtype}), converting to categorical"
            )
            # Convert to nullable Int64, then to object.
            df[col] = df[col].astype("Int64").astype("object")
    
    # 3. Enforce state-space limits (<=4 states).
    for col in df.columns:
        unique_states = df[col].dropna().unique()
        if len(unique_states) > max_states:
            raise ParameterLearningError(
                f"Variable '{col}' has {len(unique_states)} states "
                f"(maximum {max_states} allowed). "
                f"States: {list(unique_states)[:10]}... "
                f"Consider discretizing or merging categories."
            )
    
    # 4. Validate state names match model CPDs (if available).
    for var in df.columns:
        try:
            cpd = model.get_cpds(var)
            if cpd is not None and var in cpd.state_names:
                expected_states = set(cpd.state_names[var])
                actual_states = set(df[var].dropna().unique())
                
                # Unseen states in data.
                unseen = actual_states - expected_states
                if unseen:
                    logger.warning(
                        f"Variable '{var}' has unseen states in data: {unseen}. "
                        f"Expected: {expected_states}"
                    )
                
                # Missing states (OK, just log).
                missing = expected_states - actual_states
                if missing:
                    logger.debug(
                        f"Variable '{var}' missing states in data: {missing}"
                    )
        except Exception:
            # CPD might not exist yet for new models.
            pass
    
    logger.info(f"Preprocessed data: {len(df)} rows {len(df.columns)} variables")
    logger.info(f"Missing values: {df.isnull().sum().sum()} total")
    
    return df


def learn_parameters_mle(
    model: BayesianNetwork,
    data: pd.DataFrame,
) -> BayesianNetwork:
    """Learn parameters using Maximum Likelihood Estimation.
    
    Args:
        model: BN model with structure but initial CPTs
        data: Training data (already preprocessed)
        
    Returns:
        Model with learned CPTs
    """
    _ensure_pgmpy()
    
    logger.info("Learning parameters with MLE...")
    
    try:
        model.fit(data, estimator=MaximumLikelihoodEstimator)
        logger.info("MLE learning complete")
        
        return model
        
    except Exception as exc:
        raise ParameterLearningError(f"MLE learning failed: {exc}") from exc


def learn_parameters_bayesian(
    model: BayesianNetwork,
    data: pd.DataFrame,
    prior_type: str = "BDeu",
    equivalent_sample_size: int = 5,
) -> BayesianNetwork:
    """Learn parameters using Bayesian estimation.
    
    This method is recommended for most use cases as it:
    - Incorporates expert priors
    - Handles moderate missingness well
    - Provides regularization against overfitting
    
    Args:
        model: BN model with structure
        data: Training data (already preprocessed)
        prior_type: Type of prior ("BDeu", "K2", "dirichlet")
        equivalent_sample_size: Strength of prior (higher = more conservative)
        
    Returns:
        Model with learned CPTs
    """
    _ensure_pgmpy()
    
    logger.info(f"Learning parameters with Bayesian estimation ({prior_type})...")
    logger.info(f"Equivalent sample size: {equivalent_sample_size}")
    
    try:
        model.fit(
            data,
            estimator=BayesianEstimator,
            prior_type=prior_type,
            equivalent_sample_size=equivalent_sample_size,
        )
        logger.info("Bayesian learning complete")
        
        return model
        
    except Exception as exc:
        raise ParameterLearningError(f"Bayesian learning failed: {exc}") from exc


def learn_parameters_em(
    model: BayesianNetwork,
    data: pd.DataFrame,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> BayesianNetwork:
    """Learn parameters using Expectation-Maximization (for incomplete data).
    
    WARNING Warning: Use EM only if you have significant missing data (>20%).
    For moderate missingness, Bayesian estimation is more robust and stable.
    
    Args:
        model: BN model with structure
        data: Training data (may have missing values)
        max_iterations: Maximum EM iterations
        tolerance: Convergence tolerance
        
    Returns:
        Model with learned CPTs
    """
    _ensure_pgmpy()
    
    # Check missingness percentage.
    missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
    logger.info(f"Learning parameters with EM (missing data: {missing_pct:.1%})...")
    
    if missing_pct < 0.2:
        logger.warning(
            f"Data has only {missing_pct:.1%} missing values. "
            "Consider using Bayesian estimation instead for better stability."
        )
    
    try:
        em = ExpectationMaximization(model, data)
        learned_cpds = em.get_parameters(
            max_iter=max_iterations,
            atol=tolerance,
        )
        
        # EM returns a dict of CPDs - need to properly update the model.
        if isinstance(learned_cpds, dict):
            # Replace CPDs in model.
            new_cpds = []
            for var in model.nodes():
                if var in learned_cpds:
                    new_cpds.append(learned_cpds[var])
                else:
                    # Keep original CPD if EM didn't update it.
                    orig_cpd = model.get_cpds(var)
                    if orig_cpd:
                        new_cpds.append(orig_cpd)
            
            model.cpds = new_cpds
        else:
            # Fallback: assume it's a list.
            model.cpds = list(learned_cpds.values()) if hasattr(learned_cpds, 'values') else learned_cpds
        
        logger.info("EM learning complete")
        
        # Validate after EM.
        if not model.check_model():
            raise ParameterLearningError("Model validation failed after EM")
        
        return model
        
    except Exception as exc:
        raise ParameterLearningError(f"EM learning failed: {exc}") from exc


def learn_parameters(
    model: BayesianNetwork,
    data: pd.DataFrame,
    config: Optional[LearningConfig] = None,
) -> BayesianNetwork:
    """Main function for parameter learning with automatic preprocessing.
    
    This function:
    1. Validates data size
    2. Preprocesses data (discrete categories, state limits, etc.)
    3. Learns parameters using specified method
    4. Validates learned model
    
    Args:
        model: BN model with structure
        data: Training data DataFrame (will be preprocessed automatically)
        config: Learning configuration
        
    Returns:
        Model with learned CPTs
        
    Raises:
        ParameterLearningError: If learning fails
    """
    _ensure_pgmpy()
    
    if config is None:
        config = LearningConfig()
    
    # Validate data size.
    if len(data) < config.min_data_size:
        raise ParameterLearningError(
            f"Insufficient data: {len(data)} rows (minimum {config.min_data_size})"
        )
    
    logger.info(f"Learning parameters from {len(data)} records")
    logger.info(f"Method: {config.method}")
    
    # Preprocess data (ensures discrete categories, state limits, etc.).
    data = preprocess_training_data(data, model, max_states=4)
    
    # Choose learning method.
    if config.method == "mle":
        learned_model = learn_parameters_mle(model, data)
    elif config.method == "bayesian":
        learned_model = learn_parameters_bayesian(
            model,
            data,
            prior_type=config.prior_type,
            equivalent_sample_size=config.equivalent_sample_size,
        )
    elif config.method == "em":
        # Check if EM is really needed.
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct < 0.2:
            logger.warning(
                f"Using EM with only {missing_pct:.1%} missing data. "
                "Consider --method bayesian for better results."
            )
        
        learned_model = learn_parameters_em(
            model,
            data,
            max_iterations=config.max_em_iterations,
            tolerance=config.em_tolerance,
        )
    else:
        raise ParameterLearningError(f"Unknown method: {config.method}")
    
    # Validate learned model.
    try:
        if not learned_model.check_model():
            raise ParameterLearningError("Learned model validation failed")
    except Exception as exc:
        raise ParameterLearningError(f"Learned model invalid: {exc}") from exc
    
    logger.info("Parameter learning successful")
    return learned_model


def evaluate_learned_parameters(
    original_model: BayesianNetwork,
    learned_model: BayesianNetwork,
    threshold: float = 0.3,
) -> Dict[str, any]:
    """Evaluate quality of learned parameters.
    
    Args:
        original_model: Model with expert/prior CPTs
        learned_model: Model with learned CPTs
        threshold: Threshold for flagging large changes
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating learned parameters...")
    
    evaluation = {
        'nodes_evaluated': 0,
        'large_changes': [],
        'avg_change': 0.0,
        'max_change': 0.0,
    }
    
    total_change = 0.0
    max_change = 0.0
    
    for node in learned_model.nodes():
        try:
            orig_cpd = original_model.get_cpds(node)
            learned_cpd = learned_model.get_cpds(node)
            
            # Compare CPT values.
            orig_values = orig_cpd.values.flatten()
            learned_values = learned_cpd.values.flatten()
            
            # Compute max absolute difference.
            diffs = abs(orig_values - learned_values)
            max_diff = float(diffs.max())
            avg_diff = float(diffs.mean())
            
            total_change += avg_diff
            max_change = max(max_change, max_diff)
            
            evaluation['nodes_evaluated'] += 1
            
            if max_diff > threshold:
                evaluation['large_changes'].append({
                    'node': node,
                    'max_change': max_diff,
                    'avg_change': avg_diff,
                })
        
        except Exception as exc:
            logger.warning(f"Could not evaluate node {node}: {exc}")
            continue
    
    if evaluation['nodes_evaluated'] > 0:
        evaluation['avg_change'] = total_change / evaluation['nodes_evaluated']
        evaluation['max_change'] = max_change
    
    logger.info(f"Evaluation complete: {evaluation['nodes_evaluated']} nodes")
    logger.info(f"Average change: {evaluation['avg_change']:.4f}")
    logger.info(f"Max change: {evaluation['max_change']:.4f}")
    logger.info(f"Large changes (>{threshold}): {len(evaluation['large_changes'])}")
    
    return evaluation


def save_learned_model(
    model: BayesianNetwork,
    output_path: Path,
    evaluation: Optional[Dict] = None,
) -> None:
    """Save learned model and evaluation results.
    
    Args:
        model: Learned model
        output_path: Output file path
        evaluation: Optional evaluation metrics
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model.
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Learned model saved to {output_path}")
    
    # Save evaluation if provided.
    if evaluation is not None:
        eval_path = output_path.with_suffix('.eval.json')
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2)
        logger.info(f"Evaluation saved to {eval_path}")


def export_cpds_as_json(
    model: BayesianNetwork,
    output_path: Path,
) -> None:
    """Export all CPDs to structured JSON format.
    
    Args:
        model: Bayesian Network with learned CPTs
        output_path: Output JSON file path
    """
    _ensure_pgmpy()
    
    logger.info("Exporting CPDs to JSON...")
    
    cpds_export = {}
    
    for node in sorted(model.nodes()):
        cpd = model.get_cpds(node)
        
        # Get state names.
        state_names = cpd.state_names[node]
        
        # Get parent information.
        parents = list(cpd.variables[1:]) if len(cpd.variables) > 1 else []
        
        # Get CPT values.
        values = cpd.values
        
        node_export = {
            'node': node,
            'states': state_names,
            'parents': parents,
            'cardinality': cpd.cardinality[0],
        }
        
        if not parents:
            # Root node: simple prior.
            node_export['type'] = 'prior'
            node_export['probabilities'] = {
                state: float(values[i][0])
                for i, state in enumerate(state_names)
            }
        else:
            # Child node: conditional probability table.
            node_export['type'] = 'conditional'
            
            # Get parent states.
            parent_states = {}
            for parent in parents:
                parent_states[parent] = cpd.state_names[parent]
            
            node_export['parent_states'] = parent_states
            
            # Build conditional probability table.
            cpt = []
            
            # Iterate over all parent configurations.
            parent_cards = [len(parent_states[p]) for p in parents]
            num_configs = 1
            for card in parent_cards:
                num_configs *= card
            
            for config_idx in range(num_configs):
                # Decode configuration index to parent state indices.
                config = []
                temp_idx = config_idx
                for card in reversed(parent_cards):
                    config.insert(0, temp_idx % card)
                    temp_idx //= card
                
                # Build parent assignment.
                parent_assignment = {
                    parent: parent_states[parent][config[i]]
                    for i, parent in enumerate(parents)
                }
                
                # Get probabilities for this configuration.
                probs = {
                    state: float(values[state_idx][config_idx])
                    for state_idx, state in enumerate(state_names)
                }
                
                cpt.append({
                    'given': parent_assignment,
                    'probabilities': probs,
                })
            
            node_export['cpt'] = cpt
        
        cpds_export[node] = node_export
    
    # Save to JSON.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cpds_export, f, indent=2)
    
    logger.info(f"CPDs exported to {output_path}")


def export_cpds_as_markdown(
    model: BayesianNetwork,
    output_path: Path,
) -> None:
    """Export all CPDs to human-readable Markdown format.
    
    Args:
        model: Bayesian Network with learned CPTs
        output_path: Output Markdown file path
    """
    _ensure_pgmpy()
    
    logger.info("Exporting CPDs to Markdown...")
    
    lines = [
        "# Learned Conditional Probability Tables (CPTs)",
        "",
        "This document contains all learned probability parameters for the Bayesian Network.",
        "",
        f"**Total nodes**: {len(model.nodes())}",
        f"**Total edges**: {len(model.edges())}",
        "",
        "---",
        "",
    ]
    
    # Export each node.
    for node in sorted(model.nodes()):
        cpd = model.get_cpds(node)
        
        lines.append(f"## {node}")
        lines.append("")
        
        # Get state names.
        state_names = cpd.state_names[node]
        parents = list(cpd.variables[1:]) if len(cpd.variables) > 1 else []
        
        if not parents:
            # Root node.
            lines.append("**Type**: Prior probability distribution")
            lines.append("")
            lines.append("**States**: " + ", ".join(state_names))
            lines.append("")
            lines.append("| State | Probability |")
            lines.append("|-------|-------------|")
            
            values = cpd.values
            for i, state in enumerate(state_names):
                prob = values[i][0]
                lines.append(f"| {state} | {prob:.4f} |")
            
            lines.append("")
        else:
            # Child node.
            lines.append("**Type**: Conditional probability table")
            lines.append("")
            lines.append("**States**: " + ", ".join(state_names))
            lines.append("")
            lines.append("**Parents**: " + ", ".join(parents))
            lines.append("")
            
            # Get parent states.
            parent_states = {p: cpd.state_names[p] for p in parents}
            
            # Build table header.
            header = ["| " + " | ".join(parents) + " |"]
            for state in state_names:
                header[0] += f" P({state}) |"
            lines.append(header[0])
            
            # Build separator.
            sep = "|" + "---|" * len(parents)
            for _ in state_names:
                sep += "----------|"
            lines.append(sep)
            
            # Build rows.
            values = cpd.values
            parent_cards = [len(parent_states[p]) for p in parents]
            num_configs = 1
            for card in parent_cards:
                num_configs *= card
            
            for config_idx in range(num_configs):
                # Decode configuration.
                config = []
                temp_idx = config_idx
                for card in reversed(parent_cards):
                    config.insert(0, temp_idx % card)
                    temp_idx //= card
                
                # Build row.
                row = "| "
                for i, parent in enumerate(parents):
                    parent_state = parent_states[parent][config[i]]
                    row += f"{parent_state} | "
                
                for state_idx in range(len(state_names)):
                    prob = values[state_idx][config_idx]
                    row += f"{prob:.4f} | "
                
                lines.append(row)
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Add summary statistics.
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("### Node Types")
    lines.append("")
    
    root_nodes = [n for n in model.nodes() if len(list(model.predecessors(n))) == 0]
    lines.append(f"- **Root nodes** (priors): {len(root_nodes)}")
    lines.append(f"- **Child nodes** (conditional): {len(model.nodes()) - len(root_nodes)}")
    lines.append("")
    
    lines.append("### Parent Distribution")
    lines.append("")
    parent_counts = {}
    for node in model.nodes():
        num_parents = len(list(model.predecessors(node)))
        parent_counts[num_parents] = parent_counts.get(num_parents, 0) + 1
    
    for num_parents in sorted(parent_counts.keys()):
        count = parent_counts[num_parents]
        lines.append(f"- **{num_parents} parents**: {count} nodes")
    
    lines.append("")
    
    # Save to Markdown.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"CPDs exported to {output_path}")


def main():
    """Command-line interface for parameter learning."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Learn BN parameters from data"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("analysis_results/bn_model_initial.pkl"),
        help="Path to initial model",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data (CSV, JSON, or SQLite)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results/bn_model_learned.pkl"),
        help="Output path for learned model",
    )
    parser.add_argument(
        "--method",
        choices=["mle", "bayesian", "em"],
        default="bayesian",
        help="Learning method",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate learned parameters against original",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Export learned CPDs to JSON",
    )
    
    parser.add_argument(
        "--export-markdown",
        type=Path,
        help="Export learned CPDs to Markdown",
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
    
    # Load model.
    logger.info(f"Loading model from {args.model}")
    with open(args.model, 'rb') as f:
        original_model = pickle.load(f)
    
    # Make a copy for learning.
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    
    # Load data.
    if args.data.suffix == '.csv':
        data = pd.read_csv(args.data)
    elif args.data.suffix == '.json':
        # Try to extract evidence from corpus.
        data = extract_evidence_from_corpus(args.data, model)
    elif args.data.suffix in ['.db', '.sqlite', '.sqlite3']:
        data = extract_evidence_from_corpus(args.data, model)
    else:
        raise ValueError(f"Unsupported data format: {args.data.suffix}")
    
    logger.info(f"Loaded {len(data)} records")
    
    # Learn parameters.
    config = LearningConfig(method=args.method)
    learned_model = learn_parameters(model, data, config)
    
    # Evaluate if requested.
    evaluation = None
    if args.evaluate:
        evaluation = evaluate_learned_parameters(original_model, learned_model)
    
    # Save results.
    save_learned_model(learned_model, args.output, evaluation)
    
    # Export CPDs if requested.
    if args.export_json:
        export_cpds_as_json(learned_model, args.export_json)
    
    if args.export_markdown:
        export_cpds_as_markdown(learned_model, args.export_markdown)
    
    print(f"\n{'='*60}")
    print("Parameter Learning Complete")
    print(f"{'='*60}")
    print(f"Method: {config.method}")
    print(f"Training records: {len(data)}")
    print(f"Model saved to: {args.output}")
    
    if args.export_json:
        print(f"CPDs exported (JSON): {args.export_json}")
    
    if args.export_markdown:
        print(f"CPDs exported (Markdown): {args.export_markdown}")
    
    if evaluation:
        print(f"\nEvaluation:")
        print(f" Nodes evaluated: {evaluation['nodes_evaluated']}")
        print(f" Average change: {evaluation['avg_change']:.4f}")
        print(f" Max change: {evaluation['max_change']:.4f}")
        print(f" Large changes: {len(evaluation['large_changes'])}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

