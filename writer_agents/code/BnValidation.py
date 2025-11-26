"""Performance and quality validation for Bayesian Networks.

This module provides tools to validate BN models for:
- Structural constraints (node count, parent limits)
- Inference performance
- CPT quality and extreme value detection
- Model pruning based on performance metrics
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from pgmpy.inference import VariableElimination
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork
    
    HAVE_PGMPY = True
except ImportError:
    HAVE_PGMPY = False
    BayesianNetwork = None # type: ignore
    VariableElimination = None # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    
    max_nodes: int = 200
    max_parents: int = 3
    max_inference_time: float = 5.0 # seconds
    min_cpd_variance: float = 0.01
    max_slowdown_factor: float = 2.0
    extreme_value_threshold: float = 0.95 # Flag probs > 0.95 or < 0.05


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    passed: bool
    num_nodes: int
    num_edges: int
    avg_parents: float
    max_parents: int
    inference_time: float
    structure_valid: bool
    cpd_issues: List[Dict]
    performance_issues: List[str]
    recommendations: List[str]
    detailed_metrics: Dict


class ValidationError(RuntimeError):
    """Raised when validation fails."""


def _ensure_pgmpy() -> None:
    """Check that pgmpy is available."""
    if not HAVE_PGMPY:
        raise ValidationError(
            "pgmpy is required. Install with: pip install pgmpy networkx"
        )


def validate_structure(
    model: BayesianNetwork,
    config: ValidationConfig,
) -> Tuple[bool, List[str]]:
    """Validate BN structure against constraints.
    
    Args:
        model: Bayesian Network to validate
        config: Validation configuration
        
    Returns:
        Tuple of (valid, issues list)
    """
    logger.info("Validating BN structure...")
    
    issues = []
    
    # Check node count.
    num_nodes = len(model.nodes())
    if num_nodes > config.max_nodes:
        issues.append(f"Too many nodes: {num_nodes} > {config.max_nodes}")
    
    # Check parent counts.
    parent_counts = []
    for node in model.nodes():
        parents = list(model.predecessors(node))
        parent_counts.append(len(parents))
        
        if len(parents) > config.max_parents:
            issues.append(
                f"Node '{node}' has {len(parents)} parents (max {config.max_parents})"
            )
    
    avg_parents = sum(parent_counts) / len(parent_counts) if parent_counts else 0
    max_parents = max(parent_counts) if parent_counts else 0
    
    logger.info(f"Structure: {num_nodes} nodes, avg {avg_parents:.2f} parents, max {max_parents}")
    
    # Check model validity.
    try:
        if not model.check_model():
            issues.append("Model.check_model() failed")
    except Exception as exc:
        issues.append(f"Model validation error: {exc}")
    
    valid = len(issues) == 0
    return valid, issues


def validate_inference_performance(
    model: BayesianNetwork,
    config: ValidationConfig,
    test_evidence: Optional[Dict[str, str]] = None,
) -> Tuple[float, List[str]]:
    """Validate inference performance.
    
    Args:
        model: Bayesian Network
        config: Validation configuration
        test_evidence: Optional test evidence (uses empty if None)
        
    Returns:
        Tuple of (inference_time, issues list)
    """
    logger.info("Testing inference performance...")
    
    issues = []
    
    if test_evidence is None:
        test_evidence = {}
    
    try:
        inference = VariableElimination(model)
        
        # Time full inference on all nodes.
        all_nodes = list(model.nodes())
        
        start_time = time.time()
        
        for node in all_nodes:
            try:
                inference.query(
                    variables=[node],
                    evidence=test_evidence,
                    show_progress=False,
                )
            except Exception as exc:
                issues.append(f"Inference failed for node '{node}': {exc}")
        
        elapsed = time.time() - start_time
        
        logger.info(f"Full inference completed in {elapsed:.2f}s")
        
        if elapsed > config.max_inference_time:
            issues.append(
                f"Inference too slow: {elapsed:.2f}s > {config.max_inference_time}s"
            )
        
        return elapsed, issues
        
    except Exception as exc:
        issues.append(f"Inference engine creation failed: {exc}")
        return -1.0, issues


def validate_cpd_quality(
    model: BayesianNetwork,
    config: ValidationConfig,
) -> List[Dict]:
    """Validate CPD quality and detect issues.
    
    Args:
        model: Bayesian Network
        config: Validation configuration
        
    Returns:
        List of CPD issues found
    """
    logger.info("Validating CPD quality...")
    
    issues = []
    
    for node in model.nodes():
        try:
            cpd = model.get_cpds(node)
            values = cpd.values.flatten()
            
            # Check for extreme probabilities.
            extreme_high = values[values > config.extreme_value_threshold]
            extreme_low = values[values < (1 - config.extreme_value_threshold)]
            
            if len(extreme_high) > 0:
                issues.append({
                    'node': node,
                    'issue': 'extreme_high_probabilities',
                    'count': len(extreme_high),
                    'max_value': float(values.max()),
                })
            
            if len(extreme_low) > 0:
                issues.append({
                    'node': node,
                    'issue': 'extreme_low_probabilities',
                    'count': len(extreme_low),
                    'min_value': float(values.min()),
                })
            
            # Check variance.
            variance = float(values.var())
            if variance < config.min_cpd_variance:
                issues.append({
                    'node': node,
                    'issue': 'low_variance',
                    'variance': variance,
                })
            
        except Exception as exc:
            issues.append({
                'node': node,
                'issue': 'validation_error',
                'error': str(exc),
            })
    
    logger.info(f"Found {len(issues)} CPD quality issues")
    return issues


def identify_pruning_candidates(
    model: BayesianNetwork,
    config: ValidationConfig,
) -> List[str]:
    """Identify nodes that could be pruned.
    
    Nodes are candidates for pruning if:
    - Low CPD variance (not informative)
    - No paths to outcome nodes
    
    Args:
        model: Bayesian Network
        config: Validation configuration
        
    Returns:
        List of node IDs that could be pruned
    """
    logger.info("Identifying pruning candidates...")
    
    candidates = []
    
    # Check CPD variance for each node.
    for node in model.nodes():
        try:
            cpd = model.get_cpds(node)
            values = cpd.values.flatten()
            variance = float(values.var())
            
            if variance < config.min_cpd_variance:
                candidates.append(node)
                logger.debug(f"Node '{node}' has low variance: {variance:.6f}")
        
        except Exception as exc:
            logger.warning(f"Could not check variance for '{node}': {exc}")
    
    logger.info(f"Identified {len(candidates)} pruning candidates")
    return candidates


def run_full_validation(
    model: BayesianNetwork,
    config: Optional[ValidationConfig] = None,
    test_evidence: Optional[Dict[str, str]] = None,
) -> ValidationReport:
    """Run comprehensive validation on BN model.
    
    Args:
        model: Bayesian Network to validate
        config: Validation configuration
        test_evidence: Optional test evidence for performance testing
        
    Returns:
        Complete validation report
    """
    _ensure_pgmpy()
    
    if config is None:
        config = ValidationConfig()
    
    logger.info("Running full validation...")
    
    # Structure validation.
    structure_valid, structure_issues = validate_structure(model, config)
    
    # Performance validation.
    inference_time, performance_issues = validate_inference_performance(
        model, config, test_evidence
    )
    
    # CPD quality validation.
    cpd_issues = validate_cpd_quality(model, config)
    
    # Compute metrics.
    num_nodes = len(model.nodes())
    num_edges = len(model.edges())
    
    parent_counts = [len(list(model.predecessors(n))) for n in model.nodes()]
    avg_parents = sum(parent_counts) / len(parent_counts) if parent_counts else 0
    max_parents = max(parent_counts) if parent_counts else 0
    
    # Generate recommendations.
    recommendations = []
    
    if not structure_valid:
        recommendations.extend(structure_issues)
    
    if inference_time > config.max_inference_time:
        recommendations.append(
            f"Consider pruning nodes to improve inference speed ({inference_time:.2f}s)"
        )
    
    if len(cpd_issues) > 0:
        recommendations.append(
            f"Review {len(cpd_issues)} CPD quality issues for extreme values"
        )
    
    # Identify pruning candidates.
    pruning_candidates = identify_pruning_candidates(model, config)
    if len(pruning_candidates) > 0:
        recommendations.append(
            f"Consider pruning {len(pruning_candidates)} low-variance nodes"
        )
    
    # Overall pass/fail.
    passed = (
        structure_valid and
        len(performance_issues) == 0 and
        inference_time > 0
    )
    
    report = ValidationReport(
        passed=passed,
        num_nodes=num_nodes,
        num_edges=num_edges,
        avg_parents=avg_parents,
        max_parents=max_parents,
        inference_time=inference_time,
        structure_valid=structure_valid,
        cpd_issues=cpd_issues,
        performance_issues=performance_issues + structure_issues,
        recommendations=recommendations,
        detailed_metrics={
            'parent_counts': parent_counts,
            'pruning_candidates': pruning_candidates,
        }
    )
    
    logger.info(f"Validation complete: {'PASSED' if passed else 'FAILED'}")
    
    return report


def save_validation_report(
    report: ValidationReport,
    output_path: Path,
) -> None:
    """Save validation report to file.
    
    Args:
        report: Validation report
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_dict = {
        'passed': report.passed,
        'num_nodes': report.num_nodes,
        'num_edges': report.num_edges,
        'avg_parents': report.avg_parents,
        'max_parents': report.max_parents,
        'inference_time': report.inference_time,
        'structure_valid': report.structure_valid,
        'cpd_issues': report.cpd_issues,
        'performance_issues': report.performance_issues,
        'recommendations': report.recommendations,
        'detailed_metrics': report.detailed_metrics,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2)
    
    logger.info(f"Validation report saved to {output_path}")


def print_validation_summary(report: ValidationReport) -> None:
    """Print validation report summary to console.
    
    Args:
        report: Validation report
    """
    print(f"\n{'='*60}")
    print("BN VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Status: {'PASSED' if report.passed else 'FAILED'}")
    print(f"\nStructure:")
    print(f" Nodes: {report.num_nodes}")
    print(f" Edges: {report.num_edges}")
    print(f" Avg parents: {report.avg_parents:.2f}")
    print(f" Max parents: {report.max_parents}")
    print(f"\nPerformance:")
    print(f" Inference time: {report.inference_time:.2f}s")
    print(f"\nQuality:")
    print(f" Structure valid: {report.structure_valid}")
    print(f" CPD issues: {len(report.cpd_issues)}")
    print(f" Performance issues: {len(report.performance_issues)}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f" {i}. {rec}")
    
    print(f"{'='*60}\n")


def main():
    """Command-line interface for validation."""
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(
        description="Validate BN model"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model (.pkl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results/validation_report.json"),
        help="Output path for validation report",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=5.0,
        help="Maximum allowed inference time (seconds)",
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
        model = pickle.load(f)
    
    # Run validation.
    config = ValidationConfig(max_inference_time=args.max_time)
    report = run_full_validation(model, config)
    
    # Print summary.
    print_validation_summary(report)
    
    # Save report.
    save_validation_report(report, args.output)


if __name__ == "__main__":
    main()

