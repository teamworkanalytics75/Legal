"""Integration module for the enhanced BN system.

This module integrates the new scalable BN infrastructure with the existing
bn_adapter.py and WizardWeb system, providing automatic parameter learning
and seamless fallback to expert-initialized models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork
    
    HAVE_PGMPY = True
except ImportError:
    HAVE_PGMPY = False
    BayesianNetwork = None # type: ignore

from .bn_adapter import run_bn_inference, run_bn_inference_with_fallback
from .insights import CaseInsights

try:
    from .bn_constructor import build_bn_model, load_model, save_model
    from .parameter_learning import learn_parameters, LearningConfig
    from .bn_validation import run_full_validation, ValidationConfig
except ImportError:
    # Graceful degradation if new modules not available.
    build_bn_model = None # type: ignore
    load_model = None # type: ignore
    save_model = None # type: ignore
    learn_parameters = None # type: ignore
    LearningConfig = None # type: ignore
    run_full_validation = None # type: ignore
    ValidationConfig = None # type: ignore


logger = logging.getLogger(__name__)


class IntegratedBNError(RuntimeError):
    """Raised when integrated BN operations fail."""


def run_inference_with_learned_model(
    model_path: Path,
    evidence: Dict[str, str],
    summary: str,
    *,
    reference_id: str = "case",
    data_path: Optional[Path] = None,
    force_relearn: bool = False,
) -> Tuple[CaseInsights, Dict[str, Dict[str, float]]]:
    """Run inference using learned or expert-initialized model.
    
    This is the main entry point that:
    1. Checks if a learned model exists
    2. Optionally relearns from data
    3. Falls back to expert model if needed
    4. Runs inference with automatic fallback (PySMILE -> pgmpy)
    
    Args:
        model_path: Path to .xdsl or .pkl model file
        evidence: Evidence dictionary
        summary: Case summary
        reference_id: Case identifier
        data_path: Optional path to training data for parameter learning
        force_relearn: Force relearning even if learned model exists
        
    Returns:
        Tuple of (CaseInsights, posterior_data)
    """
    logger.info("Running integrated BN inference...")
    
    # Check for learned model.
    learned_model_path = _get_learned_model_path(model_path)
    
    if learned_model_path.exists() and not force_relearn:
        logger.info(f"Using learned model: {learned_model_path}")
        
        # Check if it's a pgmpy model.
        if learned_model_path.suffix == '.pkl' and HAVE_PGMPY:
            try:
                return _run_pgmpy_inference_from_pkl(
                    learned_model_path,
                    evidence,
                    summary,
                    reference_id,
                )
            except Exception as exc:
                logger.warning(f"Learned model failed: {exc}, falling back to expert model")
    
    # Optionally learn from data.
    if data_path is not None and data_path.exists():
        logger.info("Learning parameters from data...")
        try:
            learned_model_path = _learn_and_save_model(
                model_path,
                data_path,
                learned_model_path,
            )
            
            # Use the newly learned model.
            return _run_pgmpy_inference_from_pkl(
                learned_model_path,
                evidence,
                summary,
                reference_id,
            )
            
        except Exception as exc:
            logger.warning(f"Parameter learning failed: {exc}, using expert model")
    
    # Fall back to standard inference with expert model.
    logger.info("Using expert-initialized model")
    return run_bn_inference_with_fallback(
        model_path,
        evidence,
        summary,
        reference_id=reference_id,
    )


def _get_learned_model_path(base_path: Path) -> Path:
    """Get path for learned model based on base model path."""
    return base_path.with_stem(base_path.stem + "_learned").with_suffix('.pkl')


def _run_pgmpy_inference_from_pkl(
    model_path: Path,
    evidence: Dict[str, str],
    summary: str,
    reference_id: str,
) -> Tuple[CaseInsights, Dict[str, Dict[str, float]]]:
    """Run inference using a saved pgmpy model."""
    if not HAVE_PGMPY:
        raise IntegratedBNError("pgmpy not available")
    
    from .pgmpy_inference import run_pgmpy_inference
    
    # pgmpy inference expects .xdsl, but we have .pkl
    # We need to adapt this.
    try:
        import pickle
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Run inference directly on the model.
        from .pgmpy_inference import run_inference
        from .bn_bridge import build_insights
        from .insights import EvidenceItem
        
        posterior_data = run_inference(model, evidence)
        
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
        
        return insights, posterior_data
        
    except Exception as exc:
        raise IntegratedBNError(f"pgmpy inference failed: {exc}") from exc


def _learn_and_save_model(
    base_model_path: Path,
    data_path: Path,
    output_path: Path,
) -> Path:
    """Learn parameters and save learned model."""
    if not HAVE_PGMPY:
        raise IntegratedBNError("pgmpy not available for learning")
    
    if learn_parameters is None:
        raise IntegratedBNError("parameter_learning module not available")
    
    import pandas as pd
    import pickle
    
    # Load base model.
    if base_model_path.suffix == '.pkl':
        with open(base_model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        # Parse XDSL and build model.
        from .xdsl_parser import parse_xdsl
        from .pgmpy_inference import build_bn_from_xdsl
        
        xdsl_network = parse_xdsl(base_model_path)
        model = build_bn_from_xdsl(xdsl_network)
    
    # Load data.
    if data_path.suffix == '.csv':
        data = pd.read_csv(data_path)
    elif data_path.suffix == '.json':
        from .parameter_learning import extract_evidence_from_corpus
        data = extract_evidence_from_corpus(data_path, model)
    else:
        raise IntegratedBNError(f"Unsupported data format: {data_path.suffix}")
    
    # Learn parameters.
    config = LearningConfig(method="bayesian")
    learned_model = learn_parameters(model, data, config)
    
    # Save learned model.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(learned_model, f)
    
    logger.info(f"Learned model saved to {output_path}")
    
    return output_path


def validate_and_report(
    model_path: Path,
    output_dir: Path,
) -> bool:
    """Validate model and generate report.
    
    Args:
        model_path: Path to model (.pkl)
        output_dir: Directory for validation reports
        
    Returns:
        True if validation passed
    """
    if not HAVE_PGMPY:
        logger.warning("pgmpy not available, skipping validation")
        return True
    
    if run_full_validation is None:
        logger.warning("bn_validation module not available")
        return True
    
    import pickle
    
    # Load model.
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Run validation.
    config = ValidationConfig()
    report = run_full_validation(model, config)
    
    # Save report.
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{model_path.stem}_validation.json"
    
    from .bn_validation import save_validation_report, print_validation_summary
    save_validation_report(report, report_path)
    print_validation_summary(report)
    
    return report.passed


def build_enhanced_model_from_kg(
    entities_path: Path,
    graph_path: Path,
    data_path: Optional[Path],
    output_dir: Path,
    max_nodes: int = 150,
    learn_from_data: bool = True,
) -> Path:
    """Build complete BN model from knowledge graph with optional learning.
    
    This is the end-to-end pipeline:
    1. Select nodes from KG
    2. Build BN structure
    3. Initialize with expert priors
    4. Optionally learn from data
    5. Validate
    6. Save
    
    Args:
        entities_path: Path to entities_all.json
        graph_path: Path to cooccurrence graph
        data_path: Optional training data
        output_dir: Output directory
        max_nodes: Maximum nodes to select
        learn_from_data: Whether to learn parameters
        
    Returns:
        Path to final model
    """
    logger.info("Building enhanced BN model from knowledge graph...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Build structure.
    logger.info("Step 1/5: Selecting nodes and building structure...")
    from .build_bn_structure_from_kg import (
        build_structure_from_kg,
        save_structure,
        NodeSelectionConfig,
    )
    
    config = NodeSelectionConfig(max_nodes=max_nodes)
    structure = build_structure_from_kg(entities_path, graph_path, config)
    
    structure_dir = output_dir / "structure"
    save_structure(structure, structure_dir)
    logger.info(f"Structure saved to {structure_dir}")
    
    # Step 2: Build initial model.
    logger.info("Step 2/5: Building initial BN model...")
    if build_bn_model is None:
        raise IntegratedBNError("bn_constructor module not available")
    
    initial_model = build_bn_model(structure, use_expert_priors=True)
    initial_model_path = output_dir / "bn_model_initial.pkl"
    save_model(initial_model, initial_model_path)
    logger.info(f"Initial model saved to {initial_model_path}")
    
    # Step 3: Learn from data (optional).
    if learn_from_data and data_path is not None and data_path.exists():
        logger.info("Step 3/5: Learning parameters from data...")
        learned_model_path = _learn_and_save_model(
            initial_model_path,
            data_path,
            output_dir / "bn_model_learned.pkl",
        )
        final_model_path = learned_model_path
    else:
        logger.info("Step 3/5: Skipping parameter learning")
        final_model_path = initial_model_path
    
    # Step 4: Validate.
    logger.info("Step 4/5: Validating model...")
    validation_passed = validate_and_report(final_model_path, output_dir)
    
    if not validation_passed:
        logger.warning("Validation failed - review validation report")
    
    # Step 5: Export to XDSL.
    logger.info("Step 5/5: Exporting to XDSL format...")
    xdsl_path = final_model_path.with_suffix('.xdsl')
    try:
        from .xdsl_exporter import export_to_xdsl
        import pickle
        
        with open(final_model_path, 'rb') as f:
            final_model = pickle.load(f)
        
        export_to_xdsl(final_model, structure, xdsl_path)
        logger.info(f"XDSL exported to {xdsl_path}")
    except ImportError:
        logger.warning("XDSL export not available")
    
    logger.info(f"Enhanced BN model complete: {final_model_path}")
    
    return final_model_path


def main():
    """Command-line interface for integrated BN system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integrated BN system with automatic learning"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build command.
    build_parser = subparsers.add_parser('build', help='Build BN from KG')
    build_parser.add_argument('--entities', type=Path, required=True)
    build_parser.add_argument('--graph', type=Path, required=True)
    build_parser.add_argument('--data', type=Path)
    build_parser.add_argument('--output', type=Path, required=True)
    build_parser.add_argument('--max-nodes', type=int, default=150)
    build_parser.add_argument('--no-learning', action='store_true')
    
    # Inference command.
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model', type=Path, required=True)
    infer_parser.add_argument('--evidence', type=str, required=True)
    infer_parser.add_argument('--summary', type=str, default="Case summary")
    infer_parser.add_argument('--data', type=Path)
    infer_parser.add_argument('--relearn', action='store_true')
    
    # Validate command.
    validate_parser = subparsers.add_parser('validate', help='Validate model')
    validate_parser.add_argument('--model', type=Path, required=True)
    validate_parser.add_argument('--output', type=Path, required=True)
    
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # Configure logging.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    if args.command == 'build':
        build_enhanced_model_from_kg(
            entities_path=args.entities,
            graph_path=args.graph,
            data_path=args.data,
            output_dir=args.output,
            max_nodes=args.max_nodes,
            learn_from_data=not args.no_learning,
        )
    
    elif args.command == 'infer':
        # Parse evidence.
        evidence = {}
        for pair in args.evidence.split(','):
            if '=' in pair:
                node, state = pair.split('=', 1)
                evidence[node.strip()] = state.strip()
        
        insights, posteriors = run_inference_with_learned_model(
            model_path=args.model,
            evidence=evidence,
            summary=args.summary,
            data_path=args.data,
            force_relearn=args.relearn,
        )
        
        print("\nInference Results:")
        print(f"Reference: {insights.reference_id}")
        print(f"Posteriors computed for {len(posteriors)} nodes")
    
    elif args.command == 'validate':
        passed = validate_and_report(args.model, args.output)
        exit(0 if passed else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
