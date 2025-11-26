"""
Wizard's Web BN Integration Module.

This module connects AutoGen group chat results to the Bayesian Network,
updating CPT values based on evidence scores.

Usage:
    python wizardweb_integration.py path/to/groupchat_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import BN integration tools
from autogen_groupchat.tools.bn_integration import bn_write_cpt, bn_batch_update, bn_read_structure
from autogen_groupchat.schemas.evidence_card import AnalysisOutput, EdgeEvidenceCard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_groupchat_results(json_path: str) -> Dict[str, Any]:
    """
    Load AutoGen group chat results from JSON file.
    
    Args:
        json_path: Path to groupchat_results_*.json
    
    Returns:
        Dictionary with analysis results
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded group chat results from {json_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        raise


def validate_analysis_output(data: Dict[str, Any]) -> AnalysisOutput:
    """
    Validate data against AnalysisOutput schema.
    
    Args:
        data: Raw dictionary from JSON
    
    Returns:
        Validated AnalysisOutput object
    
    Raises:
        ValidationError if data doesn't match schema
    """
    try:
        analysis = AnalysisOutput(**data)
        logger.info("[ok] Data validated against AnalysisOutput schema")
        return analysis
        
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise


def update_bn_from_json(
    json_path: str,
    bn_model_path: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Update Bayesian Network CPT values from group chat JSON results.
    
    Args:
        json_path: Path to groupchat_results_*.json
        bn_model_path: Path to BN model (.xdsl or .json)
                      If None, uses default mock model
        dry_run: If True, don't actually write to BN (just validate)
    
    Returns:
        Dictionary with update results
    
    Example:
        >>> results = update_bn_from_json(
        ... "autogen_outputs/groupchat_results_20251008_132456.json",
        ... bn_model_path="data/mock_bn_model.json"
        ... )
        >>> print(f"Updated {results['success_count']} nodes")
    """
    
    logger.info("=" * 80)
    logger.info("WIZARD'S WEB: BN INTEGRATION")
    logger.info("=" * 80)
    
    # Load and validate data
    logger.info(f"Loading results from: {json_path}")
    data = load_groupchat_results(json_path)
    
    try:
        analysis = validate_analysis_output(data)
    except Exception:
        logger.warning("Schema validation failed, attempting to work with raw data...")
        analysis = None
    
    # Determine BN model path
    if bn_model_path is None:
        bn_model_path = "data/mock_bn_model.json"
        logger.info(f"Using default BN model: {bn_model_path}")
    else:
        logger.info(f"Using BN model: {bn_model_path}")
    
    # Read current BN structure
    logger.info("Reading BN structure...")
    bn_structure = bn_read_structure(bn_model_path)
    if "error" in bn_structure:
        logger.warning(f"Could not read BN structure: {bn_structure['error']}")
    else:
        logger.info(f"BN has {len(bn_structure.get('nodes', []))} nodes")
    
    # Extract causal edges
    if analysis:
        edges = analysis.causal_edges
    elif "causal_edges" in data:
        edges = [EdgeEvidenceCard(**edge) for edge in data["causal_edges"]]
    else:
        logger.error("No causal_edges found in data")
        return {"success": False, "error": "No causal_edges in data"}
    
    logger.info(f"Processing {len(edges)} causal edges...")
    
    # Batch update results
    update_results = {
        "success_count": 0,
        "failed_count": 0,
        "edge_updates": [],
        "dry_run": dry_run
    }
    
    for edge in edges:
        edge_id = edge.edge_id
        edge_desc = edge.edge_description
        
        logger.info(f"\nProcessing: {edge_id} - {edge_desc}")
        logger.info(f" Temporality: {edge.temporality_score}/3")
        logger.info(f" Reuse: {edge.reuse_score}/3")
        logger.info(f" Channel: {edge.channel_score}/3")
        logger.info(f" Confidence: {edge.confidence:.2f}")
        
        # Prepare CPT data
        cpt_data = {
            "temporality_score": edge.temporality_score,
            "reuse_score": edge.reuse_score,
            "channel_score": edge.channel_score,
            "replication_count": edge.replication_count,
            "confounder_pressure": edge.confounder_pressure,
            "confidence": edge.confidence,
        }
        
        # Update BN (unless dry run)
        if not dry_run:
            # Use edge_id as node name (or edge_description if preferred)
            node_name = edge_id
            success = bn_write_cpt(node_name, cpt_data, bn_model_path)
            
            if success:
                logger.info(f" [ok] Updated CPT for {node_name}")
                update_results["success_count"] += 1
            else:
                logger.warning(f" Failed to update CPT for {node_name}")
                update_results["failed_count"] += 1
        else:
            logger.info(f" [DRY RUN] Would update CPT for {edge_id}")
            update_results["success_count"] += 1
        
        update_results["edge_updates"].append({
            "edge_id": edge_id,
            "edge_description": edge_desc,
            "cpt_data": cpt_data,
            "success": not dry_run # All succeed in dry run
        })
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BN INTEGRATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total edges: {len(edges)}")
    logger.info(f"Successfully updated: {update_results['success_count']}")
    logger.info(f"Failed: {update_results['failed_count']}")
    
    if not dry_run:
        logger.info(f"BN model updated: {bn_model_path}")
    else:
        logger.info("[DRY RUN] No changes made to BN model")
    
    # Add red bottleneck info if available
    if analysis and analysis.red_bottleneck:
        logger.info("\n" + "-" * 80)
        logger.info("RED BOTTLENECK (Weakest Link)")
        logger.info("-" * 80)
        logger.info(f"Edge: {analysis.red_bottleneck.edge_id}")
        logger.info(f"Reason: {analysis.red_bottleneck.reason}")
        logger.info(f"Confidence: {analysis.red_bottleneck.confidence:.2f}")
        
        update_results["red_bottleneck"] = {
            "edge_id": analysis.red_bottleneck.edge_id,
            "confidence": analysis.red_bottleneck.confidence,
            "reason": analysis.red_bottleneck.reason
        }
    
    # Overall confidence
    if analysis:
        logger.info(f"\nOverall Causal Confidence: {analysis.overall_causal_confidence:.2f}")
        update_results["overall_confidence"] = analysis.overall_causal_confidence
    
    update_results["success"] = True
    return update_results


def main():
    """Command-line interface for BN integration."""
    parser = argparse.ArgumentParser(
        description="Update Bayesian Network from AutoGen group chat results"
    )
    parser.add_argument(
        "json_path",
        help="Path to groupchat_results_*.json file"
    )
    parser.add_argument(
        "--bn-model",
        default=None,
        help="Path to BN model (.xdsl or .json). Defaults to data/mock_bn_model.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data but don't update BN model"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save update results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.json_path).exists():
        logger.error(f"File not found: {args.json_path}")
        sys.exit(1)
    
    try:
        # Run integration
        results = update_bn_from_json(
            args.json_path,
            bn_model_path=args.bn_model,
            dry_run=args.dry_run
        )
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n[ok] Results saved to: {args.output}")
        
        # Exit with appropriate code
        if results.get("success"):
            logger.info("\n[ok] BN integration completed successfully")
            sys.exit(0)
        else:
            logger.error("\n BN integration failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
