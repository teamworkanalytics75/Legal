#!/usr/bin/env python3
"""
Analyze Statement of Claim for US National Security Matter Classification

Uses the full workflow system (Autogen → SK → CatBoost) to analyze whether
allegations in a statement of claim would constitute a US national security matter.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add writer_agents to path
writer_agents_path = Path(__file__).parent.parent / "writer_agents"
sys.path.insert(0, str(writer_agents_path))
sys.path.insert(0, str(writer_agents_path / "code"))

from WorkflowOrchestrator import Conductor
from workflow_config import WorkflowStrategyConfig
# Backward compatibility alias
WorkflowStrategyExecutor = Conductor
from sk_config import SKConfig
from insights import CaseInsights, EvidenceItem
from agents import ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def analyze_national_security_matter(
    statement_of_claim: str,
    jurisdiction: str = "US Federal",
    autogen_model: str = "gpt-4o",
    sk_use_local: bool = False,
    max_iterations: int = 3,
    target_confidence: float = 0.80,
) -> Dict[str, Any]:
    """
    Analyze whether allegations in statement of claim constitute US national security matter.

    Args:
        statement_of_claim: Full text of the statement of claim
        jurisdiction: Court jurisdiction
        autogen_model: Model for Autogen agents
        sk_use_local: Whether to use local model for Semantic Kernel
        max_iterations: Maximum workflow iterations
        target_confidence: Desired confidence/quality threshold for workflow auto-commit

    Returns:
        Dictionary with analysis results, national security classification, and legal reasoning
    """
    logger.info("="*70)
    logger.info("NATIONAL SECURITY MATTER ANALYSIS")
    logger.info("="*70)

    # Create case insights with national security question
    case_summary = f"""
    ANALYSIS REQUEST: To what extent do the allegations in the following Statement of Claim,
    if true as stated, constitute a matter of US national security?

    STATEMENT OF CLAIM:
    {statement_of_claim[:5000]}  # Limit to first 5000 chars for initial analysis
    """

    insights = CaseInsights(
        reference_id=f"NS-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        summary=case_summary,
        jurisdiction=jurisdiction,
        evidence=[
            EvidenceItem(
                node_id="statement_of_claim",
                state="provided",
                description=f"Statement of Claim text ({len(statement_of_claim)} characters)"
            ),
            EvidenceItem(
                node_id="analysis_type",
                state="national_security_classification",
                description="Analysis type: National Security Matter Classification"
            )
        ],
        posteriors=[],
        case_style="National Security Matter Analysis"
    )

    # Configure workflow for analysis (not motion generation)
    # For local models, configure use_local flag
    if sk_use_local:
        autogen_config = ModelConfig(use_local=True, local_model="qwen2.5:14b")
    else:
        autogen_config = ModelConfig(model=autogen_model)
    sk_config = SKConfig(use_local=sk_use_local)  # Note: parameter is 'use_local', not 'use_local_llm'

    if not 0 < target_confidence <= 1:
        raise ValueError("target_confidence must be between 0 and 1")

    workflow_config = WorkflowStrategyConfig(
        autogen_config=autogen_config,
        sk_config=sk_config,
        max_iterations=max_iterations,
        enable_sk_planner=True,
        enable_quality_gates=True,
        auto_commit_threshold=target_confidence,  # Higher threshold for analysis quality
        enable_autogen_review=True,
        validation_strict=True,
        enable_chroma_integration=True,
        chroma_persist_directory="./chroma_collections"
    )

    # Initialize conductor
    executor = Conductor(config=workflow_config)

    # Run workflow
    logger.info("Starting analysis workflow...")
    try:
        deliverable = await executor.run_hybrid_workflow(insights)

        # Extract analysis document
        analysis_document = deliverable.edited_document or (
            deliverable.sections[0].body if deliverable.sections else ""
        )

        # Extract national security indicators
        national_security_indicators = _extract_national_security_indicators(
            statement_of_claim, analysis_document
        )

        # Compile comprehensive analysis
        results = {
            "success": True,
            "analysis_document": analysis_document,
            "national_security_classification": national_security_indicators,
            "statement_of_claim_preview": statement_of_claim[:500] + "..." if len(statement_of_claim) > 500 else statement_of_claim,
            "full_statement_length": len(statement_of_claim),
            "deliverable": {
                "plan": str(deliverable.plan) if hasattr(deliverable, 'plan') else None,
                "sections": [{"title": s.title, "body": s.body[:500]} for s in deliverable.sections] if deliverable.sections else []
            },
            "target_confidence": target_confidence
        }

        logger.info("="*70)
        logger.info("Analysis Complete")
        logger.info("="*70)
        logger.info(f"National Security Indicators Found: {len(national_security_indicators.get('indicators', []))}")

        return results

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "analysis_document": "",
            "national_security_classification": {}
        }


def _extract_national_security_indicators(
    statement_text: str,
    analysis_document: str
) -> Dict[str, Any]:
    """Extract national security indicators from statement and analysis."""
    import re

    statement_lower = statement_text.lower()
    analysis_lower = analysis_document.lower()

    # National security keywords
    natsec_keywords = [
        "national security",
        "national defense",
        "classified information",
        "state secrets",
        "foreign agent",
        "foreign interference",
        "cyber security",
        "critical infrastructure",
        "defense contractor",
        "intelligence",
        "espionage",
        "terrorism",
        "weapons of mass destruction",
        "export control",
        "cfius",
        "foreign investment",
        "prc", "china",
        "russia",
        "iran",
        "north korea",
        "federal funding",
        "defense grant",
        "research security"
    ]

    # Legal frameworks
    legal_frameworks = [
        "fara", "foreign agents registration act",
        "itar", "international traffic in arms",
        "ear", "export administration regulations",
        "cfius", "committee on foreign investment",
        "ndaa", "national defense authorization",
        "executive order",
        "presidential directive"
    ]

    indicators_found = []
    frameworks_found = []

    for keyword in natsec_keywords:
        if keyword in statement_lower or keyword in analysis_lower:
            indicators_found.append(keyword)

    for framework in legal_frameworks:
        if framework in statement_lower or framework in analysis_lower:
            frameworks_found.append(framework)

    # Count mentions
    mention_counts = {}
    for keyword in natsec_keywords + legal_frameworks:
        count = statement_lower.count(keyword) + analysis_lower.count(keyword)
        if count > 0:
            mention_counts[keyword] = count

    # Determine classification strength
    total_indicators = len(indicators_found)
    total_frameworks = len(frameworks_found)

    if total_indicators >= 5 or total_frameworks >= 2:
        classification_strength = "strong"
        confidence = min(0.95, 0.70 + (total_indicators * 0.05))
    elif total_indicators >= 2 or total_frameworks >= 1:
        classification_strength = "moderate"
        confidence = 0.60 + (total_indicators * 0.05)
    elif total_indicators >= 1:
        classification_strength = "weak"
        confidence = 0.40 + (total_indicators * 0.05)
    else:
        classification_strength = "none"
        confidence = 0.20

    return {
        "classification_strength": classification_strength,
        "confidence": confidence,
        "indicators": list(set(indicators_found)),
        "legal_frameworks": list(set(frameworks_found)),
        "mention_counts": mention_counts,
        "total_indicators": total_indicators,
        "total_frameworks": total_frameworks,
        "assessment": f"Based on analysis, the allegations {'DO' if confidence >= 0.70 else 'LIKELY DO' if confidence >= 0.50 else 'MAY' if confidence >= 0.40 else 'DO NOT'} constitute a matter of US national security (confidence: {confidence:.1%})"
    }


def main():
    """Command-line interface for national security analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Statement of Claim for US National Security Matter Classification"
    )
    parser.add_argument(
        "--statement",
        type=str,
        help="Full text of statement of claim (or path to file)"
    )
    parser.add_argument(
        "--statement-file",
        type=str,
        help="Path to file containing statement of claim"
    )
    parser.add_argument(
        "--jurisdiction",
        default="US Federal",
        help="Court jurisdiction (default: US Federal)"
    )
    parser.add_argument(
        "--autogen-model",
        default="gpt-4o",
        help="Model for Autogen agents (default: gpt-4o)"
    )
    parser.add_argument(
        "--target-confidence",
        type=float,
        default=0.80,
        help="Target analysis confidence threshold (default: 0.80)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum workflow iterations (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (JSON format)"
    )

    args = parser.parse_args()

    # Load statement of claim
    statement_of_claim = ""

    if args.statement_file:
        statement_path = Path(args.statement_file)
        if statement_path.exists():
            with open(statement_path, 'r', encoding='utf-8') as f:
                statement_of_claim = f.read()
            logger.info(f"Loaded statement of claim from {statement_path} ({len(statement_of_claim)} chars)")
        else:
            logger.error(f"Statement file not found: {statement_path}")
            sys.exit(1)
    elif args.statement:
        statement_of_claim = args.statement
    else:
        logger.error("Must provide either --statement or --statement-file")
        parser.print_help()
        sys.exit(1)

    if not statement_of_claim or len(statement_of_claim.strip()) < 100:
        logger.error("Statement of claim is too short (minimum 100 characters)")
        sys.exit(1)

    # Run analysis
    logger.info(f"Analyzing statement of claim ({len(statement_of_claim)} characters)...")
    results = asyncio.run(analyze_national_security_matter(
        statement_of_claim=statement_of_claim,
        jurisdiction=args.jurisdiction,
        autogen_model=args.autogen_model,
        target_confidence=args.target_confidence,
        max_iterations=args.max_iterations
    ))

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

        # Also save analysis document as text
        analysis_path = output_path.with_suffix('.analysis.txt')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(results.get("analysis_document", ""))
        logger.info(f"Analysis document saved to {analysis_path}")

    # Print summary
    if results.get("success"):
        print("\n" + "="*70)
        print("NATIONAL SECURITY MATTER ANALYSIS - RESULTS")
        print("="*70)

        classification = results.get("national_security_classification", {})
        target_conf = results.get("target_confidence")
        print(f"\nClassification Strength: {classification.get('classification_strength', 'unknown').upper()}")
        print(f"Confidence Level: {classification.get('confidence', 0):.1%}")
        if target_conf is not None:
            meets_target = classification.get('confidence', 0) >= target_conf
            print(f"Target Confidence Threshold: {target_conf:.1%} ({'met' if meets_target else 'not met'})")
        print(f"\nAssessment: {classification.get('assessment', 'Analysis incomplete')}")

        print(f"\nNational Security Indicators Found: {classification.get('total_indicators', 0)}")
        if classification.get('indicators'):
            print(f"  - {', '.join(classification['indicators'][:10])}")

        print(f"\nLegal Frameworks Identified: {classification.get('total_frameworks', 0)}")
        if classification.get('legal_frameworks'):
            print(f"  - {', '.join(classification['legal_frameworks'])}")

        print("\n" + "-"*70)
        print("Full Analysis Document Preview:")
        print("-"*70)
        analysis_doc = results.get("analysis_document", "")
        print(analysis_doc[:1000] + "..." if len(analysis_doc) > 1000 else analysis_doc)
    else:
        print("\n" + "="*70)
        print("ERROR - Analysis Failed")
        print("="*70)
        print(f"Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
