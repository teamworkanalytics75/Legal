#!/usr/bin/env python3
"""
Demo script for Hybrid SK-AutoGen Writing System.

This script demonstrates how to use the hybrid orchestration system
for legal document drafting.
"""

import asyncio
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add writer_agents to path
import sys
sys.path.append(str(Path(__file__).parent / "code"))

from EnhancedOrchestrator import EnhancedWriterOrchestrator, EnhancedOrchestratorConfig
from insights import CaseInsights


async def demo_privacy_harm_analysis():
    """Demonstrate privacy harm analysis using hybrid workflow."""

    logger.info("🔍 Privacy Harm Analysis Demo")
    logger.info("=" * 50)

    # Create realistic case insights
    insights = CaseInsights(
        summary="""
        This case involves allegations that Harvard University's April 2019 alumni statements
        contributed to privacy violations and data exposure. The plaintiff alleges that
        Harvard's public statements about student safety led to increased scrutiny and
        potential privacy harm through unauthorized data access and disclosure.
        """,
        evidence={
            "OGC_Email_Apr18_2025": "Sent",
            "PRC_Awareness": "Direct",
            "Privacy_Violation": "Confirmed",
            "Data_Exposure": "Personal_Information",
            "Unauthorized_Access": "Student_Records"
        },
        posteriors={
            "PrivacyHarm": 0.85,
            "Causation": 0.78,
            "LegalSuccess": 0.72,
            "DataBreach": 0.68,
            "InstitutionalLiability": 0.75
        },
        jurisdiction="US",
        case_style="Motion for Sealing"  # This triggers hybrid_sk workflow
    )

    # Create enhanced orchestrator with hybrid SK enabled
    config = EnhancedOrchestratorConfig()
    config.enable_sk_hybrid = True
    config.enable_performance_monitoring = True

    orchestrator = EnhancedWriterOrchestrator(config)

    try:
        logger.info("🚀 Starting hybrid workflow...")

        # Run the intelligent workflow
        result = await orchestrator.run_intelligent_workflow(insights)

        # Display results
        logger.info("✅ Workflow completed successfully!")
        logger.info(f"📊 Selected workflow: {result.metadata.get('workflow_type', 'unknown')}")
        logger.info(f"📈 Complexity score: {result.metadata.get('complexity_score', 'unknown')}")

        if 'performance_metrics' in result.metadata:
            metrics = result.metadata['performance_metrics']
            logger.info(f"⏱️  Execution time: {metrics.get('execution_time', 'unknown')}s")
            logger.info(f"🤖 Agent interactions: {metrics.get('agent_interactions', 'unknown')}")

        # Display the generated document
        logger.info("\n📄 Generated Document:")
        logger.info("-" * 50)
        print(result.edited_document)
        logger.info("-" * 50)

        # Display section details
        if result.sections:
            logger.info(f"\n📋 Generated Sections ({len(result.sections)}):")
            for i, section in enumerate(result.sections, 1):
                logger.info(f"  {i}. {section.title} ({len(section.body)} chars)")

        # Display reviews
        if result.reviews:
            logger.info(f"\n🔍 Reviews ({len(result.reviews)}):")
            for review in result.reviews:
                logger.info(f"  - [{review.severity}] {review.message}")

        return result

    finally:
        await orchestrator.close()


async def demo_workflow_comparison():
    """Demonstrate different workflow types for comparison."""

    logger.info("\n🔄 Workflow Comparison Demo")
    logger.info("=" * 50)

    # Create test case
    insights = CaseInsights(
        summary="Simple privacy case for workflow comparison",
        evidence={"PrivacyViolation": "Confirmed"},
        posteriors={"PrivacyHarm": 0.7},
        jurisdiction="US",
        case_style="Memorandum"  # This should trigger traditional workflow
    )

    config = EnhancedOrchestratorConfig()
    config.enable_sk_hybrid = True
    config.enable_performance_monitoring = True

    orchestrator = EnhancedWriterOrchestrator(config)

    try:
        result = await orchestrator.run_intelligent_workflow(insights)

        logger.info(f"📊 Workflow selected: {result.metadata.get('workflow_type', 'unknown')}")
        logger.info(f"📈 Complexity score: {result.metadata.get('complexity_score', 'unknown')}")

        if 'performance_metrics' in result.metadata:
            metrics = result.metadata['performance_metrics']
            logger.info(f"⏱️  Execution time: {metrics.get('execution_time', 'unknown')}s")

        return result

    finally:
        await orchestrator.close()


async def demo_quality_gates():
    """Demonstrate quality gate validation."""

    logger.info("\n🛡️  Quality Gates Demo")
    logger.info("=" * 50)

    # Create case that should trigger hybrid_sk workflow
    insights = CaseInsights(
        summary="Complex privacy harm case requiring structured output",
        evidence={
            "OGC_Email_Apr18_2025": "Sent",
            "PRC_Awareness": "Direct",
            "Privacy_Violation": "Confirmed"
        },
        posteriors={
            "PrivacyHarm": 0.85,
            "Causation": 0.78,
            "LegalSuccess": 0.72
        },
        jurisdiction="US",
        case_style="Motion for Sealing"
    )

    config = EnhancedOrchestratorConfig()
    config.enable_sk_hybrid = True
    config.enable_performance_monitoring = True

    orchestrator = EnhancedWriterOrchestrator(config)

    try:
        result = await orchestrator.run_intelligent_workflow(insights)

        # Check if validation results are available
        if 'validation_results' in result.metadata:
            validation = result.metadata['validation_results']
            logger.info("🛡️  Quality Gate Results:")
            logger.info(f"  Overall Score: {validation.get('overall_score', 'unknown')}")
            logger.info(f"  Passed Gates: {len(validation.get('passed_gates', []))}")
            logger.info(f"  Failed Gates: {len(validation.get('failed_gates', []))}")

            if validation.get('failed_gates'):
                logger.info(f"  Failed: {', '.join(validation['failed_gates'])}")

        return result

    finally:
        await orchestrator.close()


async def main():
    """Run all demos."""

    logger.info("🎯 Hybrid SK-AutoGen Writing System Demo")
    logger.info("=" * 60)
    logger.info("This demo shows the hybrid orchestration system in action.")
    logger.info("The system combines AutoGen for exploration with Semantic Kernel for structured drafting.")
    logger.info("=" * 60)

    try:
        # Demo 1: Privacy Harm Analysis
        await demo_privacy_harm_analysis()

        # Demo 2: Workflow Comparison
        await demo_workflow_comparison()

        # Demo 3: Quality Gates
        await demo_quality_gates()

        logger.info("\n" + "=" * 60)
        logger.info("🎉 Demo completed successfully!")
        logger.info("The Hybrid SK-AutoGen Writing System is working correctly.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error(f"💥 Demo failed: {e}")
        logger.error("Check your OpenAI API key and dependencies.")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    asyncio.run(main())
