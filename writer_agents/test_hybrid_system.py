#!/usr/bin/env python3
"""
Test script for Hybrid SK-AutoGen Writing System.

This script tests the integration between Semantic Kernel and AutoGen
for legal document drafting.
"""

import asyncio
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add writer_agents to path
import sys
sys.path.append(str(Path(__file__).parent / "code"))

from sk_config import create_sk_kernel
from sk_plugins.DraftingPlugin.privacy_harm_function import PrivacyHarmPlugin
from WorkflowOrchestrator import Conductor as WorkflowOrchestrator
from workflow_config import WorkflowStrategyConfig
from insights import CaseInsights


async def test_sk_kernel_creation():
    """Test Semantic Kernel creation and basic functionality."""
    logger.info("Testing SK Kernel creation...")

    try:
        kernel = create_sk_kernel()
        logger.info("✅ SK Kernel created successfully")
        return kernel
    except Exception as e:
        logger.error(f"❌ SK Kernel creation failed: {e}")
        raise


async def test_privacy_harm_plugin(kernel):
    """Test Privacy Harm plugin registration and execution."""
    logger.info("Testing Privacy Harm plugin...")

    try:
        # Register plugin
        plugin = PrivacyHarmPlugin(kernel)
        await plugin.initialize()
        logger.info("✅ Privacy Harm plugin initialized")

        # Test with sample data
        test_evidence = {
            "OGC_Email_Apr18_2025": "Sent",
            "PRC_Awareness": "Direct",
            "Privacy_Violation": "Confirmed"
        }

        test_posteriors = {
            "PrivacyHarm": 0.85,
            "Causation": 0.78,
            "LegalSuccess": 0.72
        }

        test_summary = "Case involving privacy violations and data exposure"

        # Test native function
        native_result = await plugin.native_function.execute(
            evidence=test_evidence,
            posteriors=test_posteriors,
            case_summary=test_summary,
            jurisdiction="US"
        )

        if native_result.success:
            logger.info("✅ Privacy Harm native function executed successfully")
            logger.info(f"Generated section: {native_result.value.title}")
            logger.info(f"Word count: {native_result.value.word_count}")
        else:
            logger.error(f"❌ Native function failed: {native_result.error}")

        # Test semantic function
        semantic_result = await plugin.semantic_function.execute(
            kernel=kernel,
            evidence=test_evidence,
            posteriors=test_posteriors,
            case_summary=test_summary,
            jurisdiction="US"
        )

        if semantic_result.success:
            logger.info("✅ Privacy Harm semantic function executed successfully")
            logger.info(f"Generated content length: {len(semantic_result.value)} characters")
        else:
            logger.error(f"❌ Semantic function failed: {semantic_result.error}")

        return True

    except Exception as e:
        logger.error(f"❌ Privacy Harm plugin test failed: {e}")
        raise


async def test_hybrid_orchestrator():
    """Test the complete Hybrid Orchestrator workflow."""
    logger.info("Testing Hybrid Orchestrator...")

    try:
        # Create test insights
        insights = CaseInsights(
            summary="Test case involving privacy harm analysis",
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

        # Create hybrid orchestrator
        config = HybridOrchestratorConfig()
        orchestrator = HybridOrchestrator(config)

        try:
            # Run hybrid workflow
            result = await orchestrator.run_hybrid_workflow(insights)

            logger.info("✅ Hybrid workflow completed successfully")
            logger.info(f"Workflow type: {result.metadata.get('workflow_type', 'unknown')}")
            logger.info(f"Document length: {len(result.edited_document)} characters")
            logger.info(f"Sections generated: {len(result.sections)}")

            # Print sample output
            if result.sections:
                sample_section = result.sections[0]
                logger.info(f"Sample section title: {sample_section.title}")
                logger.info(f"Sample section preview: {sample_section.body[:200]}...")

            return True

        finally:
            await orchestrator.close()

    except Exception as e:
        logger.error(f"❌ Hybrid Orchestrator test failed: {e}")
        raise


async def test_enhanced_orchestrator_integration():
    """Test integration with Enhanced Orchestrator."""
    logger.info("Testing Enhanced Orchestrator integration...")

    try:
        from writer_agents.code.EnhancedOrchestrator import EnhancedWriterOrchestrator, EnhancedOrchestratorConfig

        # Create test insights that should trigger hybrid_sk workflow
        insights = CaseInsights(
            summary="Motion for sealing involving privacy harm",
            evidence={
                "OGC_Email_Apr18_2025": "Sent",
                "PRC_Awareness": "Direct"
            },
            posteriors={
                "PrivacyHarm": 0.85,
                "Causation": 0.78
            },
            jurisdiction="US",
            case_style="Motion for Sealing"  # This should trigger hybrid_sk
        )

        # Create enhanced orchestrator with SK hybrid enabled
        config = EnhancedOrchestratorConfig()
        config.enable_sk_hybrid = True

        orchestrator = EnhancedWriterOrchestrator(config)

        try:
            # Run intelligent workflow
            result = await orchestrator.run_intelligent_workflow(insights)

            logger.info("✅ Enhanced Orchestrator integration successful")
            logger.info(f"Selected workflow: {result.metadata.get('workflow_type', 'unknown')}")
            logger.info(f"Complexity score: {result.metadata.get('complexity_score', 'unknown')}")

            return True

        finally:
            await orchestrator.close()

    except Exception as e:
        logger.error(f"❌ Enhanced Orchestrator integration test failed: {e}")
        raise


async def main():
    """Run all tests."""
    logger.info("🚀 Starting Hybrid SK-AutoGen Writing System Tests")
    logger.info("=" * 60)

    try:
        # Test 1: SK Kernel Creation
        kernel = await test_sk_kernel_creation()

        # Test 2: Privacy Harm Plugin
        await test_privacy_harm_plugin(kernel)

        # Test 3: Hybrid Orchestrator
        await test_hybrid_orchestrator()

        # Test 4: Enhanced Orchestrator Integration
        await test_enhanced_orchestrator_integration()

        logger.info("=" * 60)
        logger.info("🎉 All tests passed successfully!")
        logger.info("The Hybrid SK-AutoGen Writing System is ready for use.")

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"💥 Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
