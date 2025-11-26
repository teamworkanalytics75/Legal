#!/usr/bin/env python3
"""
Test script for Hybrid SK-AutoGen Writing System Integration.

This script tests the orchestrator integration without requiring API keys.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add writer_agents to path
sys.path.append(str(Path(__file__).parent / "code"))

def test_imports():
    """Test that all orchestrator components can be imported."""
    logger.info("Testing orchestrator imports...")

    try:
        # Test basic imports
        from sk_config import SKConfig
        logger.info("✅ SK config import successful")

        from sk_plugins.base_plugin import BaseSKPlugin, PluginMetadata
        logger.info("✅ Base plugin import successful")

        from sk_plugins.DraftingPlugin.privacy_harm_function import PrivacyHarmPlugin
        logger.info("✅ Privacy harm plugin import successful")

        from insights import CaseInsights
        logger.info("✅ Insights import successful")

        from tasks import WriterDeliverable, DraftSection, PlanDirective
        logger.info("✅ Tasks import successful")

        return True

    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False


def test_hybrid_orchestrator():
    """Test HybridOrchestrator imports and basic functionality."""
    logger.info("Testing HybridOrchestrator...")

    try:
        import importlib

        # Dynamically resolve orchestrator + config to avoid stale imports
        orchestrator_module = importlib.import_module("WorkflowOrchestrator")
        HybridOrchestrator = getattr(orchestrator_module, "Conductor", None)
        HybridOrchestratorConfig = getattr(orchestrator_module, "WorkflowStrategyConfig", None)

        if HybridOrchestrator is None:
            raise AttributeError("Conductor class not found in WorkflowOrchestrator")
        if HybridOrchestratorConfig is None:
            raise AttributeError("WorkflowStrategyConfig not found in WorkflowOrchestrator")

        logger.info("✅ HybridOrchestrator import successful")

        # Test configuration creation
        hybrid_config = HybridOrchestratorConfig()
        logger.info("✅ WorkflowStrategyConfig created successfully")

        # Test orchestrator instantiation (will fail without API key, but that's expected)
        try:
            hybrid_orchestrator = HybridOrchestrator(hybrid_config)
            logger.info("✅ HybridOrchestrator instantiated successfully")
        except Exception as e:
            if "API key" in str(e) or "OPENAI_API_KEY" in str(e):
                logger.info("✅ HybridOrchestrator instantiation failed as expected (no API key)")
            else:
                raise

        return True

    except Exception as e:
        logger.error(f"❌ HybridOrchestrator test failed: {e}")
        return False


def test_insights_creation():
    """Test CaseInsights creation."""
    logger.info("Testing CaseInsights creation...")

    try:
        from insights import CaseInsights

        # Create test insights
        insights = CaseInsights(
            reference_id="test_001",
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

        logger.info("✅ CaseInsights created successfully")
        logger.info(f"   - Reference ID: {insights.reference_id}")
        logger.info(f"   - Summary: {insights.summary[:50]}...")
        logger.info(f"   - Evidence keys: {list(insights.evidence.keys())}")
        logger.info(f"   - Posterior keys: {list(insights.posteriors.keys())}")

        return True

    except Exception as e:
        logger.error(f"❌ CaseInsights creation test failed: {e}")
        return False


def test_plugin_structure():
    """Test plugin structure and metadata."""
    logger.info("Testing plugin structure...")

    try:
        from sk_plugins.base_plugin import PluginMetadata, FunctionResult

        # Test PluginMetadata creation
        metadata = PluginMetadata(
            name="TestPlugin",
            description="Test plugin for validation",
            version="1.0.0",
            functions=["test_function"]
        )
        logger.info("✅ PluginMetadata created successfully")

        # Test FunctionResult creation
        result = FunctionResult(
            success=True,
            value="Test result",
            error=None
        )
        logger.info("✅ FunctionResult created successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Plugin structure test failed: {e}")
        return False


def test_sk_config_structure():
    """Test SK configuration structure."""
    logger.info("Testing SK configuration structure...")

    try:
        from sk_config import SKConfig

        # Test that we can create a config (will fail without API key, but that's expected)
        try:
            config = SKConfig()
            logger.info("✅ SKConfig created successfully")
        except Exception as e:
            if "API key" in str(e) or "OPENAI_API_KEY" in str(e):
                logger.info("✅ SKConfig creation failed as expected (no API key)")
            else:
                raise

        return True

    except Exception as e:
        logger.error(f"❌ SK config structure test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    logger.info("🚀 Starting Hybrid SK-AutoGen Integration Tests")
    logger.info("=" * 60)

    tests = [
        ("Basic Imports", test_imports),
        ("HybridOrchestrator", test_hybrid_orchestrator),
        ("CaseInsights Creation", test_insights_creation),
        ("Plugin Structure", test_plugin_structure),
        ("SK Config Structure", test_sk_config_structure),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 INTEGRATION TEST RESULTS")
    logger.info("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nTotal: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        logger.info("🎉 All integration tests passed!")
        logger.info("The Hybrid SK-AutoGen system is properly integrated and ready for use.")
        logger.info("\n📝 Next Steps:")
        logger.info("   1. Set OPENAI_API_KEY environment variable")
        logger.info("   2. Run full workflow tests with API calls")
        logger.info("   3. Test with real case data")
        logger.info("   4. Fix EnhancedOrchestrator imports for advanced workflows")
        return True
    else:
        logger.error("⚠️  Some integration tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
