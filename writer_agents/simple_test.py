#!/usr/bin/env python3
"""
Simple test script for Hybrid SK-AutoGen Writing System.

This script tests basic functionality without complex imports.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add writer_agents to path
sys.path.append(str(Path(__file__).parent / "code"))

def test_sk_config():
    """Test Semantic Kernel configuration."""
    logger.info("Testing SK configuration...")

    try:
        from sk_config import create_sk_kernel, SKConfig
        logger.info("✅ SK config import successful")

        # Test kernel creation (without API key for now)
        try:
            config = SKConfig()
            logger.info("✅ SK config creation successful")
        except Exception as e:
            if "API key" in str(e) or "OPENAI_API_KEY" in str(e):
                logger.info("✅ SK config creation failed as expected (no API key)")
                return True
            else:
                raise

        # Test that we can create a config without API key
        try:
            # This should fail gracefully without API key
            kernel = create_sk_kernel(config)
            logger.info("✅ SK kernel creation successful")
        except Exception as e:
            if "API key" in str(e) or "OPENAI_API_KEY" in str(e):
                logger.info("✅ SK kernel creation failed as expected (no API key)")
            else:
                raise

        return True
    except Exception as e:
        logger.error(f"❌ SK config test failed: {e}")
        return False


def test_base_plugin():
    """Test base plugin classes."""
    logger.info("Testing base plugin classes...")

    try:
        from sk_plugins.base_plugin import BaseSKPlugin, PluginMetadata, FunctionResult
        logger.info("✅ Base plugin import successful")

        # Test metadata creation
        metadata = PluginMetadata(
            name="TestPlugin",
            description="Test plugin",
            version="1.0.0",
            functions=["test_function"]
        )
        logger.info("✅ Plugin metadata creation successful")

        return True
    except Exception as e:
        logger.error(f"❌ Base plugin test failed: {e}")
        return False


def test_privacy_harm_plugin():
    """Test privacy harm plugin."""
    logger.info("Testing privacy harm plugin...")

    try:
        from sk_plugins.DraftingPlugin.privacy_harm_function import PrivacyHarmPlugin
        logger.info("✅ Privacy harm plugin import successful")

        return True
    except Exception as e:
        logger.error(f"❌ Privacy harm plugin test failed: {e}")
        return False


def test_insights():
    """Test insights module."""
    logger.info("Testing insights module...")

    try:
        from insights import CaseInsights
        logger.info("✅ Insights import successful")

        # Test insights creation
        insights = CaseInsights(
            reference_id="test_001",
            summary="Test case",
            posteriors=[],
            evidence=[],
            jurisdiction="US",
            case_style="Test Case"
        )
        logger.info("✅ Insights creation successful")

        return True
    except Exception as e:
        logger.error(f"❌ Insights test failed: {e}")
        return False


def test_tasks():
    """Test tasks module."""
    logger.info("Testing tasks module...")

    try:
        from tasks import WriterDeliverable, DraftSection, PlanDirective
        logger.info("✅ Tasks import successful")

        # Test task creation
        plan = PlanDirective(
            objective="Test objective",
            deliverable_format="Test format",
            tone="Test tone"
        )
        section = DraftSection(
            section_id="test",
            title="Test Section",
            body="Test content"
        )
        deliverable = WriterDeliverable(
            plan=plan,
            sections=[section],
            edited_document="Test document",
            reviews=[],
            metadata={}
        )
        logger.info("✅ Task creation successful")

        return True
    except Exception as e:
        logger.error(f"❌ Tasks test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting Hybrid SK-AutoGen System Tests")
    logger.info("=" * 60)

    tests = [
        ("SK Configuration", test_sk_config),
        ("Base Plugin Classes", test_base_plugin),
        ("Privacy Harm Plugin", test_privacy_harm_plugin),
        ("Insights Module", test_insights),
        ("Tasks Module", test_tasks),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nTotal: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        logger.info("🎉 All tests passed! The system is ready for use.")
        return True
    else:
        logger.error("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
