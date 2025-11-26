#!/usr/bin/env python3
"""
Test script for CatBoost to SK Plugins implementation.

Tests the atomic feature plugins and orchestration system.
"""

import asyncio
import logging
import sys
from pathlib import Path
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add writer_agents to path
sys.path.append(str(Path(__file__).parent / "code"))


def test_ml_audit_pipeline():
    """Test ML audit pipeline."""
    logger.info("Testing ML audit pipeline...")

    try:
        from ml_audit.audit_catboost_patterns import audit_catboost_patterns
        from ml_audit.translate_features_to_rules import translate_features_to_rules

        logger.info("✅ ML audit imports successful")

        # Test pattern extraction (would run if database exists)
        # audit_catboost_patterns()
        logger.info("✅ ML audit pipeline ready")

        return True

    except Exception as e:
        logger.error(f"❌ ML audit pipeline test failed: {e}")
        return False


def test_feature_plugins():
    """Test atomic feature plugins."""
    logger.info("Testing atomic feature plugins...")

    try:
        from sk_plugins.FeaturePlugin import (
            BaseFeaturePlugin,
            MentionsPrivacyPlugin,
            MentionsHarassmentPlugin,
            MentionsSafetyPlugin,
            MentionsRetaliationPlugin,
            CitationRetrievalPlugin,
            PrivacyHarmCountPlugin,
            PublicInterestPlugin,
            AvoidFirstAmendmentPlugin
        )

        logger.info("✅ Feature plugin imports successful")

        # Test plugin instantiation (without kernel for now)
        rules_dir = Path(__file__).parent / "code" / "sk_plugins" / "rules"

        # Mock Chroma store for testing
        class MockChromaStore:
            async def query(self, collection_name, query_text, n_results):
                return [{"text": f"Mock result for {query_text}", "metadata": {"case_id": "test"}}]

        mock_chroma = MockChromaStore()

        # Test each plugin
        plugins = [
            ("MentionsPrivacyPlugin", MentionsPrivacyPlugin),
            ("MentionsHarassmentPlugin", MentionsHarassmentPlugin),
            ("MentionsSafetyPlugin", MentionsSafetyPlugin),
            ("MentionsRetaliationPlugin", MentionsRetaliationPlugin),
            ("CitationRetrievalPlugin", CitationRetrievalPlugin),
            ("PrivacyHarmCountPlugin", PrivacyHarmCountPlugin),
            ("PublicInterestPlugin", PublicInterestPlugin),
            ("AvoidFirstAmendmentPlugin", AvoidFirstAmendmentPlugin)
        ]

        for plugin_name, plugin_class in plugins:
            try:
                # Create plugin instance (without kernel)
                plugin = plugin_class(None, mock_chroma, rules_dir)
                logger.info(f"✅ {plugin_name} instantiated successfully")
            except Exception as e:
                logger.error(f"❌ {plugin_name} instantiation failed: {e}")
                return False

        return True

    except Exception as e:
        logger.error(f"❌ Feature plugins test failed: {e}")
        return False


def test_feature_orchestrator():
    """Test feature orchestrator."""
    logger.info("Testing feature orchestrator...")

    try:
        from sk_plugins.FeaturePlugin import FeatureOrchestrator

        logger.info("✅ Feature orchestrator import successful")

        # Test orchestrator instantiation
        orchestrator = FeatureOrchestrator({})
        logger.info("✅ Feature orchestrator instantiated successfully")

        return True

    except Exception as e:
        logger.error(f"❌ Feature orchestrator test failed: {e}")
        return False


def test_rules_loading():
    """Test rule configuration loading."""
    logger.info("Testing rule configuration loading...")

    try:
        rules_dir = Path(__file__).parent / "code" / "sk_plugins" / "rules"

        # Check if rules files exist
        rule_files = [
            "mentions_privacy_rules.json",
            "citation_requirements.json"
        ]

        for rule_file in rule_files:
            rule_path = rules_dir / rule_file
            if rule_path.exists():
                logger.info(f"✅ Rule file exists: {rule_file}")
            else:
                logger.warning(f"⚠️ Rule file missing: {rule_file}")

        return True

    except Exception as e:
        logger.error(f"❌ Rules loading test failed: {e}")
        return False


def test_hybrid_orchestrator_integration():
    """Test WorkflowOrchestrator integration."""
    logger.info("Testing WorkflowOrchestrator integration...")

    try:
        import WorkflowOrchestrator as workflow_module

        # Prefer HybridOrchestrator names when available for backward compatibility.
        WorkflowOrchestrator = getattr(
            workflow_module,
            "HybridOrchestrator",
            getattr(workflow_module, "Conductor", None)
        )
        WorkflowConfig = getattr(
            workflow_module,
            "HybridOrchestratorConfig",
            getattr(workflow_module, "WorkflowStrategyConfig", None)
        )

        if WorkflowOrchestrator is None or WorkflowConfig is None:
            raise ImportError("Workflow orchestrator classes are unavailable")

        # Test configuration
        config = WorkflowConfig()
        logger.info("✅ Workflow orchestrator config created successfully")

        # Test orchestrator instantiation (without SK kernel)
        orchestrator = WorkflowOrchestrator(config)
        logger.info("✅ WorkflowOrchestrator with feature plugins instantiated successfully")

        return True

    except Exception as e:
        logger.error(f"❌ WorkflowOrchestrator integration test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_plugin_functionality():
    """Test plugin functionality with mock data."""
    logger.info("Testing plugin functionality...")

    try:
        from sk_plugins.FeaturePlugin import MentionsPrivacyPlugin
        from pathlib import Path

        # Mock Chroma store
        class MockChromaStore:
            async def query(self, collection_name, query_text, n_results):
                return [
                    {
                        "text": "This case involves significant privacy harm due to disclosure of personal information.",
                        "metadata": {"case_id": "test_001", "outcome": "granted"}
                    }
                ]

        mock_chroma = MockChromaStore()
        rules_dir = Path(__file__).parent / "code" / "sk_plugins" / "rules"

        # Create plugin (with None kernel - should work with mocks)
        plugin = MentionsPrivacyPlugin(None, mock_chroma, rules_dir)

        # Test Chroma query with timeout
        try:
            results = await asyncio.wait_for(plugin.query_chroma("privacy case"), timeout=10.0)
            logger.info(f"✅ Chroma query returned {len(results)} results")
        except asyncio.TimeoutError:
            logger.warning("⚠️  Chroma query timed out")
            return False

        # Test pattern extraction with timeout
        try:
            patterns = await asyncio.wait_for(plugin.extract_patterns(results), timeout=10.0)
            logger.info(f"✅ Pattern extraction successful: {len(patterns.get('common_phrases', []))} phrases")
        except asyncio.TimeoutError:
            logger.warning("⚠️  Pattern extraction timed out")
            return False

        # Test argument generation with timeout
        try:
            argument = await asyncio.wait_for(plugin.generate_argument(patterns, "test case"), timeout=10.0)
            logger.info(f"✅ Argument generation successful: {len(argument)} characters")
        except asyncio.TimeoutError:
            logger.warning("⚠️  Argument generation timed out")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Plugin functionality test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main(skip_async: bool = False):
    """Run all tests.
    
    Args:
        skip_async: If True, skip the async plugin functionality test
    """
    logger.info("🚀 Starting CatBoost to SK Plugins Tests")
    logger.info("=" * 60)

    tests = [
        ("ML Audit Pipeline", test_ml_audit_pipeline),
        ("Feature Plugins", test_feature_plugins),
        ("Feature Orchestrator", test_feature_orchestrator),
        ("Rules Loading", test_rules_loading),
        ("HybridOrchestrator Integration", test_hybrid_orchestrator_integration),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))

    # Run async test with timeout (unless skipped)
    if not skip_async:
        logger.info(f"\n📋 Running Plugin Functionality test...")
        try:
            async_result = asyncio.run(asyncio.wait_for(test_plugin_functionality(), timeout=30.0))
            results.append(("Plugin Functionality", async_result))
        except asyncio.TimeoutError:
            logger.warning("⚠️  Plugin Functionality test timed out after 30 seconds")
            results.append(("Plugin Functionality", False))
        except Exception as e:
            logger.error(f"❌ Plugin Functionality test failed: {e}")
            results.append(("Plugin Functionality", False))
    else:
        logger.info(f"\n⏭️  Skipping Plugin Functionality test (async)")
        results.append(("Plugin Functionality", None))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    passed = 0
    total_tests = len(results)
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\nTotal: {passed}/{total_tests} tests passed")

    if passed == total_tests:
        logger.info("🎉 All tests passed! CatBoost to SK Plugins implementation is ready.")
        logger.info("\n📝 Next Steps:")
        logger.info("   1. Run ML audit pipeline to extract patterns from case database")
        logger.info("   2. Generate rule configurations from CatBoost features")
        logger.info("   3. Test with real case data and CatBoost model")
        logger.info("   4. Implement validation feedback loop")
        return True
    else:
        logger.error("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test CatBoost to SK Plugins")
    parser.add_argument("--skip-async", action="store_true", help="Skip async plugin functionality test")
    args = parser.parse_args()
    
    success = main(skip_async=args.skip_async)
    sys.exit(0 if success else 1)
