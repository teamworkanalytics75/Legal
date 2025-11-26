import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Provide lightweight stubs for modules that WorkflowStrategyExecutor expects during import.
if "agents" not in sys.modules:  # pragma: no cover - import shim
    agents_stub = types.ModuleType("agents")

    class _DummyAgentFactory:
        def __init__(self, *_args, **_kwargs):
            pass

        def create(self, *args, **kwargs):
            return None

    class _DummyModelConfig:
        pass

    class _DummyBaseAutoGenAgent:
        async def run(self, *args, **kwargs):
            return types.SimpleNamespace(content="")

    agents_stub.AgentFactory = _DummyAgentFactory
    agents_stub.ModelConfig = _DummyModelConfig
    agents_stub.BaseAutoGenAgent = _DummyBaseAutoGenAgent
    sys.modules["agents"] = agents_stub

if "insights" not in sys.modules:  # pragma: no cover - import shim
    insights_stub = types.ModuleType("insights")

    class _DummyCaseInsights:
        summary = ""
        jurisdiction = ""
        evidence = []
        posteriors = []
        case_style = ""

    insights_stub.CaseInsights = _DummyCaseInsights
    sys.modules["insights"] = insights_stub

if "tasks" not in sys.modules:  # pragma: no cover - import shim
    tasks_stub = types.ModuleType("tasks")

    class _DummyTask:
        def __init__(self, *args, **kwargs):
            pass

    tasks_stub.WriterDeliverable = _DummyTask
    tasks_stub.DraftSection = _DummyTask
    tasks_stub.PlanDirective = _DummyTask
    tasks_stub.ReviewFindings = _DummyTask
    sys.modules["tasks"] = tasks_stub

if "sk_config" not in sys.modules:  # pragma: no cover - import shim
    sk_config_stub = types.ModuleType("sk_config")

    class _DummySKConfig:
        def __init__(self, *args, **kwargs):
            pass

    def _create_sk_kernel(*_args, **_kwargs):
        return None

    sk_config_stub.SKConfig = _DummySKConfig
    sk_config_stub.create_sk_kernel = _create_sk_kernel
    sys.modules["sk_config"] = sk_config_stub

if "sk_plugins" not in sys.modules:  # pragma: no cover - import shim
    sk_plugins_stub = types.ModuleType("sk_plugins")

    class _DummyBaseSKPlugin:
        pass

    class _DummyPluginRegistry:
        def __init__(self):
            self._plugins = {}

        def get_plugin(self, name: str):
            return self._plugins.get(name)

        def register_plugin(self, name: str, plugin):
            self._plugins[name] = plugin

    sk_plugins_stub.BaseSKPlugin = _DummyBaseSKPlugin
    sk_plugins_stub.PluginRegistry = _DummyPluginRegistry
    sk_plugins_stub.plugin_registry = _DummyPluginRegistry()
    sys.modules["sk_plugins"] = sk_plugins_stub

    # Provide nested DraftingPlugin + FeaturePlugin modules referenced during import.
    drafting_pkg = types.ModuleType("sk_plugins.DraftingPlugin")
    privacy_module = types.ModuleType("sk_plugins.DraftingPlugin.privacy_harm_function")

    class _DummyPrivacyHarmPlugin:
        pass

    privacy_module.PrivacyHarmPlugin = _DummyPrivacyHarmPlugin
    drafting_pkg.privacy_harm_function = privacy_module
    sys.modules["sk_plugins.DraftingPlugin"] = drafting_pkg
    sys.modules["sk_plugins.DraftingPlugin.privacy_harm_function"] = privacy_module

    feature_pkg = types.ModuleType("sk_plugins.FeaturePlugin")
    case_facts_module = types.ModuleType("sk_plugins.FeaturePlugin.CaseFactsProvider")

    def _get_case_facts_provider():
        return None

    case_facts_module.get_case_facts_provider = _get_case_facts_provider
    feature_pkg.CaseFactsProvider = case_facts_module
    feature_orchestrator_module = types.ModuleType("sk_plugins.FeaturePlugin.feature_orchestrator")

    class _DummyRefinementLoop:
        pass

    feature_orchestrator_module.RefinementLoop = _DummyRefinementLoop
    feature_pkg.feature_orchestrator = feature_orchestrator_module
    sys.modules["sk_plugins.FeaturePlugin"] = feature_pkg
    sys.modules["sk_plugins.FeaturePlugin.CaseFactsProvider"] = case_facts_module
    sys.modules["sk_plugins.FeaturePlugin.feature_orchestrator"] = feature_orchestrator_module

from writer_agents.code.WorkflowStrategyExecutor import Conductor as ExecutorConductor


def _build_conductor(tmp_path: Path) -> ExecutorConductor:
    """Create a lightweight Conductor instance that skips heavy initialization."""
    conductor = ExecutorConductor.__new__(ExecutorConductor)  # type: ignore[call-arg]
    conductor._constraint_system_cache = {}
    conductor._constraint_format_cache = {}
    conductor._constraint_dir_override = tmp_path
    conductor.constraint_system_version = "test"
    conductor.feature_orchestrator = None
    conductor.config = SimpleNamespace(constraint_system_version="test")
    return conductor


@pytest.fixture
def constraint_dir(tmp_path: Path) -> Path:
    """Create a temporary constraint system directory with a baseline file."""
    data = {
        "document_level": {
            "word_count": {"min": 200, "max": 300, "ideal": 250, "unit": "words", "enforcement": "required"},
            "char_count": {"min": 1200, "max": 1800, "ideal": 1500, "unit": "chars", "enforcement": "priority"},
            "required_sections": ["introduction", "analysis", "conclusion"]
        },
        "sections": {
            "introduction": {
                "word_count": {"min": 50, "max": 80, "ideal": 65, "unit": "words", "enforcement": "priority"}
            }
        },
        "feature_constraints": {
            "high_impact": [
                {
                    "feature": "mentions_confidentiality",
                    "constraint": {"min": 3, "max": 5, "ideal": 4, "unit": "mentions"},
                    "enforcement": "required",
                    "description": "Mention confidentiality safeguards"
                }
            ]
        }
    }
    (tmp_path / "vtest_base.json").write_text(json.dumps(data))
    return tmp_path


def test_load_constraint_system_handles_missing_file(tmp_path: Path):
    conductor = _build_conductor(tmp_path)
    result = conductor._load_constraint_system("missing")
    assert result == {}


def test_format_quality_constraints_includes_icons_and_cached_blocks(constraint_dir: Path):
    conductor = _build_conductor(constraint_dir)
    text = conductor._format_quality_constraints({})

    assert "❌ Total word count: 200-300 words (ideal: 250)" in text
    assert "⚠️ Required sections: introduction, analysis, conclusion" in text
    assert "CONSTRAINT-SYSTEM FEATURE GUARDRAILS" in text
    assert conductor._constraint_format_cache, "Static constraint text should be cached"


def test_format_constraint_violation_lines_include_icons():
    conductor = _build_conductor(Path("/tmp"))
    violations = {
        "warnings": ["Word count below minimum"],
        "details": {
            "section_level": [
                {"section": "analysis", "message": "Missing balancing test"}
            ]
        }
    }

    lines = conductor._format_constraint_violation_lines(violations)
    assert any(line.startswith("  ⚠️") for line in lines)
    assert any("❌ Analysis: Missing balancing test" in line for line in lines)
