"""Behavioral tests for ValidationPlugin and MotionSectionsPlugin helpers."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from writer_agents.code.sk_plugins.ValidationPlugin.validation_functions import (
    EvidenceGroundingValidatorFunction,
    ToneConsistencySemanticFunction,
)
from writer_agents.code.sk_plugins.DraftingPlugin import motion_sections_plugin
from writer_agents.code.sk_plugins.DraftingPlugin import privacy_harm_function as privacy_module
from writer_agents.code.sk_plugins.DraftingPlugin import factual_timeline_function as factual_module
from writer_agents.code.sk_plugins.DraftingPlugin import causation_analysis_function as causation_module
from writer_agents.code.sk_plugins.DraftingPlugin.privacy_harm_function import (
    PrivacyHarmSemanticFunction,
)
from writer_agents.code.sk_plugins.DraftingPlugin.factual_timeline_function import (
    FactualTimelineSemanticFunction,
)
from writer_agents.code.sk_plugins.DraftingPlugin.causation_analysis_function import (
    CausationAnalysisSemanticFunction,
)
from writer_agents.code.sk_plugins.FeaturePlugin.base_feature_plugin import BaseFeaturePlugin
from writer_agents.code.sk_plugins.utils import extract_prompt_text


def test_evidence_grounding_normalizes_dict_and_list_inputs():
    """EvidenceGrounding validator should treat dict and list payloads consistently."""
    validator = EvidenceGroundingValidatorFunction()

    dict_payload = {"Node:MA": "Sensitive medical records"}
    list_payload = [
        {"node_id": "Node:MA", "description": "Sensitive medical records"},
        {"id": "Node:CA", "text": "Financial disclosures"},
    ]

    normalized_dict = validator._normalize_evidence_input(dict_payload)
    normalized_list = validator._normalize_evidence_input(list_payload)

    assert normalized_dict == {"Node:MA": "Sensitive medical records"}
    assert normalized_list["Node:MA"] == "Sensitive medical records"
    assert "Node:CA" in normalized_list
    assert all(isinstance(v, str) for v in normalized_list.values())


def test_evidence_grounding_validation_scores_partial_matches():
    """Evidence grounding score should reflect how many nodes appear in the document."""
    validator = EvidenceGroundingValidatorFunction()
    evidence = {"Node:MA": "Fact A", "Node:CA": "Fact B"}
    document = "Only Node:MA is cited in this draft."

    result = validator._validate_evidence_grounding(document, evidence)

    assert result.score == 0.5
    assert result.passed is False
    assert any("Include more evidence" in suggestion for suggestion in result.suggestions)


@pytest.mark.asyncio
async def test_tone_consistency_semantic_invokes_prompt_function(monkeypatch):
    """Tone consistency semantic validator should use KernelFunctionFromPrompt helpers."""
    func = ToneConsistencySemanticFunction()

    class StubPromptFunction:
        def __init__(self):
            self.invocations = 0

        async def invoke(self, kernel, arguments):
            self.invocations += 1
            assert "document" in arguments
            payload = {
                "passed": True,
                "score": 0.95,
                "details": "Professional tone detected",
                "suggestions": [],
                "errors": [],
            }
            return SimpleNamespace(value=json.dumps(payload), usage_metadata={"input_tokens": 42})

    stub_prompt = StubPromptFunction()
    monkeypatch.setattr(func, "_get_prompt_function", lambda: stub_prompt)

    class DummyKernel:
        pass

    result = await func.execute(kernel=DummyKernel(), document="Sample motion text.")

    assert result.success
    assert result.value.passed is True
    assert stub_prompt.invocations == 1


@pytest.mark.asyncio
async def test_motion_section_semantic_uses_prompt_class_when_kernel_lacks_factory(monkeypatch):
    """Motion section drafting should fall back to KernelFunctionFromPrompt when needed."""
    captured = {}

    class StubPromptFunction:
        def __init__(self, *args, **kwargs):
            captured["init"] = {"args": args, "kwargs": kwargs}

        async def invoke(self, kernel, arguments):
            captured["arguments"] = arguments
            return SimpleNamespace(value=" Draft output ", usage_metadata=None)

    monkeypatch.setattr(
        motion_sections_plugin,
        "KernelFunctionFromPrompt",
        lambda *args, **kwargs: StubPromptFunction(*args, **kwargs),
    )
    monkeypatch.setattr(
        motion_sections_plugin,
        "KernelArguments",
        lambda **kwargs: kwargs,
    )

    class DummyKernel:
        """Kernel stub without create_function_from_prompt to trigger fallback."""

        pass

    section_func = motion_sections_plugin.MotionSectionSemanticFunction(
        section_key="introduction",
        prompt_template="Prompt body",
    )

    result = await section_func.execute(
        kernel=DummyKernel(),
        evidence=[],
        posteriors={},
        case_summary="Case summary",
        jurisdiction="US",
        research_findings={},
        quality_constraints="Apply all constraints",
        structured_facts="Fact block",
        autogen_notes="Notes",
    )

    assert result.success
    assert result.value == "Draft output"
    assert "init" in captured
    assert captured["init"]["kwargs"]["function_name"].startswith("DraftIntroduction")
    assert "quality_constraints" in captured["arguments"]


@pytest.mark.asyncio
@pytest.mark.xfail(reason="Tone validation still needs ChatMessageContent parsing fix (Agent 1)", strict=False)
async def test_tone_consistency_handles_chat_message_payload(monkeypatch):
    """Tone validator should parse ChatMessageContent payloads returned as lists."""
    func = ToneConsistencySemanticFunction()

    class StubPromptFunction:
        async def invoke(self, kernel, arguments):
            payload = {
                "passed": True,
                "score": 0.9,
                "details": "Consistent tone detected",
                "suggestions": [],
                "errors": [],
            }
            message = SimpleNamespace(text=json.dumps(payload))
            return SimpleNamespace(value=[message], usage_metadata=None)

    monkeypatch.setattr(func, "_get_prompt_function", lambda: StubPromptFunction())

    class DummyKernel:
        pass

    result = await func.execute(kernel=DummyKernel(), document="Formal filing.")
    assert result.success
    assert result.value.passed is True


@pytest.mark.asyncio
async def test_feature_plugin_registers_kernel_functions(tmp_path):
    """Feature plugins should register kernel_function-decorated callables."""

    class DummyKernel:
        def __init__(self):
                self.plugins = {}

        def add_plugin(self, plugin=None, plugin_name=None, **_kwargs):
            self.plugins[plugin_name] = plugin or {}
            return SimpleNamespace(functions=self.plugins[plugin_name])

    class DummyFeaturePlugin(BaseFeaturePlugin):
        def __init__(self, kernel):
            super().__init__(kernel, "dummy_feature", chroma_store=None, rules_dir=tmp_path)

        def _initialize_database_access(self, *args, **kwargs):
            # Skip expensive DB wiring for tests
            self.db_paths = []
            self.sqlite_searcher = None
            self.langchain_agent = None
            self.courtlistener_searcher = None
            self.storm_researcher = None

        async def query_chroma(self, case_context: str, **kwargs):
            return []

        async def extract_patterns(self, results, **kwargs):
            return []

        async def generate_argument(self, patterns, case_context: str, **kwargs):
            return "argument"

    kernel = DummyKernel()
    plugin = DummyFeaturePlugin(kernel)
    await plugin._register_functions()

    plugin_name = plugin.metadata.name
    assert plugin_name in kernel.plugins, "Kernel should hold registered plugin functions"
    for func in kernel.plugins[plugin_name].values():
        assert hasattr(func, "__kernel_function_name__"), "Functions must be decorated with kernel_function"


def test_extract_prompt_text_handles_nested_messages():
    class Item:
        def __init__(self, text):
            self.text = text

    class Message:
        def __init__(self, text):
            self.items = [Item(text)]

    payload = [Message("Resolved output")]
    assert extract_prompt_text(payload) == "Resolved output"


@pytest.mark.asyncio
async def test_privacy_harm_semantic_uses_prompt_text(monkeypatch):
    func = PrivacyHarmSemanticFunction()

    class StubPrompt:
        def __init__(self, *args, **kwargs):
            pass

        async def invoke(self, kernel, arguments):
            message = SimpleNamespace(items=[SimpleNamespace(text="Privacy draft")])
            return SimpleNamespace(value=[message], usage_metadata=None)

    monkeypatch.setattr(privacy_module, "KernelFunctionFromPrompt", StubPrompt)

    class DummyKernel:
        pass

    result = await func.execute(
        kernel=DummyKernel(),
        evidence=[],
        posteriors={},
        case_summary="Summary",
    )

    assert result.success
    assert result.value == "Privacy draft"


@pytest.mark.asyncio
async def test_factual_timeline_semantic_uses_prompt_text(monkeypatch):
    func = FactualTimelineSemanticFunction()

    class StubPrompt:
        def __init__(self, *args, **kwargs):
            pass

        async def invoke(self, kernel, arguments):
            message = SimpleNamespace(text="Timeline draft")
            return SimpleNamespace(value=message, usage_metadata=None)

    monkeypatch.setattr(factual_module, "KernelFunctionFromPrompt", StubPrompt)

    class DummyKernel:
        pass

    result = await func.execute(
        kernel=DummyKernel(),
        evidence=[],
        posteriors={},
        case_summary="Summary",
    )

    assert result.success
    assert result.value == "Timeline draft"


@pytest.mark.asyncio
async def test_causation_analysis_semantic_uses_prompt_text(monkeypatch):
    func = CausationAnalysisSemanticFunction()

    class StubPrompt:
        def __init__(self, *args, **kwargs):
            pass

        async def invoke(self, kernel, arguments):
            message = SimpleNamespace(items=[SimpleNamespace(text="Causation draft")])
            return SimpleNamespace(value=[message], usage_metadata=None)

    monkeypatch.setattr(causation_module, "KernelFunctionFromPrompt", StubPrompt)

    class DummyKernel:
        pass

    result = await func.execute(
        kernel=DummyKernel(),
        evidence=[],
        posteriors={},
        case_summary="Summary",
    )

    assert result.success
    assert result.value == "Causation draft"
