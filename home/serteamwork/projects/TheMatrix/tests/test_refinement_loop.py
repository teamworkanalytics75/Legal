import types
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from writer_agents.code.sk_plugins.FeaturePlugin.feature_orchestrator import RefinementLoop


@pytest.fixture()
def minimal_loop() -> RefinementLoop:
    """Return a RefinementLoop instance without running heavy initializers."""
    loop = object.__new__(RefinementLoop)
    loop.plugins = {}
    loop.catboost_model = None
    loop.model_path = None
    loop.shap_importance = {}
    loop.baseline_score = None
    loop.feature_targets = {}
    loop.label_encoder = None
    loop.debug_mode = False
    loop.outline_manager = None
    loop.plugin_calibrator = None
    loop.feature_extractor = None
    loop.catboost_predictor = SimpleNamespace(model=None, shap_importance={})
    loop.shap_analyzer = None
    loop.memory_manager = None
    loop._feature_to_plugin_map = {}
    loop._feature_target_lookup = {}
    loop._successful_case_averages = {}
    loop._debug_output_dir = None
    return loop


@pytest.mark.asyncio()
async def test_merge_improvements_uses_llm_result(monkeypatch: pytest.MonkeyPatch, minimal_loop: RefinementLoop) -> None:
    class DummyKernel:
        def __init__(self, service: Any) -> None:
            self._service = service

        def get_service(self, type_id: str | None = None) -> Any:
            return self._service

        @property
        def services(self) -> Dict[str, Any]:
            return {"default": self._service}

        def get_prompt_execution_settings_from_service_id(self, service_id: str) -> Dict[str, Any]:
            return {"temperature": 0}

    import semantic_kernel.connectors.ai as sk_ai

    class DummyChatCompletionBase:  # Dynamically injected base class
        pass

    monkeypatch.setattr(sk_ai, "ChatCompletionClientBase", DummyChatCompletionBase, raising=False)

    class DummyChatService(DummyChatCompletionBase):
        service_id = "dummy"

        async def get_chat_message_contents(self, chat_history: Any, settings: Any = None) -> List[Any]:
            class Response:
                content = "IMPROVED DRAFT WITH CITATIONS"

            return [Response()]

    minimal_loop.plugins = {"citation": SimpleNamespace(kernel=DummyKernel(DummyChatService()))}

    improvements = [
        {"feature": "citations", "argument": "Add Harvard v. Doe cite", "priority": "high", "type": "content"}
    ]

    merged = await minimal_loop._merge_improvements("Original draft", improvements)
    assert merged == "IMPROVED DRAFT WITH CITATIONS"
    assert "AI-Generated Improvements" not in merged


@pytest.mark.asyncio()
async def test_merge_improvements_fallback_includes_structural_guidance(minimal_loop: RefinementLoop) -> None:
    improvements = [
        {"feature": "privacy_balance", "argument": "Explain why privacy interests win", "priority": "medium", "type": "content"}
    ]
    structural = [{"feature": "paragraph_structure", "recommendation": "Use shorter paragraphs"}]

    merged = await minimal_loop._merge_improvements("Original draft", improvements, structural_features=structural)
    assert "AI-Generated Improvements" in merged
    assert "Structural Feature Recommendations" in merged
    assert "paragraph_structure" not in merged  # Confirm human-friendly casing applied
    assert "Paragraph Structure" in merged


@pytest.mark.asyncio()
async def test_feedback_loop_exits_when_no_weak_features(minimal_loop: RefinementLoop) -> None:
    async def fake_analyze(self: RefinementLoop, _: str) -> Dict[str, Any]:
        return {"weak_features": {}}

    async def fake_strengthen(self: RefinementLoop, draft: str, weak: Dict[str, Any], context: Dict[str, Any] | None = None) -> str:
        return draft

    async def fake_validate(self: RefinementLoop, draft: str) -> Dict[str, Any]:
        return {"confidence": 0.0, "improvement_percent": 0.0, "improved": False}

    minimal_loop.analyze_draft = types.MethodType(fake_analyze, minimal_loop)
    minimal_loop.strengthen_draft = types.MethodType(fake_strengthen, minimal_loop)
    minimal_loop.validate_with_catboost = types.MethodType(fake_validate, minimal_loop)

    result = await minimal_loop.run_feedback_loop("Draft text", max_iterations=3)
    assert result["iterations_completed"] == 0
    assert result["stop_reason"] == "No weak features found"


@pytest.mark.asyncio()
async def test_feedback_loop_tracks_confidence_and_stops_on_threshold(minimal_loop: RefinementLoop) -> None:
    validate_results = [
        {"confidence": 0.35, "improvement_percent": 0.0, "improved": False},
        {"confidence": 0.55, "improvement_percent": 0.5, "improved": True},
        {"confidence": 0.91, "improvement_percent": 12.0, "improved": True},
    ]

    async def fake_analyze(self: RefinementLoop, _: str) -> Dict[str, Any]:
        return {"weak_features": {"citations": {"current": 0.1, "target": 0.8, "gap": 0.7}}}

    async def fake_strengthen(self: RefinementLoop, draft: str, weak: Dict[str, Any], context: Dict[str, Any] | None = None) -> str:
        return draft  # Keep length stable to avoid bloat-based stop reasons

    async def fake_validate(self: RefinementLoop, draft: str) -> Dict[str, Any]:
        return validate_results.pop(0)

    minimal_loop.analyze_draft = types.MethodType(fake_analyze, minimal_loop)
    minimal_loop.strengthen_draft = types.MethodType(fake_strengthen, minimal_loop)
    minimal_loop.validate_with_catboost = types.MethodType(fake_validate, minimal_loop)
    minimal_loop.catboost_model = True

    result = await minimal_loop.run_feedback_loop("Initial draft", max_iterations=4)

    assert result["iterations_completed"] == 2
    assert result["stop_reason"].startswith("High confidence")
    assert result["final_confidence"] == pytest.approx(0.91)
    assert result["success"] is True
    assert result["iteration_results"][0]["improvement_percent"] == 0.5


@pytest.mark.asyncio()
async def test_feedback_loop_respects_max_iterations(minimal_loop: RefinementLoop) -> None:
    validate_results = [
        {"confidence": 0.3, "improvement_percent": 0.0, "improved": False},
        {"confidence": 0.6, "improvement_percent": 2.0, "improved": True},
    ]

    async def fake_analyze(self: RefinementLoop, _: str) -> Dict[str, Any]:
        return {"weak_features": {"structure": {"current": 0.2, "target": 0.6, "gap": 0.4}}}

    async def fake_strengthen(self: RefinementLoop, draft: str, weak: Dict[str, Any], context: Dict[str, Any] | None = None) -> str:
        return draft

    async def fake_validate(self: RefinementLoop, draft: str) -> Dict[str, Any]:
        return validate_results.pop(0)

    minimal_loop.analyze_draft = types.MethodType(fake_analyze, minimal_loop)
    minimal_loop.strengthen_draft = types.MethodType(fake_strengthen, minimal_loop)
    minimal_loop.validate_with_catboost = types.MethodType(fake_validate, minimal_loop)
    minimal_loop.catboost_model = True

    result = await minimal_loop.run_feedback_loop("Initial draft", max_iterations=1)

    assert result["iterations_completed"] == 1
    assert result["stop_reason"] == "Reached max iterations (1)"
    assert result["iteration_results"][0]["confidence"] == pytest.approx(0.6)
