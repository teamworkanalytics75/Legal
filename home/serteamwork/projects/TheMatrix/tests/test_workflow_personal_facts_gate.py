import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_MODULE_ALIASES = {
    "agents": "writer_agents.agents",
    "insights": "writer_agents.insights",
    "tasks": "writer_agents.tasks",
    "sk_config": "writer_agents.sk_config",
    "sk_plugins": "writer_agents.sk_plugins",
}

for alias, target in _MODULE_ALIASES.items():
    if alias not in sys.modules:
        sys.modules[alias] = importlib.import_module(target)

from writer_agents.code import WorkflowOrchestrator as orchestrator


class _DetectorStub:
    """Simple contradiction detector stub for orchestrator tests."""

    def __init__(
        self,
        source_docs_dir: Optional[Path] = None,
        lawsuit_facts_db: Optional[Path] = None,
        fact_registry: Optional[Dict[str, Any]] = None,
        knowledge_graph: Any = None,
        fact_graph_query: Any = None,
    ) -> None:
        self.source_docs_dir = source_docs_dir
        self.lawsuit_facts_db = lawsuit_facts_db
        self.fact_registry = fact_registry

    def detect_contradictions(self, motion_text: str) -> List[Dict[str, Any]]:
        return [
            {
                "claim": motion_text,
                "contradiction_type": "DIRECT_CONTRADICTION",
                "severity": "critical",
                "location": motion_text,
                "fact_type": "citizenship",
            }
        ]


def test_run_personal_facts_verifier_reports_contradictions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sample_snapshot = {"structured_facts": {"citizenship": "US citizen"}}

    def fake_snapshot(context: Optional[Dict[str, Any]] = None, provider: Any = None) -> Dict[str, Any]:
        if context is not None:
            context["personal_corpus_facts"] = sample_snapshot
        return sample_snapshot

    def fake_verifier(
        document: str,
        personal_facts: Dict[str, Any],
        negative_rules: Any = None,
        fact_graph_query: Any = None,
    ) -> Any:
        return (
            True,
            [],
            [],
            {
                "coverage": 0.9,
                "matches": {"citizenship": "US citizen"},
            },
        )

    docs_dir = tmp_path / "source_docs"
    docs_dir.mkdir()

    monkeypatch.setattr(orchestrator, "_personal_facts_snapshot", fake_snapshot)
    monkeypatch.setattr(orchestrator, "verify_motion_uses_personal_facts", fake_verifier)
    monkeypatch.setattr(orchestrator, "ContradictionDetector", _DetectorStub)

    motion = "The plaintiff is a PRC citizen."
    context: Dict[str, Any] = {"source_docs_dir": docs_dir}
    result = orchestrator._run_personal_facts_verifier(motion, context=context)

    assert result is not None
    assert result["has_contradictions"] is True
    assert result["contradictions"]
    assert result["contradictions"][0]["severity"] == "critical"
    assert result["is_valid"] is False


def test_workflow_router_blocks_commit_when_critical_failures_present() -> None:
    config = orchestrator.WorkflowStrategyConfig(max_iterations=1)
    router = orchestrator.WorkflowRouter(config)
    state = orchestrator.WorkflowState(
        phase=orchestrator.WorkflowPhase.VALIDATE,
        iteration=config.max_iterations,
        validation_results={
            "overall_score": 0.6,
            "personal_facts_verification": {
                "has_violations": True,
                "has_contradictions": False,
                "critical_facts_missing": True,
                "facts_missing": ["citizenship"],
            },
        },
    )

    with pytest.raises(orchestrator.WorkflowCommitBlocked) as exc:
        router.get_next_phase(orchestrator.WorkflowPhase.VALIDATE, state)

    assert "Blocking commit" in str(exc.value)


def test_router_blocks_commit_when_critical_failures_present() -> None:
    config = orchestrator.WorkflowStrategyConfig(max_iterations=1, auto_commit_threshold=0.9)
    router = orchestrator.WorkflowRouter(config=config)
    state = orchestrator.WorkflowState(
        phase=orchestrator.WorkflowPhase.VALIDATE,
        iteration=config.max_iterations,
    )
    state.validation_results = {
        "meets_threshold": False,
        "overall_score": 0.5,
        "personal_facts_verification": {
            "has_violations": True,
            "has_contradictions": False,
            "critical_facts_missing": False,
            "facts_missing": ["citizenship"],
        },
    }

    with pytest.raises(orchestrator.WorkflowCommitBlocked) as exc:
        router.get_next_phase(orchestrator.WorkflowPhase.VALIDATE, state)

    assert "Blocking commit" in str(exc.value)
