from __future__ import annotations

import asyncio
import importlib
import json
import sys
import textwrap
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple

import pytest

from writer_agents.code.sk_plugins.FeaturePlugin.CaseFactsProvider import CaseFactsProvider
from writer_agents.code.validation.personal_facts_verifier import verify_motion_uses_personal_facts

_original_code_module = sys.modules.get("code")
sys.modules["code"] = importlib.import_module("writer_agents.code")
from writer_agents.scripts.generate_optimized_motion import generate_optimized_motion
if _original_code_module is not None:
    sys.modules["code"] = _original_code_module
else:  # pragma: no cover - cleanup when stdlib module absent
    sys.modules.pop("code", None)


class _StubSection:
    def __init__(self, title: str, body: str) -> None:
        self.title = title
        self.body = body

    def dict(self) -> Dict[str, str]:
        return {"title": self.title, "body": self.body}


class _StubPlan:
    def __init__(self, title: str = "personal-facts-plan") -> None:
        self.title = title

    def dict(self) -> Dict[str, str]:
        return {"title": self.title}


class _StubDeliverable:
    def __init__(self, text: str) -> None:
        self.edited_document = text
        self.sections = [_StubSection("Findings", text)]
        self.plan = _StubPlan()
        self.reviews: list[Any] = []


def _patch_conductor(
    monkeypatch: pytest.MonkeyPatch,
    builder: Callable[[Any], Tuple[str, Dict[str, Any]]],
) -> None:
    class _FakeExecutor:
        def __init__(self, config: Any) -> None:
            self.config = config
            self._last_state = SimpleNamespace(validation_results={})

        async def run_hybrid_workflow(self, insights: Any, **_: Any) -> _StubDeliverable:
            motion_text, validation = builder(insights)
            self._last_state = SimpleNamespace(validation_results=validation)
            return _StubDeliverable(motion_text)

    monkeypatch.setattr(
        "writer_agents.scripts.generate_optimized_motion.Conductor",
        _FakeExecutor,
    )


def _build_success_motion_builder(personal_facts_fixture):
    def _builder(insights: Any) -> Tuple[str, Dict[str, Any]]:
        provider = CaseFactsProvider(
            case_insights=insights,
            personal_corpus_dir=personal_facts_fixture.corpus_dir,
            lawsuit_facts_db_path=personal_facts_fixture.lawsuit_facts_db_path,
            enable_factuality_filter=False,
        )
        facts_section = provider.format_facts_for_autogen() or ""
        truths_section = provider.format_truths_for_prompt()
        motion_text = textwrap.dedent(
            f"""
            The Hong Kong Statement of Claim filed June 2, 2025 (HK Statement) proves defamation,
            privacy breach, harassment, and retaliation. Harvard's OGC emails on April 7, 2025 and April 18, 2025
            concede privacy violations while threatening the plaintiff. The Office of General Counsel's April 2025
            correspondence, together with June 2025 arrests culminating on June 4, 2025, show a retaliation timeline.
            These events document harassment, retaliation, and privacy breaches tied directly to the HK Statement.
            {facts_section}

            Timeline Evidence:
            - April 2025 OGC emails confirm retaliation and privacy breaches.
            - June 2025 arrests and June 4, 2025 detentions show escalating retaliation.

            {truths_section}
            """
        ).strip()
        is_valid, missing, _violations, details = verify_motion_uses_personal_facts(
            motion_text,
            personal_facts_fixture.personal_facts,
        )
        assert is_valid, f"Fixture builder should satisfy all facts, missing={missing}"
        validation = {
            "personal_facts_verification": {
                "is_valid": is_valid,
                "missing": missing,
                **details,
            }
        }
        return motion_text, validation

    return _builder


def _build_incomplete_motion_builder(personal_facts_fixture):
    def _builder(insights: Any) -> Tuple[str, Dict[str, Any]]:
        motion_text = (
            "This draft repeats generic case background without citing the plaintiff's personal filings "
            "or Harvard communications. It mentions a privacy breach abstractly but omits specific dates, "
            "emails, or arrests from the personal corpus."
        )
        is_valid, missing, _violations, details = verify_motion_uses_personal_facts(
            motion_text,
            personal_facts_fixture.personal_facts,
        )
        assert not is_valid
        validation = {
            "personal_facts_verification": {
                "is_valid": is_valid,
                "missing": missing,
                **details,
            }
        }
        return motion_text, validation

    return _builder


def _run_generate(case_summary: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(
        generate_optimized_motion(
            case_summary=case_summary,
            jurisdiction="D. Mass.",
            evidence=evidence,
            autogen_model="test-local",
            workflow_overrides={
                "max_iterations": 1,
                "enable_iterative_refinement": False,
                "enable_quality_gates": False,
                "enable_autogen_review": False,
            },
        )
    )


def test_generate_motion_references_personal_facts(personal_facts_fixture, monkeypatch):
    _patch_conductor(monkeypatch, _build_success_motion_builder(personal_facts_fixture))

    payload = json.loads(personal_facts_fixture.lawsuit_facts_extracted_path.read_text())
    assert payload["fact_blocks"], "lawsuit_facts_extracted.json fixture missing fact blocks"
    results = _run_generate("Seal motion relying on personal corpus", personal_facts_fixture.evidence_payload)

    assert results["success"], results.get("error")
    motion_text = results["final_motion"]
    assert motion_text
    assert "lorem ipsum" not in motion_text.lower()
    assert "placeholder" not in motion_text.lower()

    is_valid, missing, _violations, details = verify_motion_uses_personal_facts(
        motion_text,
        personal_facts_fixture.personal_facts,
    )
    assert is_valid, f"Missing personal facts: {missing}"
    assert pytest.approx(1.0, rel=1e-3) == details["coverage"]

    workflow_verification = results["validation_results"]["personal_facts_verification"]
    assert workflow_verification["is_valid"]
    assert workflow_verification["missing"] == []


def test_generate_motion_detects_missing_personal_facts(personal_facts_fixture, monkeypatch):
    _patch_conductor(monkeypatch, _build_incomplete_motion_builder(personal_facts_fixture))

    payload = json.loads(personal_facts_fixture.lawsuit_facts_extracted_path.read_text())
    assert payload["fact_blocks"]
    results = _run_generate("Seal motion missing facts", personal_facts_fixture.evidence_payload)

    assert results["success"], "Stub executor still returns motion text"
    motion_text = results["final_motion"]
    is_valid, missing, _violations, _details = verify_motion_uses_personal_facts(
        motion_text,
        personal_facts_fixture.personal_facts,
    )
    assert not is_valid
    assert "hk_statement" in missing

    workflow_verification = results["validation_results"]["personal_facts_verification"]
    assert workflow_verification["is_valid"] is False
    assert "hk_statement" in workflow_verification["missing"]
