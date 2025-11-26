import json
import textwrap

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRITER_AGENTS_SRC = PROJECT_ROOT / "writer_agents"
if str(WRITER_AGENTS_SRC) not in sys.path:
    sys.path.append(str(WRITER_AGENTS_SRC))

from writer_agents.code.WorkflowOrchestrator import (  # noqa: E402
    QualityGatePipeline,
    _run_personal_facts_verifier,
)
from writer_agents.code.validation.personal_facts_verifier import (
    DEFAULT_FACT_RULES,
    verify_motion_with_case_insights,
    verify_motion_uses_personal_facts,
)


PRC_TEST_MOTION = "The plaintiff, a PRC citizen, seeks protection from retaliation."


def _personal_facts_context() -> dict:
    # Provide a minimal snapshot so the orchestrator avoids hitting the provider.
    return {"personal_corpus_facts": {"aliases": {"hk_statement": ["Action No. 771"]}}}


def test_verifier_detects_all_required_facts():
    motion_text = textwrap.dedent(
        """
        The Hong Kong Statement of Claim filed on June 2, 2025 documents repeated defamation,
        harassment, and retaliation campaigns. Harvard's Office of General Counsel (OGC)
        received emails on April 7, 2025 and April 18, 2025 that warned of privacy breaches and
        ongoing retaliation. By June 4, 2025 the plaintiff faced arrests in June 2025 tied to
        the same defamation. This timeline shows April 2025 OGC emails and June 2025 arrests.
        """
    ).strip()

    is_valid, missing, violations, details = verify_motion_uses_personal_facts(motion_text, personal_corpus_facts={})

    assert is_valid is True
    assert missing == []
    assert violations == []
    assert details["coverage"] == 1.0
    assert details["total_required"] == len(DEFAULT_FACT_RULES)
    assert "hk_statement" in details["matches"]
    assert "timeline_april_ogc_emails" in details["matches"]
    assert "timeline_june_2025_arrests" in details["matches"]


def test_verifier_identifies_missing_dates_and_timeline():
    motion_text = textwrap.dedent(
        """
        The HK Statement generally shows defamation and privacy breaches.
        Harvard OGC ignored warnings, but the motion forgot to mention specific dates.
        """
    ).strip()

    is_valid, missing, violations, details = verify_motion_uses_personal_facts(motion_text, personal_corpus_facts={})

    assert is_valid is False
    assert "date_april_7_2025" in missing
    assert "date_april_18_2025" in missing
    assert "timeline_april_ogc_emails" in missing
    assert details["coverage"] < 1.0
    assert violations == []


def test_verifier_respects_aliases_from_personal_corpus_data():
    motion_text = (
        "Action No. 771 details retaliation while OGC emails proved harassment. "
        "April 2025 summary notes apply here."
    )
    personal_facts = {
        "aliases": {
            "hk_statement": ["Action No. 771"],
            "timeline_april_ogc_emails": ["April 2025"],
        }
    }

    is_valid, missing, violations, details = verify_motion_uses_personal_facts(motion_text, personal_facts)

    assert "hk_statement" in details["matches"]
    assert "timeline_april_ogc_emails" in details["matches"]
    assert "ogc_emails" in details["matches"]
    assert "allegation_defamation" in missing  # defamation term not present
    assert violations == []


def test_verify_motion_with_case_insights_loads_aliases(tmp_path):
    lawsuit_facts_extracted_path = tmp_path / "lawsuit_facts_extracted.json"
    case_insights_path = tmp_path / "case_insights.json"  # Also create old name for backward compatibility
    corpus_dir = tmp_path / "lawsuit_source_documents"
    corpus_dir.mkdir()
    (corpus_dir / "Exhibit 2 — Certified Statement of Claim (Hong Kong, 2 Jun 2025).txt").write_text("dummy")
    (corpus_dir / "3 Emails to Harvard OGC.txt").write_text("dummy")
    case_insights_data = {
        "case_summary": "Action No. 771 Hong Kong Statement of Claim references Office of General Counsel.",
        "fact_blocks": {
            "hk_allegation_defamation": "Action No. 771 shows defamatory conduct.",
            "hk_retaliation_events": "Timeline referencing June 2025 arrests.",
            "ogc_email_1_threat": "April 7, 2025 OGC email notice.",
            "ogc_email_2_non_response": "April 18, 2025 follow-up to Office of General Counsel.",
            "ogc_email_3_meet_confer": "August 2025 meet-and-confer.",
            "safety_concerns": "June 2025 arrests in Hong Kong.",
            "harvard_retaliation_events": "Continuing retaliation with June 2025 arrests.",
            "privacy_leak_events": "Privacy breach via disclosures.",
        },
    }
    lawsuit_facts_extracted_path.write_text(json.dumps(case_insights_data))
    case_insights_path.write_text(json.dumps(case_insights_data))  # Also create old name

    motion_text = textwrap.dedent(
        """
        Action No. 771 demonstrates defamation, harassment, and retaliation.
        OGC staff ignored privacy breaches despite emails on April 7, 2025 and April 18, 2025.
        The filing on June 2, 2025 and resulting arrests on June 4, 2025 show June 2025 arrests.
        This timeline highlights April 2025 OGC emails tied to June 2025 arrests.
        """
    ).strip()

    is_valid, missing, violations, details = verify_motion_with_case_insights(
        motion_text, case_insights_path=lawsuit_facts_extracted_path, corpus_dir=corpus_dir
    )

    assert is_valid is True
    assert missing == []
    assert violations == []
    assert details["coverage"] == 1.0


def test_verify_motion_with_case_insights_missing_file(tmp_path):
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        verify_motion_with_case_insights("text", case_insights_path=missing_path)


def test_negative_rule_detects_citizenship_violation():
    is_valid, missing, violations, details = verify_motion_uses_personal_facts(PRC_TEST_MOTION, {})

    assert is_valid is False
    assert violations == ["not_prc_citizen"]
    assert details["violations"] == ["not_prc_citizen"]


def test_negative_rule_reports_alongside_missing_facts():
    motion_text = "PRC citizen seeks sealing relief."

    is_valid, missing, violations, details = verify_motion_uses_personal_facts(motion_text, {})

    assert is_valid is False
    assert "not_prc_citizen" in violations
    assert missing  # should still report missing required positives


def test_negative_rule_skipped_when_not_present():
    motion_text = "The plaintiff, a US citizen, seeks protection."

    is_valid, missing, violations, details = verify_motion_uses_personal_facts(motion_text, {})

    assert "not_prc_citizen" not in violations


def test_negative_rule_general_location_violation():
    motion_text = "The District of Hong Kong federal court should grant this motion."

    is_valid, missing, violations, details = verify_motion_uses_personal_facts(motion_text, {})

    assert "not_wrong_court_location" in violations


def test_graph_fact_coverage_reported_with_fact_query():
    motion_text = "April 7, 2025 OGC emails documented a privacy breach."
    fact_query = _FakeFactGraphQuery()

    is_valid, missing, violations, details = verify_motion_uses_personal_facts(
        motion_text,
        personal_corpus_facts={},
        fact_graph_query=fact_query,
    )

    coverage = details.get("graph_fact_coverage")
    assert coverage is not None
    assert "date" in coverage
    assert coverage["date"]["present"]
    assert any("privacy breach" in entry.lower() for entry in coverage["allegation"]["present"])
    summary = details.get("graph_coverage_summary")
    assert summary
    assert summary["total_types"] >= 2


def test_run_personal_facts_verifier_returns_violation_details(monkeypatch):
    monkeypatch.setattr("writer_agents.code.WorkflowOrchestrator.ContradictionDetector", None)
    context = _personal_facts_context()

    result = _run_personal_facts_verifier(PRC_TEST_MOTION, context)

    assert result is not None
    assert result["has_violations"] is True
    assert result["violations"][0]["name"] == "not_prc_citizen"
    assert result["details"]["violations"] == ["not_prc_citizen"]
    assert result["is_valid"] is False


def test_quality_gate_rejects_motion_with_prohibited_fact(monkeypatch):
    monkeypatch.setattr("writer_agents.code.WorkflowOrchestrator.ContradictionDetector", None)
    pipeline = QualityGatePipeline(sk_kernel=None, executor=None)
    context = _personal_facts_context()

    gate_result = pipeline._validate_personal_facts_coverage(PRC_TEST_MOTION, context)

    assert gate_result["score"] == 0.0
    assert gate_result["has_violations"] is True
    assert gate_result["violations"][0]["name"] == "not_prc_citizen"
    assert any("prohibited" in warning.lower() for warning in gate_result["warnings"])
class _FakeFactGraphQuery:
    def __init__(self) -> None:
        self._hierarchy = {
            "date": ["April 7, 2025", "April 18, 2025"],
            "allegation": ["privacy breach", "retaliation"],
        }

    def get_fact_hierarchy(self):
        return self._hierarchy

    def find_similar_facts(self, *args, **kwargs):
        return []
