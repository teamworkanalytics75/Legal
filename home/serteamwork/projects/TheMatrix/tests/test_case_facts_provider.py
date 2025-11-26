import textwrap
from pathlib import Path
from typing import Dict

import pytest

from writer_agents.code.insights import CaseInsights, EvidenceItem, Posterior
from writer_agents.code.sk_plugins.FeaturePlugin.CaseFactsProvider import CaseFactsProvider


def _build_insights() -> CaseInsights:
    """Create a minimal CaseInsights instance for testing."""
    evidence = [
        EvidenceItem(
            node_id="fact_block_privacy_leak_events",
            state="Disclosure of private information (2019).",
            description="Privacy leak"
        ),
        EvidenceItem(
            node_id="unrelated_node",
            state="Irrelevant",
            description="Noise"
        ),
    ]
    return CaseInsights(
        reference_id="case",
        summary="Motion to seal sensitive information.",
        posteriors=[Posterior(node_id="outcome", probabilities={"win": 0.7, "loss": 0.3})],
        evidence=evidence,
        jurisdiction="D. Mass.",
        case_style="Doe v. Harvard",
    )


def _insights_from_fact_blocks(fact_blocks: Dict[str, str]) -> CaseInsights:
    evidence = [
        EvidenceItem(
            node_id=f"fact_block_{key}",
            state=value,
            description="fixture"
        )
        for key, value in fact_blocks.items()
    ]
    return CaseInsights(
        reference_id="fixture_case",
        summary="Fixture insights assembled from personal corpus.",
        posteriors=[Posterior(node_id="privacy", probabilities={"win": 0.8, "loss": 0.2})],
        evidence=evidence,
        jurisdiction="D. Mass.",
        case_style="Doe v. Harvard",
    )


def test_personal_corpus_loading(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "Harvard Emails.txt").write_text(
        "Email correspondence with Harvard OGC confirming reception of April 7 notice.",
        encoding="utf-8"
    )

    provider = CaseFactsProvider(
        case_insights=_build_insights(),
        personal_corpus_dir=corpus_dir,
        enable_factuality_filter=False,
    )

    facts = provider.get_all_facts()
    personal_keys = [key for key in facts if key.startswith("personal_corpus_")]
    assert personal_keys, "expected personal corpus facts to be loaded"


def test_filter_evidence_for_lawsuit_drops_irrelevant(tmp_path: Path) -> None:
    provider = CaseFactsProvider(
        case_insights=_build_insights(),
        personal_corpus_dir=tmp_path,
        enable_factuality_filter=False,
    )

    evidence = [
        {"node_id": "fact_block_privacy_leak_events", "state": "present"},
        {"node_id": "extraneous_node", "state": "none"},
        {"state": "missing node id"},
    ]

    filtered = provider.filter_evidence_for_lawsuit(evidence)
    assert any(item.get("node_id") == "fact_block_privacy_leak_events" for item in filtered)
    assert all(item.get("node_id") != "extraneous_node" for item in filtered)

    stats = provider.get_last_filter_stats()
    assert stats["original_count"] == len(evidence)
    assert stats["filtered_count"] < stats["original_count"]
    assert stats["dropped_count"] == stats["original_count"] - stats["filtered_count"]


def test_filter_evidence_returns_original_when_strict_disabled(tmp_path: Path) -> None:
    provider = CaseFactsProvider(
        case_insights=_build_insights(),
        personal_corpus_dir=tmp_path,
        enable_factuality_filter=False,
        strict_filtering=False,
    )

    evidence = [{"node_id": "unknown", "state": "none"}]
    filtered = provider.filter_evidence_for_lawsuit(evidence)
    assert filtered, "filter should fall back to original evidence when nothing matches"


def test_filter_evidence_strict_mode_returns_empty_when_personal_loaded(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "HK Statement.txt").write_text(
        "Hong Kong Statement of Claim dated June 4, 2025 referencing arrests.",
        encoding="utf-8"
    )

    provider = CaseFactsProvider(
        case_insights=_build_insights(),
        personal_corpus_dir=corpus_dir,
        enable_factuality_filter=False,
        strict_filtering=True,
    )

    evidence = [{"node_id": "unknown", "state": "none"}]
    filtered = provider.filter_evidence_for_lawsuit(evidence)
    assert filtered == [], "strict filtering should return empty list when nothing matches"

    stats = provider.get_last_filter_stats()
    assert stats.get("strict_filtering_failed") is True


def test_case_facts_provider_uses_real_personal_corpus(personal_facts_fixture) -> None:
    provider = CaseFactsProvider(
        case_insights=_insights_from_fact_blocks(personal_facts_fixture.fact_blocks),
        personal_corpus_dir=personal_facts_fixture.corpus_dir,
        lawsuit_facts_db_path=personal_facts_fixture.lawsuit_facts_db_path,
        enable_factuality_filter=False,
    )

    facts = provider.get_all_facts()
    personal_entries = [key for key in facts if key.startswith("personal_corpus_")]
    assert personal_entries, "expected personal corpus entries to be present"

    formatted = provider.format_facts_for_autogen()
    assert "Hong Kong Statement" in formatted

    truths_block = provider.format_truths_for_prompt()
    assert "Verified Lawsuit Facts" in truths_block
    assert "HK Statement" in truths_block


def test_filter_evidence_handles_missing_personal_corpus(tmp_path: Path) -> None:
    provider = CaseFactsProvider(
        case_insights=_build_insights(),
        personal_corpus_dir=tmp_path / "missing",
        enable_factuality_filter=False,
        strict_filtering=True,
    )
    evidence = [{"node_id": "unknown", "state": "none"}]
    filtered = provider.filter_evidence_for_lawsuit(evidence)
    assert filtered == evidence, "When corpus missing provider should fall back to original evidence"

    stats = provider.get_last_filter_stats()
    assert stats["strict_filtering"] is True
    assert stats["used_personal_corpus"] is False
