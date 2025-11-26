from pathlib import Path

import pytest

from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph
from writer_agents.code.validation.fact_graph_query import FactGraphQuery
from writer_agents.code.validation.kg_fact_adapter import KGFactAdapter


def _build_sample_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_entity(
        "fact::date::april_7_2025",
        entity_type="fact:date",
        fact_value="April 7, 2025",
        description="Initial OGC notification",
        source_doc="docs/notice.txt",
    )
    kg.add_entity(
        "fact::citizenship::us_citizen",
        entity_type="fact:citizenship",
        fact_value="US citizen",
        description="Declarative citizenship fact",
        source_doc="docs/profile.txt",
    )
    kg.add_entity(
        "fact::citizenship::not_us_citizen",
        entity_type="fact:citizenship",
        fact_value="not US citizen",
        description="Conflicting statement",
        source_doc="docs/contradiction.txt",
    )
    kg.add_relation("fact::date::april_7_2025", "related_to", "fact::citizenship::us_citizen")
    return kg


def test_fact_graph_query_basic_lookup(tmp_path: Path) -> None:
    kg = _build_sample_graph()
    query = FactGraphQuery(kg, fact_registry_db=tmp_path / "facts.db")

    assert query.verify_fact_exists("date", "April 7, 2025")
    relationships = query.get_fact_relationships("date", "April 7, 2025")
    assert relationships and relationships[0][1] == "related_to"
    related = query.find_related_facts("date", "April 7, 2025")
    assert any(node.endswith("us_citizen") for node in related)


def test_fact_graph_query_semantic_fallback() -> None:
    kg = _build_sample_graph()
    query = FactGraphQuery(kg)

    matches = query.find_similar_facts("April 7, 2025 notice", fact_type="date", similarity_threshold=0.3)
    assert matches
    validation = query.validate_fact_claim("Plaintiff served April 7, 2025 notice", expected_fact_type="date")
    assert validation["valid"] is True


def test_fact_graph_hierarchy_and_contradictions() -> None:
    kg = _build_sample_graph()
    query = FactGraphQuery(kg)

    hierarchy = query.get_fact_hierarchy()
    assert "date" in hierarchy and "citizenship" in hierarchy

    contradictions = query.detect_fact_contradictions("The motion states US citizen but later says not us citizen.")
    assert contradictions


def test_fact_query_date_range_and_timeline() -> None:
    kg = _build_sample_graph()
    query = FactGraphQuery(kg)

    date_matches = query.find_facts_by_date_range("April 1, 2025", "April 30, 2025")
    assert len(date_matches) == 1
    timeline = query.get_fact_timeline()
    assert timeline[0]["value"] == "April 7, 2025"


def test_fact_query_source_doc_and_contradictions() -> None:
    kg = _build_sample_graph()
    query = FactGraphQuery(kg)

    from_doc = query.find_facts_by_source_document("docs/notice.txt")
    assert from_doc and from_doc[0]["value"] == "April 7, 2025"

    contradictions = query.find_contradictory_facts("citizenship", "US citizen")
    assert contradictions


def test_fact_query_contextual_search_and_metrics() -> None:
    kg = _build_sample_graph()
    query = FactGraphQuery(kg)

    contextual_matches = query.find_similar_facts("OGC notification", fact_type="date", similarity_threshold=0.2)
    assert contextual_matches and contextual_matches[0].fact_value == "April 7, 2025"

    query.find_facts_by_date_range("April 1, 2025", "April 30, 2025")
    metrics = query.get_query_metrics()
    assert metrics.get("date_range_queries", 0) >= 1
    assert metrics.get("similarity_results", 0) >= 1


def test_fact_query_handles_empty_graph() -> None:
    kg = KnowledgeGraph()
    query = FactGraphQuery(kg)

    assert query.find_facts_by_date_range("Jan 1, 2024", "Jan 2, 2024") == []
    assert query.find_similar_facts("anything") == []
    assert query.get_query_metrics()


def test_kg_fact_adapter_round_trip(tmp_path: Path) -> None:
    kg = _build_sample_graph()
    adapter = KGFactAdapter()

    facts = adapter.load_facts_from_graph(kg)
    assert len(facts) == 3

    db_path = tmp_path / "facts.db"
    inserted = adapter.sync_graph_to_registry(kg, db_path)
    assert inserted == 3
