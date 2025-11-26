from pathlib import Path
from typing import Any, Dict

from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph

from writer_agents.scripts.extract_fact_registry import FactEntry
from writer_agents.scripts.map_fact_relationships import build_fact_relationships
from writer_agents.scripts.map_facts_to_kg import fact_node_id, map_facts_to_knowledge_graph


def _make_fact(
    fact_type: str,
    fact_value: str,
    source_doc: str,
    description: str | None = None,
    *,
    fact_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> FactEntry:
    return FactEntry(
        fact_type=fact_type,
        fact_value=fact_value,
        description=description,
        source_doc=source_doc,
        extraction_method="test",
        confidence=1.0,
        fact_id=fact_id,
        metadata=metadata or {},
    )


def test_map_facts_adds_fact_nodes_and_document_relations() -> None:
    facts = [
        _make_fact("date", "April 7, 2025", "case_a.txt"),
        _make_fact("allegation", "Privacy breach on April 7, 2025", "case_a.txt"),
    ]
    kg = KnowledgeGraph()

    map_facts_to_knowledge_graph(facts, kg)

    fact_nodes = [node for node in kg.graph.nodes if node.startswith("fact::")]
    assert len(fact_nodes) == 2
    assert any(edge_data["relation"] == "extracted_from" for _, _, edge_data in kg.graph.edges(data=True))


def test_build_fact_relationships_links_related_facts() -> None:
    facts = [
        _make_fact("date", "April 7, 2025", "case_a.txt"),
        _make_fact("date", "April 18, 2025", "case_a.txt"),
        _make_fact("timeline_event", "Notice issued April 7, 2025 by Harvard OGC", "case_a.txt"),
        _make_fact("organization", "Harvard OGC", "case_a.txt"),
    ]
    kg = KnowledgeGraph()
    map_facts_to_knowledge_graph(facts, kg)

    build_fact_relationships(facts, kg)

    relations = list(kg.graph.edges(data=True))
    relation_types = {edge_data.get("relation") for _, _, edge_data in relations}
    assert "relates_to" in relation_types
    assert "before" in relation_types
    assert "involves" in relation_types


def test_map_facts_attaches_metadata_nodes() -> None:
    fact = _make_fact(
        "timeline_event",
        "Harvard emails PRC contacts about plaintiff",
        "case_a.txt",
        fact_id="FACT-001",
        metadata={"subject": "Harvard", "actorrole": "University", "causal_salience_score": 0.9},
    )
    kg = KnowledgeGraph()

    map_facts_to_knowledge_graph([fact], kg)

    node_id = fact_node_id(fact)
    assert node_id in kg.graph.nodes
    assert "subject::Harvard" in kg.graph.nodes
    edge_data = kg.graph.get_edge_data(node_id, "subject::Harvard")
    assert edge_data is not None
    assert edge_data.get("relation") == "involves_subject"
    assert kg.graph.nodes[node_id]["metadata"]["causal_salience_score"] == 0.9
