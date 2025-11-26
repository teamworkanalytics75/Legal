from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph

from writer_agents.scripts.validate_knowledge_graph import compute_graph_report


def test_compute_graph_report_detects_orphans_and_invalid_relations() -> None:
    kg = KnowledgeGraph()
    kg.add_entity("fact::date::april_7_2025", entity_type="fact:date")
    kg.add_entity("doc::case_a.txt", entity_type="document")
    kg.add_entity("doc::case_b.txt", entity_type="document")
    kg.add_entity("orphan_node", entity_type="UNKNOWN")

    kg.add_relation("fact::date::april_7_2025", "extracted_from", "doc::case_a.txt")
    kg.add_relation("fact::date::april_7_2025", "unknown_relation", "doc::case_b.txt")

    report = compute_graph_report(kg)

    assert "orphan_node" in report["orphan_nodes"]
    assert report["unlinked_fact_nodes"] == []
    invalid = report["invalid_relations"]
    assert ("fact::date::april_7_2025", "unknown_relation", "doc::case_b.txt") in invalid
