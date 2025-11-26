#!/usr/bin/env python3
"""
Integration tests that cover the final-facts pipeline:
CSV → fact_registry → document_facts → knowledge graph → CaseFactsProvider.
"""

from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, cast

import pytest
from _pytest.monkeypatch import MonkeyPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph
from scripts import import_final_facts_to_database as importer
from writer_agents.scripts import map_facts_to_kg, sync_fact_registry_to_master

FINAL_CSV = REPO_ROOT / "case_law_data" / "top_1000_facts_for_chatgpt_final.csv"

pytestmark = pytest.mark.integration

if TYPE_CHECKING:
    from writer_agents.code.sk_plugins.FeaturePlugin.CaseFactsProvider import CaseFactsProvider


@pytest.fixture(scope="module")
def final_facts_pipeline(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, object]:
    """Build a throwaway pipeline using the curated CSV."""
    tmp_dir = tmp_path_factory.mktemp("final_facts_pipeline")
    db_path = (tmp_dir / "lawsuit_facts_database.db").resolve()
    monkeypatcher = MonkeyPatch()

    rows = importer.load_csv_rows(FINAL_CSV)
    entries = importer.prepare_registry_entries(rows)
    with sqlite3.connect(db_path) as conn:
        importer.insert_entries(conn, entries, mode="replace")

    # Build a KG and sync facts so CaseFactsProvider can read document_facts.
    facts = map_facts_to_kg.fetch_facts_from_db(db_path)
    kg = KnowledgeGraph()
    map_facts_to_kg.map_facts_to_knowledge_graph(facts, kg)
    sync_fact_registry_to_master.sync_fact_registry_to_document_facts(db_path, db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS legal_implications (
                implication_id TEXT PRIMARY KEY,
                derived_from TEXT,
                legal_implication TEXT,
                supporting_fact_ids TEXT,
                jurisdiction_assumption TEXT,
                rationale TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        (doc_count,) = conn.execute("SELECT COUNT(*) FROM document_facts").fetchone()
        assert doc_count == len(entries)

    casefacts_module = importlib.import_module("writer_agents.code.sk_plugins.FeaturePlugin.CaseFactsProvider")
    monkeypatcher.setattr(casefacts_module, "EMBEDDING_RETRIEVER_AVAILABLE", False, raising=False)
    monkeypatcher.setattr(casefacts_module, "EmbeddingRetriever", None, raising=False)

    empty_personal = tmp_dir / "personal_corpus"
    empty_personal.mkdir()
    provider = casefacts_module.CaseFactsProvider(
        case_insights=None,
        enable_factuality_filter=False,
        lawsuit_facts_db_path=db_path,
        personal_corpus_dir=empty_personal,
        strict_filtering=False,
    )

    try:
        yield {
            "db_path": db_path,
            "entries": entries,
            "facts": facts,
            "knowledge_graph": kg,
            "provider": provider,
        }
    finally:
        monkeypatcher.undo()


def test_csv_rows_import_into_fact_registry(final_facts_pipeline: Dict[str, object]) -> None:
    entries = cast(List[importer.RegistryEntry], final_facts_pipeline["entries"])
    db_path = cast(Path, final_facts_pipeline["db_path"])

    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM fact_registry").fetchone()
        sample_row = conn.execute(
            "SELECT fact_type, fact_value, source_doc FROM fact_registry WHERE fact_id = ?",
            (entries[0].fact_id,),
        ).fetchone()

    assert count == len(entries) == 605
    assert sample_row is not None
    assert entries[0].fact_value[:40] in sample_row[1]
    assert sample_row[0] == entries[0].fact_type


def test_knowledge_graph_contains_final_fact_nodes(final_facts_pipeline: Dict[str, object]) -> None:
    kg = cast(KnowledgeGraph, final_facts_pipeline["knowledge_graph"])
    entries = cast(List[importer.RegistryEntry], final_facts_pipeline["entries"])

    stats = kg.get_summary_stats()
    assert stats["num_entities"] >= len(entries)
    assert stats["num_relations"] >= len(entries)

    sample_entry = entries[0]
    node_id = map_facts_to_kg.fact_node_id(sample_entry)
    assert node_id in kg.graph.nodes
    node_data = kg.graph.nodes[node_id]
    assert node_data["fact_value"].startswith(sample_entry.fact_value[:60])


def test_case_facts_provider_reads_synced_records(final_facts_pipeline: Dict[str, object]) -> None:
    provider = cast("CaseFactsProvider", final_facts_pipeline["provider"])
    entries = cast(List[importer.RegistryEntry], final_facts_pipeline["entries"])
    sample_entry = entries[0]

    assert provider.has_lawsuit_facts_database()
    registry = provider.get_fact_registry(sample_entry.fact_type)
    assert registry
    assert sample_entry.fact_value in registry.get(sample_entry.fact_type, [])

    records = provider.get_lawsuit_facts_records(include_implications=False)
    assert len(records) >= 605
    assert any(sample_entry.fact_value in row.get("fact_text", "") for row in records)
