#!/usr/bin/env python3
"""
Rebuild the facts knowledge graph from the curated fact_registry dataset.

This script reads the final SQLite database, creates KnowledgeGraph nodes
for every fact, adds provenance edges, and persists the graph to the format
requested by --output (json/gexf/pkl).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nlp_analysis.code.KnowledgeGraph import KnowledgeGraph
from writer_agents.scripts import map_facts_to_kg

DEFAULT_DATABASE = REPO_ROOT / "case_law_data" / "lawsuit_facts_database.db"
DEFAULT_OUTPUT = REPO_ROOT / "case_law_data" / "facts_knowledge_graph.json"
LOGGER = logging.getLogger(__name__)


def rebuild_from_fact_registry(
    database: Path,
    output: Path,
    *,
    merge_similar: bool = False,
    merge_threshold: float = 0.9,
    fact_limit: int | None = None,
) -> KnowledgeGraph:
    """
    Rebuild the knowledge graph from fact_registry entries.

    Args:
        database: Path to lawsuit_facts_database.db populated with the 605 curated facts.
        output: Destination path for the serialized graph.
        merge_similar: Merge near-duplicate entity labels when True.
        merge_threshold: Similarity threshold for merge_similar.
        fact_limit: Optional cap for debugging subsets.

    Returns:
        KnowledgeGraph populated with final facts and metadata nodes.
    """
    facts = map_facts_to_kg.fetch_facts_from_db(database)
    if fact_limit is not None:
        facts = facts[:fact_limit]
    if not facts:
        raise ValueError(f"No fact_registry rows found in {database}")

    LOGGER.info("Loaded %d fact_registry row(s) from %s", len(facts), database)

    kg = KnowledgeGraph()
    map_facts_to_kg.map_facts_to_knowledge_graph(facts, kg)

    if merge_similar:
        kg.merge_similar_entities(similarity_threshold=merge_threshold)

    output.parent.mkdir(parents=True, exist_ok=True)
    kg.save_to_file(str(output))

    stats = kg.get_summary_stats()
    LOGGER.info(
        "Knowledge graph saved to %s (%d entities / %d relations)",
        output,
        stats["num_entities"],
        stats["num_relations"],
    )
    return kg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild facts_knowledge_graph.json directly from fact_registry entries."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="Path to lawsuit_facts_database.db with the final curated facts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for the rebuilt graph (json/gexf/pkl).",
    )
    parser.add_argument(
        "--merge-similar",
        action="store_true",
        help="Merge highly similar node labels after import.",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold used when --merge-similar is set.",
    )
    parser.add_argument(
        "--fact-limit",
        type=int,
        default=None,
        help="Optional limit for smoke-testing smaller batches.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    rebuild_from_fact_registry(
        args.database,
        args.output,
        merge_similar=args.merge_similar,
        merge_threshold=args.merge_threshold,
        fact_limit=args.fact_limit,
    )


if __name__ == "__main__":
    main()
