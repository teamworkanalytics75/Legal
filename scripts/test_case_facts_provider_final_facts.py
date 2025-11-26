#!/usr/bin/env python3
"""
Quick verification harness for CaseFactsProvider + final fact_registry.

Usage examples:
  python scripts/test_case_facts_provider_final_facts.py
  python scripts/test_case_facts_provider_final_facts.py --fact-id 1927
  python scripts/test_case_facts_provider_final_facts.py --sample 5
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

cfp_module = importlib.import_module("writer_agents.code.sk_plugins.FeaturePlugin.CaseFactsProvider")
cfp_module.EMBEDDING_RETRIEVER_AVAILABLE = False  # type: ignore[attr-defined]
CaseFactsProvider = cfp_module.CaseFactsProvider  # type: ignore[attr-defined]

DEFAULT_DB = REPO_ROOT / "case_law_data" / "lawsuit_facts_database.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that CaseFactsProvider can load the final fact_registry rows and surface metadata.\n"
            "\n"
            "Outputs the total fact count plus a few sample entries. Provide --fact-id to inspect a specific fact."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DB,
        help="Path to case_law_data/lawsuit_facts_database.db",
    )
    parser.add_argument(
        "--fact-id",
        help="Optional FactID to inspect (e.g., 1927); prints fact text + metadata",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=3,
        help="Number of sample facts to print when --fact-id is not set",
    )
    return parser.parse_args()


def format_metadata(metadata: Dict[str, object]) -> str:
    if not metadata:
        return "n/a"
    return json.dumps(metadata, indent=2, sort_keys=True)


def print_samples(provider: CaseFactsProvider, sample_ids: Iterable[str]) -> None:
    print("\nSample facts:")
    for fact_id in sample_ids:
        fact_text = provider.get_fact_block(fact_id)
        metadata = provider.get_fact_metadata(fact_id) or {}
        print(f"- {fact_id}: {fact_text}")
        if metadata:
            print(f"  metadata: {format_metadata(metadata)}")


def main() -> None:
    args = parse_args()
    database = args.database.expanduser()
    if not database.exists():
        raise SystemExit(f"Database not found: {database}")

    provider = CaseFactsProvider(
        lawsuit_facts_db_path=database,
        enable_factuality_filter=False,
        strict_filtering=False,
    )
    fact_blocks = provider.get_all_facts()
    if not fact_blocks:
        raise SystemExit("CaseFactsProvider did not load any facts. Verify Session 1 import is complete.")

    print("CaseFactsProvider final facts verification")
    print("-----------------------------------------")
    print(f"Database : {database}")
    print(f"Total facts loaded : {len(fact_blocks)}")

    if args.fact_id:
        fact_text = provider.get_fact_block(args.fact_id)
        if not fact_text:
            raise SystemExit(f"FactID {args.fact_id} not found in CaseFactsProvider cache.")
        metadata = provider.get_fact_metadata(args.fact_id) or {}
        print(f"\nFact {args.fact_id}:\n{fact_text}")
        print("\nMetadata:")
        print(format_metadata(metadata))
        return

    sample_count = max(1, args.sample)
    sample_ids: List[str] = sorted(fact_blocks.keys())[:sample_count]
    print_samples(provider, sample_ids)


if __name__ == "__main__":
    main()
