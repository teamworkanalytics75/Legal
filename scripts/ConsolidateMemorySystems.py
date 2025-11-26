#!/usr/bin/env python3
"""One-time migration script to consolidate all memory systems into EpisodicMemoryBank.

This script migrates data from:
- DatabaseQueryRecorder (langchain_meta_memory.sqlite)
- ConversationTranscriptRecorder (lawsuit_memory.sqlite)
- DocumentEditRecorder (version_history.db)
- DocumentMetadataRecorder (document_tracker.db)

Into the unified EpisodicMemoryBank.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

from EpisodicMemoryBank import EpisodicMemoryBank
from MemorySystemConsolidator import MemorySystemConsolidator


def main():
    """Run memory system consolidation."""
    print("=" * 70)
    print("Memory System Consolidation")
    print("=" * 70)
    print()

    # Load existing EpisodicMemoryBank
    print("Loading EpisodicMemoryBank...")
    store = EpisodicMemoryBank(storage_path=Path("memory_store"))
    print(f"✓ Current memories in store: {len(store)}")
    print()

    # Create consolidator
    print("Initializing consolidator...")
    consolidator = MemorySystemConsolidator(store)
    print("✓ Consolidator ready")
    print()

    # Run migrations
    print("Starting migrations...")
    print("-" * 70)
    results = consolidator.migrate_all()
    print("-" * 70)
    print()

    # Display results
    print("Migration Results:")
    print(f"  Database queries:      {results['queries']:>6} memories")
    print(f"  Conversations:         {results['conversations']:>6} memories")
    print(f"  Document edits:        {results['edits']:>6} memories")
    print(f"  Document metadata:     {results['documents']:>6} memories")
    print(f"  {'─' * 40}")
    print(f"  Total migrated:        {sum(results.values()):>6} memories")
    print()

    print(f"✓ Total memories in store: {len(store)}")
    print()
    print("=" * 70)
    print("Consolidation complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Verify consolidated memories with: python -c \"from writer_agents.code.EpisodicMemoryBank import *; store = EpisodicMemoryBank(storage_path=Path('memory_store')); print(f'Total: {len(store)}'); print(store.get_all_stats())\"")
    print("  2. Update your code to use the new recorder classes")
    print("  3. Test memory retrieval across all types")
    print()


if __name__ == "__main__":
    main()

