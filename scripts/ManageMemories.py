#!/usr/bin/env python3
"""Memory management utility for EpisodicMemoryBank."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))
from EpisodicMemoryBank import EpisodicMemoryBank

def main():
    parser = argparse.ArgumentParser(description='Manage EpisodicMemoryBank')
    parser.add_argument('--memory-store-path', type=str, default='memory_store',
                       help='Path to memory store directory')
    parser.add_argument('--list-sources', action='store_true',
                       help='List all memory sources')
    parser.add_argument('--delete-source', type=str,
                       help='Delete all memories from source')
    parser.add_argument('--delete-agent', type=str,
                       help='Delete all memories for agent type')
    parser.add_argument('--stats', action='store_true',
                       help='Show memory statistics')

    args = parser.parse_args()

    memory_bank = EpisodicMemoryBank(storage_path=Path(args.memory_store_path))

    if args.list_sources:
        sources = memory_bank.list_sources()
        print("\nMemory Sources:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count} memories")

    if args.delete_source:
        count = memory_bank.delete_by_source(args.delete_source)
        print(f"Deleted {count} memories from source '{args.delete_source}'")

    if args.delete_agent:
        count = memory_bank.delete_by_agent_type(args.delete_agent)
        print(f"Deleted {count} memories for agent type '{args.delete_agent}'")

    if args.stats:
        stats = memory_bank.get_all_stats()
        print(f"\nMemory Statistics:")
        print(f"  Total agents: {stats['total_agents']}")
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Agent types: {', '.join(stats['agent_types'])}")

if __name__ == "__main__":
    main()

