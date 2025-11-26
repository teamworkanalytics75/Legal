"""Refresh agent memory snapshots from recent runs.

Usage:
    python scripts/refresh_agent_memories.py
    python scripts/refresh_agent_memories.py --days 7
    python scripts/refresh_agent_memories.py --full # Re-scan everything
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from writer_agents.code.job_persistence import JobManager
from writer_agents.code.memory_system import MemoryStore
from writer_agents.code.extract_memories import MemoryBuilder


async def refresh_memories(
    days: int = None,
    full_rescan: bool = False,
    use_openai: bool = False
):
    """Extract and refresh agent memories.

    Args:
        days: Number of days to scan back (None for default)
        full_rescan: Whether to rescan all historical data
    """
    print("=" * 80)
    print("AGENT MEMORY REFRESH")
    print("=" * 80)

    # Initialize components
    job_manager = JobManager("jobs.db")
    memory_builder = MemoryBuilder()
    memory_store = MemoryStore(
        use_local_embeddings=not use_openai
    )

    # Determine time range
    min_date = None
    if days and not full_rescan:
        min_date = datetime.now() - timedelta(days=days)
        print(f"Scanning last {days} days (since {min_date.strftime('%Y-%m-%d')})")
    elif full_rescan:
        print("Full rescan - extracting all historical data")
    else:
        min_date = datetime.now() - timedelta(days=30)
        print(f"Default: scanning last 30 days")

    # Extract from jobs.db
    print("\n1. Extracting from jobs.db...")
    try:
        job_memories = memory_builder.extract_from_jobs_db(job_manager, min_date=min_date)
        print(f" [ok] Found {len(job_memories)} job-based memories")
    except Exception as e:
        print(f" WARNING Error extracting from jobs.db: {e}")
        job_memories = []

    # Extract from analysis_outputs/
    print("\n2. Extracting from analysis_outputs/...")
    try:
        artifacts_dir = Path("analysis_outputs")
        if artifacts_dir.exists():
            artifact_memories = memory_builder.extract_from_artifacts(artifacts_dir)
            print(f" [ok] Found {len(artifact_memories)} artifact-based memories")
        else:
            print(f" WARNING analysis_outputs/ directory not found")
            artifact_memories = []
    except Exception as e:
        print(f" WARNING Error extracting from artifacts: {e}")
        artifact_memories = []

    # Combine and embed
    all_memories = job_memories + artifact_memories
    print(f"\n3. Total memories to process: {len(all_memories)}")

    if not all_memories:
        print(" WARNING No new memories found. Try --full for complete rescan.")
        return

    # Add to vector store (this embeds them)
    print("\n4. Embedding and storing...")
    try:
        memory_store.add_batch(all_memories)
        print(f" [ok] Embedded {len(all_memories)} memories")
    except Exception as e:
        print(f" x Error embedding memories: {e}")
        return

    # Print summary by agent type
    print("\n5. Memory distribution by agent type:")
    from collections import Counter
    agent_counts = Counter(m.agent_type for m in all_memories)
    for agent_type, count in agent_counts.most_common(10):
        print(f" {agent_type:30s}: {count:4d} memories")

    # Estimate cost (if using OpenAI embeddings)
    embedding_service = memory_store.embeddings_service
    if embedding_service.mode == "openai":
        avg_tokens_per_summary = 50
        total_tokens = len(all_memories) * avg_tokens_per_summary
        cost = total_tokens * 0.00002 / 1000 # $0.02 per 1M tokens
        print(f"\n6. Embedding cost: ${cost:.4f}")
    else:
        print(f"\n6. Embedding cost: $0.00 (local model)")

    print("\n[ok] Memory refresh complete!")
    print(f" Saved to: memory_snapshots/vector_store.pkl")

    # Print system stats
    stats = memory_store.get_all_stats()
    print(f"\nSystem-wide stats:")
    print(f" Total agents: {stats['total_agents']}")
    print(f" Total memories: {stats['total_memories']}")
    print(f" Sources: {stats['sources']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Refresh agent memory snapshots")
    parser.add_argument("--days", type=int, help="Scan last N days (default: 30)")
    parser.add_argument("--full", action="store_true", help="Full rescan (ignore date filter)")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI embeddings (paid, higher quality)")
    args = parser.parse_args()

    asyncio.run(refresh_memories(days=args.days, full_rescan=args.full, use_openai=args.openai))


if __name__ == "__main__":
    main()
