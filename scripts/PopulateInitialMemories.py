"""Populate initial agent memories from existing data.

This script extracts memories from jobs.db and analysis_outputs/ and uses
LLM-driven memory writing to create high-quality agent memories.
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from writer_agents.code.job_persistence import JobManager
from writer_agents.code.memory_system import MemoryStore, AgentMemory
from writer_agents.code.extract_memories import MemoryBuilder
from writer_agents.code.agents import AgentFactory, ModelConfig


async def populate_initial_memories(
    days: int = None,
    full_rescan: bool = False,
    use_openai: bool = False,
    max_memories_per_agent: int = 20
):
    """Populate initial memories using LLM-driven writing.

    Args:
        days: Number of days to scan back (None for default)
        full_rescan: Whether to rescan all historical data
        use_openai: Whether to use OpenAI embeddings
        max_memories_per_agent: Maximum memories to create per agent
    """
    print("=" * 80)
    print("POPULATING INITIAL AGENT MEMORIES")
    print("=" * 80)

    # Initialize components
    job_manager = JobManager("jobs.db")
    memory_builder = MemoryBuilder()
    memory_store = MemoryStore(use_local_embeddings=not use_openai)

    # Initialize LLM for memory writing
    factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))

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

    # Extract raw data
    print("\n1. Extracting raw data...")
    try:
        job_memories = memory_builder.extract_from_jobs_db(job_manager, min_date=min_date)
        print(f"   [OK] Found {len(job_memories)} job-based patterns")
    except Exception as e:
        print(f"   [ERROR] Error extracting from jobs.db: {e}")
        job_memories = []

    try:
        artifacts_dir = Path("analysis_outputs")
        if artifacts_dir.exists():
            artifact_memories = memory_builder.extract_from_artifacts(artifacts_dir)
            print(f"   [OK] Found {len(artifact_memories)} artifact-based patterns")
        else:
            print(f"   [WARNING] analysis_outputs/ directory not found")
            artifact_memories = []
    except Exception as e:
        print(f"   [ERROR] Error extracting from artifacts: {e}")
        artifact_memories = []

    # Extract from lawsuit database
    try:
        lawsuit_memories = memory_builder.extract_from_lawsuit_db(max_documents=50)
        print(f"   [OK] Found {len(lawsuit_memories)} lawsuit database patterns")
    except Exception as e:
        print(f"   [ERROR] Error extracting from lawsuit database: {e}")
        lawsuit_memories = []

    # Extract from analysis results
    try:
        analysis_memories = memory_builder.extract_from_analysis_results()
        print(f"   [OK] Found {len(analysis_memories)} analysis result patterns")
    except Exception as e:
        print(f"   [ERROR] Error extracting from analysis results: {e}")
        analysis_memories = []

    # Combine and group by agent type
    all_patterns = job_memories + artifact_memories + lawsuit_memories + analysis_memories
    agent_patterns: Dict[str, List[Any]] = {}

    for pattern in all_patterns:
        agent_type = pattern.agent_type
        if agent_type not in agent_patterns:
            agent_patterns[agent_type] = []
        agent_patterns[agent_type].append(pattern)

    print(f"\n2. Found patterns for {len(agent_patterns)} agent types")

    # Create enhanced memories using LLM
    print(f"\n3. Creating enhanced memories (max {max_memories_per_agent} per agent)...")

    total_cost = 0.0
    total_memories = 0

    for agent_type, patterns in agent_patterns.items():
        print(f"\n   Processing {agent_type} ({len(patterns)} patterns)...")

        # Limit patterns per agent
        selected_patterns = patterns[:max_memories_per_agent]

        # Determine if agent is deterministic
        is_deterministic = _is_deterministic_agent(agent_type)

        agent_memories = []

        for i, pattern in enumerate(selected_patterns, 1):
            try:
                if is_deterministic:
                    # Auto-generate memory (FREE)
                    summary = _auto_generate_memory_summary(agent_type, pattern)
                    cost = 0.0
                else:
                    # Use LLM to enhance memory (CHEAP)
                    summary = await _llm_enhance_memory(factory, agent_type, pattern)
                    cost = 0.0001  # ~100 tokens with gpt-4o-mini

                if summary:
                    memory = AgentMemory(
                        agent_type=agent_type,
                        memory_id=f"{agent_type}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        summary=summary,
                        context=pattern.context,
                        embedding=None,  # Will be computed by MemoryStore
                        source="initial_population",
                        timestamp=datetime.now()
                    )
                    agent_memories.append(memory)
                    total_cost += cost

                if i % 5 == 0:
                    print(f"     Processed {i}/{len(selected_patterns)} patterns")

            except Exception as e:
                print(f"     [ERROR] Error processing pattern {i}: {e}")
                continue

        # Add memories to store
        if agent_memories:
            memory_store.add_batch(agent_memories)
            total_memories += len(agent_memories)
            print(f"   [OK] Created {len(agent_memories)} memories for {agent_type}")
            print(f"   Cost: ${total_cost:.4f}")

    # Save final store
    memory_store.save()

    print(f"\n4. Summary:")
    print(f"   Total memories created: {total_memories}")
    print(f"   Total cost: ${total_cost:.4f}")
    print(f"   Agent types: {len(agent_patterns)}")
    print(f"   Saved to: memory_snapshots/vector_store.pkl")

    # Show stats
    stats = memory_store.get_all_stats()
    print(f"\n5. Memory store statistics:")
    print(f"   Total agents with memories: {stats['total_agents']}")
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Sources: {stats['sources']}")


def _is_deterministic_agent(agent_type: str) -> bool:
    """Check if agent is deterministic based on name patterns."""
    deterministic_patterns = [
        'Citation', 'Normalizer', 'Verifier', 'Locator', 'Inserter',
        'StatuteLocator', 'ExhibitFetcher', 'StyleChecker', 'ConsistencyChecker',
        'RedactionAgent', 'ComplianceAgent', 'MarkdownExporter', 'DocxExporter',
        'MetadataTagger', 'SettlementOptimizer', 'BATNAAnalyzer', 'NashEquilibrium',
        'ReputationRiskScorer'
    ]

    return any(pattern in agent_type for pattern in deterministic_patterns)


def _auto_generate_memory_summary(agent_type: str, pattern: Any) -> str:
    """Auto-generate memory summary for deterministic agents (FREE)."""
    summary = pattern.summary

    # Enhance with additional context if available
    if hasattr(pattern, 'context') and pattern.context:
        context = pattern.context

        if 'tokens' in context:
            tokens = context['tokens']
            summary += f" (processed {tokens} tokens)"

        if 'duration' in context:
            duration = context['duration']
            summary += f" in {duration:.1f}s"

        if 'citations_found' in context:
            count = context['citations_found']
            summary += f" - found {count} citations"

    return summary


async def _llm_enhance_memory(factory: AgentFactory, agent_type: str, pattern: Any) -> str:
    """Enhance memory using LLM (CHEAP)."""
    try:
        # Create a temporary agent for LLM calls
        from writer_agents.code.atomic_agent import AtomicAgent

        class TempAgent(AtomicAgent):
            duty = f"Memory enhancement for {agent_type}"
            is_deterministic = False

        temp_agent = TempAgent(factory)

        # Build prompt
        original_summary = pattern.summary
        context_str = str(pattern.context)[:300] if hasattr(pattern, 'context') else ""

        prompt = f"""Original memory: {original_summary}
Context: {context_str}

Enhance this memory to be more useful for future {agent_type} executions.
Focus on: what worked well, key patterns, what made this successful.
Keep it concise (1-2 sentences).

Enhanced memory:"""

        response = await temp_agent.run(prompt)
        enhanced = response.strip()

        # Limit length
        if len(enhanced) > 200:
            enhanced = enhanced[:200] + "..."

        return enhanced

    except Exception as e:
        # Fallback to original summary
        return pattern.summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Populate initial agent memories")
    parser.add_argument("--days", type=int, help="Scan last N days (default: 30)")
    parser.add_argument("--full", action="store_true", help="Full rescan (ignore date filter)")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI embeddings (paid, higher quality)")
    parser.add_argument("--max-per-agent", type=int, default=20, help="Max memories per agent (default: 20)")
    args = parser.parse_args()

    asyncio.run(populate_initial_memories(
        days=args.days,
        full_rescan=args.full,
        use_openai=args.openai,
        max_memories_per_agent=args.max_per_agent
    ))


if __name__ == "__main__":
    main()
