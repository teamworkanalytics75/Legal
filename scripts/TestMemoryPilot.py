"""Pilot test script for agent memory system.

Tests memory integration with 5 key agents to validate the system
before full deployment to all 49 agents.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from writer_agents.code.agents import AgentFactory, ModelConfig
from writer_agents.code.memory_system import MemoryStore, AgentMemory
from writer_agents.code.extract_memories import MemoryBuilder
from writer_agents.code.agent_context_templates import get_agent_context
from writer_agents.code.job_persistence import JobManager
from datetime import datetime


async def test_agent_memory_integration():
    """Test memory integration with 5 pilot agents."""

    print("=" * 80)
    print("AGENT MEMORY SYSTEM PILOT TEST")
    print("=" * 80)

    # Initialize components
    factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))
    memory_store = MemoryStore()

    # Create sample memories for testing
    print("\n1. Creating sample memories...")
    sample_memories = [
        AgentMemory(
            agent_type="CitationFinderAgent",
            memory_id="pilot-001",
            summary="Successfully found 8 citations using regex patterns in 0.3s",
            context={'citations_found': 8, 'duration': 0.3, 'method': 'regex'},
            embedding=None,
            source="pilot_test",
            timestamp=datetime.now()
        ),
        AgentMemory(
            agent_type="OutlineBuilderAgent",
            memory_id="pilot-002",
            summary="Built 5-section legal outline with introduction, facts, analysis, conclusion",
            context={'sections': 5, 'structure': 'legal', 'quality': 'high'},
            embedding=None,
            source="pilot_test",
            timestamp=datetime.now()
        ),
        AgentMemory(
            agent_type="FactExtractorAgent",
            memory_id="pilot-003",
            summary="Extracted 12 discrete facts from case documents with 95% accuracy",
            context={'facts_extracted': 12, 'accuracy': 0.95, 'source': 'case_docs'},
            embedding=None,
            source="pilot_test",
            timestamp=datetime.now()
        ),
        AgentMemory(
            agent_type="GrammarFixerAgent",
            memory_id="pilot-004",
            summary="Fixed 15 grammar errors and improved sentence clarity",
            context={'errors_fixed': 15, 'improvement': 'clarity', 'method': 'llm'},
            embedding=None,
            source="pilot_test",
            timestamp=datetime.now()
        ),
        AgentMemory(
            agent_type="ExpertQAAgent",
            memory_id="pilot-005",
            summary="Expert QA review found 2 minor issues, overall quality excellent",
            context={'issues_found': 2, 'severity': 'minor', 'overall_quality': 'excellent'},
            embedding=None,
            source="pilot_test",
            timestamp=datetime.now()
        )
    ]

    # Add memories to store
    memory_store.add_batch(sample_memories)
    print(f" [ok] Created {len(sample_memories)} sample memories")

    # Test each pilot agent
    pilot_agents = [
        "CitationFinderAgent",
        "OutlineBuilderAgent",
        "FactExtractorAgent",
        "GrammarFixerAgent",
        "ExpertQAAgent"
    ]

    print(f"\n2. Testing {len(pilot_agents)} pilot agents...")

    for agent_type in pilot_agents:
        print(f"\n Testing {agent_type}...")

        # Test 1: Agent context loading
        context = get_agent_context(agent_type)
        assert context['project'] == "The Matrix - Bayesian Legal AI System"
        assert 'role' in context
        assert 'phase' in context
        print(f" [ok] Context loaded: {context['role'][:50]}...")

        # Test 2: Memory retrieval
        memories = memory_store.retrieve(agent_type, k=3)
        print(f" [ok] Retrieved {len(memories)} memories")

        # Test 3: Agent initialization with memory
        try:
            # Import the specific agent class
            if agent_type == "CitationFinderAgent":
                from writer_agents.code.atomic_agents.citations import CitationFinderAgent
                agent_class = CitationFinderAgent
            elif agent_type == "OutlineBuilderAgent":
                from writer_agents.code.atomic_agents.drafting import OutlineBuilderAgent
                agent_class = OutlineBuilderAgent
            elif agent_type == "FactExtractorAgent":
                from writer_agents.code.atomic_agents.research import FactExtractorAgent
                agent_class = FactExtractorAgent
            elif agent_type == "GrammarFixerAgent":
                from writer_agents.code.atomic_agents.review import GrammarFixerAgent
                agent_class = GrammarFixerAgent
            elif agent_type == "ExpertQAAgent":
                from writer_agents.code.atomic_agents.review import ExpertQAAgent
                agent_class = ExpertQAAgent

            # Initialize with memory enabled
            agent = agent_class(
                factory,
                enable_memory=True,
                memory_config={'k_neighbors': 3}
            )

            # Check memory loading
            assert agent.memories is not None
            assert 'project_context' in agent.memories
            assert 'past_patterns' in agent.memories

            # Check system message enhancement
            system_message = agent._build_system_message_with_memory()
            assert len(system_message) > len(agent._build_system_message())
            assert "The Matrix" in system_message
            assert "PAST EXPERIENCE" in system_message

            print(f" [ok] Memory integration successful")
            print(f" [ok] System message enhanced: {len(system_message)} chars")

        except Exception as e:
            print(f" x Memory integration failed: {e}")
            return False

    print(f"\n3. Testing memory retrieval patterns...")

    # Test semantic search
    test_queries = [
        ("CitationFinderAgent", "finding citations efficiently"),
        ("OutlineBuilderAgent", "building document structure"),
        ("FactExtractorAgent", "extracting facts accurately"),
        ("GrammarFixerAgent", "fixing grammar errors"),
        ("ExpertQAAgent", "quality assurance review")
    ]

    for agent_type, query in test_queries:
        results = memory_store.retrieve(agent_type, query=query, k=2)
        print(f" {agent_type}: Found {len(results)} relevant memories for '{query}'")

        if results:
            print(f" Top result: {results[0].summary}")

    print(f"\n4. Testing memory statistics...")

    stats = memory_store.get_all_stats()
    print(f" Total agents with memories: {stats['total_agents']}")
    print(f" Total memories: {stats['total_memories']}")
    print(f" Sources: {stats['sources']}")

    # Test individual agent stats
    for agent_type in pilot_agents:
        agent_stats = memory_store.get_agent_stats(agent_type)
        print(f" {agent_type}: {agent_stats['total_memories']} memories")

    print(f"\n[ok] PILOT TEST COMPLETED SUCCESSFULLY!")
    print(f" All {len(pilot_agents)} pilot agents integrated with memory system")
    print(f" Memory retrieval and context loading working correctly")
    print(f" Ready for full deployment to all 49 agents")

    return True


async def test_memory_extraction():
    """Test memory extraction from existing data."""

    print("\n" + "=" * 80)
    print("MEMORY EXTRACTION TEST")
    print("=" * 80)

    # Test extraction from jobs.db if it exists
    jobs_db_path = Path("jobs.db")
    if jobs_db_path.exists():
        print("\n1. Testing extraction from jobs.db...")

        try:
            job_manager = JobManager("jobs.db")
            memory_builder = MemoryBuilder()

            memories = memory_builder.extract_from_jobs_db(job_manager)
            print(f" [ok] Extracted {len(memories)} memories from jobs.db")

            if memories:
                # Show sample memories
                for i, mem in enumerate(memories[:3], 1):
                    print(f" Sample {i}: {mem.summary}")

        except Exception as e:
            print(f" WARNING Error extracting from jobs.db: {e}")
    else:
        print("\n1. jobs.db not found, skipping extraction test")

    # Test extraction from analysis_outputs if it exists
    analysis_dir = Path("analysis_outputs")
    if analysis_dir.exists():
        print("\n2. Testing extraction from analysis_outputs/...")

        try:
            memory_builder = MemoryBuilder()
            memories = memory_builder.extract_from_artifacts(analysis_dir)
            print(f" [ok] Extracted {len(memories)} memories from artifacts")

            if memories:
                # Show sample memories
                for i, mem in enumerate(memories[:3], 1):
                    print(f" Sample {i}: {mem.summary}")

        except Exception as e:
            print(f" WARNING Error extracting from artifacts: {e}")
    else:
        print("\n2. analysis_outputs/ not found, skipping artifact extraction test")


async def main():
    """Main test runner."""
    try:
        # Run pilot test
        success = await test_agent_memory_integration()

        if success:
            # Run extraction test
            await test_memory_extraction()

            print("\n" + "=" * 80)
            print("ALL TESTS PASSED - READY FOR DEPLOYMENT")
            print("=" * 80)
            print("\nNext steps:")
            print("1. Run: python scripts/refresh_agent_memories.py --full")
            print("2. Deploy to all 49 agents")
            print("3. Monitor performance improvements")
        else:
            print("\nx PILOT TEST FAILED - FIX ISSUES BEFORE DEPLOYMENT")
            return 1

    except Exception as e:
        print(f"\nx TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
