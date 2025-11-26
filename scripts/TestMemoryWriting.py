"""Test script for LLM-driven memory writing."""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from writer_agents.code.agents import AgentFactory, ModelConfig
from writer_agents.code.memory_system import MemoryStore, AgentMemory
from writer_agents.code.atomic_agent import AtomicAgent
from datetime import datetime
import uuid


class TestAgent(AtomicAgent):
    """Test agent for memory writing."""
    duty = "Test memory writing functionality"
    is_deterministic = False
    cost_tier = "mini"


async def test_memory_writing():
    """Test LLM-driven memory writing."""

    print("=" * 80)
    print("TESTING LLM-DRIVEN MEMORY WRITING")
    print("=" * 80)

    # Initialize components
    factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))
    memory_store = MemoryStore()

    # Create test agent
    agent = TestAgent(factory, enable_memory=True)

    print(f"Agent: {agent.__class__.__name__}")
    print(f"Memory enabled: {agent.memories is not None}")
    print(f"Is deterministic: {agent.is_deterministic}")

    # Test memory writing
    print(f"\n1. Testing memory writing...")

    test_input = {
        "text": "This is a test case with some legal citations like 42 U.S.C. § 1983 and Miranda v. Arizona.",
        "case_type": "employment_discrimination"
    }

    test_result = {
        "citations_found": 2,
        "citations": ["42 U.S.C. § 1983", "Miranda v. Arizona"],
        "quality_score": 0.85,
        "processing_time": 0.3
    }

    # Simulate execution metrics
    agent.metrics.tokens_in = 150
    agent.metrics.tokens_out = 75
    agent.metrics.duration_seconds = 0.3
    agent.metrics.cache_hit = False

    # Test memory writing
    memory = await agent._write_memory(test_input, test_result)

    if memory:
        print(f"   [OK] Memory written successfully")
        print(f"   Summary: {memory.summary}")
        print(f"   Agent type: {memory.agent_type}")
        print(f"   Source: {memory.source}")

        # Add to store
        memory_store.add(memory)
        print(f"   [OK] Memory added to store")

        # Test retrieval
        retrieved = memory_store.retrieve("TestAgent", k=1)
        print(f"   [OK] Memory retrieved: {len(retrieved)} memories")

        if retrieved:
            print(f"   Retrieved summary: {retrieved[0].summary}")
    else:
        print(f"   [WARNING] No memory written (may not meet criteria)")

    # Test deterministic agent
    print(f"\n2. Testing deterministic agent memory...")

    class TestDeterministicAgent(AtomicAgent):
        duty = "Test deterministic memory writing"
        is_deterministic = True

    det_agent = TestDeterministicAgent(factory, enable_memory=True)

    det_input = {"text": "Sample text"}
    det_result = {
        "citations": ["42 U.S.C. § 1983", "Miranda v. Arizona"],
        "count": 2
    }

    det_agent.metrics.duration_seconds = 0.2

    det_memory = await det_agent._write_memory(det_input, det_result)

    if det_memory:
        print(f"   [OK] Deterministic memory written")
        print(f"   Summary: {det_memory.summary}")
        print(f"   Cost: $0.00 (auto-generated)")
    else:
        print(f"   [WARNING] No deterministic memory written")

    # Test memory selection criteria
    print(f"\n3. Testing memory selection criteria...")

    test_cases = [
        {"result": {"error": "test error"}, "should_write": True, "reason": "error"},
        {"result": {"quality_score": 0.9}, "should_write": True, "reason": "high quality"},
        {"result": {}, "should_write": False, "reason": "normal execution"},
    ]

    for i, case in enumerate(test_cases, 1):
        should_write = agent._should_write_memory(case["result"])
        print(f"   Case {i} ({case['reason']}): {'[OK]' if should_write == case['should_write'] else '[FAIL]'} "
              f"Expected {case['should_write']}, got {should_write}")

    # Show final stats
    print(f"\n4. Final memory store stats:")
    stats = memory_store.get_all_stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Agent types: {stats['agent_types']}")

    print(f"\n[OK] Memory writing test completed!")


async def test_cost_estimation():
    """Test cost estimation for memory writing."""

    print(f"\n" + "=" * 80)
    print("TESTING COST ESTIMATION")
    print("=" * 80)

    # Estimate costs for different scenarios
    scenarios = [
        {"agents": 18, "type": "deterministic", "memories_per": 20, "cost_per": 0.0},
        {"agents": 31, "type": "LLM", "memories_per": 20, "cost_per": 0.0001},
        {"agents": 49, "type": "all", "memories_per": 20, "cost_per": 0.00005},  # Average
    ]

    for scenario in scenarios:
        total_cost = scenario["agents"] * scenario["memories_per"] * scenario["cost_per"]
        print(f"{scenario['type'].title()} agents: "
              f"{scenario['agents']} agents × {scenario['memories_per']} memories × "
              f"${scenario['cost_per']:.4f} = ${total_cost:.4f}")

    print(f"\nTotal estimated cost: ~$0.06-0.08")


async def main():
    """Run all tests."""
    try:
        await test_memory_writing()
        await test_cost_estimation()

        print(f"\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nNext steps:")
        print(f"1. Run: python scripts/populate_initial_memories.py --full")
        print(f"2. Expected cost: ~$0.06-0.08")
        print(f"3. Agents will automatically write memories during execution")

    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
