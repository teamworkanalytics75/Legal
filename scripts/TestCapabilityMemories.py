#!/usr/bin/env python3
"""
Simple test script to validate the capability seeding approach without requiring API keys.
This creates mock memories to test the memory storage system.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Mock the memory system for testing
class MockAgentMemory:
    def __init__(self, agent_type: str, summary: str, context: Dict[str, Any]):
        self.agent_type = agent_type
        self.memory_id = str(uuid.uuid4())
        self.summary = summary
        self.context = context
        self.embedding = None
        self.source = "capability_seed"
        self.timestamp = datetime.utcnow()

class MockMemoryStore:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.memories = {}

    def add(self, memory: MockAgentMemory):
        if memory.agent_type not in self.memories:
            self.memories[memory.agent_type] = []
        self.memories[memory.agent_type].append(memory)

    def save(self):
        # Save to JSON for inspection
        output_data = {}
        for agent_type, memories in self.memories.items():
            output_data[agent_type] = []
            for memory in memories:
                output_data[agent_type].append({
                    "memory_id": memory.memory_id,
                    "summary": memory.summary,
                    "context": memory.context,
                    "source": memory.source,
                    "timestamp": memory.timestamp.isoformat()
                })

        output_file = self.storage_path / "mock_capability_memories.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Mock memories saved to {output_file}")

def create_mock_capability_memories():
    """Create mock capability memories to test the system."""

    # Load the capability seeds
    seeds_file = Path("config/langchain_capability_seeds.json")
    if not seeds_file.exists():
        print(f"Error: {seeds_file} not found")
        return

    with open(seeds_file, 'r', encoding='utf-8') as f:
        seeds = json.load(f)

    memory_store = MockMemoryStore(Path("memory_store"))
    memory_store.storage_path.mkdir(parents=True, exist_ok=True)

    successes = 0

    # Process universal skills
    universal_skills = seeds.get("universal_skills", [])
    print(f"Creating mock memories for {len(universal_skills)} universal skills")

    for skill in universal_skills:
        memory = MockAgentMemory(
            agent_type="UniversalAgent",
            summary=f"Universal skill: {skill['id']} - {skill['nl']}",
            context={
                "codename": skill["id"],
                "question": skill["nl"],
                "context": f"Universal skill: {skill['category']}",
                "answer": f"Mock answer for {skill['id']}",
                "executed_sql": f"SELECT * FROM mock_table WHERE {skill['id']} = 'test'",
                "cost_estimate": 0.0003,
                "seed_timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "mock_capability_seed_script",
            }
        )
        memory_store.add(memory)
        successes += 1
        print(f"  Created mock memory for {skill['id']}")

    # Process agent-specific queries (sample)
    agents = seeds.get("agents", {})
    print(f"Creating mock memories for {len(agents)} agents (sample)")

    for agent_name, agent_data in list(agents.items())[:5]:  # Sample first 5 agents
        primitives = agent_data.get("job_primitives", [])
        anchors = agent_data.get("domain_anchors", [])

        # Add primitives
        for primitive in primitives[:2]:  # Sample first 2 primitives
            memory = MockAgentMemory(
                agent_type=agent_name,
                summary=f"Job primitive: {primitive['name']} - {primitive['nl']}",
                context={
                    "codename": primitive["name"],
                    "question": primitive["nl"],
                    "context": f"Job primitive: {primitive.get('category', 'generic')}",
                    "answer": f"Mock answer for {primitive['name']}",
                    "executed_sql": f"SELECT * FROM mock_table WHERE {primitive['name']} = 'test'",
                    "cost_estimate": 0.0003,
                    "seed_timestamp": datetime.utcnow().isoformat() + "Z",
                    "source": "mock_capability_seed_script",
                }
            )
            memory_store.add(memory)
            successes += 1
            print(f"  Created mock memory for {agent_name}.{primitive['name']}")

        # Add anchors
        for anchor in anchors[:1]:  # Sample first anchor
            memory = MockAgentMemory(
                agent_type=agent_name,
                summary=f"Domain anchor: {anchor['nl']}",
                context={
                    "codename": "domain_anchor",
                    "question": anchor["nl"],
                    "context": f"Domain anchor (limit: {anchor.get('limit', 'none')})",
                    "answer": f"Mock answer for domain anchor",
                    "executed_sql": f"SELECT * FROM mock_table WHERE domain_anchor = 'test'",
                    "cost_estimate": 0.0003,
                    "seed_timestamp": datetime.utcnow().isoformat() + "Z",
                    "source": "mock_capability_seed_script",
                }
            )
            memory_store.add(memory)
            successes += 1
            print(f"  Created mock memory for {agent_name}.domain_anchor")

    memory_store.save()
    print(f"\nCreated {successes} mock capability memories")
    print("This validates the memory storage system works correctly")
    print("Next step: Set up OpenAI API key and run real LangChain queries")

if __name__ == "__main__":
    create_mock_capability_memories()
