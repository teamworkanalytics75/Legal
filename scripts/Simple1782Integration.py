#!/usr/bin/env python3
"""
Simple 1782 Memory Integration - Batch Processing

Integrates 1782 memories in smaller batches to avoid timeouts.
"""

import json
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the writer_agents code to path
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = PROJECT_ROOT / "writer_agents" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

try:
    from memory_system import AgentMemory, MemoryStore
except ImportError as exc:
    print(f"Failed to import The Matrix modules: {exc}")
    sys.exit(1)

def load_1782_memory_seeds():
    """Load 1782 memory seeds."""
    with open("config/1782_memory_seeds.json", 'r') as f:
        return json.load(f)

def integrate_universal_skills():
    """Integrate universal skills only."""
    print("🔄 Integrating Universal Skills...")

    memory_seeds = load_1782_memory_seeds()
    memory_store = MemoryStore()

    universal_skills = memory_seeds['universal_skills']
    print(f"📚 Processing {len(universal_skills)} universal skills...")

    for i, skill in enumerate(universal_skills, 1):
        try:
            memory = AgentMemory(
                agent_type="UniversalAgent",
                memory_id=str(uuid.uuid4()),
                summary=f"1782 Universal Skill: {skill['id']} - {skill['nl']}",
                context={
                    "skill_id": skill['id'],
                    "question": skill['nl'],
                    "priority": skill['priority'],
                    "category": skill['category'],
                    "source": "1782_analysis",
                    "database": "1782_pdf_analysis.db",
                    "integration_date": datetime.now().isoformat()
                },
                source="1782_capability_seed"
            )
            memory_store.add(memory)
            print(f"   ✅ {i}/{len(universal_skills)}: {skill['id']}")
        except Exception as e:
            print(f"   ❌ Error adding {skill['id']}: {e}")

    # Save batch
    memory_store.save_to_file("memory_store/1782_universal_skills.json")
    print(f"✅ Universal skills integrated: {len(universal_skills)} memories")
    return len(universal_skills)

def integrate_agent_memories(agent_name: str):
    """Integrate memories for a specific agent."""
    print(f"🔄 Integrating {agent_name} memories...")

    memory_seeds = load_1782_memory_seeds()
    memory_store = MemoryStore()

    if agent_name not in memory_seeds['agents']:
        print(f"❌ No memories found for {agent_name}")
        return 0

    memories = memory_seeds['agents'][agent_name]
    print(f"📚 Processing {len(memories)} memories for {agent_name}...")

    for i, memory_data in enumerate(memories, 1):
        try:
            memory = AgentMemory(
                agent_type=agent_name,
                memory_id=str(uuid.uuid4()),
                summary=f"1782 {agent_name}: {memory_data['id']} - {memory_data['nl']}",
                context={
                    "memory_id": memory_data['id'],
                    "question": memory_data['nl'],
                    "priority": memory_data['priority'],
                    "category": memory_data['category'],
                    "source": "1782_analysis",
                    "database": "1782_pdf_analysis.db",
                    "integration_date": datetime.now().isoformat()
                },
                source="1782_capability_seed"
            )
            memory_store.add(memory)
            print(f"   ✅ {i}/{len(memories)}: {memory_data['id']}")
        except Exception as e:
            print(f"   ❌ Error adding {memory_data['id']}: {e}")

    # Save batch
    memory_store.save_to_file(f"memory_store/1782_{agent_name.lower()}_memories.json")
    print(f"✅ {agent_name} memories integrated: {len(memories)} memories")
    return len(memories)

def main():
    """Main execution function."""
    print("🚀 Simple 1782 Memory Integration (Batch Processing)")
    print("="*60)

    # Step 1: Universal skills
    universal_count = integrate_universal_skills()

    # Step 2: Agent memories (one at a time)
    memory_seeds = load_1782_memory_seeds()
    agent_names = list(memory_seeds['agents'].keys())

    print(f"\n🔄 Processing {len(agent_names)} agents...")
    total_agent_memories = 0

    for agent_name in agent_names:
        count = integrate_agent_memories(agent_name)
        total_agent_memories += count

    # Summary
    print(f"\n✅ Memory Integration Complete!")
    print(f"="*60)
    print(f"📊 Integration Summary:")
    print(f"   Universal Skills: {universal_count}")
    print(f"   Agent Memories: {total_agent_memories}")
    print(f"   Total Memories: {universal_count + total_agent_memories}")

    print(f"\n🎯 Your agents now have 1782 memories!")
    print(f"📁 Files created in memory_store/ directory")

if __name__ == "__main__":
    main()
