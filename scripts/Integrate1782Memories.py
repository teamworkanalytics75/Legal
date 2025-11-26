#!/usr/bin/env python3
"""
Integrate 1782 Memories with Existing Agent Memory System

Merges 1782 case law memories with existing lawsuit database memories
so agents have access to both databases.
"""

import json
import sys
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
    from agents import ModelConfig
except ImportError as exc:
    print(f"Failed to import The Matrix modules: {exc}")
    sys.exit(1)

def load_existing_memories():
    """Load existing agent memories from memory store."""
    try:
        memory_store = MemoryStore()
        # Try to load from existing memory store
        if Path("memory_store/system_meta.json").exists():
            with open("memory_store/system_meta.json", 'r') as f:
                meta = json.load(f)
            print(f"📊 Existing memories: {meta['total_memories']} across {meta['agent_count']} agents")
            return memory_store, meta
        else:
            print("📊 No existing memory store found, creating new one")
            return MemoryStore(), {}
    except Exception as e:
        print(f"⚠️ Error loading existing memories: {e}")
        return MemoryStore(), {}

def load_1782_memory_seeds():
    """Load 1782 memory seeds."""
    with open("config/1782_memory_seeds.json", 'r') as f:
        return json.load(f)

def integrate_1782_memories():
    """Integrate 1782 memories with existing agent memory system."""

    print("🔄 Integrating 1782 Memories with Existing Agent System...")
    print("="*60)

    # Load existing memories
    memory_store, existing_meta = load_existing_memories()

    # Load 1782 memory seeds
    memory_seeds = load_1782_memory_seeds()

    print(f"📚 1782 Memory Seeds:")
    print(f"   Universal Skills: {len(memory_seeds['universal_skills'])}")
    print(f"   Agent-Specific: {sum(len(memories) for memories in memory_seeds['agents'].values())}")
    print(f"   Total Agents: {len(memory_seeds['agents'])}")

    # Track integration stats
    integration_stats = {
        "universal_skills_added": 0,
        "agent_memories_added": 0,
        "agents_updated": 0,
        "total_memories_before": existing_meta.get('total_memories', 0),
        "total_memories_after": 0
    }

    # Add universal skills
    print(f"\n📚 Adding Universal Skills...")
    for skill in memory_seeds['universal_skills']:
        try:
            memory = AgentMemory(
                agent_id="UniversalAgent",
                memory_type="capability",
                content=skill['nl'],
                metadata={
                    "skill_id": skill['id'],
                    "priority": skill['priority'],
                    "category": skill['category'],
                    "source": "1782_analysis",
                    "database": "1782_pdf_analysis.db",
                    "integration_date": datetime.now().isoformat()
                }
            )
            memory_store.store_memory(memory)
            integration_stats['universal_skills_added'] += 1
            print(f"   ✅ {skill['id']}: {skill['nl'][:50]}...")
        except Exception as e:
            print(f"   ❌ Error adding {skill['id']}: {e}")

    # Add agent-specific memories
    print(f"\n🤖 Adding Agent-Specific Memories...")
    for agent_name, memories in memory_seeds['agents'].items():
        print(f"\n   {agent_name}:")
        agent_memories_added = 0

        for memory_data in memories:
            try:
                memory = AgentMemory(
                    agent_id=agent_name,
                    memory_type="capability",
                    content=memory_data['nl'],
                    metadata={
                        "memory_id": memory_data['id'],
                        "priority": memory_data['priority'],
                        "category": memory_data['category'],
                        "source": "1782_analysis",
                        "database": "1782_pdf_analysis.db",
                        "integration_date": datetime.now().isoformat()
                    }
                )
                memory_store.store_memory(memory)
                agent_memories_added += 1
                integration_stats['agent_memories_added'] += 1
                print(f"     ✅ {memory_data['id']}: {memory_data['nl'][:40]}...")
            except Exception as e:
                print(f"     ❌ Error adding {memory_data['id']}: {e}")

        if agent_memories_added > 0:
            integration_stats['agents_updated'] += 1

    # Save integrated memory store
    memory_store.save_to_file("memory_store/integrated_memories.json")

    # Update system metadata
    integration_stats['total_memories_after'] = (
        integration_stats['total_memories_before'] +
        integration_stats['universal_skills_added'] +
        integration_stats['agent_memories_added']
    )

    # Create updated system metadata
    updated_meta = {
        "total_memories": integration_stats['total_memories_after'],
        "agent_count": existing_meta.get('agent_count', 0),
        "last_updated": datetime.now().isoformat(),
        "databases": {
            "lawsuit_docs": {
                "memories": integration_stats['total_memories_before'],
                "last_updated": existing_meta.get('last_updated', 'unknown')
            },
            "1782_pdf_analysis": {
                "memories": integration_stats['universal_skills_added'] + integration_stats['agent_memories_added'],
                "last_updated": datetime.now().isoformat()
            }
        },
        "integration_stats": integration_stats
    }

    # Save updated metadata
    with open("memory_store/system_meta.json", 'w') as f:
        json.dump(updated_meta, f, indent=2)

    return integration_stats, updated_meta

def create_agent_database_mapping():
    """Create mapping of which agents can access which databases."""

    mapping = {
        "databases": {
            "lawsuit_docs": {
                "type": "MySQL",
                "description": "General lawsuit database",
                "memories": "analysis_artifact, langchain_seed",
                "agents": "All 49 agents"
            },
            "1782_pdf_analysis": {
                "type": "SQLite",
                "description": "1782 case law analysis database",
                "memories": "1782_analysis",
                "agents": "All 49 agents"
            }
        },
        "agent_capabilities": {
            "UniversalAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["schema_discovery", "text_search", "1782_outcome_analysis", "1782_intel_factors"]
            },
            "OutcomePredictorAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["outcome_prediction", "1782_success_probability"]
            },
            "DocumentClassifierAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["document_classification", "1782_case_classification"]
            },
            "PatternRecognizerAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["pattern_recognition", "1782_pattern_recognition"]
            },
            "LegalResearchAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["legal_research", "1782_legal_research"]
            },
            "CaseAnalysisAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["case_analysis", "1782_case_analysis"]
            },
            "CitationAnalysisAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["citation_analysis", "1782_citation_analysis"]
            },
            "TextAnalysisAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["text_analysis", "1782_text_analysis"]
            },
            "StatisticalAnalysisAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["statistical_analysis", "1782_statistical_analysis"]
            },
            "ReportGenerationAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["report_generation", "1782_report_generation"]
            },
            "MemoryManagementAgent": {
                "databases": ["lawsuit_docs", "1782_pdf_analysis"],
                "capabilities": ["memory_management", "1782_memory_management"]
            }
        }
    }

    return mapping

def main():
    """Main execution function."""
    print("🚀 1782 Memory Integration with Existing Agent System")
    print("="*60)

    # Integrate memories
    integration_stats, updated_meta = integrate_1782_memories()

    # Create database mapping
    mapping = create_agent_database_mapping()

    # Save mapping
    with open("memory_store/agent_database_mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)

    # Print summary
    print(f"\n✅ Memory Integration Complete!")
    print(f"="*60)
    print(f"📊 Integration Summary:")
    print(f"   Memories Before: {integration_stats['total_memories_before']}")
    print(f"   Universal Skills Added: {integration_stats['universal_skills_added']}")
    print(f"   Agent Memories Added: {integration_stats['agent_memories_added']}")
    print(f"   Agents Updated: {integration_stats['agents_updated']}")
    print(f"   Total Memories After: {integration_stats['total_memories_after']}")

    print(f"\n🗄️ Database Access:")
    print(f"   Lawsuit Database: {updated_meta['databases']['lawsuit_docs']['memories']} memories")
    print(f"   1782 Database: {updated_meta['databases']['1782_pdf_analysis']['memories']} memories")

    print(f"\n🎯 Your 49 Agents Now Have Access To:")
    print(f"   ✅ Lawsuit Database (MySQL) - General legal analysis")
    print(f"   ✅ 1782 Database (SQLite) - Specific 1782 case law analysis")
    print(f"   ✅ Integrated Memory Store - Combined knowledge base")
    print(f"   ✅ Cross-Database Queries - Can query both databases")

    print(f"\n💡 Example Queries Your Agents Can Now Run:")
    print(f"   'Show me successful 1782 cases with Factor 2'")
    print(f"   'Find lawsuit patterns similar to 1782 denials'")
    print(f"   'Compare 1782 outcomes with general lawsuit outcomes'")
    print(f"   'Analyze citation patterns across both databases'")

    print(f"\n📁 Files Created/Updated:")
    print(f"   memory_store/integrated_memories.json")
    print(f"   memory_store/system_meta.json")
    print(f"   memory_store/agent_database_mapping.json")

if __name__ == "__main__":
    main()
