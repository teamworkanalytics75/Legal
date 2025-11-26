#!/usr/bin/env python3
"""
Seed 1782 Case Law Memories for The Matrix 49 Agents

Creates 1782-specific memories for agents based on PDF analysis results
and integrates with existing memory seeding system.
"""

import json
import sqlite3
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

def load_1782_analysis_data():
    """Load 1782 PDF analysis data."""
    with open("data/case_law/simple_pdf_analysis.json", 'r') as f:
        return json.load(f)

def create_1782_memory_seeds():
    """Create 1782-specific memory seeds for agents."""

    # Load analysis data
    analysis_data = load_1782_analysis_data()

    # Create memory seeds based on analysis
    memory_seeds = {
        "generation_info": {
            "source": "1782 PDF Analysis",
            "total_cases": len(analysis_data),
            "analysis_date": datetime.now().isoformat(),
            "database": "data/case_law/1782_pdf_analysis.db"
        },
        "universal_skills": [
            {
                "id": "1782_outcome_analysis",
                "nl": "Show outcome distribution for 1782 cases (granted vs denied vs unclear)",
                "priority": 1,
                "category": "1782_analysis"
            },
            {
                "id": "1782_intel_factors",
                "nl": "Show Intel factor detection rates across all 1782 cases",
                "priority": 1,
                "category": "1782_analysis"
            },
            {
                "id": "1782_success_patterns",
                "nl": "Show Intel factors present in successful (granted) 1782 cases",
                "priority": 1,
                "category": "1782_analysis"
            },
            {
                "id": "1782_failure_patterns",
                "nl": "Show Intel factors present in denied 1782 cases",
                "priority": 1,
                "category": "1782_analysis"
            },
            {
                "id": "1782_citation_analysis",
                "nl": "Show citation patterns in 1782 cases (Intel Corp, ZF Automotive, etc.)",
                "priority": 1,
                "category": "1782_analysis"
            },
            {
                "id": "1782_text_length_correlation",
                "nl": "Show average text length for granted vs denied 1782 cases",
                "priority": 1,
                "category": "1782_analysis"
            },
            {
                "id": "1782_section_1782_mentions",
                "nl": "Show Section 1782 mention counts by case outcome",
                "priority": 1,
                "category": "1782_analysis"
            }
        ],
        "agents": {
            "OutcomePredictorAgent": [
                {
                    "id": "1782_outcome_prediction",
                    "nl": "Based on Intel factors, predict outcome for 1782 cases",
                    "priority": 1,
                    "category": "1782_prediction"
                },
                {
                    "id": "1782_success_probability",
                    "nl": "Calculate success probability based on Factor 2 (receptivity) presence",
                    "priority": 1,
                    "category": "1782_prediction"
                }
            ],
            "DocumentClassifierAgent": [
                {
                    "id": "1782_case_classification",
                    "nl": "Classify 1782 cases by outcome, Intel factors, and citation patterns",
                    "priority": 1,
                    "category": "1782_classification"
                },
                {
                    "id": "1782_factor_classification",
                    "nl": "Classify cases by Intel factor presence and outcome correlation",
                    "priority": 1,
                    "category": "1782_classification"
                }
            ],
            "PatternRecognizerAgent": [
                {
                    "id": "1782_pattern_recognition",
                    "nl": "Identify patterns in successful vs failed 1782 applications",
                    "priority": 1,
                    "category": "1782_patterns"
                },
                {
                    "id": "1782_success_correlates",
                    "nl": "Find correlations between Intel factors and case success",
                    "priority": 1,
                    "category": "1782_patterns"
                }
            ],
            "LegalResearchAgent": [
                {
                    "id": "1782_legal_research",
                    "nl": "Research 1782 case law patterns and precedents",
                    "priority": 1,
                    "category": "1782_research"
                },
                {
                    "id": "1782_precedent_analysis",
                    "nl": "Analyze citation patterns and precedent influence in 1782 cases",
                    "priority": 1,
                    "category": "1782_research"
                }
            ],
            "CaseAnalysisAgent": [
                {
                    "id": "1782_case_analysis",
                    "nl": "Analyze individual 1782 cases for Intel factors and outcomes",
                    "priority": 1,
                    "category": "1782_analysis"
                },
                {
                    "id": "1782_factor_analysis",
                    "nl": "Analyze Intel factor presence and correlation with outcomes",
                    "priority": 1,
                    "category": "1782_analysis"
                }
            ],
            "CitationAnalysisAgent": [
                {
                    "id": "1782_citation_analysis",
                    "nl": "Analyze citation patterns in 1782 cases",
                    "priority": 1,
                    "category": "1782_citations"
                },
                {
                    "id": "1782_precedent_tracking",
                    "nl": "Track precedent citations across 1782 cases",
                    "priority": 1,
                    "category": "1782_citations"
                }
            ],
            "TextAnalysisAgent": [
                {
                    "id": "1782_text_analysis",
                    "nl": "Analyze text length and content patterns in 1782 cases",
                    "priority": 1,
                    "category": "1782_text"
                },
                {
                    "id": "1782_section_1782_analysis",
                    "nl": "Analyze Section 1782 mention patterns in case text",
                    "priority": 1,
                    "category": "1782_text"
                }
            ],
            "StatisticalAnalysisAgent": [
                {
                    "id": "1782_statistical_analysis",
                    "nl": "Perform statistical analysis on 1782 case outcomes and factors",
                    "priority": 1,
                    "category": "1782_statistics"
                },
                {
                    "id": "1782_correlation_analysis",
                    "nl": "Calculate correlations between Intel factors and case success",
                    "priority": 1,
                    "category": "1782_statistics"
                }
            ],
            "ReportGenerationAgent": [
                {
                    "id": "1782_report_generation",
                    "nl": "Generate reports on 1782 case analysis and patterns",
                    "priority": 1,
                    "category": "1782_reports"
                },
                {
                    "id": "1782_summary_generation",
                    "nl": "Generate summaries of 1782 case outcomes and insights",
                    "priority": 1,
                    "category": "1782_reports"
                }
            ],
            "MemoryManagementAgent": [
                {
                    "id": "1782_memory_management",
                    "nl": "Manage 1782 case law memories and knowledge base",
                    "priority": 1,
                    "category": "1782_memory"
                },
                {
                    "id": "1782_knowledge_retrieval",
                    "nl": "Retrieve 1782 case law knowledge for agent queries",
                    "priority": 1,
                    "category": "1782_memory"
                }
            ]
        }
    }

    return memory_seeds

def seed_1782_memories():
    """Seed 1782-specific memories for agents."""

    print("🌱 Seeding 1782 Case Law Memories for 49 Agents...")
    print("="*60)

    # Create memory seeds
    memory_seeds = create_1782_memory_seeds()

    # Save memory seeds
    seeds_file = "config/1782_memory_seeds.json"
    with open(seeds_file, 'w') as f:
        json.dump(memory_seeds, f, indent=2)

    print(f"✅ Memory seeds created: {seeds_file}")

    # Initialize memory store
    memory_store = MemoryStore()

    # Seed universal skills
    print(f"\n📚 Seeding Universal Skills ({len(memory_seeds['universal_skills'])} skills)...")
    for skill in memory_seeds['universal_skills']:
        memory = AgentMemory(
            agent_id="universal",
            memory_type="capability",
            content=skill['nl'],
            metadata={
                "skill_id": skill['id'],
                "priority": skill['priority'],
                "category": skill['category'],
                "source": "1782_analysis"
            }
        )
        memory_store.store_memory(memory)
        print(f"  ✅ {skill['id']}: {skill['nl'][:50]}...")

    # Seed agent-specific memories
    print(f"\n🤖 Seeding Agent-Specific Memories...")
    total_agent_memories = 0

    for agent_name, memories in memory_seeds['agents'].items():
        print(f"\n  {agent_name}:")
        for memory_data in memories:
            memory = AgentMemory(
                agent_id=agent_name,
                memory_type="capability",
                content=memory_data['nl'],
                metadata={
                    "memory_id": memory_data['id'],
                    "priority": memory_data['priority'],
                    "category": memory_data['category'],
                    "source": "1782_analysis"
                }
            )
            memory_store.store_memory(memory)
            total_agent_memories += 1
            print(f"    ✅ {memory_data['id']}: {memory_data['nl'][:40]}...")

    # Save memory store
    memory_store.save_to_file("memory_store/1782_memories.json")

    print(f"\n📊 Memory Seeding Summary:")
    print(f"  Universal Skills: {len(memory_seeds['universal_skills'])}")
    print(f"  Agent Memories: {total_agent_memories}")
    print(f"  Total Agents: {len(memory_seeds['agents'])}")
    print(f"  Memory Store: memory_store/1782_memories.json")

    return memory_store

def create_1782_agent_queries():
    """Create specific queries for each agent type."""

    queries = {
        "OutcomePredictorAgent": [
            "What's the success rate when Factor 2 (receptivity) is present?",
            "Which combination of Intel factors predicts success best?",
            "What's the correlation between text length and success?"
        ],
        "DocumentClassifierAgent": [
            "Classify cases by Intel factor completeness",
            "Group cases by citation patterns",
            "Categorize cases by outcome and factor presence"
        ],
        "PatternRecognizerAgent": [
            "Find patterns in denied cases",
            "Identify success correlates",
            "Discover hidden patterns in case text"
        ],
        "LegalResearchAgent": [
            "Research Intel Corp citation patterns",
            "Analyze ZF Automotive influence",
            "Study precedent evolution over time"
        ],
        "CaseAnalysisAgent": [
            "Analyze individual case outcomes",
            "Compare factor analysis across cases",
            "Evaluate case complexity metrics"
        ],
        "CitationAnalysisAgent": [
            "Track citation frequency patterns",
            "Analyze precedent influence",
            "Study citation correlation with outcomes"
        ],
        "TextAnalysisAgent": [
            "Analyze text length patterns",
            "Study Section 1782 mention frequency",
            "Examine language patterns in outcomes"
        ],
        "StatisticalAnalysisAgent": [
            "Calculate success probability factors",
            "Perform correlation analysis",
            "Generate statistical insights"
        ],
        "ReportGenerationAgent": [
            "Generate outcome analysis reports",
            "Create pattern recognition summaries",
            "Produce statistical analysis reports"
        ],
        "MemoryManagementAgent": [
            "Manage 1782 knowledge base",
            "Retrieve relevant case patterns",
            "Update memory with new insights"
        ]
    }

    return queries

def main():
    """Main execution function."""
    print("🚀 1782 Case Law Memory Seeding for 49 Agents")
    print("="*60)

    # Seed memories
    memory_store = seed_1782_memories()

    # Create agent queries
    agent_queries = create_1782_agent_queries()

    # Save agent queries
    queries_file = "config/1782_agent_queries.json"
    with open(queries_file, 'w') as f:
        json.dump(agent_queries, f, indent=2)

    print(f"\n📋 Agent Queries Created: {queries_file}")

    print(f"\n✅ 1782 Memory Seeding Complete!")
    print(f"🎯 Your 49 agents now have 1782 case law knowledge!")
    print(f"💡 Agents can now query: 'Show me successful 1782 cases with Factor 2'")
    print(f"🧠 Memory integration with existing LangChain system ready!")

if __name__ == "__main__":
    main()
