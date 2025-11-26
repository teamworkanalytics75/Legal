#!/usr/bin/env python3
"""
1782 Case Law Query Interface for 49 Agents

Demonstrates how agents can query the 1782 PDF analysis database
using natural language queries.
"""

import json
import sqlite3
from pathlib import Path

def load_1782_memory_seeds():
    """Load the 1782 memory seeds."""
    with open("config/1782_memory_seeds.json", 'r') as f:
        return json.load(f)

def query_1782_database(query, db_path="data/case_law/1782_pdf_analysis.db"):
    """Query the 1782 database with SQL."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]

        # Convert to list of dictionaries
        data = []
        for row in results:
            data.append(dict(zip(columns, row)))

        return data
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

def demonstrate_agent_queries():
    """Demonstrate how agents can query 1782 data."""

    print("🤖 1782 Agent Query Demonstrations")
    print("="*50)

    # Load memory seeds
    memory_seeds = load_1782_memory_seeds()

    # Universal skills queries
    print("\n📚 Universal Skills Queries:")
    print("-" * 30)

    universal_queries = {
        "1782_outcome_analysis": "SELECT outcome, COUNT(*) as count FROM cases GROUP BY outcome",
        "1782_intel_factors": "SELECT factor_name, COUNT(*) as detected_count FROM intel_factors WHERE detected = 1 GROUP BY factor_name",
        "1782_success_patterns": "SELECT c.outcome, COUNT(if.factor_name) as factors_discussed FROM cases c JOIN intel_factors if ON c.cluster_id = if.cluster_id WHERE if.detected = 1 GROUP BY c.outcome",
        "1782_citation_analysis": "SELECT citation_pattern, SUM(citation_count) as total_mentions FROM citations GROUP BY citation_pattern ORDER BY total_mentions DESC",
        "1782_text_length_correlation": "SELECT outcome, AVG(text_length) as avg_length FROM cases GROUP BY outcome"
    }

    for skill_id, query in universal_queries.items():
        print(f"\n🔍 {skill_id}:")
        print(f"   Query: {query}")
        results = query_1782_database(query)
        if "error" not in results:
            print(f"   Results: {results}")
        else:
            print(f"   Error: {results['error']}")

    # Agent-specific queries
    print(f"\n🤖 Agent-Specific Queries:")
    print("-" * 30)

    agent_queries = {
        "OutcomePredictorAgent": {
            "query": "SELECT c.case_name, c.outcome, COUNT(if.factor_name) as factors_discussed FROM cases c JOIN intel_factors if ON c.cluster_id = if.cluster_id WHERE if.detected = 1 GROUP BY c.cluster_id ORDER BY c.outcome",
            "description": "Predict outcomes based on Intel factors"
        },
        "DocumentClassifierAgent": {
            "query": "SELECT c.outcome, COUNT(ci.citation_pattern) as citation_types FROM cases c JOIN citations ci ON c.cluster_id = ci.cluster_id GROUP BY c.outcome",
            "description": "Classify cases by citation patterns"
        },
        "PatternRecognizerAgent": {
            "query": "SELECT if.factor_name, c.outcome, COUNT(*) as count FROM intel_factors if JOIN cases c ON if.cluster_id = c.cluster_id WHERE if.detected = 1 GROUP BY if.factor_name, c.outcome",
            "description": "Recognize patterns in Intel factors"
        },
        "StatisticalAnalysisAgent": {
            "query": "SELECT outcome, AVG(section_1782_mentions) as avg_mentions, AVG(text_length) as avg_length FROM cases GROUP BY outcome",
            "description": "Statistical analysis of case outcomes"
        }
    }

    for agent_name, query_info in agent_queries.items():
        print(f"\n🎯 {agent_name}:")
        print(f"   Description: {query_info['description']}")
        print(f"   Query: {query_info['query']}")
        results = query_1782_database(query_info['query'])
        if "error" not in results:
            print(f"   Results: {results}")
        else:
            print(f"   Error: {results['error']}")

def create_agent_query_examples():
    """Create example queries for each agent type."""

    examples = {
        "OutcomePredictorAgent": [
            "What's the success rate when Factor 2 is present?",
            "Which cases are most likely to succeed?",
            "What factors predict denial?"
        ],
        "DocumentClassifierAgent": [
            "Classify cases by Intel factor completeness",
            "Group cases by citation patterns",
            "Categorize by outcome and complexity"
        ],
        "PatternRecognizerAgent": [
            "Find patterns in denied cases",
            "Identify success correlates",
            "Discover hidden patterns"
        ],
        "LegalResearchAgent": [
            "Research Intel Corp citation patterns",
            "Analyze ZF Automotive influence",
            "Study precedent evolution"
        ],
        "CaseAnalysisAgent": [
            "Analyze individual case outcomes",
            "Compare factor analysis across cases",
            "Evaluate case complexity"
        ],
        "CitationAnalysisAgent": [
            "Track citation frequency patterns",
            "Analyze precedent influence",
            "Study citation correlation with outcomes"
        ],
        "TextAnalysisAgent": [
            "Analyze text length patterns",
            "Study Section 1782 mention frequency",
            "Examine language patterns"
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

    return examples

def main():
    """Main execution function."""
    print("🚀 1782 Case Law Query Interface for 49 Agents")
    print("="*60)

    # Demonstrate queries
    demonstrate_agent_queries()

    # Create query examples
    examples = create_agent_query_examples()

    # Save examples
    examples_file = "config/1782_agent_query_examples.json"
    with open(examples_file, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f"\n📋 Query Examples Saved: {examples_file}")

    print(f"\n✅ 1782 Agent Query Interface Ready!")
    print(f"🎯 Your 49 agents can now query 1782 case law data!")
    print(f"💡 Example: 'Show me successful cases with Factor 2'")
    print(f"🧠 Memory seeds: config/1782_memory_seeds.json")
    print(f"🗄️ Database: data/case_law/1782_pdf_analysis.db")

if __name__ == "__main__":
    main()
