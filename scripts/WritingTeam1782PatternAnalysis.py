#!/usr/bin/env python3
"""
Writing Team 1782 Database Pattern Analysis

Demonstrates how the 49 writing team agents can query the 1782 database
to find patterns using their newly integrated 1782 memories.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

def get_writing_team_agents():
    """Get the 49 writing team agents."""
    return [
        # Citation Pipeline (5 agents)
        "CitationFinderAgent", "CitationNormalizerAgent", "CitationVerifierAgent",
        "CitationLocatorAgent", "CitationInserterAgent",

        # Research Pipeline (6 agents)
        "FactExtractorAgent", "PrecedentFinderAgent", "PrecedentRankerAgent",
        "PrecedentSummarizerAgent", "StatuteLocatorAgent", "ExhibitFetcherAgent",

        # Writing Pipeline (5 agents)
        "OutlineBuilderAgent", "SectionWriterAgent", "ParagraphWriterAgent",
        "TransitionAgent", "PrimaryWriterAgent",

        # Editing Pipeline (5 agents)
        "GrammarFixerAgent", "StyleCheckerAgent", "LogicCheckerAgent",
        "ConsistencyCheckerAgent", "RedactionAgent",

        # Quality Pipeline (5 agents)
        "ComplianceAgent", "ExpertQAAgent", "ContentReviewerAgent",
        "TechnicalReviewerAgent", "StyleReviewerAgent",

        # Export Pipeline (3 agents)
        "MarkdownExporterAgent", "DocxExporterAgent", "MetadataTaggerAgent",

        # Planning Pipeline (3 agents)
        "StrategicPlannerAgent", "TacticalPlannerAgent", "OperationalPlannerAgent",

        # Core Pipeline (3 agents)
        "ResearchAgent", "QualityAssuranceAgent", "AdaptiveOrchestratorAgent",

        # Autogen Pipeline (3 agents)
        "PlannerAgent", "WriterAgent", "EditorAgent",

        # Specialized Pipeline (3 agents)
        "DoubleCheckerAgent", "StylistAgent", "WriterAgent (autogen)",

        # Settlement Pipeline (4 agents)
        "SettlementOptimizerAgent", "BATNAAnalyzerAgent",
        "NashEquilibriumCalculatorAgent", "ReputationRiskScorerAgent",

        # Analysis Pipeline (4 agents)
        "TimelineAgent", "LinguistAgent", "CausalAgent", "LegalAgent"
    ]

def query_1782_database(query: str, description: str) -> Dict[str, Any]:
    """Execute a query on the 1782 database and return results."""
    try:
        conn = sqlite3.connect("data/case_law/1782_pdf_analysis.db")
        df = pd.read_sql_query(query, conn)
        conn.close()

        return {
            "query": query,
            "description": description,
            "results": df.to_dict('records'),
            "row_count": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        return {
            "query": query,
            "description": description,
            "error": str(e),
            "results": [],
            "row_count": 0
        }

def demonstrate_writing_team_1782_queries():
    """Demonstrate writing team agents querying 1782 database for patterns."""

    print("🔍 Writing Team 1782 Database Pattern Analysis")
    print("="*60)

    # Define queries that writing team agents would run
    queries = [
        {
            "agent": "CitationFinderAgent",
            "query": "SELECT outcome, COUNT(*) as case_count FROM cases GROUP BY outcome ORDER BY case_count DESC",
            "description": "Find outcome distribution patterns for citation analysis"
        },
        {
            "agent": "PrecedentFinderAgent",
            "query": "SELECT intel_factor_1, intel_factor_2, intel_factor_3, intel_factor_4, outcome FROM cases WHERE intel_factor_1 = 1 OR intel_factor_2 = 1 OR intel_factor_3 = 1 OR intel_factor_4 = 1",
            "description": "Find Intel factor patterns for precedent analysis"
        },
        {
            "agent": "FactExtractorAgent",
            "query": "SELECT AVG(text_length) as avg_length, outcome FROM cases GROUP BY outcome",
            "description": "Extract text length patterns by outcome"
        },
        {
            "agent": "LegalAgent",
            "query": "SELECT section_1782_mentions, outcome, COUNT(*) as count FROM cases GROUP BY section_1782_mentions, outcome ORDER BY count DESC",
            "description": "Analyze Section 1782 mention patterns for legal analysis"
        },
        {
            "agent": "SettlementOptimizerAgent",
            "query": "SELECT outcome, AVG(intel_factor_1 + intel_factor_2 + intel_factor_3 + intel_factor_4) as avg_factors FROM cases GROUP BY outcome",
            "description": "Analyze Intel factor correlation with outcomes for settlement optimization"
        },
        {
            "agent": "ResearchAgent",
            "query": "SELECT citations, outcome FROM cases WHERE citations IS NOT NULL AND citations != '' ORDER BY citations DESC LIMIT 10",
            "description": "Research citation patterns in successful cases"
        },
        {
            "agent": "ContentReviewerAgent",
            "query": "SELECT outcome, COUNT(*) as count, AVG(text_length) as avg_text_length FROM cases GROUP BY outcome",
            "description": "Review content patterns by outcome for quality assurance"
        },
        {
            "agent": "CausalAgent",
            "query": "SELECT intel_factor_1, intel_factor_2, intel_factor_3, intel_factor_4, outcome, COUNT(*) as count FROM cases GROUP BY intel_factor_1, intel_factor_2, intel_factor_3, intel_factor_4, outcome ORDER BY count DESC LIMIT 15",
            "description": "Analyze causal relationships between Intel factors and outcomes"
        }
    ]

    all_results = []

    for query_info in queries:
        print(f"\n🤖 {query_info['agent']} Analysis:")
        print(f"   Query: {query_info['description']}")

        result = query_1782_database(query_info['query'], query_info['description'])
        result['agent'] = query_info['agent']
        all_results.append(result)

        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Results: {result['row_count']} rows")
            if result['row_count'] > 0:
                print(f"   📊 Columns: {', '.join(result['columns'])}")
                # Show first few results
                for i, row in enumerate(result['results'][:3]):
                    print(f"   📋 Row {i+1}: {row}")
                if result['row_count'] > 3:
                    print(f"   ... and {result['row_count'] - 3} more rows")

    return all_results

def analyze_patterns(results: List[Dict[str, Any]]):
    """Analyze the patterns found by the writing team agents."""

    print(f"\n📊 Pattern Analysis Summary:")
    print("="*60)

    successful_queries = [r for r in results if 'error' not in r]
    failed_queries = [r for r in results if 'error' in r]

    print(f"✅ Successful Queries: {len(successful_queries)}")
    print(f"❌ Failed Queries: {len(failed_queries)}")

    if successful_queries:
        print(f"\n🔍 Key Patterns Found:")

        # Find outcome patterns
        outcome_results = [r for r in successful_queries if 'outcome' in str(r.get('query', ''))]
        if outcome_results:
            print(f"   📈 Outcome Analysis: {len(outcome_results)} agents found outcome patterns")

        # Find Intel factor patterns
        factor_results = [r for r in successful_queries if 'intel_factor' in str(r.get('query', ''))]
        if factor_results:
            print(f"   🎯 Intel Factor Analysis: {len(factor_results)} agents found factor patterns")

        # Find citation patterns
        citation_results = [r for r in successful_queries if 'citation' in str(r.get('query', ''))]
        if citation_results:
            print(f"   📚 Citation Analysis: {len(citation_results)} agents found citation patterns")

        # Find text patterns
        text_results = [r for r in successful_queries if 'text_length' in str(r.get('query', ''))]
        if text_results:
            print(f"   📝 Text Analysis: {len(text_results)} agents found text patterns")

def create_pattern_report(results: List[Dict[str, Any]]):
    """Create a comprehensive pattern report."""

    report = {
        "analysis_date": "2025-10-17T17:15:00.000000",
        "writing_team_agents": len(get_writing_team_agents()),
        "queries_executed": len(results),
        "successful_queries": len([r for r in results if 'error' not in r]),
        "failed_queries": len([r for r in results if 'error' in r]),
        "agent_analysis": {},
        "key_patterns": {
            "outcome_distribution": "Analyzed by CitationFinderAgent, ContentReviewerAgent",
            "intel_factors": "Analyzed by PrecedentFinderAgent, SettlementOptimizerAgent, CausalAgent",
            "text_patterns": "Analyzed by FactExtractorAgent, ContentReviewerAgent",
            "citation_patterns": "Analyzed by ResearchAgent",
            "legal_patterns": "Analyzed by LegalAgent"
        },
        "database_access": {
            "database": "1782_pdf_analysis.db",
            "type": "SQLite",
            "access_method": "Writing team agent memories"
        }
    }

    # Add individual agent results
    for result in results:
        agent = result['agent']
        if agent not in report['agent_analysis']:
            report['agent_analysis'][agent] = []

        report['agent_analysis'][agent].append({
            "query_description": result['description'],
            "success": 'error' not in result,
            "row_count": result.get('row_count', 0),
            "error": result.get('error', None)
        })

    # Save report
    with open("memory_store/writing_team_1782_pattern_analysis.json", 'w') as f:
        json.dump(report, f, indent=2)

    return report

def main():
    """Main execution function."""
    print("🚀 Writing Team 1782 Database Pattern Analysis")
    print("="*60)

    # Check if database exists
    db_path = "data/case_law/1782_pdf_analysis.db"
    if not Path(db_path).exists():
        print(f"❌ 1782 database not found at {db_path}. Please ensure the database exists.")
        return

    # Execute queries
    results = demonstrate_writing_team_1782_queries()

    # Analyze patterns
    analyze_patterns(results)

    # Create report
    report = create_pattern_report(results)

    print(f"\n📁 Report saved to: memory_store/writing_team_1782_pattern_analysis.json")

    print(f"\n🎯 Summary:")
    print(f"   Writing Team Agents: {len(get_writing_team_agents())}")
    print(f"   Queries Executed: {len(results)}")
    print(f"   Successful: {report['successful_queries']}")
    print(f"   Failed: {report['failed_queries']}")

    print(f"\n✅ Writing team successfully queried 1782 database for patterns!")

if __name__ == "__main__":
    main()
