#!/usr/bin/env python3
"""
LangChain Integration Test

Tests the enhanced research agents with LangChain SQLDatabaseToolkit integration.
Compares performance and results with the original agents.
"""

import asyncio
import os
from pathlib import Path

# Set up environment
os.environ["OPENAI_API_KEY"] = "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"

# Add paths for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import agents
try:
    from writer_agents.code.agents import AgentFactory, ModelConfig
    from writer_agents.code.atomic_agents.research import PrecedentFinderAgent, FactExtractorAgent
    from writer_agents.code.atomic_agents.enhanced_research import (
        EnhancedPrecedentFinderAgent,
        EnhancedFactExtractorAgent
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    exit(1)


async def test_precedent_finder():
    """Test precedent finder agents."""
    print("=" * 80)
    print("TESTING PRECEDENT FINDER AGENTS")
    print("=" * 80)

    # Initialize agents
    factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))

    # Original agent
    original_agent = PrecedentFinderAgent(factory)

    # Enhanced agent with LangChain
    enhanced_agent = EnhancedPrecedentFinderAgent(
        factory,
        db_path=Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"),
        enable_langchain=True
    )

    # Test query
    test_input = {
        "legal_issue": "Harvard University admissions practices and alumni club relationships"
    }

    print(f"Test Query: {test_input['legal_issue']}")
    print()

    # Test original agent
    print("--- ORIGINAL AGENT ---")
    try:
        original_result = await original_agent.execute(test_input)
        print(f"Precedents found: {original_result.get('precedent_count', 0)}")
        print(f"Precedents: {original_result.get('precedents', [])}")
        print(f"Cost tier: {original_agent.cost_tier}")
    except Exception as e:
        print(f"Original agent failed: {e}")

    print()

    # Test enhanced agent
    print("--- ENHANCED AGENT (LangChain) ---")
    try:
        enhanced_result = await enhanced_agent.execute(test_input)
        print(f"Precedents found: {enhanced_result.get('precedent_count', 0)}")
        print(f"Precedents: {enhanced_result.get('precedents', [])}")
        print(f"LangChain used: {enhanced_result.get('langchain_used', False)}")
        print(f"Evidence retrieval method: {enhanced_result.get('evidence_retrieval_method', 'none')}")
        print(f"Cost tier: {enhanced_agent.cost_tier}")

        # Show enhanced precedents with evidence
        if enhanced_result.get('enhanced_precedents'):
            print("\nEnhanced precedents with evidence:")
            for precedent in enhanced_result['enhanced_precedents']:
                print(f"  - {precedent['case']}")
                print(f"    Evidence found: {precedent['evidence_found']}")
                print(f"    Evidence count: {precedent['evidence_count']}")
                if precedent.get('evidence_preview'):
                    print(f"    Preview: {precedent['evidence_preview'][:100]}...")
                print()

    except Exception as e:
        print(f"Enhanced agent failed: {e}")


async def test_fact_extractor():
    """Test fact extractor agents."""
    print("=" * 80)
    print("TESTING FACT EXTRACTOR AGENTS")
    print("=" * 80)

    # Initialize agents
    factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))

    # Original agent
    original_agent = FactExtractorAgent(factory)

    # Enhanced agent with LangChain
    enhanced_agent = EnhancedFactExtractorAgent(
        factory,
        db_path=Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"),
        enable_langchain=True
    )

    # Test query
    test_input = {
        "query": "Harvard alumni club activities and admissions officer communications"
    }

    print(f"Test Query: {test_input['query']}")
    print()

    # Test original agent (no documents provided)
    print("--- ORIGINAL AGENT (No Documents) ---")
    try:
        original_result = await original_agent.execute(test_input)
        print(f"Facts extracted: {original_result.get('fact_count', 0)}")
        print(f"Cost tier: {original_agent.cost_tier}")
    except Exception as e:
        print(f"Original agent failed: {e}")

    print()

    # Test enhanced agent (with LangChain document retrieval)
    print("--- ENHANCED AGENT (LangChain Document Retrieval) ---")
    try:
        enhanced_result = await enhanced_agent.execute(test_input)
        print(f"Facts extracted: {enhanced_result.get('fact_count', 0)}")
        print(f"LangChain used: {enhanced_result.get('langchain_used', False)}")
        print(f"Evidence retrieval method: {enhanced_result.get('evidence_retrieval_method', 'none')}")
        print(f"Cost tier: {enhanced_agent.cost_tier}")

        # Show extracted facts
        if enhanced_result.get('facts'):
            print("\nExtracted facts:")
            for i, fact in enumerate(enhanced_result['facts'][:3], 1):  # Show first 3
                print(f"  {i}. {fact.get('fact', '')}")
                print(f"     Confidence: {fact.get('confidence', 0)}")
                print(f"     Source: {fact.get('source', 'Unknown')}")
                print()

    except Exception as e:
        print(f"Enhanced agent failed: {e}")


async def test_evidence_retrieval():
    """Test direct evidence retrieval."""
    print("=" * 80)
    print("TESTING DIRECT EVIDENCE RETRIEVAL")
    print("=" * 80)

    try:
        from writer_agents.code.langchain_integration import EvidenceRetrievalAgent
        from writer_agents.code.agents import AgentFactory, ModelConfig

        factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))

        evidence_agent = EvidenceRetrievalAgent(
            db_path=Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"),
            factory=factory,
            enable_langchain=True,
            fallback_to_manual=True
        )

        # Test queries
        test_queries = [
            "What evidence exists about Marlyn McGrath's role in Harvard alumni statements?",
            "Find documents about Xi Mingze slide presentation",
            "Harvard admissions practices and alumni club relationships"
        ]

        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 60)

            result = evidence_agent.search_evidence(query, limit=3)

            print(f"Success: {result.get('success', False)}")
            print(f"Query type: {result.get('query_type', 'unknown')}")

            if result.get('success'):
                if result.get('answer'):
                    print(f"Answer: {result['answer'][:200]}...")
                if result.get('results'):
                    print(f"Results count: {len(result['results'])}")
                    for i, res in enumerate(result['results'][:2], 1):
                        print(f"  {i}. Doc #{res.get('id', '?')} ({res.get('length', 0)} chars)")
                        print(f"     Preview: {res.get('preview', '')[:100]}...")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Evidence retrieval test failed: {e}")


async def main():
    """Run all tests."""
    print("LANGCHAIN INTEGRATION TEST")
    print("Testing enhanced research agents with LangChain SQLDatabaseToolkit")
    print()

    # Test evidence retrieval first
    await test_evidence_retrieval()

    print("\n" + "=" * 80)
    print("COMPARISON TESTS")
    print("=" * 80)

    # Test precedent finder comparison
    await test_precedent_finder()

    # Test fact extractor comparison
    await test_fact_extractor()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("Check the results above to see:")
    print("- LangChain vs manual query performance")
    print("- Evidence retrieval capabilities")
    print("- Cost implications")
    print("- Integration success/failure points")


if __name__ == "__main__":
    asyncio.run(main())
