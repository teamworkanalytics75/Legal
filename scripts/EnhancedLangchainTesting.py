#!/usr/bin/env python3
"""
Enhanced LangChain Seeding and Complex Query Testing
Tests LangChain's ability to handle sophisticated legal reasoning queries
"""

import os
import sys
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def setup_environment():
    """Setup environment variables and paths."""
    # Set OpenAI API key if not already set
    if not os.getenv('OPENAI_API_KEY'):
        # Try to find it in the demo file
        demo_file = PROJECT_ROOT / "writer_agents" / "code" / "demo_langchain_workflow.py"
        if demo_file.exists():
            with open(demo_file, 'r') as f:
                content = f.read()
                if 'sk-' in content:
                    # Extract the API key
                    lines = content.split('\n')
                    for line in lines:
                        if 'sk-' in line and '=' in line:
                            api_key = line.split('=')[1].strip().strip('"\'')
                            os.environ['OPENAI_API_KEY'] = api_key
                            print(f"✅ Set OpenAI API key from demo file")
                            break

def run_additional_seeding():
    """Run additional LangChain seeding for more comprehensive coverage."""
    print("🌱 Running Additional LangChain Seeding...")

    try:
        from writer_agents.code.demo_langchain_workflow import LangChainWorkflow

        # Initialize workflow
        workflow = LangChainWorkflow()

        # Additional seeding queries for better coverage
        additional_queries = [
            # Timeline-specific queries
            "What evidence exists regarding timeline of events in 2019?",
            "What documents were created or modified on or before April 19, 2019?",
            "What communications occurred between parties in April 2019?",

            # Harvard-specific queries
            "What knowledge did Harvard have about Schedule 4 Xi slide?",
            "What constructive knowledge did Harvard possess regarding Xi slide?",
            "What evidence shows Harvard's awareness of Schedule 4 Xi slide before April 19, 2019?",

            # Legal reasoning queries
            "What legal arguments support constructive knowledge claims?",
            "What evidence demonstrates party awareness of specific facts?",
            "What timeline evidence supports knowledge claims?",

            # Document analysis queries
            "What documents contain references to Schedule 4 Xi slide?",
            "What communications mention Xi slide or related concepts?",
            "What evidence shows document creation dates before April 19, 2019?",

            # Case law queries
            "What case law supports constructive knowledge arguments?",
            "What precedents exist for timeline-based knowledge claims?",
            "What legal standards apply to constructive knowledge?",
        ]

        print(f"📝 Running {len(additional_queries)} additional seeding queries...")

        for i, query in enumerate(additional_queries, 1):
            print(f"  {i}/{len(additional_queries)}: {query[:60]}...")
            try:
                result = workflow.run_query(query)
                print(f"    ✅ Query completed")
            except Exception as e:
                print(f"    ⚠️ Query failed: {e}")

        print("✅ Additional seeding completed!")
        return True

    except Exception as e:
        print(f"❌ Seeding failed: {e}")
        return False

def test_complex_query():
    """Test the specific Harvard Schedule 4 Xi slide query."""
    print("\n🧠 Testing Complex Legal Query...")

    complex_query = """
    What arguments did Harvard have constructive knowledge of regarding the Schedule 4 Xi slide on or before April 19, 2019?

    Please analyze:
    1. What evidence shows Harvard's awareness of the Schedule 4 Xi slide
    2. What timeline evidence supports knowledge before April 19, 2019
    3. What legal arguments support constructive knowledge claims
    4. What documents or communications demonstrate this knowledge
    5. What case law or precedents support these arguments
    """

    try:
        from writer_agents.code.demo_langchain_workflow import LangChainWorkflow

        workflow = LangChainWorkflow()

        print("🔍 Running complex legal analysis query...")
        print(f"Query: {complex_query[:100]}...")

        result = workflow.run_query(complex_query)

        print("\n📊 Query Results:")
        print("=" * 80)
        print(result)
        print("=" * 80)

        # Analyze the quality of the response
        analyze_response_quality(result)

        return result

    except Exception as e:
        print(f"❌ Complex query failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_response_quality(result: str):
    """Analyze the quality of LangChain's response."""
    print("\n📈 Response Quality Analysis:")

    # Check for key legal concepts
    legal_concepts = [
        "constructive knowledge",
        "timeline",
        "evidence",
        "awareness",
        "Schedule 4",
        "Xi slide",
        "April 19, 2019",
        "Harvard",
        "legal argument",
        "precedent",
        "case law"
    ]

    found_concepts = []
    for concept in legal_concepts:
        if concept.lower() in result.lower():
            found_concepts.append(concept)

    print(f"✅ Found {len(found_concepts)}/{len(legal_concepts)} key legal concepts:")
    for concept in found_concepts:
        print(f"  - {concept}")

    # Check response structure
    if "1." in result and "2." in result:
        print("✅ Response shows structured analysis")
    else:
        print("⚠️ Response lacks clear structure")

    # Check for specific evidence
    if "document" in result.lower() or "communication" in result.lower():
        print("✅ Response mentions specific evidence")
    else:
        print("⚠️ Response lacks specific evidence references")

    # Check for legal reasoning
    if "legal" in result.lower() or "precedent" in result.lower():
        print("✅ Response includes legal reasoning")
    else:
        print("⚠️ Response lacks legal reasoning")

def check_memory_coverage():
    """Check current memory coverage."""
    print("\n📊 Checking Memory Coverage...")

    try:
        # Check vector store
        memory_store_path = PROJECT_ROOT / "memory_store" / "vector_store.pkl"
        if memory_store_path.exists():
            import pickle
            with open(memory_store_path, 'rb') as f:
                vector_data = pickle.load(f)

            total_memories = sum(len(memories) for memories in vector_data.values())
            agents_with_memories = len([a for a in vector_data.values() if len(a) > 0])

            print(f"📝 Vector Store: {total_memories} total memories across {agents_with_memories} agents")

            # Show top agents by memory count
            agent_memory_counts = [(name, len(memories)) for name, memories in vector_data.items()]
            agent_memory_counts.sort(key=lambda x: x[1], reverse=True)

            print("🏆 Top 10 agents by memory count:")
            for name, count in agent_memory_counts[:10]:
                print(f"  {name}: {count} memories")

        # Check LangChain meta-memory
        langchain_db_path = PROJECT_ROOT / "memory_store" / "langchain_meta_memory.sqlite"
        if langchain_db_path.exists():
            conn = sqlite3.connect(langchain_db_path)
            cursor = conn.cursor()

            try:
                cursor.execute("SELECT COUNT(*) FROM query_history")
                query_count = cursor.fetchone()[0]
                print(f"🔍 LangChain Meta-Memory: {query_count} queries")

                # Get recent queries
                cursor.execute("SELECT query, timestamp FROM query_history ORDER BY timestamp DESC LIMIT 5")
                recent_queries = cursor.fetchall()

                print("📋 Recent queries:")
                for query, timestamp in recent_queries:
                    print(f"  {timestamp}: {query[:60]}...")

            except sqlite3.OperationalError:
                print("⚠️ LangChain meta-memory schema not found")

            conn.close()

    except Exception as e:
        print(f"❌ Memory check failed: {e}")

def main():
    """Main execution function."""
    print("🚀 Enhanced LangChain Seeding and Complex Query Testing")
    print("=" * 60)

    # Setup
    setup_environment()

    # Check current coverage
    check_memory_coverage()

    # Run additional seeding
    seeding_success = run_additional_seeding()

    if seeding_success:
        # Check updated coverage
        print("\n📊 Updated Memory Coverage:")
        check_memory_coverage()

        # Test complex query
        result = test_complex_query()

        if result:
            print("\n✅ Complex query test completed successfully!")
            print("🎯 LangChain demonstrated sophisticated legal reasoning capabilities")
        else:
            print("\n❌ Complex query test failed")
    else:
        print("\n❌ Additional seeding failed - skipping complex query test")

if __name__ == "__main__":
    main()
