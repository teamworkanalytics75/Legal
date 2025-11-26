#!/usr/bin/env python3
"""
Direct Harvard Query Test
Test LangChain's ability to handle the specific Harvard Schedule 4 Xi slide query
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up API key
os.environ["OPENAI_API_KEY"] = "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"

def test_harvard_query():
    """Test the specific Harvard Schedule 4 Xi slide query."""

    print("🧠 Testing Harvard Schedule 4 Xi Slide Query")
    print("=" * 60)

    try:
        from writer_agents.code.demo_langchain_workflow import LangChainWorkflow

        # Initialize workflow
        workflow = LangChainWorkflow()

        # The specific query
        query = """
        What arguments did Harvard have constructive knowledge of regarding the Schedule 4 Xi slide on or before April 19, 2019?

        Please provide a comprehensive analysis including:
        1. Evidence of Harvard's awareness of the Schedule 4 Xi slide
        2. Timeline evidence supporting knowledge before April 19, 2019
        3. Legal arguments supporting constructive knowledge claims
        4. Specific documents or communications demonstrating this knowledge
        5. Relevant case law or precedents that support these arguments
        6. Analysis of the legal standard for constructive knowledge
        """

        print("🔍 Running query...")
        print(f"Query: {query[:100]}...")
        print()

        # Run the query
        result = workflow.run_query(query)

        print("📊 Query Results:")
        print("=" * 80)
        print(result)
        print("=" * 80)

        # Analyze the response
        analyze_harvard_response(result)

        return result

    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_harvard_response(result: str):
    """Analyze the quality of the Harvard query response."""
    print("\n📈 Response Analysis:")

    # Key elements to look for
    key_elements = {
        "constructive knowledge": "constructive knowledge" in result.lower(),
        "timeline analysis": "april 19, 2019" in result.lower() or "2019" in result.lower(),
        "schedule 4 xi slide": "schedule 4" in result.lower() and "xi slide" in result.lower(),
        "harvard awareness": "harvard" in result.lower() and ("aware" in result.lower() or "knowledge" in result.lower()),
        "legal arguments": "legal" in result.lower() and "argument" in result.lower(),
        "evidence": "evidence" in result.lower() or "document" in result.lower(),
        "case law": "case law" in result.lower() or "precedent" in result.lower(),
        "structured analysis": result.count("1.") >= 2 and result.count("2.") >= 1
    }

    print("✅ Key Elements Found:")
    for element, found in key_elements.items():
        status = "✅" if found else "❌"
        print(f"  {status} {element}")

    # Overall quality score
    score = sum(key_elements.values()) / len(key_elements) * 100
    print(f"\n🎯 Overall Quality Score: {score:.1f}%")

    if score >= 80:
        print("🌟 Excellent - LangChain demonstrated sophisticated legal reasoning")
    elif score >= 60:
        print("👍 Good - LangChain showed solid legal analysis capabilities")
    elif score >= 40:
        print("⚠️ Fair - LangChain provided basic legal information")
    else:
        print("❌ Poor - LangChain struggled with complex legal reasoning")

    # Detailed analysis
    print("\n🔍 Detailed Analysis:")

    # Check for specific legal concepts
    legal_concepts = [
        "constructive knowledge",
        "actual knowledge",
        "imputed knowledge",
        "timeline",
        "evidence",
        "awareness",
        "Schedule 4",
        "Xi slide",
        "April 19, 2019",
        "Harvard",
        "legal argument",
        "precedent",
        "case law",
        "legal standard"
    ]

    found_concepts = []
    for concept in legal_concepts:
        if concept.lower() in result.lower():
            found_concepts.append(concept)

    print(f"📚 Legal Concepts Found: {len(found_concepts)}/{len(legal_concepts)}")
    for concept in found_concepts:
        print(f"  - {concept}")

    # Check for reasoning structure
    reasoning_indicators = [
        "because",
        "therefore",
        "thus",
        "consequently",
        "as a result",
        "in conclusion",
        "analysis",
        "reasoning"
    ]

    reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in result.lower())
    print(f"🧠 Reasoning Indicators: {reasoning_count}/{len(reasoning_indicators)}")

    # Check for specific evidence types
    evidence_types = [
        "document",
        "communication",
        "email",
        "meeting",
        "correspondence",
        "record",
        "file",
        "report"
    ]

    evidence_count = sum(1 for evidence_type in evidence_types if evidence_type in result.lower())
    print(f"📄 Evidence Types Mentioned: {evidence_count}/{len(evidence_types)}")

def main():
    """Main execution."""
    print("🎯 Harvard Schedule 4 Xi Slide Query Test")
    print("Testing LangChain's complex legal reasoning capabilities")
    print()

    result = test_harvard_query()

    if result:
        print("\n✅ Test completed successfully!")
        print("🎉 LangChain successfully handled the complex legal query")
    else:
        print("\n❌ Test failed")
        print("🔧 Consider running additional seeding or checking database connections")

if __name__ == "__main__":
    main()
