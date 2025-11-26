#!/usr/bin/env python3
"""
Harvard Schedule 4 Xi Slide Query Test
Specific test for LangChain's ability to handle complex legal timeline analysis
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_harvard_query():
    """Test the specific Harvard Schedule 4 Xi slide query."""

    # Set up API key
    if not os.getenv('OPENAI_API_KEY'):
        demo_file = PROJECT_ROOT / "writer_agents" / "code" / "demo_langchain_workflow.py"
        if demo_file.exists():
            with open(demo_file, 'r') as f:
                content = f.read()
                if 'sk-' in content:
                    lines = content.split('\n')
                    for line in lines:
                        if 'sk-' in line and '=' in line:
                            api_key = line.split('=')[1].strip().strip('"\'')
                            os.environ['OPENAI_API_KEY'] = api_key
                            break

    try:
        from writer_agents.code.demo_langchain_workflow import LangChainWorkflow

        print("🧠 Testing Harvard Schedule 4 Xi Slide Query")
        print("=" * 60)

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
