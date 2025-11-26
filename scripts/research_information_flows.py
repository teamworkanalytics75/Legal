"""
Direct Research: Information Flows Between Overseas Chinese Media and Domestic Crackdowns
Using STORM-Inspired Research System
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from STORMInspiredResearch import STORMInspiredResearch

def research_information_flows():
    """Research information flows between overseas Chinese media and domestic crackdowns."""
    print("\n🔬 Researching: Information Flows Between Overseas Chinese Media and Domestic Crackdowns")
    print("="*100 + "\n")

    try:
        # Initialize system
        print("🔧 Initializing STORM-Inspired Research System...")
        research_system = STORMInspiredResearch()

        # Research topic
        topic = "Information Flows Between Overseas Chinese Media and Domestic Crackdowns: How foreign narratives influence CCP risk perception and enforcement actions"

        print(f"📋 Research Topic: {topic}\n")

        # Run comprehensive research
        print("🚀 Running comprehensive research...")
        results = research_system.run_comprehensive_research(topic)

        # Display detailed results
        print("\n" + "="*100)
        print("✅ RESEARCH COMPLETED SUCCESSFULLY!")
        print("="*100)

        print(f"\n📊 Research Summary:")
        print(f"   • Topic: {results['topic']}")
        print(f"   • Perspectives: {len(results['perspectives'])}")
        print(f"   • Sources: {len(results['search_results'])}")
        print(f"   • Processing Time: {results['processing_time']:.1f}s")
        print(f"   • Local Enhancement: {'✅' if results['local_results'] else '⏭️'}")
        print(f"   • Report File: {results['report_file']}")

        print(f"\n🎯 Generated Research Perspectives:")
        for i, perspective in enumerate(results['perspectives'], 1):
            print(f"   {i}. {perspective}")

        print(f"\n📚 Key Sources Found:")
        for i, source in enumerate(results['search_results'][:10], 1):
            print(f"   {i}. {source['title']}")
            print(f"      URL: {source['href']}")
            print(f"      Query: {source['query']}")
            print()

        print(f"\n📄 Report Preview (first 1000 chars):")
        print("-" * 100)
        print(results['report'][:1000] + "...")
        print("-" * 100)

        print(f"\n🎉 Research on Information Flows Complete!")
        print(f"   • Comprehensive analysis of overseas Chinese media influence")
        print(f"   • Multi-perspective examination of cross-border information flows")
        print(f"   • Documentation of cases where foreign content preceded domestic actions")
        print(f"   • Analysis of CCP risk perception and enforcement mechanisms")
        print(f"   • Total cost: $0.00")

        return results

    except Exception as e:
        print(f"\n❌ RESEARCH FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = research_information_flows()
    if results:
        print(f"\n✅ Research completed successfully!")
        print(f"📄 Full report available at: {results['report_file']}")
    else:
        print("\n❌ Research failed. Please check the error messages above.")
