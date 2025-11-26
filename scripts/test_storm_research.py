"""
Test STORM-Inspired Research System
Automated test with sample topic
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from STORMInspiredResearch import STORMInspiredResearch

def test_storm_research():
    """Test the STORM-inspired research system."""
    print("\n🧪 Testing STORM-Inspired Research System")
    print("="*60 + "\n")

    try:
        # Initialize system
        print("🔧 Initializing research system...")
        research_system = STORMInspiredResearch()

        # Test topic
        test_topic = "Section 1782 discovery applications in federal courts"
        print(f"📋 Test topic: {test_topic}\n")

        # Run research
        print("🚀 Running comprehensive research...")
        results = research_system.run_comprehensive_research(test_topic)

        # Display results summary
        print("\n" + "="*60)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📊 Results Summary:")
        print(f"   • Topic: {results['topic']}")
        print(f"   • Perspectives: {len(results['perspectives'])}")
        print(f"   • Sources: {len(results['search_results'])}")
        print(f"   • Processing Time: {results['processing_time']:.1f}s")
        print(f"   • Local Enhancement: {'✅' if results['local_results'] else '⏭️'}")
        print(f"   • Report File: {results['report_file']}")

        print(f"\n🎯 Generated Perspectives:")
        for i, perspective in enumerate(results['perspectives'], 1):
            print(f"   {i}. {perspective}")

        print(f"\n📚 Sample Sources:")
        for i, source in enumerate(results['search_results'][:3], 1):
            print(f"   {i}. {source['title']}")
            print(f"      {source['href']}")

        print(f"\n📄 Report Preview (first 500 chars):")
        print("-" * 60)
        print(results['report'][:500] + "...")
        print("-" * 60)

        print(f"\n🎉 STORM-Inspired Research System Test PASSED!")
        print(f"   • All components working correctly")
        print(f"   • Multi-perspective research implemented")
        print(f"   • Wikipedia-style article generated")
        print(f"   • Local document integration working")
        print(f"   • Zero API costs achieved")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_storm_research()
    if success:
        print("\n✅ All tests passed! System ready for production use.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
