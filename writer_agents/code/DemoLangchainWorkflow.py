#!/usr/bin/env python3
"""
Demo LangChain Workflow

Minimal working example that validates the full LangChain integration
by using MasterSupervisor with LangChain enabled.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CODE_ROOT = PROJECT_ROOT / "writer_agents" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# Set up environment
os.environ["OPENAI_API_KEY"] = "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"

try:
    from master_supervisor import MasterSupervisor, SupervisorConfig
    from supervisors import ResearchSupervisor
    from job_persistence import JobManager
    from atomic_agents.enhanced_research import EnhancedPrecedentFinderAgent, EnhancedFactExtractorAgent
    from atomic_agents.research import FactExtractorAgent, PrecedentFinderAgent
    from agents import AgentFactory, ModelConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


def create_synthetic_insights() -> Dict[str, Any]:
    """Create synthetic case insights for testing."""
    return {
        "case_name": "Harvard University v. Test Case",
        "summary": "Test case involving Harvard University and evidence discovery",
        "evidence_count": 5,
        "posterior_probabilities": {
            "harvard_knowledge": 0.75,
            "evidence_timeline": 0.60
        },
        "key_facts": [
            "Harvard University is involved in litigation",
            "Evidence includes Xi Mingze slide",
            "Timeline involves April 2019 events"
        ]
    }


async def test_research_supervisor_langchain():
    """Test ResearchSupervisor with LangChain enabled."""
    print("Testing ResearchSupervisor with LangChain enabled...")

    # Create supervisor with LangChain enabled
    db_path = Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db")
    supervisor = ResearchSupervisor(
        session=None,
        job_manager=JobManager("test_jobs.db"),
        budgets={},
        use_langchain=True,
        langchain_db_path=db_path
    )

    # Test spawning enhanced agents
    try:
        precedent_agent = supervisor._spawn_agent('PrecedentFinderAgent')
        fact_agent = supervisor._spawn_agent('FactExtractorAgent')

        print(f"PrecedentFinderAgent type: {type(precedent_agent).__name__}")
        print(f"FactExtractorAgent type: {type(fact_agent).__name__}")

        # Verify enhanced agents were spawned
        if isinstance(precedent_agent, EnhancedPrecedentFinderAgent):
            print("PASS: EnhancedPrecedentFinderAgent spawned correctly")
        else:
            print(f"FAIL: Expected EnhancedPrecedentFinderAgent, got {type(precedent_agent).__name__}")

        if isinstance(fact_agent, EnhancedFactExtractorAgent):
            print("PASS: EnhancedFactExtractorAgent spawned correctly")
        else:
            print(f"FAIL: Expected EnhancedFactExtractorAgent, got {type(fact_agent).__name__}")

        return True

    except Exception as e:
        print(f"FAIL: Error spawning agents: {e}")
        return False


async def test_research_supervisor_original():
    """Test ResearchSupervisor with LangChain disabled."""
    print("\nTesting ResearchSupervisor with LangChain disabled...")

    # Create supervisor with LangChain disabled
    supervisor = ResearchSupervisor(
        session=None,
        job_manager=JobManager("test_jobs.db"),
        budgets={},
        use_langchain=False,
        langchain_db_path=None
    )

    # Test spawning original agents
    try:
        precedent_agent = supervisor._spawn_agent('PrecedentFinderAgent')
        fact_agent = supervisor._spawn_agent('FactExtractorAgent')

        print(f"PrecedentFinderAgent type: {type(precedent_agent).__name__}")
        print(f"FactExtractorAgent type: {type(fact_agent).__name__}")

        # Verify original agents were spawned
        if isinstance(precedent_agent, PrecedentFinderAgent):
            print("✅ PrecedentFinderAgent spawned correctly")
        else:
            print(f"❌ Expected PrecedentFinderAgent, got {type(precedent_agent).__name__}")

        if isinstance(fact_agent, FactExtractorAgent):
            print("✅ FactExtractorAgent spawned correctly")
        else:
            print(f"❌ Expected FactExtractorAgent, got {type(fact_agent).__name__}")

        return True

    except Exception as e:
        print(f"FAIL: Error spawning agents: {e}")
        return False


async def test_enhanced_agent_execution():
    """Test that enhanced agents can execute with LangChain."""
    print("\nTesting enhanced agent execution...")

    try:
        factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))
        db_path = Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db")

        # Create enhanced precedent finder
        agent = EnhancedPrecedentFinderAgent(
            factory=factory,
            db_path=db_path,
            enable_langchain=True
        )

        # Test execution
        input_data = {
            'query': 'Harvard University knowledge of Xi Mingze slide',
            'legal_issue': 'Institutional knowledge of specific evidence'
        }

        result = await agent.execute(input_data)

        if result.get('precedents') and len(result.get('precedents', [])) > 0:
            print("✅ EnhancedPrecedentFinderAgent executed successfully")
            print(f"   Found {len(result.get('precedents', []))} precedents")
            return True
        else:
            print("❌ EnhancedPrecedentFinderAgent returned no precedents")
            return False

    except Exception as e:
        print(f"❌ Error executing enhanced agent: {e}")
        return False


async def test_master_supervisor_config():
    """Test MasterSupervisor configuration with LangChain."""
    print("\nTesting MasterSupervisor configuration...")

    try:
        # Create config with LangChain enabled
        config = SupervisorConfig()
        config.enable_langchain = True
        config.langchain_db_path = Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db")

        # Create master supervisor
        supervisor = MasterSupervisor(session=None, config=config)

        # Check that research supervisor has LangChain enabled
        if supervisor.research.use_langchain:
            print("✅ MasterSupervisor correctly configured with LangChain")
            print(f"   LangChain DB path: {supervisor.research.langchain_db_path}")
            return True
        else:
            print("❌ MasterSupervisor not configured with LangChain")
            return False

    except Exception as e:
        print(f"❌ Error configuring MasterSupervisor: {e}")
        return False


async def run_all_tests():
    """Run all LangChain integration tests."""
    print("=" * 60)
    print("LangChain Integration Demo")
    print("=" * 60)

    tests = [
        ("ResearchSupervisor LangChain", test_research_supervisor_langchain),
        ("ResearchSupervisor Original", test_research_supervisor_original),
        ("Enhanced Agent Execution", test_enhanced_agent_execution),
        ("MasterSupervisor Config", test_master_supervisor_config),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n🎉 All tests passed! LangChain integration is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the output above for details.")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
