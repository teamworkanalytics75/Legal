#!/usr/bin/env python3
"""
Comprehensive Import Test for 49-Agent System
==============================================

Tests all critical imports to ensure the system can run from scripts.
"""

import sys
from pathlib import Path

# Add writer_agents/code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

def test_imports():
    """Test all critical imports."""
    print("Testing critical imports...")

    try:
        # Test core modules
        print("  Testing master_supervisor...")
        from master_supervisor import MasterSupervisor, SupervisorConfig
        print("    [OK] MasterSupervisor imported successfully")

        print("  Testing session_manager...")
        from session_manager import SessionManager
        print("    [OK] SessionManager imported successfully")

        print("  Testing atomic_agent...")
        from atomic_agent import AtomicAgent
        print("    [OK] AtomicAgent imported successfully")

        print("  Testing memory_system...")
        from memory_system import MemoryStore
        print("    [OK] MemoryStore imported successfully")

        print("  Testing all atomic agents...")
        from atomic_agents import (
            CitationFinderAgent, CitationNormalizerAgent, CitationVerifierAgent,
            CitationLocatorAgent, CitationInserterAgent,
            FactExtractorAgent, PrecedentFinderAgent, PrecedentRankerAgent,
            PrecedentSummarizerAgent, StatuteLocatorAgent, ExhibitFetcherAgent,
            OutlineBuilderAgent, SectionWriterAgent, ParagraphWriterAgent,
            TransitionAgent, GrammarFixerAgent, StyleCheckerAgent,
            LogicCheckerAgent, ConsistencyCheckerAgent, RedactionAgent,
            ComplianceAgent, ExpertQAAgent, MarkdownExporterAgent,
            DocxExporterAgent, MetadataTaggerAgent
        )
        print("    [OK] All 25 atomic agents imported successfully")

        print("  Testing supervisors...")
        from supervisors import (
            ResearchSupervisor, DraftingSupervisor, CitationSupervisor, QASupervisor
        )
        print("    [OK] All supervisors imported successfully")

        print("  Testing insights...")
        from insights import CaseInsights
        print("    [OK] CaseInsights imported successfully")

        print("  Testing job_persistence...")
        from job_persistence import JobManager
        print("    [OK] JobManager imported successfully")

        print("\n[SUCCESS] ALL IMPORTS SUCCESSFUL!")
        return True

    except ImportError as e:
        print(f"\n[ERROR] IMPORT FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        return False

def test_agent_instantiation():
    """Test that agents can be instantiated."""
    print("\nTesting agent instantiation...")

    try:
        from atomic_agents import CitationFinderAgent
        from agents import AgentFactory, ModelConfig

        # Create a simple factory
        factory = AgentFactory(ModelConfig())

        # Try to instantiate an agent
        agent = CitationFinderAgent(factory)
        print("    [OK] CitationFinderAgent instantiated successfully")

        return True

    except Exception as e:
        print(f"    [ERROR] Agent instantiation failed: {e}")
        return False

def test_master_supervisor_init():
    """Test MasterSupervisor initialization."""
    print("\nTesting MasterSupervisor initialization...")

    try:
        from master_supervisor import MasterSupervisor, SupervisorConfig

        # Try to initialize (without actually running)
        config = SupervisorConfig()
        print("    [OK] SupervisorConfig created successfully")

        # Don't actually initialize MasterSupervisor as it requires AutoGen
        print("    [OK] MasterSupervisor import successful (not initializing)")

        return True

    except Exception as e:
        print(f"    [ERROR] MasterSupervisor test failed: {e}")
        return False

if __name__ == "__main__":
    print("49-AGENT SYSTEM IMPORT TEST")
    print("=" * 40)

    success = True
    success &= test_imports()
    success &= test_agent_instantiation()
    success &= test_master_supervisor_init()

    if success:
        print("\n[SUCCESS] ALL TESTS PASSED - System ready for full execution!")
    else:
        print("\n[ERROR] SOME TESTS FAILED - Need to fix remaining import issues")
        sys.exit(1)
