#!/usr/bin/env python
"""Test atomic agents with simplified approach."""

import os
import time
from pathlib import Path
from writer_agents.code.agents import AgentFactory, ModelConfig

def test_atomic_agent_imports():
    """Test that atomic agents can be imported without errors."""
    print("\n" + "="*60)
    print("TESTING ATOMIC AGENT IMPORTS")
    print("="*60)

    import_results = {}

    # Test citation agents
    print("\n1. Testing Citation Agents:")
    print("-" * 30)

    try:
        from writer_agents.code.atomic_agents.citations import (
            CitationFinderAgent,
            CitationNormalizerAgent,
            CitationVerifierAgent,
            CitationLocatorAgent,
            CitationInserterAgent,
        )
        print("  Citation agents imported successfully")
        import_results["citations"] = True
    except Exception as e:
        print(f"  Citation agents import failed: {e}")
        import_results["citations"] = False

    # Test research agents
    print("\n2. Testing Research Agents:")
    print("-" * 30)

    try:
        from writer_agents.code.atomic_agents.research import (
            FactExtractorAgent,
            PrecedentFinderAgent,
            PrecedentRankerAgent,
            PrecedentSummarizerAgent,
            StatuteLocatorAgent,
            ExhibitFetcherAgent,
        )
        print("  Research agents imported successfully")
        import_results["research"] = True
    except Exception as e:
        print(f"  Research agents import failed: {e}")
        import_results["research"] = False

    # Test drafting agents
    print("\n3. Testing Drafting Agents:")
    print("-" * 30)

    try:
        from writer_agents.code.atomic_agents.drafting import (
            OutlineBuilderAgent,
            SectionWriterAgent,
            ParagraphWriterAgent,
            TransitionAgent,
        )
        print("  Drafting agents imported successfully")
        import_results["drafting"] = True
    except Exception as e:
        print(f"  Drafting agents import failed: {e}")
        import_results["drafting"] = False

    # Test review agents
    print("\n4. Testing Review Agents:")
    print("-" * 30)

    try:
        from writer_agents.code.atomic_agents.review import (
            GrammarFixerAgent,
            StyleCheckerAgent,
            LogicCheckerAgent,
            ConsistencyCheckerAgent,
            RedactionAgent,
            ComplianceAgent,
            ExpertQAAgent,
        )
        print("  Review agents imported successfully")
        import_results["review"] = True
    except Exception as e:
        print(f"  Review agents import failed: {e}")
        import_results["review"] = False

    # Test output agents
    print("\n5. Testing Output Agents:")
    print("-" * 30)

    try:
        from writer_agents.code.atomic_agents.output import (
            MarkdownExporterAgent,
            DocxExporterAgent,
            MetadataTaggerAgent,
        )
        print("  Output agents imported successfully")
        import_results["output"] = True
    except Exception as e:
        print(f"  Output agents import failed: {e}")
        import_results["output"] = False

    return import_results

def test_supervisor_imports():
    """Test that supervisors can be imported."""
    print("\n" + "="*60)
    print("TESTING SUPERVISOR IMPORTS")
    print("="*60)

    try:
        from writer_agents.code.supervisors import (
            ResearchSupervisor,
            DraftingSupervisor,
            CitationSupervisor,
            QASupervisor,
        )
        print("  All supervisors imported successfully")
        return True
    except Exception as e:
        print(f"  Supervisor import failed: {e}")
        return False

def test_agent_factory():
    """Test AgentFactory functionality."""
    print("\n" + "="*60)
    print("TESTING AGENT FACTORY")
    print("="*60)

    try:
        # Test factory creation
        factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))
        print("  AgentFactory created successfully")

        # Test client creation
        client = factory.client()
        print(f"  Model client created: {type(client)}")

        # Test agent creation
        agent = factory.create("test_agent", "You are a test agent.")
        print(f"  Test agent created: {type(agent)}")

        return True

    except Exception as e:
        print(f"  AgentFactory test failed: {e}")
        return False

def test_master_supervisor():
    """Test MasterSupervisor functionality."""
    print("\n" + "="*60)
    print("TESTING MASTER SUPERVISOR")
    print("="*60)

    try:
        from writer_agents.code.master_supervisor import MasterSupervisor

        factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))
        supervisor = MasterSupervisor(factory)

        print("  MasterSupervisor created successfully")
        print(f"  Supervisor type: {type(supervisor)}")

        return True

    except Exception as e:
        print(f"  MasterSupervisor test failed: {e}")
        return False

def test_task_decomposer():
    """Test TaskDecomposer functionality."""
    print("\n" + "="*60)
    print("TESTING TASK DECOMPOSER")
    print("="*60)

    try:
        from writer_agents.code.task_decomposer import TaskDecomposer

        decomposer = TaskDecomposer()

        print("  TaskDecomposer created successfully")
        print(f"  Decomposer type: {type(decomposer)}")

        # Test decomposition
        test_task = "Analyze Harvard discrimination case"
        subtasks = decomposer.decompose(test_task)

        print(f"  Decomposed task into {len(subtasks)} subtasks")
        for i, subtask in enumerate(subtasks[:3], 1):  # Show first 3
            print(f"    {i}. {subtask}")

        return True

    except Exception as e:
        print(f"  TaskDecomposer test failed: {e}")
        return False

def test_workflow_runner():
    """Test the atomic workflow runner."""
    print("\n" + "="*60)
    print("TESTING WORKFLOW RUNNER")
    print("="*60)

    try:
        from writer_agents.code.run_atomic_workflow import main as run_workflow

        print("  Workflow runner imported successfully")
        print("  Note: Full workflow test requires test case file")

        return True

    except Exception as e:
        print(f"  Workflow runner test failed: {e}")
        return False

def main():
    """Main test function."""
    print("ATOMIC AGENTS SIMPLIFIED TEST")
    print("="*60)

    try:
        # Test imports
        import_results = test_atomic_agent_imports()
        supervisor_success = test_supervisor_imports()

        # Test core components
        factory_success = test_agent_factory()
        master_success = test_master_supervisor()
        decomposer_success = test_task_decomposer()
        workflow_success = test_workflow_runner()

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        print(f"Citation Agents: {'PASS' if import_results.get('citations') else 'FAIL'}")
        print(f"Research Agents: {'PASS' if import_results.get('research') else 'FAIL'}")
        print(f"Drafting Agents: {'PASS' if import_results.get('drafting') else 'FAIL'}")
        print(f"Review Agents: {'PASS' if import_results.get('review') else 'FAIL'}")
        print(f"Output Agents: {'PASS' if import_results.get('output') else 'FAIL'}")
        print(f"Supervisors: {'PASS' if supervisor_success else 'FAIL'}")
        print(f"Agent Factory: {'PASS' if factory_success else 'FAIL'}")
        print(f"Master Supervisor: {'PASS' if master_success else 'FAIL'}")
        print(f"Task Decomposer: {'PASS' if decomposer_success else 'FAIL'}")
        print(f"Workflow Runner: {'PASS' if workflow_success else 'FAIL'}")

        # Overall success
        all_imports = all(import_results.values())
        all_components = all([supervisor_success, factory_success, master_success, decomposer_success, workflow_success])

        if all_imports and all_components:
            print("\nSUCCESS: All atomic agent components are working!")
            return True
        else:
            print("\nPARTIAL: Some components need attention.")
            return False

    except Exception as e:
        print(f"\nERROR: Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
