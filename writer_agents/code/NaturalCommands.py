"""Natural language command interface for the advanced writing system."""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util
import sys
from pathlib import Path

# Import the WizardWeb module with dots in filename
wizardweb_path = Path(__file__).parent.parent / "experiments" / "WizardWeb1.1.4_STABLE.py"
spec = importlib.util.spec_from_file_location("wizardweb", wizardweb_path)
wizardweb_module = importlib.util.module_from_spec(spec)
sys.modules["wizardweb"] = wizardweb_module
spec.loader.exec_module(wizardweb_module)
parse_evidence_input = wizardweb_module.parse_evidence_input
from writer_agents.bn_integration import BNIntegrationConfig, BNWritingIntegrator
from writer_agents.advanced_agents import AdvancedAgentConfig, AdvancedWriterOrchestrator
from writer_agents.enhanced_orchestrator import EnhancedOrchestratorConfig, EnhancedWriterOrchestrator
from writer_agents.insights import CaseInsights, EvidenceItem, Posterior

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NaturalCommandProcessor:
    """Processes natural language commands for the writing system."""
    
    def __init__(self) -> None:
        self._command_handlers = {
            "run full workflow": self._run_full_workflow,
            "run wizardweb": self._run_full_workflow,
            "run bn": self._run_bn_only,
            "run writer": self._run_writer_only,
            "generate memo": self._run_writer_only,
            "run advanced": self._run_advanced_workflow,
            "run hybrid": self._run_hybrid_workflow,
            "run traditional": self._run_traditional_workflow,
            "test system": self._test_system,
            "demo": self._run_demo,
            "run analysis chat": self._run_group_chat,
        }
        
        # Default configuration
        self._default_config = BNIntegrationConfig(
            model_path=Path("experiments/WizardWeb1.1.3.xdsl"),
            enable_pysmile=True,
            fallback_to_mock=True,
            use_advanced_workflow=True,
            enable_hybrid_mode=True,
        )
    
    async def process_command(self, command: str, *args) -> bool:
        """Process a natural language command."""
        command = command.lower().strip()
        
        print(f" Processing command: '{command}'")
        if args:
            print(f" Arguments: {args}")
        
        # Find matching handler
        handler = None
        for cmd_pattern, cmd_handler in self._command_handlers.items():
            if cmd_pattern in command:
                handler = cmd_handler
                break
        
        if not handler:
            print(f"x Unknown command: '{command}'")
            print(f"Available commands: {list(self._command_handlers.keys())}")
            return False
        
        try:
            result = await handler(command, *args)
            print(f"[ok] Command completed successfully")
            return result
        except Exception as e:
            logger.error(f"Command failed: {e}")
            print(f"x Command failed: {e}")
            return False
    
    async def _run_full_workflow(self, command: str, *args) -> bool:
        """Run the complete BN-integrated writing workflow."""
        print(" Running full WizardWeb workflow with advanced writing system...")
        
        # Parse evidence from arguments
        evidence = {}
        if args:
            evidence_str = " ".join(args)
            evidence = parse_evidence_input(evidence_str)
        
        # Create default case insights
        summary = "Legal case analysis using advanced multi-agent writing system"
        reference_id = "WIZARDWEB-001"
        
        # Configure and run
        integrator = BNWritingIntegrator(self._default_config)
        
        try:
            result = await integrator.run_bn_writing_workflow(
                evidence=evidence,
                summary=summary,
                reference_id=reference_id,
                jurisdiction="US",
                case_style="Memorandum"
            )
            
            # Display results
            print(f"\n Results:")
            print(f" Reference ID: {result.case_insights.reference_id}")
            print(f" Evidence items: {len(result.case_insights.evidence)}")
            print(f" Posterior nodes: {len(result.case_insights.posteriors)}")
            print(f" Document length: {len(result.writing_result.edited_document)} characters")
            print(f" Sections generated: {len(result.writing_result.sections)}")
            
            # Show document preview
            print(f"\n Document Preview:")
            print("-" * 50)
            preview = result.writing_result.edited_document[:500]
            print(preview)
            if len(result.writing_result.edited_document) > 500:
                print("... [document continues]")
            print("-" * 50)
            
            # Show integration metrics
            if result.integration_metrics:
                print(f"\n Integration Metrics:")
                for metric, value in result.integration_metrics.items():
                    print(f" {metric}: {value}")
            
            return True
            
        finally:
            await integrator.close()
    
    async def _run_bn_only(self, command: str, *args) -> bool:
        """Run only the Bayesian network inference."""
        print(" Running Bayesian network inference...")
        
        # Parse evidence
        evidence = {}
        if args:
            evidence_str = " ".join(args)
            evidence = parse_evidence_input(evidence_str)
        
        # Create simple integrator for BN only
        config = BNIntegrationConfig(
            model_path=self._default_config.model_path,
            enable_pysmile=True,
            fallback_to_mock=True,
            use_advanced_workflow=False, # Skip writing for BN-only
        )
        
        integrator = BNWritingIntegrator(config)
        
        try:
            # Run BN inference
            result = await integrator.run_bn_writing_workflow(
                evidence=evidence,
                summary="BN inference test",
                reference_id="BN-ONLY-001"
            )
            
            # Display BN results
            print(f"\n Bayesian Network Results:")
            print(f" Reference ID: {result.case_insights.reference_id}")
            print(f" Evidence: {len(result.case_insights.evidence)} items")
            print(f" Posteriors: {len(result.case_insights.posteriors)} nodes")
            
            # Show posterior probabilities
            print(f"\n Posterior Probabilities:")
            for posterior in result.case_insights.posteriors:
                probs_str = ", ".join(f"{state}: {prob:.2%}" for state, prob in posterior.probabilities.items())
                print(f" {posterior.node_id}: {probs_str}")
            
            # Show evidence
            if result.case_insights.evidence:
                print(f"\n Evidence:")
                for item in result.case_insights.evidence:
                    print(f" {item.node_id} = {item.state}")
            
            return True
            
        finally:
            await integrator.close()
    
    async def _run_writer_only(self, command: str, *args) -> bool:
        """Run only the writing workflow with mock BN data."""
        print(" Running advanced writing system...")
        
        # Create mock case insights
        insights = CaseInsights(
            reference_id="WRITER-ONLY-001",
            summary="Legal memorandum generation using advanced multi-agent writing system",
            posteriors=[
                Posterior(
                    node_id="LegalSuccess_US",
                    probabilities={"High": 0.25, "Moderate": 0.55, "Low": 0.20}
                ),
                Posterior(
                    node_id="FinancialDamage",
                    probabilities={"Severe": 0.15, "Material": 0.60, "Nominal": 0.25}
                ),
                Posterior(
                    node_id="ReputationalHarm",
                    probabilities={"Elevated": 0.40, "Base": 0.60}
                )
            ],
            evidence=[
                EvidenceItem(node_id="OGC_Email_Apr18_2025", state="Sent"),
                EvidenceItem(node_id="PRC_Awareness", state="Direct")
            ],
            jurisdiction="US",
            case_style="Memorandum"
        )
        
        # Configure advanced writing system
        config = AdvancedAgentConfig(
            max_review_rounds=2,
            enable_research_agents=True,
            enable_quality_gates=True,
            enable_adaptive_workflow=True,
        )
        
        orchestrator = AdvancedWriterOrchestrator(config)
        
        try:
            result = await orchestrator.run_advanced_workflow(insights)
            
            # Display results
            print(f"\n Writing Results:")
            print(f" Plan objective: {result.plan.objective}")
            print(f" Deliverable format: {result.plan.deliverable_format}")
            print(f" Tone: {result.plan.tone}")
            print(f" Sections generated: {len(result.sections)}")
            print(f" Document length: {len(result.edited_document)} characters")
            
            # Show document
            print(f"\n Generated Memorandum:")
            print("=" * 60)
            print(result.edited_document)
            print("=" * 60)
            
            # Show metadata
            if result.metadata:
                print(f"\n Workflow Metadata:")
                for key, value in result.metadata.items():
                    print(f" {key}: {value}")
            
            return True
            
        finally:
            await orchestrator.close()
    
    async def _run_advanced_workflow(self, command: str, *args) -> bool:
        """Run the advanced workflow specifically."""
        print(" Running advanced multi-agent workflow...")
        
        # Parse evidence if provided
        evidence = {}
        if args:
            evidence_str = " ".join(args)
            evidence = parse_evidence_input(evidence_str)
        
        # Create case insights
        insights = CaseInsights(
            reference_id="ADVANCED-001",
            summary="Advanced workflow demonstration with comprehensive multi-agent analysis",
            posteriors=[
                Posterior(
                    node_id="LegalSuccess_US",
                    probabilities={"High": 0.30, "Moderate": 0.50, "Low": 0.20}
                ),
                Posterior(
                    node_id="FinancialDamage",
                    probabilities={"Severe": 0.20, "Material": 0.50, "Nominal": 0.30}
                )
            ],
            evidence=[EvidenceItem(node_id=k, state=v) for k, v in evidence.items()] if evidence else [
                EvidenceItem(node_id="TestEvidence", state="Present")
            ],
            jurisdiction="US",
            case_style="Memorandum"
        )
        
        # Configure advanced system
        config = AdvancedAgentConfig(
            max_review_rounds=3,
            enable_research_agents=True,
            enable_quality_gates=True,
            enable_adaptive_workflow=True,
        )
        
        orchestrator = AdvancedWriterOrchestrator(config)
        
        try:
            result = await orchestrator.run_advanced_workflow(insights)
            
            print(f"\n Advanced Workflow Results:")
            print(f" Workflow type: {result.metadata.get('workflow_type', 'advanced')}")
            print(f" Research enabled: {result.metadata.get('research_enabled', False)}")
            print(f" Quality gates enabled: {result.metadata.get('quality_gates_enabled', False)}")
            print(f" Adaptive workflow: {result.metadata.get('adaptive_workflow_enabled', False)}")
            print(f" Sections: {len(result.sections)}")
            print(f" Document length: {len(result.edited_document)} characters")
            
            # Show document preview
            print(f"\n Advanced Document Preview:")
            print("-" * 50)
            preview = result.edited_document[:800]
            print(preview)
            if len(result.edited_document) > 800:
                print("... [document continues]")
            print("-" * 50)
            
            return True
            
        finally:
            await orchestrator.close()
    
    async def _run_hybrid_workflow(self, command: str, *args) -> bool:
        """Run the hybrid workflow."""
        print(" Running hybrid workflow (traditional + advanced review)...")
        
        # Parse evidence if provided
        evidence = {}
        if args:
            evidence_str = " ".join(args)
            evidence = parse_evidence_input(evidence_str)
        
        # Configure enhanced orchestrator for hybrid mode
        config = EnhancedOrchestratorConfig(
            use_advanced_workflow=True,
            complexity_threshold=0.5, # Lower threshold to trigger hybrid
            enable_hybrid_mode=True,
            enable_performance_monitoring=True,
        )
        
        orchestrator = EnhancedWriterOrchestrator(config)
        
        try:
            # Create case insights
            insights = CaseInsights(
                reference_id="HYBRID-001",
                summary="Hybrid workflow demonstration combining traditional planning with advanced review",
                posteriors=[
                    Posterior(
                        node_id="TestNode",
                        probabilities={"High": 0.4, "Medium": 0.4, "Low": 0.2}
                    )
                ],
                evidence=[EvidenceItem(node_id=k, state=v) for k, v in evidence.items()] if evidence else [
                    EvidenceItem(node_id="TestEvidence", state="Present")
                ],
                jurisdiction="US",
                case_style="Memorandum"
            )
            
            # Get recommendations
            recommendations = orchestrator.get_workflow_recommendations(insights)
            
            # Run workflow
            result = await orchestrator.run_intelligent_workflow(insights)
            
            print(f"\n Hybrid Workflow Results:")
            print(f" Complexity score: {recommendations['complexity_score']:.2f}")
            print(f" Recommended workflow: {recommendations['recommended_workflow']}")
            print(f" Actual workflow: {result.metadata.get('workflow_type', 'unknown')}")
            print(f" Sections: {len(result.sections)}")
            print(f" Document length: {len(result.edited_document)} characters")
            
            # Show performance metrics
            if "performance_metrics" in result.metadata:
                metrics = result.metadata["performance_metrics"]
                print(f"\n Performance Metrics:")
                for metric, value in metrics.items():
                    print(f" {metric}: {value}")
            
            return True
            
        finally:
            await orchestrator.close()
    
    async def _run_traditional_workflow(self, command: str, *args) -> bool:
        """Run the traditional workflow."""
        print(" Running traditional workflow...")
        
        # Parse evidence if provided
        evidence = {}
        if args:
            evidence_str = " ".join(args)
            evidence = parse_evidence_input(evidence_str)
        
        # Create case insights
        insights = CaseInsights(
            reference_id="TRADITIONAL-001",
            summary="Traditional workflow demonstration with standard multi-agent system",
            posteriors=[
                Posterior(
                    node_id="TestNode",
                    probabilities={"High": 0.5, "Low": 0.5}
                )
            ],
            evidence=[EvidenceItem(node_id=k, state=v) for k, v in evidence.items()] if evidence else [
                EvidenceItem(node_id="TestEvidence", state="Present")
            ],
            jurisdiction="US",
            case_style="Memorandum"
        )
        
        # Configure traditional system
        from writer_agents.orchestrator import WriterOrchestrator, WriterOrchestratorConfig
        
        config = WriterOrchestratorConfig()
        orchestrator = WriterOrchestrator(config)
        
        try:
            result = await orchestrator.run(insights)
            
            print(f"\n Traditional Workflow Results:")
            print(f" Plan objective: {result.plan.objective}")
            print(f" Sections: {len(result.sections)}")
            print(f" Reviews: {len(result.reviews)}")
            print(f" Document length: {len(result.edited_document)} characters")
            
            # Show document preview
            print(f"\n Traditional Document Preview:")
            print("-" * 50)
            preview = result.edited_document[:600]
            print(preview)
            if len(result.edited_document) > 600:
                print("... [document continues]")
            print("-" * 50)
            
            return True
            
        finally:
            await orchestrator.close()
    
    async def _test_system(self, command: str, *args) -> bool:
        """Run the comprehensive test suite."""
        print(" Running comprehensive test suite...")
        
        # Import and run the test suite
        from writer_agents.test_advanced_system import AdvancedSystemTester
        
        tester = AdvancedSystemTester()
        
        try:
            success = await tester.run_all_tests()
            
            if success:
                print(" All tests passed!")
                return True
            else:
                print("WARNING Some tests failed. Check the test report for details.")
                return False
                
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            print(f"x Test suite failed: {e}")
            return False
    
    async def _run_group_chat(self, command: str, *args) -> bool:
        """Launch the group analysis chat workflow."""
        print("Launching group analysis chat workflow...")
        loop = asyncio.get_running_loop()

        def _execute() -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                [sys.executable, "writer_agents/group_analysis_chat.py"],
                check=False,
            )

        result = await loop.run_in_executor(None, _execute)
        if result.returncode == 0:
            print("Group analysis chat completed.")
            return True
        print(f"Group analysis chat exited with status {result.returncode}.")
        return False

    async def _run_demo(self, command: str, *args) -> bool:
        """Run the demonstration script."""
        print(" Running demonstration script...")
        
        # Import and run the demo
        from writer_agents.demo_advanced_system import main as demo_main
        
        try:
            exit_code = await demo_main()
            return exit_code == 0
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"x Demo failed: {e}")
            return False


async def main():
    """Main entry point for natural commands."""
    parser = argparse.ArgumentParser(description="Natural language commands for the advanced writing system")
    parser.add_argument("command", help="The command to execute")
    parser.add_argument("args", nargs="*", help="Additional arguments for the command")
    
    args = parser.parse_args()
    
    processor = NaturalCommandProcessor()
    
    try:
        success = await processor.process_command(args.command, *args.args)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n Command interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        print(f"x Command execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
