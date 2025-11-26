"""Demonstration script for the advanced multi-agent writing system."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Add the parent directory to the path to import from experiments
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.Wweb1 import build_case_insights, parse_evidence_input
from writer_agents.advanced_agents import AdvancedAgentConfig, AdvancedWriterOrchestrator
from writer_agents.enhanced_orchestrator import EnhancedOrchestratorConfig, EnhancedWriterOrchestrator
from writer_agents.insights import CaseInsights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_advanced_workflow():
    """Demonstrate the advanced multi-agent writing workflow."""
    print("=" * 80)
    print("ADVANCED MULTI-AGENT WRITING SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Create sample case insights
    evidence = parse_evidence_input("OGC_Email_Apr18_2025=Sent,PRC_Awareness=Direct")
    insights = build_case_insights(
        evidence=evidence,
        summary="Complex legal case involving regulatory compliance and potential liability exposure",
        reference_id="DEMO-2024-001"
    )
    
    print(f"\nCase Insights:")
    print(f"Reference ID: {insights.reference_id}")
    print(f"Summary: {insights.summary}")
    print(f"Evidence: {len(insights.evidence)} items")
    print(f"Posteriors: {len(insights.posteriors)} nodes")
    print(f"Jurisdiction: {insights.jurisdiction}")
    
    # Configure advanced system
    config = AdvancedAgentConfig(
        model_config=None, # Will use defaults
        max_review_rounds=2, # Reduced for demo
        enable_research_agents=True,
        enable_quality_gates=True,
        enable_adaptive_workflow=True,
    )
    
    print(f"\nAdvanced System Configuration:")
    print(f"- Max Review Rounds: {config.max_review_rounds}")
    print(f"- Research Agents: {config.enable_research_agents}")
    print(f"- Quality Gates: {config.enable_quality_gates}")
    print(f"- Adaptive Workflow: {config.enable_adaptive_workflow}")
    print(f"- Review Levels: {[level.value for level in config.review_levels]}")
    print(f"- Planning Orders: {[order.value for order in config.planning_orders]}")
    
    # Initialize advanced orchestrator
    orchestrator = AdvancedWriterOrchestrator(config)
    
    try:
        print(f"\n{'='*60}")
        print("EXECUTING ADVANCED WORKFLOW")
        print(f"{'='*60}")
        
        # Run the advanced workflow
        result = await orchestrator.run_advanced_workflow(insights)
        
        print(f"\nWorkflow Results:")
        print(f"- Plan Objective: {result.plan.objective}")
        print(f"- Deliverable Format: {result.plan.deliverable_format}")
        print(f"- Tone: {result.plan.tone}")
        print(f"- Sections Generated: {len(result.sections)}")
        print(f"- Reviews Conducted: {len(result.reviews)}")
        print(f"- Metadata: {result.metadata}")
        
        print(f"\nGenerated Document Preview:")
        print("-" * 40)
        # Show first 500 characters of the document
        preview = result.edited_document[:500]
        print(preview)
        if len(result.edited_document) > 500:
            print("... [document continues]")
        print("-" * 40)
        
        return result
        
    except Exception as e:
        logger.error(f"Advanced workflow failed: {e}")
        raise
    finally:
        await orchestrator.close()


async def demo_enhanced_orchestrator():
    """Demonstrate the enhanced orchestrator with intelligent workflow selection."""
    print("\n" + "=" * 80)
    print("ENHANCED ORCHESTRATOR WITH INTELLIGENT WORKFLOW SELECTION")
    print("=" * 80)
    
    # Create sample case insights with varying complexity
    test_cases = [
        {
            "name": "Simple Case",
            "evidence": "OGC_Email_Apr18_2025=Sent",
            "summary": "Basic regulatory compliance issue",
            "reference_id": "SIMPLE-001"
        },
        {
            "name": "Complex Case", 
            "evidence": "OGC_Email_Apr18_2025=Sent,PRC_Awareness=Direct,FinancialImpact=Severe",
            "summary": "Complex multi-jurisdictional legal matter involving regulatory compliance, potential liability exposure, financial implications, and reputational risks requiring comprehensive analysis and strategic recommendations",
            "reference_id": "COMPLEX-001"
        }
    ]
    
    # Configure enhanced orchestrator
    config = EnhancedOrchestratorConfig(
        use_advanced_workflow=True,
        complexity_threshold=0.6,
        enable_hybrid_mode=True,
        enable_performance_monitoring=True,
    )
    
    print(f"\nEnhanced Orchestrator Configuration:")
    print(f"- Use Advanced Workflow: {config.use_advanced_workflow}")
    print(f"- Complexity Threshold: {config.complexity_threshold}")
    print(f"- Hybrid Mode: {config.enable_hybrid_mode}")
    print(f"- Performance Monitoring: {config.enable_performance_monitoring}")
    
    orchestrator = EnhancedWriterOrchestrator(config)
    
    try:
        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"TESTING: {test_case['name']}")
            print(f"{'='*60}")
            
            # Build insights
            evidence = parse_evidence_input(test_case["evidence"])
            insights = build_case_insights(
                evidence=evidence,
                summary=test_case["summary"],
                reference_id=test_case["reference_id"]
            )
            
            # Get workflow recommendations
            recommendations = orchestrator.get_workflow_recommendations(insights)
            
            print(f"\nWorkflow Analysis:")
            print(f"- Complexity Score: {recommendations['complexity_score']:.2f}")
            print(f"- Recommended Workflow: {recommendations['recommended_workflow']}")
            print(f"- Workflow Characteristics:")
            for key, value in recommendations['workflow_characteristics'].items():
                print(f" * {key}: {value}")
            
            # Execute the recommended workflow
            print(f"\nExecuting {recommendations['recommended_workflow']} workflow...")
            result = await orchestrator.run_intelligent_workflow(insights)
            
            print(f"\nResults:")
            print(f"- Workflow Type Used: {result.metadata.get('workflow_type', 'unknown')}")
            print(f"- Complexity Score: {result.metadata.get('complexity_score', 0):.2f}")
            print(f"- Performance Metrics: {result.metadata.get('performance_metrics', {})}")
            print(f"- Document Length: {len(result.edited_document)} characters")
            print(f"- Sections Generated: {len(result.sections)}")
            
    except Exception as e:
        logger.error(f"Enhanced orchestrator demo failed: {e}")
        raise
    finally:
        await orchestrator.close()


async def demo_workflow_comparison():
    """Demonstrate comparison between traditional and advanced workflows."""
    print("\n" + "=" * 80)
    print("WORKFLOW COMPARISON: TRADITIONAL vs ADVANCED")
    print("=" * 80)
    
    # Create a moderately complex case for comparison
    evidence = parse_evidence_input("OGC_Email_Apr18_2025=Sent,PRC_Awareness=Direct")
    insights = build_case_insights(
        evidence=evidence,
        summary="Moderate complexity legal case requiring thorough analysis and strategic recommendations",
        reference_id="COMPARISON-001"
    )
    
    print(f"\nTest Case:")
    print(f"- Reference ID: {insights.reference_id}")
    print(f"- Summary: {insights.summary}")
    print(f"- Evidence Items: {len(insights.evidence)}")
    print(f"- Posterior Nodes: {len(insights.posteriors)}")
    
    # Test traditional workflow
    print(f"\n{'='*40}")
    print("TRADITIONAL WORKFLOW")
    print(f"{'='*40}")
    
    from writer_agents.orchestrator import WriterOrchestrator, WriterOrchestratorConfig
    
    traditional_config = WriterOrchestratorConfig()
    traditional_orchestrator = WriterOrchestrator(traditional_config)
    
    try:
        traditional_result = await traditional_orchestrator.run(insights)
        print(f"Traditional Results:")
        print(f"- Sections: {len(traditional_result.sections)}")
        print(f"- Reviews: {len(traditional_result.reviews)}")
        print(f"- Document Length: {len(traditional_result.edited_document)} chars")
    finally:
        await traditional_orchestrator.close()
    
    # Test advanced workflow
    print(f"\n{'='*40}")
    print("ADVANCED WORKFLOW")
    print(f"{'='*40}")
    
    advanced_config = AdvancedAgentConfig(max_review_rounds=1) # Reduced for demo
    advanced_orchestrator = AdvancedWriterOrchestrator(advanced_config)
    
    try:
        advanced_result = await advanced_orchestrator.run_advanced_workflow(insights)
        print(f"Advanced Results:")
        print(f"- Sections: {len(advanced_result.sections)}")
        print(f"- Reviews: {len(advanced_result.reviews)}")
        print(f"- Document Length: {len(advanced_result.edited_document)} chars")
        print(f"- Metadata: {advanced_result.metadata}")
    finally:
        await advanced_orchestrator.close()
    
    # Compare results
    print(f"\n{'='*40}")
    print("COMPARISON SUMMARY")
    print(f"{'='*40}")
    print(f"Traditional Workflow:")
    print(f" - Simpler, faster execution")
    print(f" - Basic review processes")
    print(f" - Standard quality assurance")
    print(f"\nAdvanced Workflow:")
    print(f" - Multi-order planning (strategic, tactical, operational)")
    print(f" - Nested review processes with multiple levels")
    print(f" - Research agents for fact-checking")
    print(f" - Quality gates and adaptive optimization")
    print(f" - More comprehensive but slower execution")


async def main():
    """Main demonstration function."""
    print("Starting Advanced Multi-Agent Writing System Demonstrations...")
    
    try:
        # Demo 1: Advanced workflow
        await demo_advanced_workflow()
        
        # Demo 2: Enhanced orchestrator
        await demo_enhanced_orchestrator()
        
        # Demo 3: Workflow comparison
        await demo_workflow_comparison()
        
        print("\n" + "=" * 80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\nDemonstration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
