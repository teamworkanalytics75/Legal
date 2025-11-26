"""Benchmark comparison tests for different workflow approaches."""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from writer_agents.code.WorkflowOrchestrator import Conductor as WorkflowOrchestrator, WorkflowStrategyConfig
from writer_agents.code.EnhancedOrchestrator import EnhancedWriterOrchestrator, EnhancedOrchestratorConfig
from writer_agents.code.insights import CaseInsights
from writer_agents.code.tasks import WriterDeliverable


@pytest.fixture
def benchmark_insights():
    """Standardized case insights for benchmarking."""
    return CaseInsights(
        case_id="benchmark_case_001",
        summary="A complex privacy violation case requiring comprehensive legal analysis and motion to seal documents",
        jurisdiction="US",
        case_style="Motion to Seal Documents",
        evidence={
            "OGC_Email": "Sent",
            "PRC_Awareness": "Direct",
            "PrivacyViolation": "Confirmed",
            "TimelineEvent": "Apr18_2025",
            "CausationLink": "Established",
            "LegalPrecedent": "Supporting"
        },
        posteriors={
            "PrivacyHarm": 0.85,
            "Causation": 0.78,
            "LegalSuccess": 0.72,
            "TimelineEvent": 0.90,
            "SealingSuccess": 0.80
        },
        research_gaps=["Additional precedent research", "Causation analysis refinement"]
    )


@pytest.fixture
async def mock_hybrid_orchestrator():
    """Mock hybrid orchestrator for benchmarking."""

    # Mock the SK kernel
    mock_kernel = Mock(spec=Kernel)
    mock_kernel.invoke_function = AsyncMock()

    # Mock AutoGen factory
    mock_autogen_factory = Mock()
    mock_autogen_factory.close = AsyncMock()

    # Mock AutoGen-SK bridge
    mock_bridge = Mock()
    mock_bridge.register_sk_function_as_tool = Mock()

    # Create orchestrator with mocked components
    orchestrator = HybridOrchestrator(HybridOrchestratorConfig())
    orchestrator.sk_kernel = mock_kernel
    orchestrator.autogen_factory = mock_autogen_factory
    orchestrator.bridge = mock_bridge

    return orchestrator


@pytest.fixture
async def mock_enhanced_orchestrator():
    """Mock enhanced orchestrator for benchmarking."""

    # Mock AutoGen factory
    mock_autogen_factory = Mock()
    mock_autogen_factory.close = AsyncMock()

    # Create orchestrator with mocked components
    config = EnhancedOrchestratorConfig()
    orchestrator = EnhancedWriterOrchestrator(config)
    orchestrator.autogen_factory = mock_autogen_factory

    return orchestrator


class TestWorkflowBenchmarks:
    """Benchmark tests comparing different workflow approaches."""

    @pytest.mark.asyncio
    async def test_hybrid_sk_workflow_benchmark(self, mock_hybrid_orchestrator, benchmark_insights):
        """Benchmark hybrid SK workflow performance."""

        # Mock all phases with realistic timing
        mock_hybrid_orchestrator._execute_exploration_phase = AsyncMock(
            return_value="Exploration completed: Privacy harm analysis needed"
        )

        mock_hybrid_orchestrator._execute_drafting_phase = AsyncMock(
            return_value="# Privacy Harm Analysis\n\nTest content with structured output..."
        )

        mock_hybrid_orchestrator._execute_validation_phase = AsyncMock(
            return_value={
                "passed": True,
                "overall_score": 0.9,
                "gate_results": []
            }
        )

        # Measure execution time
        start_time = time.time()
        result = await mock_hybrid_orchestrator.run_hybrid_workflow(benchmark_insights)
        execution_time = time.time() - start_time

        # Verify results
        assert result is not None
        assert isinstance(result, WriterDeliverable)
        assert result.edited_document is not None
        assert result.metadata["workflow_type"] == "hybrid_sk"

        # Performance assertions
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert result.metadata["performance_metrics"]["execution_time"] > 0

        print(f"Hybrid SK Workflow Execution Time: {execution_time:.2f}s")
        print(f"Document Length: {len(result.edited_document)} characters")
        print(f"Quality Score: {result.metadata.get('quality_metrics', {}).get('overall_score', 'N/A')}")

    @pytest.mark.asyncio
    async def test_traditional_autogen_workflow_benchmark(self, mock_enhanced_orchestrator, benchmark_insights):
        """Benchmark traditional AutoGen workflow performance."""

        # Mock AutoGen agent response
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value="# Legal Analysis\n\nTraditional AutoGen content...")

        mock_enhanced_orchestrator.autogen_factory.create_exploration_agent.return_value = mock_agent

        # Measure execution time
        start_time = time.time()
        result = await mock_enhanced_orchestrator.run_intelligent_workflow(benchmark_insights)
        execution_time = time.time() - start_time

        # Verify results
        assert result is not None
        assert isinstance(result, WriterDeliverable)
        assert result.edited_document is not None

        # Performance assertions
        assert execution_time < 5.0  # Should complete within 5 seconds

        print(f"Traditional AutoGen Workflow Execution Time: {execution_time:.2f}s")
        print(f"Document Length: {len(result.edited_document)} characters")

    @pytest.mark.asyncio
    async def test_workflow_comparison(self, mock_hybrid_orchestrator, mock_enhanced_orchestrator, benchmark_insights):
        """Compare different workflow approaches."""

        results = {}

        # Test Hybrid SK Workflow
        mock_hybrid_orchestrator._execute_exploration_phase = AsyncMock(
            return_value="Exploration completed"
        )
        mock_hybrid_orchestrator._execute_drafting_phase = AsyncMock(
            return_value="# Privacy Harm Analysis\n\nStructured SK content with citations [OGC_Email:Sent]..."
        )
        mock_hybrid_orchestrator._execute_validation_phase = AsyncMock(
            return_value={"passed": True, "overall_score": 0.9, "gate_results": []}
        )

        start_time = time.time()
        hybrid_result = await mock_hybrid_orchestrator.run_hybrid_workflow(benchmark_insights)
        hybrid_time = time.time() - start_time

        results["hybrid_sk"] = {
            "execution_time": hybrid_time,
            "document_length": len(hybrid_result.edited_document),
            "quality_score": hybrid_result.metadata.get("quality_metrics", {}).get("overall_score", 0),
            "validation_passed": hybrid_result.metadata.get("validation_passed", False),
            "workflow_type": hybrid_result.metadata.get("workflow_type", "unknown")
        }

        # Test Traditional AutoGen Workflow
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value="# Legal Analysis\n\nTraditional AutoGen content...")
        mock_enhanced_orchestrator.autogen_factory.create_exploration_agent.return_value = mock_agent

        start_time = time.time()
        traditional_result = await mock_enhanced_orchestrator.run_intelligent_workflow(benchmark_insights)
        traditional_time = time.time() - start_time

        results["traditional_autogen"] = {
            "execution_time": traditional_time,
            "document_length": len(traditional_result.edited_document),
            "quality_score": 0.7,  # Mock quality score
            "validation_passed": False,  # Traditional workflow doesn't have validation
            "workflow_type": traditional_result.metadata.get("workflow_type", "traditional")
        }

        # Compare results
        print("\n" + "="*60)
        print("WORKFLOW COMPARISON RESULTS")
        print("="*60)

        for workflow_name, metrics in results.items():
            print(f"\n{workflow_name.upper()}:")
            print(f"  Execution Time: {metrics['execution_time']:.2f}s")
            print(f"  Document Length: {metrics['document_length']} characters")
            print(f"  Quality Score: {metrics['quality_score']:.2f}")
            print(f"  Validation Passed: {metrics['validation_passed']}")
            print(f"  Workflow Type: {metrics['workflow_type']}")

        # Performance assertions
        assert results["hybrid_sk"]["execution_time"] < 5.0
        assert results["traditional_autogen"]["execution_time"] < 5.0

        # Quality assertions
        assert results["hybrid_sk"]["quality_score"] >= results["traditional_autogen"]["quality_score"]
        assert results["hybrid_sk"]["validation_passed"] is True
        assert results["traditional_autogen"]["validation_passed"] is False


class TestQualityBenchmarks:
    """Quality-focused benchmark tests."""

    @pytest.mark.asyncio
    async def test_citation_quality_benchmark(self, mock_hybrid_orchestrator, benchmark_insights):
        """Benchmark citation quality in different workflows."""

        # Mock SK drafting with proper citations
        mock_hybrid_orchestrator._execute_drafting_phase = AsyncMock(
            return_value="""
            # Privacy Harm Analysis

            The privacy harm [OGC_Email:Sent] demonstrates [PRC_Awareness:Direct]
            the violation [PrivacyViolation:Confirmed] as established in the timeline [TimelineEvent:Apr18_2025].
            """
        )

        # Mock validation with high citation quality
        mock_hybrid_orchestrator._execute_validation_phase = AsyncMock(
            return_value={
                "passed": True,
                "overall_score": 0.95,
                "gate_results": [
                    {"gate": "citation_validity", "score": 1.0, "details": "All citations valid"},
                    {"gate": "evidence_grounding", "score": 0.9, "details": "Strong evidence grounding"}
                ]
            }
        )

        result = await mock_hybrid_orchestrator.run_hybrid_workflow(benchmark_insights)

        # Verify citation quality
        document = result.edited_document
        citation_count = document.count("[")
        evidence_refs = len(benchmark_insights.evidence)

        assert citation_count >= evidence_refs  # Should cite all evidence
        assert result.metadata["quality_metrics"]["overall_score"] >= 0.9

        print(f"Citation Quality Benchmark:")
        print(f"  Citations Found: {citation_count}")
        print(f"  Evidence Items: {evidence_refs}")
        print(f"  Citation Coverage: {citation_count/evidence_refs:.1%}")
        print(f"  Quality Score: {result.metadata['quality_metrics']['overall_score']:.2f}")

    @pytest.mark.asyncio
    async def test_structure_quality_benchmark(self, mock_hybrid_orchestrator, benchmark_insights):
        """Benchmark document structure quality."""

        # Mock SK drafting with proper structure
        mock_hybrid_orchestrator._execute_drafting_phase = AsyncMock(
            return_value="""
            # Privacy Harm Analysis

            ## Introduction
            This section analyzes the privacy harm in this case.

            ## Analysis
            The evidence demonstrates significant privacy violations.

            ## Conclusion
            The privacy harm analysis supports the motion to seal.
            """
        )

        # Mock validation with perfect structure
        mock_hybrid_orchestrator._execute_validation_phase = AsyncMock(
            return_value={
                "passed": True,
                "overall_score": 1.0,
                "gate_results": [
                    {"gate": "structure_complete", "score": 1.0, "details": "All required sections present"}
                ]
            }
        )

        result = await mock_hybrid_orchestrator.run_hybrid_workflow(benchmark_insights)

        # Verify structure quality
        document = result.edited_document
        required_sections = ["Introduction", "Analysis", "Conclusion"]

        for section in required_sections:
            assert section in document

        assert result.metadata["quality_metrics"]["overall_score"] == 1.0

        print(f"Structure Quality Benchmark:")
        print(f"  Required Sections: {len(required_sections)}")
        print(f"  Sections Found: {sum(1 for section in required_sections if section in document)}")
        print(f"  Structure Score: {result.metadata['quality_metrics']['overall_score']:.2f}")


class TestConsistencyBenchmarks:
    """Consistency-focused benchmark tests."""

    @pytest.mark.asyncio
    async def test_output_consistency_benchmark(self, mock_hybrid_orchestrator, benchmark_insights):
        """Benchmark output consistency across multiple runs."""

        # Mock consistent responses
        mock_hybrid_orchestrator._execute_exploration_phase = AsyncMock(
            return_value="Exploration completed: Privacy harm analysis needed"
        )
        mock_hybrid_orchestrator._execute_drafting_phase = AsyncMock(
            return_value="# Privacy Harm Analysis\n\nConsistent structured output..."
        )
        mock_hybrid_orchestrator._execute_validation_phase = AsyncMock(
            return_value={"passed": True, "overall_score": 0.9, "gate_results": []}
        )

        # Run multiple iterations
        results = []
        for i in range(3):
            result = await mock_hybrid_orchestrator.run_hybrid_workflow(benchmark_insights)
            results.append(result)

        # Verify consistency
        assert len(results) == 3

        # Check that all results have similar quality scores
        quality_scores = [r.metadata.get("quality_metrics", {}).get("overall_score", 0) for r in results]
        assert all(score >= 0.8 for score in quality_scores)

        # Check that all results have similar document lengths
        doc_lengths = [len(r.edited_document) for r in results]
        assert all(length > 100 for length in doc_lengths)

        print(f"Consistency Benchmark:")
        print(f"  Quality Scores: {quality_scores}")
        print(f"  Document Lengths: {doc_lengths}")
        print(f"  Quality Score Variance: {max(quality_scores) - min(quality_scores):.2f}")
        print(f"  Length Variance: {max(doc_lengths) - min(doc_lengths)} characters")


class TestScalabilityBenchmarks:
    """Scalability-focused benchmark tests."""

    @pytest.mark.asyncio
    async def test_large_case_benchmark(self, mock_hybrid_orchestrator):
        """Benchmark performance with large case data."""

        # Create large case insights
        large_insights = CaseInsights(
            case_id="large_case_001",
            summary="A complex case with extensive evidence and multiple legal issues requiring comprehensive analysis",
            jurisdiction="US",
            case_style="Motion to Seal Documents",
            evidence={f"Evidence_{i}": f"State_{i}" for i in range(50)},  # 50 evidence items
            posteriors={f"Posterior_{i}": 0.8 for i in range(20)},  # 20 posteriors
            research_gaps=[f"Research gap {i}" for i in range(10)]  # 10 research gaps
        )

        # Mock responses
        mock_hybrid_orchestrator._execute_exploration_phase = AsyncMock(
            return_value="Exploration completed for large case"
        )
        mock_hybrid_orchestrator._execute_drafting_phase = AsyncMock(
            return_value="# Large Case Analysis\n\n" + "Test content " * 100
        )
        mock_hybrid_orchestrator._execute_validation_phase = AsyncMock(
            return_value={"passed": True, "overall_score": 0.9, "gate_results": []}
        )

        # Measure execution time
        start_time = time.time()
        result = await mock_hybrid_orchestrator.run_hybrid_workflow(large_insights)
        execution_time = time.time() - start_time

        # Verify results
        assert result is not None
        assert len(result.edited_document) > 1000  # Should handle large content
        assert execution_time < 10.0  # Should complete within 10 seconds

        print(f"Large Case Benchmark:")
        print(f"  Evidence Items: {len(large_insights.evidence)}")
        print(f"  Posterior Items: {len(large_insights.posteriors)}")
        print(f"  Research Gaps: {len(large_insights.research_gaps)}")
        print(f"  Execution Time: {execution_time:.2f}s")
        print(f"  Document Length: {len(result.edited_document)} characters")


if __name__ == "__main__":
    # Run benchmarks
    pytest.main([__file__, "-v", "-s"])
