"""Integration tests for hybrid workflow."""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from writer_agents.code.WorkflowOrchestrator import Conductor as WorkflowOrchestrator, WorkflowStrategyConfig
from writer_agents.code.insights import CaseInsights
from writer_agents.code.tasks import WriterDeliverable


@pytest.fixture
def sample_insights():
    """Sample case insights for testing."""
    return CaseInsights(
        case_id="test_case_001",
        summary="A case involving privacy violations requiring motion to seal documents",
        jurisdiction="US",
        case_style="Motion to Seal Documents",
        evidence={
            "OGC_Email": "Sent",
            "PRC_Awareness": "Direct",
            "PrivacyViolation": "Confirmed",
            "TimelineEvent": "Apr18_2025"
        },
        posteriors={
            "PrivacyHarm": 0.85,
            "Causation": 0.78,
            "LegalSuccess": 0.72,
            "TimelineEvent": 0.90
        },
        research_gaps=["Additional precedent research needed"]
    )


@pytest.fixture
async def mock_hybrid_orchestrator():
    """Create a mock hybrid orchestrator for testing."""

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


class TestHybridOrchestrator:
    """Test cases for Hybrid Orchestrator."""

    @pytest.mark.asyncio
    async def test_hybrid_orchestrator_initialization(self):
        """Test hybrid orchestrator initialization."""

        config = HybridOrchestratorConfig()
        orchestrator = HybridOrchestrator(config)

        assert orchestrator.config == config
        assert orchestrator.sk_kernel is not None
        assert orchestrator.autogen_factory is not None
        assert orchestrator.bridge is not None
        assert orchestrator.quality_gates is not None
        assert orchestrator.iteration_controller is not None

    @pytest.mark.asyncio
    async def test_workflow_router_selection(self, mock_hybrid_orchestrator, sample_insights):
        """Test workflow router selection logic."""

        # Test structured output requirement
        assert mock_hybrid_orchestrator._requires_structured_output(sample_insights) is True

        # Test non-structured output
        sample_insights.case_style = "General Legal Analysis"
        assert mock_hybrid_orchestrator._requires_structured_output(sample_insights) is False

    @pytest.mark.asyncio
    async def test_autogen_exploration_phase(self, mock_hybrid_orchestrator, sample_insights):
        """Test AutoGen exploration phase."""

        # Mock AutoGen agent response
        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value="Exploration completed: Privacy harm analysis needed")

        mock_hybrid_orchestrator.autogen_factory.create_exploration_agent.return_value = mock_agent

        result = await mock_hybrid_orchestrator._execute_exploration_phase(sample_insights)

        assert result is not None
        assert "privacy harm" in result.lower()
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_sk_drafting_phase(self, mock_hybrid_orchestrator, sample_insights):
        """Test SK drafting phase."""

        # Mock SK function responses
        mock_privacy_result = Mock()
        mock_privacy_result.value = "# Privacy Harm Analysis\n\nTest privacy harm content..."

        mock_timeline_result = Mock()
        mock_timeline_result.value = "# Factual Timeline\n\nTest timeline content..."

        mock_causation_result = Mock()
        mock_causation_result.value = "# Causation Analysis\n\nTest causation content..."

        mock_hybrid_orchestrator.sk_kernel.invoke_function.side_effect = [
            mock_privacy_result,
            mock_timeline_result,
            mock_causation_result
        ]

        result = await mock_hybrid_orchestrator._execute_drafting_phase(sample_insights)

        assert result is not None
        assert "privacy harm" in result.lower()
        assert "timeline" in result.lower()
        assert "causation" in result.lower()

        # Verify SK functions were called
        assert mock_hybrid_orchestrator.sk_kernel.invoke_function.call_count == 3

    @pytest.mark.asyncio
    async def test_quality_gate_validation(self, mock_hybrid_orchestrator, sample_insights):
        """Test quality gate validation."""

        # Mock validation results
        mock_citation_result = Mock()
        mock_citation_result.value = json.dumps({
            "score": 0.9,
            "passed": True,
            "details": "Valid citations found",
            "suggestions": []
        })

        mock_structure_result = Mock()
        mock_structure_result.value = json.dumps({
            "score": 1.0,
            "passed": True,
            "details": "All required sections present",
            "suggestions": []
        })

        mock_hybrid_orchestrator.sk_kernel.invoke_function.side_effect = [
            mock_citation_result,
            mock_structure_result
        ]

        document = "# Privacy Harm Analysis\n\nTest content with [OGC_Email:Sent] citations."

        result = await mock_hybrid_orchestrator._execute_validation_phase(document, sample_insights)

        assert result is not None
        assert result["overall_score"] > 0.8
        assert result["passed"] is True
        assert len(result["gate_results"]) == 2

    @pytest.mark.asyncio
    async def test_iteration_controller(self, mock_hybrid_orchestrator):
        """Test iteration controller."""

        # Test successful validation (no iteration needed)
        validation_result = {
            "passed": True,
            "overall_score": 0.9,
            "gate_results": []
        }

        should_iterate = mock_hybrid_orchestrator.iteration_controller.should_iterate(validation_result)
        assert should_iterate is False

        # Test failed validation (iteration needed)
        validation_result = {
            "passed": False,
            "overall_score": 0.6,
            "gate_results": []
        }

        should_iterate = mock_hybrid_orchestrator.iteration_controller.should_iterate(validation_result)
        assert should_iterate is True

    @pytest.mark.asyncio
    async def test_end_to_end_hybrid_workflow(self, mock_hybrid_orchestrator, sample_insights):
        """Test complete end-to-end hybrid workflow."""

        # Mock all phases
        mock_hybrid_orchestrator._execute_exploration_phase = AsyncMock(
            return_value="Exploration completed: Privacy harm analysis needed"
        )

        mock_hybrid_orchestrator._execute_drafting_phase = AsyncMock(
            return_value="# Privacy Harm Analysis\n\nTest content..."
        )

        mock_hybrid_orchestrator._execute_validation_phase = AsyncMock(
            return_value={
                "passed": True,
                "overall_score": 0.9,
                "gate_results": []
            }
        )

        # Execute workflow
        result = await mock_hybrid_orchestrator.run_hybrid_workflow(sample_insights)

        assert result is not None
        assert isinstance(result, WriterDeliverable)
        assert result.edited_document is not None
        assert "privacy harm" in result.edited_document.lower()
        assert result.metadata["workflow_type"] == "hybrid_sk"
        assert result.metadata["validation_passed"] is True


class TestQualityGatePipeline:
    """Test cases for Quality Gate Pipeline."""

    @pytest.mark.asyncio
    async def test_quality_gate_execution(self, mock_hybrid_orchestrator):
        """Test quality gate execution."""

        # Mock validation results
        mock_citation_result = Mock()
        mock_citation_result.value = json.dumps({
            "score": 0.9,
            "passed": True,
            "details": "Valid citations found",
            "suggestions": []
        })

        mock_hybrid_orchestrator.sk_kernel.invoke_function.return_value = mock_citation_result

        document = "Test document with [OGC_Email:Sent] citations."
        context = {"evidence": {"OGC_Email": "Sent"}}

        result = await mock_hybrid_orchestrator.quality_gates.run_gates(document, context)

        assert result is not None
        assert result["overall_score"] > 0.0
        assert len(result["gate_results"]) > 0

    @pytest.mark.asyncio
    async def test_quality_gate_failure_handling(self, mock_hybrid_orchestrator):
        """Test quality gate failure handling."""

        # Mock validation failure
        mock_citation_result = Mock()
        mock_citation_result.value = json.dumps({
            "score": 0.3,
            "passed": False,
            "details": "Invalid citations found",
            "suggestions": ["Fix citation format"]
        })

        mock_hybrid_orchestrator.sk_kernel.invoke_function.return_value = mock_citation_result

        document = "Test document with invalid citations."
        context = {"evidence": {"OGC_Email": "Sent"}}

        result = await mock_hybrid_orchestrator.quality_gates.run_gates(document, context)

        assert result is not None
        assert result["passed"] is False
        assert result["overall_score"] < 0.5


class TestAutoGenSKBridge:
    """Test cases for AutoGen-SK Bridge."""

    @pytest.mark.asyncio
    async def test_sk_function_registration(self, mock_hybrid_orchestrator):
        """Test SK function registration as AutoGen tools."""

        # Test function registration
        mock_hybrid_orchestrator.bridge.register_sk_function_as_tool.assert_called()

        # Verify tools are registered
        assert len(mock_hybrid_orchestrator.bridge.registered_functions) > 0

    @pytest.mark.asyncio
    async def test_sk_function_invocation(self, mock_hybrid_orchestrator):
        """Test SK function invocation through bridge."""

        # Mock SK function response
        mock_result = Mock()
        mock_result.value = "# Privacy Harm Analysis\n\nTest content..."
        mock_hybrid_orchestrator.sk_kernel.invoke_function.return_value = mock_result

        # Test function invocation
        result = await mock_hybrid_orchestrator.bridge.invoke_sk_function(
            plugin_name="PrivacyHarmPlugin",
            function_name="PrivacyHarmSemantic",
            evidence='{"OGC_Email": "Sent"}',
            posteriors='{"PrivacyHarm": 0.85}',
            case_summary="Test case"
        )

        assert result is not None
        assert "privacy harm" in result.lower()

    @pytest.mark.asyncio
    async def test_autogen_agent_creation(self, mock_hybrid_orchestrator):
        """Test AutoGen agent creation with SK tools."""

        # Mock AutoGen agent
        mock_agent = Mock()
        mock_hybrid_orchestrator.autogen_factory.create_exploration_agent.return_value = mock_agent

        # Test agent creation
        agent = mock_hybrid_orchestrator.autogen_factory.create_exploration_agent()

        assert agent is not None
        mock_hybrid_orchestrator.autogen_factory.create_exploration_agent.assert_called_once()


class TestErrorHandling:
    """Test error handling in hybrid workflow."""

    @pytest.mark.asyncio
    async def test_sk_kernel_error_handling(self, mock_hybrid_orchestrator, sample_insights):
        """Test SK kernel error handling."""

        # Mock SK kernel to raise exception
        mock_hybrid_orchestrator.sk_kernel.invoke_function.side_effect = Exception("SK kernel error")

        # Test error handling in drafting phase
        result = await mock_hybrid_orchestrator._execute_drafting_phase(sample_insights)

        # Should handle error gracefully
        assert result is not None
        assert "error" in result.lower() or "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_autogen_error_handling(self, mock_hybrid_orchestrator, sample_insights):
        """Test AutoGen error handling."""

        # Mock AutoGen agent to raise exception
        mock_agent = Mock()
        mock_agent.run = AsyncMock(side_effect=Exception("AutoGen error"))

        mock_hybrid_orchestrator.autogen_factory.create_exploration_agent.return_value = mock_agent

        # Test error handling in exploration phase
        result = await mock_hybrid_orchestrator._execute_exploration_phase(sample_insights)

        # Should handle error gracefully
        assert result is not None
        assert "error" in result.lower() or "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, mock_hybrid_orchestrator, sample_insights):
        """Test validation error handling."""

        # Mock validation to raise exception
        mock_hybrid_orchestrator.sk_kernel.invoke_function.side_effect = Exception("Validation error")

        document = "Test document"

        # Test error handling in validation phase
        result = await mock_hybrid_orchestrator._execute_validation_phase(document, sample_insights)

        # Should handle error gracefully
        assert result is not None
        assert result["passed"] is False
        assert result["overall_score"] == 0.0


class TestPerformanceMetrics:
    """Test performance metrics collection."""

    @pytest.mark.asyncio
    async def test_execution_time_metrics(self, mock_hybrid_orchestrator, sample_insights):
        """Test execution time metrics collection."""

        # Mock all phases to return quickly
        mock_hybrid_orchestrator._execute_exploration_phase = AsyncMock(
            return_value="Exploration completed"
        )
        mock_hybrid_orchestrator._execute_drafting_phase = AsyncMock(
            return_value="# Test Document\n\nTest content..."
        )
        mock_hybrid_orchestrator._execute_validation_phase = AsyncMock(
            return_value={"passed": True, "overall_score": 0.9, "gate_results": []}
        )

        # Execute workflow
        result = await mock_hybrid_orchestrator.run_hybrid_workflow(sample_insights)

        # Check performance metrics
        assert result.metadata["performance_metrics"] is not None
        assert "execution_time" in result.metadata["performance_metrics"]
        assert result.metadata["performance_metrics"]["execution_time"] > 0

    @pytest.mark.asyncio
    async def test_quality_metrics(self, mock_hybrid_orchestrator, sample_insights):
        """Test quality metrics collection."""

        # Mock validation with specific scores
        mock_hybrid_orchestrator._execute_validation_phase = AsyncMock(
            return_value={
                "passed": True,
                "overall_score": 0.85,
                "gate_results": [
                    {"gate": "citation_validity", "score": 0.9},
                    {"gate": "structure_complete", "score": 0.8}
                ]
            }
        )

        # Execute workflow
        result = await mock_hybrid_orchestrator.run_hybrid_workflow(sample_insights)

        # Check quality metrics
        assert result.metadata["quality_metrics"] is not None
        assert result.metadata["quality_metrics"]["overall_score"] == 0.85
        assert len(result.metadata["quality_metrics"]["gate_scores"]) == 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
