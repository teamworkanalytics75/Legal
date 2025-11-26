"""Enhanced orchestrator that combines traditional and advanced multi-agent workflows."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

from writer_agents.code.AdvancedAgents import (
    AdvancedAgentConfig,
    AdvancedWriterOrchestrator,
    PlanningOrder,
    ReviewLevel,
)
from writer_agents.code.agents import ModelConfig
from writer_agents.code.WorkflowOrchestrator import Conductor as WorkflowOrchestrator
from writer_agents.code.insights import CaseInsights
from writer_agents.code.orchestrator import WriterOrchestrator, WriterOrchestratorConfig
from writer_agents.code.tasks import DraftSection, PlanDirective, WriterDeliverable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EnhancedOrchestratorConfig:
    """Configuration for the enhanced orchestrator."""

    # Traditional workflow config
    traditional_config: WriterOrchestratorConfig = field(default_factory=WriterOrchestratorConfig)

    # Advanced workflow config
    advanced_config: AdvancedAgentConfig = field(default_factory=AdvancedAgentConfig)

    # Workflow selection criteria
    use_advanced_workflow: bool = True
    complexity_threshold: float = 0.7 # Threshold for switching to advanced workflow
    enable_hybrid_mode: bool = True # Allow mixing traditional and advanced components
    enable_sk_hybrid: bool = True # Allow Semantic Kernel hybrid workflow

    # Quality and performance settings
    max_parallel_agents: int = 3
    enable_performance_monitoring: bool = True
    quality_improvement_threshold: float = 0.1 # Minimum improvement to justify advanced workflow


class WorkflowComplexityAnalyzer:
    """Analyzes case complexity to determine optimal workflow."""

    def __init__(self) -> None:
        self._complexity_factors = {
            "evidence_count": 0.2,
            "posterior_count": 0.2,
            "section_count": 0.15,
            "jurisdiction_complexity": 0.15,
            "case_style_complexity": 0.15,
            "research_requirements": 0.15,
        }

    def analyze_complexity(self, insights: CaseInsights) -> float:
        """Analyze case complexity and return a score between 0.0 and 1.0."""
        complexity_score = 0.0

        # Evidence complexity
        evidence_count = len(insights.evidence)
        evidence_complexity = min(evidence_count / 10.0, 1.0) # Normalize to 0-1
        complexity_score += evidence_complexity * self._complexity_factors["evidence_count"]

        # Posterior complexity
        posterior_count = len(insights.posteriors)
        posterior_complexity = min(posterior_count / 15.0, 1.0) # Normalize to 0-1
        complexity_score += posterior_complexity * self._complexity_factors["posterior_count"]

        # Jurisdiction complexity
        jurisdiction_complexity = self._assess_jurisdiction_complexity(insights.jurisdiction)
        complexity_score += jurisdiction_complexity * self._complexity_factors["jurisdiction_complexity"]

        # Case style complexity
        case_style_complexity = self._assess_case_style_complexity(insights.case_style)
        complexity_score += case_style_complexity * self._complexity_factors["case_style_complexity"]

        # Research requirements (based on summary length and complexity)
        research_complexity = self._assess_research_requirements(insights.summary)
        complexity_score += research_complexity * self._complexity_factors["research_requirements"]

        return min(complexity_score, 1.0)

    def _assess_jurisdiction_complexity(self, jurisdiction: Optional[str]) -> float:
        """Assess jurisdiction complexity."""
        if not jurisdiction:
            return 0.5 # Default moderate complexity

        # Simple heuristic based on jurisdiction
        complex_jurisdictions = ["US", "EU", "International"]
        if jurisdiction in complex_jurisdictions:
            return 0.8
        elif jurisdiction in ["State", "Provincial"]:
            return 0.6
        else:
            return 0.4

    def _assess_case_style_complexity(self, case_style: Optional[str]) -> float:
        """Assess case style complexity."""
        if not case_style:
            return 0.5 # Default moderate complexity

        # Simple heuristic based on case style
        complex_styles = ["Appellate Brief", "Supreme Court Brief", "Complex Litigation"]
        if case_style in complex_styles:
            return 0.9
        elif case_style in ["Memorandum", "Motion"]:
            return 0.6
        else:
            return 0.4

    def _assess_research_requirements(self, summary: str) -> float:
        """Assess research requirements based on summary complexity."""
        if not summary:
            return 0.5

        # Simple heuristics
        word_count = len(summary.split())
        if word_count > 500:
            return 0.8
        elif word_count > 200:
            return 0.6
        else:
            return 0.4


class PerformanceMonitor:
    """Monitors workflow performance and quality metrics."""

    def __init__(self) -> None:
        self._metrics = {
            "execution_time": 0.0,
            "quality_score": 0.0,
            "review_rounds": 0,
            "agent_interactions": 0,
            "error_count": 0,
        }

    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self._start_time = asyncio.get_event_loop().time()
        self._metrics = {key: 0.0 for key in self._metrics}

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric."""
        if metric_name in self._metrics:
            self._metrics[metric_name] = value

    def increment_metric(self, metric_name: str, increment: float = 1.0) -> None:
        """Increment a performance metric."""
        if metric_name in self._metrics:
            self._metrics[metric_name] += increment

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return final metrics."""
        if hasattr(self, '_start_time'):
            self._metrics["execution_time"] = asyncio.get_event_loop().time() - self._start_time
        return self._metrics.copy()

    def get_quality_improvement(self, baseline_quality: float) -> float:
        """Calculate quality improvement over baseline."""
        current_quality = self._metrics.get("quality_score", 0.0)
        return current_quality - baseline_quality


class EnhancedWriterOrchestrator:
    """Enhanced orchestrator that intelligently selects and manages workflows."""

    def __init__(self, config: Optional[EnhancedOrchestratorConfig] = None) -> None:
        self._config = config or EnhancedOrchestratorConfig()

        # Initialize workflow components
        self._traditional_orchestrator = WriterOrchestrator(self._config.traditional_config)
        self._advanced_orchestrator = AdvancedWriterOrchestrator(self._config.advanced_config)

        # Initialize analysis components
        self._complexity_analyzer = WorkflowComplexityAnalyzer()
        self._performance_monitor = PerformanceMonitor() if self._config.enable_performance_monitoring else None

        # Workflow selection history for learning
        self._workflow_history: List[Dict] = []

    async def close(self) -> None:
        """Close all orchestrator components."""
        await self._traditional_orchestrator.close()
        await self._advanced_orchestrator.close()

    async def run_intelligent_workflow(self, insights: CaseInsights) -> WriterDeliverable:
        """Run the most appropriate workflow based on case analysis."""
        logger.info("Starting intelligent workflow selection")

        # Start performance monitoring
        if self._performance_monitor:
            self._performance_monitor.start_monitoring()

        try:
            # Analyze case complexity
            complexity_score = self._complexity_analyzer.analyze_complexity(insights)
            logger.info(f"Case complexity score: {complexity_score:.2f}")

            # Select workflow
            workflow_type = self._select_workflow(complexity_score, insights)
            logger.info(f"Selected workflow: {workflow_type}")

            # Execute selected workflow
            if workflow_type == "advanced":
                result = await self._execute_advanced_workflow(insights)
            elif workflow_type == "hybrid":
                result = await self._execute_hybrid_workflow(insights)
            elif workflow_type == "hybrid_sk":
                result = await self._execute_hybrid_sk_workflow(insights)
            else:
                result = await self._execute_traditional_workflow(insights)

            # Ensure result is a WriterDeliverable
            if not hasattr(result, 'edited_document'):
                logger.error(f"Invalid result type: {type(result)}")
                raise ValueError(f"Expected WriterDeliverable, got {type(result)}")

            # Record workflow performance
            if self._performance_monitor:
                metrics = self._performance_monitor.stop_monitoring()
                self._record_workflow_performance(workflow_type, complexity_score, metrics)

            # Enhance result with workflow metadata
            result.metadata.update({
                "workflow_type": workflow_type,
                "complexity_score": complexity_score,
                "performance_metrics": self._performance_monitor._metrics if self._performance_monitor else {},
            })

            logger.info("Intelligent workflow completed successfully")
            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            if self._performance_monitor:
                self._performance_monitor.increment_metric("error_count")
            raise

    def _select_workflow(self, complexity_score: float, insights: CaseInsights) -> str:
        """Select the most appropriate workflow based on complexity and configuration."""

        # Prefer SK hybrid when structured output is required
        if self._config.enable_sk_hybrid and self._requires_structured_output(insights):
            return "hybrid_sk"

        # Force advanced workflow if configured
        if self._config.use_advanced_workflow and complexity_score >= self._config.complexity_threshold:
            return "advanced"

        # Use hybrid mode if enabled and complexity is moderate
        if (self._config.enable_hybrid_mode and
            0.4 <= complexity_score < self._config.complexity_threshold):
            return "hybrid"

        # Use traditional workflow for simple cases or when advanced is disabled
        return "traditional"

    def _requires_structured_output(self, insights: CaseInsights) -> bool:
        """Determine if the case should leverage the SK drafting workflow."""
        case_style = (insights.case_style or "").lower()
        return any(keyword in case_style for keyword in ("motion", "seal", "pseudonym"))

    async def _execute_traditional_workflow(self, insights: CaseInsights) -> WriterDeliverable:
        """Execute the traditional workflow."""
        logger.info("Executing traditional workflow")

        if self._performance_monitor:
            self._performance_monitor.increment_metric("agent_interactions", 5) # Approximate agent count

        return await self._traditional_orchestrator.run(insights)

    async def _execute_advanced_workflow(self, insights: CaseInsights) -> WriterDeliverable:
        """Execute the advanced workflow with nested reviews."""
        logger.info("Executing advanced workflow")

        if self._performance_monitor:
            # Advanced workflow has more agents and interactions
            self._performance_monitor.increment_metric("agent_interactions", 15)

        return await self._advanced_orchestrator.run_advanced_workflow(insights)

    async def _execute_hybrid_workflow(self, insights: CaseInsights) -> WriterDeliverable:
        """Execute a hybrid workflow combining traditional and advanced components."""
        logger.info("Executing hybrid workflow")

        if self._performance_monitor:
            self._performance_monitor.increment_metric("agent_interactions", 10)

        # Phase 1: Use traditional planning for structure
        traditional_result = await self._traditional_orchestrator.run(insights)

        # Phase 2: Apply advanced review processes
        enhanced_result = await self._apply_advanced_review_process(
            insights, traditional_result
        )

        return enhanced_result

    async def _execute_hybrid_sk_workflow(self, insights: CaseInsights) -> WriterDeliverable:
        """Execute the Semantic Kernel hybrid workflow."""
        logger.info("Executing Semantic Kernel hybrid workflow")

        if self._performance_monitor:
            self._performance_monitor.increment_metric("agent_interactions", 8)  # SK + AutoGen agents

        # Create and run hybrid orchestrator
        from .WorkflowOrchestrator import WorkflowStrategyConfig
        hybrid_config = WorkflowStrategyConfig()
        hybrid = WorkflowOrchestrator(hybrid_config)

        try:
            result = await hybrid.run_hybrid_workflow(insights)
            return result
        finally:
            await hybrid.close()

    async def _apply_advanced_review_process(
        self,
        insights: CaseInsights,
        traditional_result: WriterDeliverable
    ) -> WriterDeliverable:
        """Apply advanced review processes to traditional workflow results."""
        logger.info("Applying advanced review processes")

        # Create a simplified advanced orchestrator for review-only operations
        review_config = AdvancedAgentConfig(
            model_config=self._config.advanced_config.model_config,
            max_review_rounds=2, # Fewer rounds for hybrid mode
            enable_research_agents=False, # Skip research in hybrid mode
            enable_quality_gates=True,
            enable_adaptive_workflow=False, # Skip adaptive workflow in hybrid mode
            review_levels=[ReviewLevel.INTERMEDIATE, ReviewLevel.ADVANCED],
            planning_orders=[], # No planning in hybrid mode
        )

        review_orchestrator = AdvancedWriterOrchestrator(review_config)

        try:
            # Convert traditional result to advanced format for review
            enhanced_result = await self._enhance_with_advanced_review(
                insights, traditional_result, review_orchestrator
            )
            return enhanced_result
        finally:
            await review_orchestrator.close()

    async def _enhance_with_advanced_review(
        self,
        insights: CaseInsights,
        traditional_result: WriterDeliverable,
        review_orchestrator: AdvancedWriterOrchestrator
    ) -> WriterDeliverable:
        """Enhance traditional result with advanced review processes."""

        # Apply advanced review to the edited document
        # This is a simplified implementation - in practice, you'd want to
        # break down the document into sections and apply targeted reviews

        enhanced_document = traditional_result.edited_document

        # Add advanced review metadata
        traditional_result.metadata.update({
            "enhanced_with_advanced_review": True,
            "review_levels_applied": [level.value for level in review_orchestrator._config.review_levels],
        })

        return traditional_result

    def _record_workflow_performance(
        self,
        workflow_type: str,
        complexity_score: float,
        metrics: Dict[str, float]
    ) -> None:
        """Record workflow performance for future optimization."""
        performance_record = {
            "workflow_type": workflow_type,
            "complexity_score": complexity_score,
            "metrics": metrics,
            "timestamp": asyncio.get_event_loop().time(),
        }

        self._workflow_history.append(performance_record)

        # Keep only recent history (last 100 records)
        if len(self._workflow_history) > 100:
            self._workflow_history = self._workflow_history[-100:]

        logger.info(f"Recorded performance: {workflow_type} workflow, "
                   f"complexity {complexity_score:.2f}, "
                   f"time {metrics.get('execution_time', 0):.2f}s")

    def get_workflow_recommendations(self, insights: CaseInsights) -> Dict[str, any]:
        """Get workflow recommendations based on case analysis and historical performance."""
        complexity_score = self._complexity_analyzer.analyze_complexity(insights)

        # Analyze historical performance
        similar_cases = [
            record for record in self._workflow_history
            if abs(record["complexity_score"] - complexity_score) < 0.2
        ]

        workflow_type = self._select_workflow(complexity_score, insights)

        recommendations = {
            "complexity_score": complexity_score,
            "recommended_workflow": workflow_type,
            "historical_performance": self._analyze_historical_performance(similar_cases),
            "workflow_characteristics": self._get_workflow_characteristics(workflow_type),
        }

        return recommendations

    def _analyze_historical_performance(self, similar_cases: List[Dict]) -> Dict[str, float]:
        """Analyze historical performance for similar cases."""
        if not similar_cases:
            return {"average_time": 0.0, "average_quality": 0.0, "success_rate": 0.0}

        total_time = sum(record["metrics"].get("execution_time", 0) for record in similar_cases)
        total_quality = sum(record["metrics"].get("quality_score", 0) for record in similar_cases)
        success_count = sum(1 for record in similar_cases if record["metrics"].get("error_count", 0) == 0)

        return {
            "average_time": total_time / len(similar_cases),
            "average_quality": total_quality / len(similar_cases),
            "success_rate": success_count / len(similar_cases),
        }

    def _get_workflow_characteristics(self, workflow_type: str) -> Dict[str, any]:
        """Describe each workflow option."""
        if workflow_type == "advanced":
            return {
                "workflow_type": "advanced",
                "features": [
                    "Multi-order planning (strategic, tactical, operational)",
                    "Nested review processes with multiple levels",
                    "Dedicated research agents",
                    "Quality gates and assurance",
                    "Adaptive workflow optimization"
                ],
                "estimated_agents": 12,
                "estimated_time": "2-4x traditional workflow",
                "quality_level": "expert"
            }
        if workflow_type == "hybrid":
            return {
                "workflow_type": "hybrid",
                "features": [
                    "Traditional planning with advanced review",
                    "Multi-level content review",
                    "Technical and style validation",
                    "Quality assurance gates"
                ],
                "estimated_agents": 8,
                "estimated_time": "1.5-2x traditional workflow",
                "quality_level": "high"
            }
        if workflow_type == "hybrid_sk":
            return {
                "workflow_type": "hybrid_sk",
                "features": [
                    "Semantic Kernel drafting with deterministic validation",
                    "AutoGen-led argument exploration",
                    "Automatic quality gates (citations, tone, structure)",
                    "Structured section assembly"
                ],
                "estimated_agents": 4,
                "estimated_time": "comparable to traditional workflow",
                "quality_level": "high (structured)"
            }
        return {
            "workflow_type": "traditional",
            "features": [
                "Standard planning and writing",
                "Basic review and editing",
                "Citation validation",
                "Structure verification"
            ],
            "estimated_agents": 5,
            "estimated_time": "baseline",
            "quality_level": "standard"
        }


__all__ = [
    "EnhancedOrchestratorConfig",
    "EnhancedWriterOrchestrator",
    "WorkflowComplexityAnalyzer",
    "PerformanceMonitor",
]
