#!/usr/bin/env python3
"""
Full Workflow Orchestrator - Triggers full Conductor workflows when needed.

Converts questions to CaseInsights and runs full research → ML → writing pipeline
to generate comprehensive reports.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .QuestionClassifier import QuestionClassification
from .BNQueryMapper import BNQueryMapper
from .insights import CaseInsights, Posterior, EvidenceItem

logger = logging.getLogger(__name__)


class FullWorkflowOrchestrator:
    """Orchestrates full Conductor workflows for complex questions."""

    def __init__(
        self,
        conductor=None,  # Conductor instance
        bn_mapper: Optional[BNQueryMapper] = None
    ):
        """
        Initialize full workflow orchestrator.

        Args:
            conductor: Conductor instance (optional, can be passed per request)
            bn_mapper: BN query mapper instance
        """
        self.conductor = conductor
        self.bn_mapper = bn_mapper or BNQueryMapper()
        logger.info("FullWorkflowOrchestrator initialized")

    async def run_full_workflow(
        self,
        question: str,
        classification: QuestionClassification,
        context: Optional[str] = None,
        conductor=None
    ) -> Dict[str, Any]:
        """
        Run full Conductor workflow for a question.

        Args:
            question: User's question
            classification: Question classification
            context: Optional conversation context
            conductor: Conductor instance (overrides self.conductor if provided)

        Returns:
            Dictionary with workflow results, deliverables, and summary
        """
        conductor_instance = conductor or self.conductor
        if not conductor_instance:
            raise ValueError("Conductor instance required for full workflow")

        try:
            # Convert question to CaseInsights
            insights = self._question_to_insights(question, classification, context)

            # Run full workflow
            logger.info(f"Running full workflow for question: {question[:100]}")
            deliverable = await conductor_instance.run_hybrid_workflow(insights)

            # Generate summary
            summary = self._generate_workflow_summary(deliverable, classification)

            return {
                "success": True,
                "insights": insights,
                "deliverable": deliverable,
                "summary": summary,
                "question": question,
                "classification": classification
            }

        except Exception as e:
            logger.error(f"Full workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }

    def _question_to_insights(
        self,
        question: str,
        classification: QuestionClassification,
        context: Optional[str]
    ) -> CaseInsights:
        """Convert question to CaseInsights for Conductor."""
        # Map question to BN query if BN component is needed
        posteriors = []
        evidence = []

        if classification.required_components and any(
            comp.value == "bn" for comp in classification.required_components
        ):
            try:
                bn_query = self.bn_mapper.map_question_to_bn_query(question, context)

                # Create evidence items from BN query
                for node_id, state in bn_query.evidence.items():
                    evidence.append(EvidenceItem(
                        node_id=node_id,
                        state=state,
                        description=f"From question: {question[:50]}"
                    ))

                # If we can run BN inference now, get posteriors
                # Otherwise, Conductor will run it
                try:
                    bn_model_path = self._find_bn_model_path()
                    if bn_model_path:
                        from .BnAdapter import run_bn_inference_with_fallback
                        insights, posterior_data = run_bn_inference_with_fallback(
                            model_path=bn_model_path,
                            evidence=bn_query.evidence,
                            summary=question,
                            reference_id="workflow_question",
                            apply_always_on=False
                        )
                        posteriors = insights.posteriors
                except Exception as e:
                    logger.debug(f"Could not pre-run BN inference: {e}")
                    # Conductor will run it later

            except Exception as e:
                logger.warning(f"BN query mapping failed: {e}")

        # Extract jurisdiction and case style from question if possible
        jurisdiction = self._extract_jurisdiction(question)
        case_style = self._extract_case_style(question)

        return CaseInsights(
            reference_id=f"question_{hash(question)}",
            summary=question,
            posteriors=posteriors,
            evidence=evidence,
            jurisdiction=jurisdiction,
            case_style=case_style
        )

    def _extract_jurisdiction(self, question: str) -> Optional[str]:
        """Extract jurisdiction from question."""
        question_lower = question.lower()

        if "hong kong" in question_lower or "hk" in question_lower:
            return "Hong Kong"
        elif "federal" in question_lower or "us" in question_lower or "united states" in question_lower:
            return "US Federal"
        elif "district" in question_lower:
            return "US District"

        return None

    def _extract_case_style(self, question: str) -> Optional[str]:
        """Extract case style from question."""
        question_lower = question.lower()

        if "section 1782" in question_lower or "1782" in question_lower:
            return "Section 1782 Discovery"
        elif "defamation" in question_lower:
            return "Defamation"
        elif "national security" in question_lower:
            return "National Security"

        return None

    def _generate_workflow_summary(
        self,
        deliverable,
        classification: QuestionClassification
    ) -> str:
        """Generate human-readable summary of workflow results."""
        summary_parts = []

        # Add draft sections if available
        if hasattr(deliverable, 'draft_sections') and deliverable.draft_sections:
            summary_parts.append(f"Generated {len(deliverable.draft_sections)} draft sections")

        # Add research results if available
        if hasattr(deliverable, 'research_results') and deliverable.research_results:
            cases = deliverable.research_results.get("cases", [])
            if cases:
                summary_parts.append(f"Found {len(cases)} relevant cases")

        # Add validation results if available
        if hasattr(deliverable, 'validation_results') and deliverable.validation_results:
            score = deliverable.validation_results.get("overall_score", 0)
            if score > 0:
                summary_parts.append(f"Validation score: {score:.0%}")

        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return "Full workflow completed successfully"

    def _find_bn_model_path(self) -> Optional[Path]:
        """Find BN model file path."""
        possible_paths = [
            Path(__file__).parent.parent.parent / "experiments" / "WizardWeb1.1.3.xdsl",
            Path(__file__).parent.parent.parent.parent / "experiments" / "WizardWeb1.1.3.xdsl",
            Path(r"C:\Users\Owner\Desktop\WizardWeb\models\WizardWeb1.1.3.xdsl"),
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return None

