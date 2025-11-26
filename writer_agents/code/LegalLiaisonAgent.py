#!/usr/bin/env python3
"""
Legal Liaison Agent - Main conversational interface coordinating all components.

Acts as the liaison between the user and the entire system, similar to ChatGPT
but connected to Conductor, Research, ML, and BN systems.

Features:
- Manages conversation history (limited context window to avoid hallucinations)
- Routes questions to appropriate components
- Decides quick answer vs full workflow
- Formats responses (conversational + structured)
- Integrates with Conductor, Research, ML, BN systems
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from .QuestionClassifier import QuestionClassifier, QuestionClassification, ComplexityLevel, RequiredComponent
from .BNQueryMapper import BNQueryMapper
from .ContextManager import ContextManager, MessageRole
from .QuickAnswerEngine import QuickAnswerEngine
from .FullWorkflowOrchestrator import FullWorkflowOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class LiaisonResponse:
    """Response from LegalLiaisonAgent."""
    answer: str
    confidence: float
    sources: List[str]
    follow_up_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    requires_full_workflow: bool = False
    workflow_result: Optional[Dict[str, Any]] = None


class LegalLiaisonAgent:
    """Main conversational interface for legal analysis system."""

    def __init__(
        self,
        conductor=None,  # Conductor instance
        case_law_researcher=None,  # CaseLawResearcher instance
        bn_adapter=None,  # BnAdapter functions
        memory_store=None,  # EpisodicMemoryBank
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Legal Liaison Agent.

        Args:
            conductor: Conductor instance (optional, can be passed per request)
            case_law_researcher: CaseLawResearcher instance (optional)
            bn_adapter: BnAdapter functions (optional)
            memory_store: EpisodicMemoryBank for conversation history (optional)
            config: Optional configuration dict
        """
        self.conductor = conductor
        self.case_law_researcher = case_law_researcher
        self.bn_adapter = bn_adapter
        self.memory_store = memory_store
        self.config = config or {}

        # Initialize components
        self.question_classifier = QuestionClassifier()
        self.bn_mapper = BNQueryMapper()
        self.context_manager = ContextManager(
            max_messages=self.config.get("max_context_messages", 10),
            max_facts=self.config.get("max_facts", 50)
        )
        self.quick_answer_engine = QuickAnswerEngine(
            bn_mapper=self.bn_mapper,
            case_law_researcher=self.case_law_researcher,
            bn_adapter=self.bn_adapter,
            context_manager=self.context_manager
        )
        self.full_workflow_orchestrator = FullWorkflowOrchestrator(
            conductor=self.conductor,
            bn_mapper=self.bn_mapper
        )

        logger.info("LegalLiaisonAgent initialized")

    async def ask(
        self,
        question: str,
        context: Optional[str] = None,
        force_full_workflow: bool = False,
        conductor=None
    ) -> LiaisonResponse:
        """
        Process a user question and return a response.

        Args:
            question: User's question
            context: Optional conversation context (auto-generated if not provided)
            force_full_workflow: Force full workflow instead of quick answer
            conductor: Conductor instance (overrides self.conductor if provided)

        Returns:
            LiaisonResponse with answer, confidence, sources, etc.
        """
        # Get context from context manager if not provided
        if not context:
            context = self.context_manager.get_context()

        # Add user message to context
        self.context_manager.add_message(MessageRole.USER, question)

        try:
            # Classify question
            classification = self.question_classifier.classify(question, context)

            logger.info(f"Question classified: {classification.question_type.value}, "
                       f"complexity: {classification.complexity.value}, "
                       f"components: {[c.value for c in classification.required_components]}")

            # Decide: quick answer vs full workflow
            should_use_full_workflow = (
                force_full_workflow or
                classification.complexity == ComplexityLevel.COMPLEX or
                RequiredComponent.WRITING in classification.required_components or
                self.config.get("always_full_workflow", False)
            )

            if should_use_full_workflow:
                # Run full workflow
                logger.info("Using full workflow for question")
                workflow_result = await self.full_workflow_orchestrator.run_full_workflow(
                    question=question,
                    classification=classification,
                    context=context,
                    conductor=conductor or self.conductor
                )

                if workflow_result.get("success"):
                    # Generate conversational summary from workflow results
                    answer = self._format_workflow_response(workflow_result, classification)
                    sources = self._extract_sources_from_workflow(workflow_result)
                    follow_ups = self._generate_follow_ups(classification, workflow_result)

                    response = LiaisonResponse(
                        answer=answer,
                        confidence=0.9,  # High confidence for full workflow
                        sources=sources,
                        follow_up_suggestions=follow_ups,
                        metadata={
                            "workflow_result": workflow_result,
                            "classification": classification.reasoning
                        },
                        requires_full_workflow=True,
                        workflow_result=workflow_result
                    )
                else:
                    # Workflow failed, fall back to quick answer
                    logger.warning("Full workflow failed, falling back to quick answer")
                    quick_answer = await self.quick_answer_engine.answer(
                        question, classification, context
                    )
                    response = LiaisonResponse(
                        answer=quick_answer.answer,
                        confidence=quick_answer.confidence,
                        sources=quick_answer.sources,
                        metadata=quick_answer.metadata,
                        requires_full_workflow=False
                    )
            else:
                # Use quick answer
                logger.info("Using quick answer for question")
                quick_answer = await self.quick_answer_engine.answer(
                    question, classification, context
                )

                follow_ups = []
                if classification.complexity == ComplexityLevel.MODERATE:
                    follow_ups.append("Would you like a full comprehensive analysis?")
                if RequiredComponent.WRITING in classification.required_components:
                    follow_ups.append("I can generate a full draft document for you.")

                response = LiaisonResponse(
                    answer=quick_answer.answer,
                    confidence=quick_answer.confidence,
                    sources=quick_answer.sources,
                    follow_up_suggestions=follow_ups,
                    metadata=quick_answer.metadata,
                    requires_full_workflow=False
                )

            # Add assistant response to context
            self.context_manager.add_message(
                MessageRole.ASSISTANT,
                response.answer,
                extract_facts=True,
                metadata={"sources": response.sources, "confidence": response.confidence}
            )

            return response

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            error_response = f"I encountered an error processing your question: {str(e)}. Please try rephrasing or asking for help."
            self.context_manager.add_message(MessageRole.ASSISTANT, error_response)
            return LiaisonResponse(
                answer=error_response,
                confidence=0.0,
                sources=[],
                metadata={"error": str(e)}
            )

    def _format_workflow_response(
        self,
        workflow_result: Dict[str, Any],
        classification: QuestionClassification
    ) -> str:
        """Format full workflow results into conversational response."""
        answer_parts = []

        # Add summary
        summary = workflow_result.get("summary", "Full workflow completed successfully")
        answer_parts.append(summary)

        # Add draft sections if available
        deliverable = workflow_result.get("deliverable")
        if deliverable and hasattr(deliverable, 'draft_sections'):
            sections = deliverable.draft_sections
            if sections:
                answer_parts.append(f"\nGenerated {len(sections)} draft sections:")
                for section in sections[:3]:  # Show first 3
                    section_name = getattr(section, 'section_name', 'Unknown')
                    answer_parts.append(f"  - {section_name}")

        # Add research results if available
        if deliverable and hasattr(deliverable, 'research_results'):
            research = deliverable.research_results
            if research and research.get("cases"):
                case_count = len(research["cases"])
                answer_parts.append(f"\nFound {case_count} relevant cases supporting your question.")

        # Add validation results if available
        if deliverable and hasattr(deliverable, 'validation_results'):
            validation = deliverable.validation_results
            if validation:
                score = validation.get("overall_score", 0)
                if score > 0:
                    answer_parts.append(f"\nValidation score: {score:.0%}")

        return "\n".join(answer_parts)

    def _extract_sources_from_workflow(self, workflow_result: Dict[str, Any]) -> List[str]:
        """Extract source components from workflow result."""
        sources = []

        classification = workflow_result.get("classification")
        if classification and hasattr(classification, 'required_components'):
            sources = [comp.value for comp in classification.required_components]

        return sources if sources else ["Full Workflow"]

    def _generate_follow_ups(
        self,
        classification: QuestionClassification,
        workflow_result: Dict[str, Any]
    ) -> List[str]:
        """Generate follow-up question suggestions."""
        follow_ups = []

        # Add context-specific follow-ups
        if RequiredComponent.BN in classification.required_components:
            follow_ups.append("Would you like to see detailed probability breakdowns?")

        if RequiredComponent.RESEARCH in classification.required_components:
            follow_ups.append("Would you like me to find more similar cases?")

        if RequiredComponent.WRITING in classification.required_components:
            follow_ups.append("Would you like me to refine or expand any sections?")

        return follow_ups

    def get_conversation_history(self, n: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        messages = self.context_manager.get_recent_messages(n)
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in messages
        ]

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.context_manager.clear()
        logger.info("Conversation history cleared")

    def get_verified_facts(self) -> List[Dict[str, Any]]:
        """Get verified facts from conversation."""
        facts = self.context_manager.verified_facts
        return [
            {
                "fact": fact.fact,
                "source": fact.source,
                "confidence": fact.confidence,
                "timestamp": fact.timestamp.isoformat()
            }
            for fact in facts
        ]

