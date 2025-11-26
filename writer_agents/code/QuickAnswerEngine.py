#!/usr/bin/env python3
"""
Quick Answer Engine - Fast-path answers without full workflow.

Provides conversational answers in 5-10 seconds by:
- Direct database queries
- Simple BN queries
- CatBoost feature lookup
- Research result summaries
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .QuestionClassifier import QuestionClassification, RequiredComponent
from .BNQueryMapper import BNQueryMapper, BNQuery
from .ContextManager import ContextManager

logger = logging.getLogger(__name__)


@dataclass
class QuickAnswer:
    """Result from quick answer engine."""
    answer: str
    confidence: float  # 0.0-1.0
    sources: List[str]  # Sources used (e.g., "BN", "Research", "ML")
    metadata: Dict[str, Any] = None  # Additional data (probabilities, case counts, etc.)


class QuickAnswerEngine:
    """Fast-path answer engine for quick queries."""

    def __init__(
        self,
        bn_mapper: Optional[BNQueryMapper] = None,
        case_law_researcher=None,  # CaseLawResearcher instance
        bn_adapter=None,  # BnAdapter functions
        context_manager: Optional[ContextManager] = None
    ):
        """
        Initialize quick answer engine.

        Args:
            bn_mapper: BN query mapper instance
            case_law_researcher: CaseLawResearcher instance (optional)
            bn_adapter: BnAdapter functions (optional)
            context_manager: Context manager instance (optional)
        """
        self.bn_mapper = bn_mapper or BNQueryMapper()
        self.case_law_researcher = case_law_researcher
        self.bn_adapter = bn_adapter
        self.context_manager = context_manager
        logger.info("QuickAnswerEngine initialized")

    async def answer(
        self,
        question: str,
        classification: QuestionClassification,
        context: Optional[str] = None
    ) -> QuickAnswer:
        """
        Generate a quick answer for a question.

        Args:
            question: User's question
            classification: Question classification result
            context: Optional conversation context

        Returns:
            QuickAnswer with answer text, confidence, and sources
        """
        sources = []
        answer_parts = []
        metadata = {}
        confidence = 0.5

        # Handle probability questions (BN queries)
        if RequiredComponent.BN in classification.required_components:
            bn_answer = await self._answer_bn_query(question, classification, context)
            if bn_answer:
                answer_parts.append(bn_answer["text"])
                sources.append("BN")
                metadata.update(bn_answer.get("metadata", {}))
                confidence = max(confidence, bn_answer.get("confidence", 0.5))

        # Handle research questions
        if RequiredComponent.RESEARCH in classification.required_components:
            research_answer = await self._answer_research_query(question, classification, context)
            if research_answer:
                answer_parts.append(research_answer["text"])
                sources.append("Research")
                metadata.update(research_answer.get("metadata", {}))
                confidence = max(confidence, research_answer.get("confidence", 0.5))

        # Handle ML/analysis questions
        if RequiredComponent.ML in classification.required_components:
            ml_answer = await self._answer_ml_query(question, classification, context)
            if ml_answer:
                answer_parts.append(ml_answer["text"])
                sources.append("ML")
                metadata.update(ml_answer.get("metadata", {}))
                confidence = max(confidence, ml_answer.get("confidence", 0.5))

        # Combine answers
        if not answer_parts:
            answer_text = "I couldn't find a quick answer for your question. Would you like me to run a full analysis?"
            confidence = 0.0
        else:
            answer_text = "\n\n".join(answer_parts)
            # Add suggestion for full report if complexity is high
            if classification.complexity.value in ["moderate", "complex"]:
                answer_text += "\n\nWould you like me to generate a full comprehensive report on this?"

        return QuickAnswer(
            answer=answer_text,
            confidence=confidence,
            sources=sources,
            metadata=metadata
        )

    async def _answer_bn_query(
        self,
        question: str,
        classification: QuestionClassification,
        context: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Answer a BN probability question."""
        if not self.bn_adapter:
            return None

        try:
            # Map question to BN query
            bn_query = self.bn_mapper.map_question_to_bn_query(question, context)

            if not bn_query.target_nodes or not bn_query.evidence:
                logger.debug("BN query mapping failed - no target nodes or evidence")
                return None

            # Find BN model path
            bn_model_path = self._find_bn_model_path()
            if not bn_model_path:
                return None

            # Run BN inference
            from .BnAdapter import run_bn_inference_with_fallback
            insights, posterior_data = run_bn_inference_with_fallback(
                model_path=bn_model_path,
                evidence=bn_query.evidence,
                summary=bn_query.question_summary,
                reference_id="quick_answer",
                apply_always_on=False
            )

            # Extract probabilities for target nodes
            probability_texts = []
            for target_node in bn_query.target_nodes:
                if target_node in posterior_data:
                    probs = posterior_data[target_node]
                    # Get top probability
                    top_state = max(probs.items(), key=lambda x: x[1])
                    prob_pct = top_state[1] * 100
                    probability_texts.append(f"{target_node}: {prob_pct:.1f}% probability of {top_state[0]}")

            if probability_texts:
                answer_text = f"Based on Bayesian Network analysis:\n\n" + "\n".join(f"  - {pt}" for pt in probability_texts)
                return {
                    "text": answer_text,
                    "confidence": bn_query.confidence,
                    "metadata": {
                        "bn_query": bn_query.question_summary,
                        "target_nodes": bn_query.target_nodes,
                        "posterior_data": posterior_data
                    }
                }

        except Exception as e:
            logger.warning(f"BN query failed: {e}")
            return None

        return None

    async def _answer_research_query(
        self,
        question: str,
        classification: QuestionClassification,
        context: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Answer a research/case law question."""
        if not self.case_law_researcher:
            return None

        try:
            # Create a simple case insights object for research
            from .insights import CaseInsights, Posterior
            insights = CaseInsights(
                reference_id="quick_research",
                summary=question,
                posteriors=[],
                jurisdiction=None,
                case_style=None
            )

            # Run research
            research_results = self.case_law_researcher.research_case_law(
                insights=insights,
                top_k=5,  # Limit to 5 for quick answer
                min_similarity=0.3
            )

            if not research_results or not research_results.get("cases"):
                return None

            cases = research_results["cases"][:3]  # Top 3 cases
            case_texts = []
            for case in cases:
                case_name = case.get("case_name", "Unknown case")
                relevance = case.get("similarity", 0)
                case_texts.append(f"  - {case_name} ({relevance:.0%} relevance)")

            if case_texts:
                answer_text = f"Found {len(research_results.get('cases', []))} relevant cases. Top matches:\n\n" + "\n".join(case_texts)
                return {
                    "text": answer_text,
                    "confidence": 0.7,
                    "metadata": {
                        "total_cases": len(research_results.get("cases", [])),
                        "top_cases": cases
                    }
                }

        except Exception as e:
            logger.warning(f"Research query failed: {e}")
            return None

        return None

    async def _answer_ml_query(
        self,
        question: str,
        classification: QuestionClassification,
        context: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Answer an ML/analysis question (limited - usually requires full workflow)."""
        # For quick answers, ML queries are limited
        # Most ML analysis requires full workflow
        return None

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

