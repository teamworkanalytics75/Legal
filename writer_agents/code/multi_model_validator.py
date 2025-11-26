"""Multi-model validation combining Legal-BERT scoring and Qwen2.5 logical review."""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from semantic_kernel import Kernel
from sk_config import SKConfig, create_sk_kernel
from .sk_compat import get_chat_components
from legal_bert_validator import LegalBERTValidator, QualityScore

logger = logging.getLogger(__name__)


@dataclass
class MultiModelValidationResult:
    """Combined validation result from multiple models."""
    overall_score: float  # 0-1 combined score
    legal_bert_score: float  # Legal-BERT quality score
    qwen_logical_score: float  # Qwen2.5 logical review score
    catboost_score: Optional[float] = None  # CatBoost feature validation score
    legal_bert_details: Optional[QualityScore] = None
    qwen_review_feedback: Optional[str] = None
    qwen_suggestions: Optional[list] = None
    meets_threshold: bool = False
    weights: Dict[str, float] = None


class MultiModelValidator:
    """Coordinates validation from Legal-BERT and Qwen2.5."""

    def __init__(
        self,
        legal_bert_weight: float = 0.40,
        qwen_review_weight: float = 0.40,
        catboost_weight: float = 0.20,
        qwen_model: str = "qwen2.5:14b",
        ollama_base_url: str = "http://localhost:11434",
        quality_threshold: float = 0.70
    ):
        """
        Initialize multi-model validator.

        Args:
            legal_bert_weight: Weight for Legal-BERT scoring (0-1)
            qwen_review_weight: Weight for Qwen2.5 logical review (0-1)
            catboost_weight: Weight for CatBoost validation (0-1)
            qwen_model: Qwen model name for logical review
            ollama_base_url: Ollama server URL
            quality_threshold: Minimum overall score to pass validation
        """
        # Validate weights sum to 1.0
        total_weight = legal_bert_weight + qwen_review_weight + catboost_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Validation weights sum to {total_weight}, normalizing to 1.0")
            legal_bert_weight /= total_weight
            qwen_review_weight /= total_weight
            catboost_weight /= total_weight

        self.legal_bert_weight = legal_bert_weight
        self.qwen_review_weight = qwen_review_weight
        self.catboost_weight = catboost_weight
        self.quality_threshold = quality_threshold

        # Initialize Legal-BERT validator
        self.legal_bert_validator = LegalBERTValidator()

        # Initialize Qwen2.5 kernel for logical review
        self.qwen_kernel: Optional[Kernel] = None
        self.qwen_model = qwen_model
        self.ollama_base_url = ollama_base_url
        self._initialize_qwen_kernel()

    def _initialize_qwen_kernel(self) -> None:
        """Initialize Qwen2.5 kernel for logical review."""
        try:
            qwen_config = SKConfig(
                use_local=True,
                local_model=self.qwen_model,
                local_base_url=self.ollama_base_url,
                temperature=0.2,  # Lower temperature for logical review
                max_tokens=2000
            )
            self.qwen_kernel = create_sk_kernel(qwen_config)
            logger.info(f"Initialized Qwen2.5 kernel for logical review: {self.qwen_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize Qwen2.5 kernel: {e}")
            self.qwen_kernel = None

    async def validate_document(
        self,
        document: str,
        context: Dict[str, Any],
        catboost_score: Optional[float] = None
    ) -> MultiModelValidationResult:
        """
        Validate document using multi-model approach.

        Args:
            document: Document text to validate
            context: Additional context for validation
            catboost_score: Optional CatBoost validation score

        Returns:
            MultiModelValidationResult with combined scores
        """
        logger.info("Starting multi-model validation...")

        # 1. Legal-BERT quality scoring
        legal_bert_score_obj = self.legal_bert_validator.score_document(document)
        legal_bert_score = legal_bert_score_obj.overall_score
        logger.info(f"Legal-BERT quality score: {legal_bert_score:.3f}")

        # 2. Qwen2.5 logical review
        qwen_review_result = await self._qwen_logical_review(document, context)
        qwen_logical_score = qwen_review_result.get("score", 0.0)
        qwen_feedback = qwen_review_result.get("feedback", "")
        qwen_suggestions = qwen_review_result.get("suggestions", [])
        logger.info(f"Qwen2.5 logical review score: {qwen_logical_score:.3f}")

        # 3. Calculate combined score
        # Only use CatBoost weight if score provided
        weights_sum = self.legal_bert_weight + self.qwen_review_weight
        if catboost_score is not None:
            weights_sum += self.catboost_weight
            effective_legal_weight = self.legal_bert_weight / weights_sum
            effective_qwen_weight = self.qwen_review_weight / weights_sum
            effective_catboost_weight = self.catboost_weight / weights_sum
        else:
            # Normalize weights without CatBoost
            effective_legal_weight = self.legal_bert_weight / weights_sum
            effective_qwen_weight = self.qwen_review_weight / weights_sum
            effective_catboost_weight = 0.0

        overall_score = (
            legal_bert_score * effective_legal_weight +
            qwen_logical_score * effective_qwen_weight +
            (catboost_score or 0.0) * effective_catboost_weight
        )

        meets_threshold = overall_score >= self.quality_threshold

        logger.info(f"Combined validation score: {overall_score:.3f} (threshold: {self.quality_threshold})")
        logger.info(f"Validation {'PASSED' if meets_threshold else 'FAILED'}")

        return MultiModelValidationResult(
            overall_score=float(overall_score),
            legal_bert_score=float(legal_bert_score),
            qwen_logical_score=float(qwen_logical_score),
            catboost_score=catboost_score,
            legal_bert_details=legal_bert_score_obj,
            qwen_review_feedback=qwen_feedback,
            qwen_suggestions=qwen_suggestions,
            meets_threshold=meets_threshold,
            weights={
                "legal_bert": effective_legal_weight,
                "qwen": effective_qwen_weight,
                "catboost": effective_catboost_weight
            }
        )

    async def _qwen_logical_review(
        self,
        document: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Qwen2.5 to review document for logical consistency and reasoning quality.

        Args:
            document: Document text to review
            context: Additional context

        Returns:
            Dictionary with score, feedback, and suggestions
        """
        if self.qwen_kernel is None:
            logger.warning("Qwen2.5 kernel not available, using rule-based logical review")
            return self._rule_based_logical_review(document)

        try:
            # Build review prompt
            review_prompt = self._build_review_prompt(document, context)

            # Get chat service from kernel
            chat_service = None
            try:
                # Try different methods to get chat service (handle API version differences)
                # Method 1: Try get_service without type_id (newer API)
                if hasattr(self.qwen_kernel, 'get_service'):
                    try:
                        # Try with service_id parameter
                        if hasattr(self.qwen_kernel, 'services') and self.qwen_kernel.services:
                            service_id = list(self.qwen_kernel.services.keys())[0]
                            chat_service = self.qwen_kernel.get_service(service_id=service_id)
                    except (TypeError, AttributeError, IndexError):
                        pass
                    
                    # Method 2: Try get_service with type_id (older API) - only if Method 1 failed
                    if not chat_service:
                        try:
                            chat_service = self.qwen_kernel.get_service(type_id="chat_completion")
                        except (TypeError, AttributeError, KeyError, ValueError):
                            pass
                
                # Method 3: Get first service directly
                if not chat_service:
                    services = getattr(self.qwen_kernel, 'services', {})
                    if services:
                        if isinstance(services, dict):
                            chat_service = list(services.values())[0]
                        elif hasattr(services, '__iter__'):
                            chat_service = next(iter(services)) if services else None
            except Exception as e:
                logger.debug(f"Error getting chat service: {e}")

            if not chat_service:
                return self._rule_based_logical_review(document)

            # Generate review using Qwen2.5
            ChatHistory, _, _ = get_chat_components()

            chat_history = ChatHistory()
            chat_history.add_user_message(review_prompt)

            response = await chat_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=None
            )

            # Extract review text
            if isinstance(response, list) and len(response) > 0:
                review_text = str(response[0].content) if hasattr(response[0], 'content') else str(response[0])
            elif hasattr(response, 'content'):
                review_text = str(response.content)
            else:
                review_text = str(response)

            # Parse review to extract score and feedback
            return self._parse_qwen_review(review_text, document)

        except Exception as e:
            logger.error(f"Qwen2.5 logical review failed: {e}")
            return self._rule_based_logical_review(document)

    def _build_review_prompt(self, document: str, context: Dict[str, Any]) -> str:
        """Build prompt for logical review."""
        case_summary = context.get('case_summary', 'N/A')
        jurisdiction = context.get('jurisdiction', 'US Federal Court')

        prompt = f"""You are a legal expert reviewing a Motion to Seal for logical consistency and reasoning quality.

CASE INFORMATION:
Jurisdiction: {jurisdiction}
Case Summary: {case_summary[:500]}

DOCUMENT TO REVIEW:
{document[:2000]}

TASK: Review this legal document and provide:
1. A quality score from 0.0 to 1.0 based on:
   - Logical consistency of arguments
   - Quality of legal reasoning
   - Structure and flow of arguments
   - Completeness of legal analysis

2. Specific feedback on strengths and weaknesses
3. Concrete suggestions for improvement

Format your response as:
SCORE: [0.0-1.0]
FEEDBACK: [your feedback]
SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]
- [etc]

Review the document:"""

        return prompt

    def _parse_qwen_review(self, review_text: str, document: str) -> Dict[str, Any]:
        """Parse Qwen2.5 review response to extract structured information."""
        import re

        # Extract score
        score_match = re.search(r'SCORE:\s*([\d.]+)', review_text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.5

        # Extract feedback
        feedback_match = re.search(r'FEEDBACK:\s*(.*?)(?=SUGGESTIONS:|$)', review_text, re.IGNORECASE | re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else review_text[:500]

        # Extract suggestions
        suggestions_match = re.search(r'SUGGESTIONS:\s*(.*?)$', review_text, re.IGNORECASE | re.DOTALL)
        suggestions_text = suggestions_match.group(1) if suggestions_match else ""

        # Parse bullet points
        suggestions = []
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                suggestion = re.sub(r'^[-•*]\s*', '', line)
                if suggestion:
                    suggestions.append(suggestion)

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        return {
            "score": float(score),
            "feedback": feedback,
            "suggestions": suggestions
        }

    def _rule_based_logical_review(self, document: str) -> Dict[str, Any]:
        """Fallback rule-based logical review when Qwen2.5 is unavailable."""
        score = 0.5
        feedback_parts = []
        suggestions = []

        # Check for logical connectors
        logical_connectors = ['therefore', 'thus', 'consequently', 'because', 'since', 'as a result']
        has_connectors = any(connector in document.lower() for connector in logical_connectors)
        if has_connectors:
            score += 0.2
            feedback_parts.append("Document uses logical connectors effectively.")
        else:
            score -= 0.1
            suggestions.append("Add logical connectors to strengthen argument flow.")

        # Check for argument structure (premise-conclusion)
        has_because = 'because' in document.lower() or 'since' in document.lower()
        has_therefore = 'therefore' in document.lower() or 'thus' in document.lower()
        if has_because and has_therefore:
            score += 0.2
            feedback_parts.append("Document demonstrates clear premise-conclusion structure.")
        else:
            suggestions.append("Strengthen argument structure with clear premises and conclusions.")

        # Check for multiple supporting points
        numbering_patterns = [r'\d+\.', r'[a-z]\)', r'\([a-z]\)']
        has_numbering = any(re.search(pattern, document) for pattern in numbering_patterns)
        if has_numbering:
            score += 0.1
            feedback_parts.append("Document uses structured argumentation with numbered points.")

        # Normalize score
        score = max(0.0, min(1.0, score))

        feedback = " ".join(feedback_parts) if feedback_parts else "Basic logical structure present."

        return {
            "score": float(score),
            "feedback": feedback,
            "suggestions": suggestions
        }


__all__ = ["MultiModelValidator", "MultiModelValidationResult"]
