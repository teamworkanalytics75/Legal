#!/usr/bin/env python3
"""
Privacy Plugin - Atomic SK plugin for privacy mentions feature.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class PrivacyPlugin(BaseFeaturePlugin):
    """Atomic plugin for privacy mentions feature analysis."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "mentions_privacy", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("PrivacyPlugin initialized")

    async def analyze_privacy_strength(self, draft_text: str) -> FunctionResult:
        """Analyze privacy argument strength in draft."""
        try:
            text_lower = draft_text.lower()

            # Count privacy-related terms
            privacy_terms = [
                "privacy", "personal information", "confidential",
                "private", "expectation of privacy", "privacy interest"
            ]

            term_counts = {}
            total_mentions = 0

            for term in privacy_terms:
                count = text_lower.count(term)
                term_counts[term] = count
                total_mentions += count

            # Check for privacy harm language
            harm_indicators = [
                "privacy harm", "invasion of privacy", "breach of privacy",
                "privacy violation", "unauthorized disclosure"
            ]

            harm_count = sum(text_lower.count(indicator) for indicator in harm_indicators)

            # Calculate strength score
            min_threshold = self.rules.get("minimum_threshold", 3)
            target_average = self.rules.get("successful_case_average", 5.0)

            strength_score = min(1.0, total_mentions / target_average)
            meets_threshold = total_mentions >= min_threshold

            return FunctionResult(
                success=True,
                value={
                    "total_privacy_mentions": total_mentions,
                    "term_breakdown": term_counts,
                    "harm_indicators": harm_count,
                    "strength_score": strength_score,
                    "meets_threshold": meets_threshold,
                    "recommendation": self._get_privacy_recommendation(total_mentions, min_threshold)
                },
                metadata={"analysis_type": "privacy_strength"}
            )

        except Exception as e:
            logger.error(f"Privacy analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _get_privacy_recommendation(self, current_count: int, min_threshold: int) -> str:
        """Get recommendation based on current privacy mention count."""
        if current_count < min_threshold:
            return f"Increase privacy mentions from {current_count} to at least {min_threshold}. Add specific privacy harm examples."
        elif current_count < min_threshold * 1.5:
            return f"Privacy mentions adequate but could be strengthened. Consider adding more specific harm descriptions."
        else:
            return "Privacy argument strength is good. Focus on quality and specificity of harm descriptions."

    async def suggest_privacy_improvements(self, draft_text: str) -> FunctionResult:
        """
        Suggest specific improvements for privacy arguments.

        Phase 7: Now uses database search to find supporting precedents.
        """
        try:
            text_lower = draft_text.lower()
            suggestions = []

            # Phase 7: Find supporting privacy precedents from database
            supporting_cases = []
            if self.sqlite_searcher:
                try:
                    # Search for privacy-related winning cases
                    winning_cases = await self.find_winning_cases(
                        keywords=["privacy", "confidential", "personal information", "privacy harm"],
                        min_keywords=1,
                        limit=5
                    )

                    # Also search by semantic similarity
                    similar_cases = await self.search_case_law(
                        query="privacy harm expectation of privacy personal information disclosure",
                        top_k=5,
                        min_similarity=0.5
                    )

                    # Combine and deduplicate
                    all_cases = {}
                    for case in winning_cases + similar_cases:
                        case_name = case.get('case_name', '')
                        if case_name and case_name not in all_cases:
                            all_cases[case_name] = case

                    supporting_cases = list(all_cases.values())[:5]  # Top 5

                    if supporting_cases:
                        logger.info(f"Found {len(supporting_cases)} supporting privacy cases from database")
                except Exception as e:
                    logger.debug(f"Database search for privacy precedents failed: {e}")

            # Check for missing privacy concepts
            required_concepts = [
                "expectation of privacy",
                "personal information",
                "privacy harm",
                "confidential information"
            ]

            missing_concepts = []
            for concept in required_concepts:
                if concept not in text_lower:
                    missing_concepts.append(concept)

            if missing_concepts:
                suggestion_msg = f"Add these privacy concepts: {', '.join(missing_concepts)}"
                if supporting_cases:
                    suggestion_msg += f"\nSupporting precedents: {', '.join([c.get('case_name', '') for c in supporting_cases[:3]])}"

                suggestions.append({
                    "type": "missing_concepts",
                    "priority": "high",
                    "message": suggestion_msg,
                    "examples": self._get_concept_examples(missing_concepts),
                    "supporting_cases": [c.get('case_name', '') for c in supporting_cases[:3]]
                })

            # Check for weak privacy language
            weak_phrases = ["privacy", "private"]
            strong_phrases = ["privacy harm", "invasion of privacy", "breach of privacy"]

            weak_count = sum(text_lower.count(phrase) for phrase in weak_phrases)
            strong_count = sum(text_lower.count(phrase) for phrase in strong_phrases)

            if weak_count > strong_count * 2:
                suggestions.append({
                    "type": "strengthen_language",
                    "priority": "medium",
                    "message": "Replace generic privacy terms with specific harm language",
                    "examples": [
                        "Instead of 'privacy', use 'privacy harm'",
                        "Instead of 'private', use 'confidential information'",
                        "Add 'invasion of privacy' or 'breach of privacy'"
                    ]
                })

            # Check for specific harm examples
            if "example" not in text_lower and "instance" not in text_lower:
                suggestions.append({
                    "type": "add_examples",
                    "priority": "medium",
                    "message": "Add specific examples of privacy harm",
                    "examples": [
                        "Describe specific personal information that would be disclosed",
                        "Explain how disclosure would cause concrete harm",
                        "Provide examples of similar cases where privacy was protected"
                    ]
                })

            return FunctionResult(
                success=True,
                value={
                    "suggestions": suggestions,
                    "total_suggestions": len(suggestions),
                    "high_priority": len([s for s in suggestions if s["priority"] == "high"])
                },
                metadata={"analysis_type": "privacy_improvements"}
            )

        except Exception as e:
            logger.error(f"Privacy improvement suggestions failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _get_concept_examples(self, concepts: List[str]) -> Dict[str, str]:
        """Get example text for missing privacy concepts."""
        examples = {
            "expectation of privacy": "Plaintiff had a reasonable expectation of privacy in their personal information.",
            "personal information": "The documents contain sensitive personal information including addresses and contact details.",
            "privacy harm": "Disclosure would cause significant privacy harm by exposing intimate personal details.",
            "confidential information": "The information is confidential and was provided with the expectation of privacy."
        }

        return {concept: examples.get(concept, f"Add discussion of {concept}") for concept in concepts}
