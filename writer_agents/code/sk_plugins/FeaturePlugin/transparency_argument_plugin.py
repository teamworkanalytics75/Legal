#!/usr/bin/env python3
"""
Transparency Argument Plugin - Atomic SK plugin for transparency feature.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class TransparencyArgumentPlugin(BaseFeaturePlugin):
    """Atomic plugin for transparency feature analysis."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, **kwargs):
        super().__init__(kernel, "mentions_transparency", chroma_store, rules_dir, **kwargs)
        logger.info("TransparencyArgumentPlugin initialized")

    async def analyze_transparency_arguments(self, draft_text: str) -> FunctionResult:
        """Analyze transparency and First Amendment arguments."""
        try:
            text_lower = draft_text.lower()

            # Count transparency-related terms
            transparency_terms = [
                "transparency", "transparent", "open court", "public access",
                "court transparency", "judicial transparency", "public right"
            ]

            first_amendment_terms = [
                "first amendment", "first amendment right", "free speech",
                "freedom of speech", "constitutional right", "public forum"
            ]

            transparency_count = sum(text_lower.count(term) for term in transparency_terms)
            first_amendment_count = sum(text_lower.count(term) for term in first_amendment_terms)

            # Check for counter-arguments
            counter_indicators = [
                "limited", "minimal", "outweighs", "outweigh", "greater",
                "more important", "private matter", "personal dispute"
            ]

            counter_count = sum(text_lower.count(indicator) for indicator in counter_indicators)

            # Calculate argument strength
            total_mentions = transparency_count + first_amendment_count
            min_threshold = self.rules.get("minimum_threshold", 1)

            # For privacy cases, should acknowledge but counter transparency concerns
            argument_score = min(1.0, counter_count / max(total_mentions, 1))
            meets_threshold = total_mentions >= min_threshold

            return FunctionResult(
                success=True,
                value={
                    "transparency_mentions": transparency_count,
                    "first_amendment_mentions": first_amendment_count,
                    "total_mentions": total_mentions,
                    "counter_arguments": counter_count,
                    "argument_score": argument_score,
                    "meets_threshold": meets_threshold,
                    "recommendation": self._get_transparency_recommendation(total_mentions, counter_count)
                },
                metadata={"analysis_type": "transparency_arguments"}
            )

        except Exception as e:
            logger.error(f"Transparency analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _get_transparency_recommendation(self, total_mentions: int, counter_count: int) -> str:
        """Get recommendation based on transparency argument strength."""
        if total_mentions == 0:
            return "Acknowledge transparency concerns but emphasize limited public interest."
        elif counter_count < total_mentions:
            return "Strengthen counter-arguments to transparency concerns. Emphasize privacy interests."
        elif counter_count < total_mentions * 1.5:
            return "Transparency arguments adequate but could be stronger. Add more privacy-focused counter-arguments."
        else:
            return "Transparency argument handling is good. Focus on specific privacy interests."
