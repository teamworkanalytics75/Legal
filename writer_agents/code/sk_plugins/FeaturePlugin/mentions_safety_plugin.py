#!/usr/bin/env python3
"""
Safety Plugin - Atomic SK plugin for safety mentions feature.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class SafetyPlugin(BaseFeaturePlugin):
    """Atomic plugin for safety mentions feature analysis."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "mentions_safety", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("SafetyPlugin initialized")

    async def analyze_safety_concerns(self, draft_text: str) -> FunctionResult:
        """Analyze safety concerns argument strength."""
        try:
            text_lower = draft_text.lower()

            # Count safety-related terms
            safety_terms = [
                "safety", "danger", "threat", "threaten", "harm", "harmful",
                "risk", "risky", "vulnerable", "vulnerability", "security"
            ]

            term_counts = {}
            total_mentions = 0

            for term in safety_terms:
                count = text_lower.count(term)
                term_counts[term] = count
                total_mentions += count

            # Check for specific harm language
            harm_indicators = [
                "physical harm", "safety concerns", "danger to", "threat of",
                "security risk", "vulnerable to", "at risk"
            ]

            harm_count = sum(text_lower.count(indicator) for indicator in harm_indicators)

            # Calculate strength score
            min_threshold = self.rules.get("minimum_threshold", 1)
            target_average = self.rules.get("successful_case_average", 1.5)

            strength_score = min(1.0, total_mentions / max(target_average, 1))
            meets_threshold = total_mentions >= min_threshold

            return FunctionResult(
                success=True,
                value={
                    "total_safety_mentions": total_mentions,
                    "term_breakdown": term_counts,
                    "harm_indicators": harm_count,
                    "strength_score": strength_score,
                    "meets_threshold": meets_threshold,
                    "recommendation": self._get_safety_recommendation(total_mentions, min_threshold)
                },
                metadata={"analysis_type": "safety_concerns"}
            )

        except Exception as e:
            logger.error(f"Safety analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _get_safety_recommendation(self, current_count: int, min_threshold: int) -> str:
        """Get recommendation based on current safety mention count."""
        if current_count < min_threshold:
            return f"Increase safety mentions from {current_count} to at least {min_threshold}. Add specific safety concerns."
        elif current_count < min_threshold * 2:
            return f"Safety mentions adequate but could be strengthened. Add specific harm scenarios."
        else:
            return "Safety argument strength is good. Focus on concrete safety risks."
