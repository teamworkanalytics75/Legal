#!/usr/bin/env python3
"""
Harassment Plugin - Atomic SK plugin for harassment mentions feature.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class HarassmentPlugin(BaseFeaturePlugin):
    """Atomic plugin for harassment mentions feature analysis."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "mentions_harassment", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("HarassmentPlugin initialized")

    async def analyze_harassment_risk(self, draft_text: str) -> FunctionResult:
        """Analyze harassment risk argument strength."""
        try:
            text_lower = draft_text.lower()

            # Count harassment-related terms
            harassment_terms = [
                "harassment", "harass", "intimidation", "intimidate",
                "retaliation", "retaliate", "reprisal", "adverse action"
            ]

            term_counts = {}
            total_mentions = 0

            for term in harassment_terms:
                count = text_lower.count(term)
                term_counts[term] = count
                total_mentions += count

            # Check for causation language
            causation_indicators = [
                "would likely", "risk of", "potential for", "may result",
                "could lead", "threat of", "fear of"
            ]

            causation_count = sum(text_lower.count(indicator) for indicator in causation_indicators)

            # Calculate strength score
            min_threshold = self.rules.get("minimum_threshold", 1)
            target_average = self.rules.get("successful_case_average", 2.0)

            strength_score = min(1.0, total_mentions / max(target_average, 1))
            meets_threshold = total_mentions >= min_threshold

            return FunctionResult(
                success=True,
                value={
                    "total_harassment_mentions": total_mentions,
                    "term_breakdown": term_counts,
                    "causation_indicators": causation_count,
                    "strength_score": strength_score,
                    "meets_threshold": meets_threshold,
                    "recommendation": self._get_harassment_recommendation(total_mentions, min_threshold)
                },
                metadata={"analysis_type": "harassment_risk"}
            )

        except Exception as e:
            logger.error(f"Harassment analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _get_harassment_recommendation(self, current_count: int, min_threshold: int) -> str:
        """Get recommendation based on current harassment mention count."""
        if current_count < min_threshold:
            return f"Increase harassment mentions from {current_count} to at least {min_threshold}. Add specific risk examples."
        elif current_count < min_threshold * 2:
            return f"Harassment mentions adequate but could be strengthened. Add causation language."
        else:
            return "Harassment argument strength is good. Focus on specific risk scenarios."
