#!/usr/bin/env python3
"""
Retaliation Plugin - Atomic SK plugin for retaliation mentions feature.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class RetaliationPlugin(BaseFeaturePlugin):
    """Atomic plugin for retaliation mentions feature analysis."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "mentions_retaliation", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("RetaliationPlugin initialized")

    async def analyze_retaliation_risk(self, draft_text: str) -> FunctionResult:
        """Analyze retaliation risk argument strength."""
        try:
            text_lower = draft_text.lower()

            # Count retaliation-related terms
            retaliation_terms = [
                "retaliation", "retaliate", "reprisal", "adverse action",
                "punitive", "punishment", "retaliatory", "reprisal"
            ]

            term_counts = {}
            total_mentions = 0

            for term in retaliation_terms:
                count = text_lower.count(term)
                term_counts[term] = count
                total_mentions += count

            # Check for causal connection language
            causal_indicators = [
                "because of", "due to", "as a result", "in response to",
                "caused by", "resulting from", "following", "after"
            ]

            causal_count = sum(text_lower.count(indicator) for indicator in causal_indicators)

            # Calculate strength score
            min_threshold = self.rules.get("minimum_threshold", 1)
            target_average = self.rules.get("successful_case_average", 1.2)

            strength_score = min(1.0, total_mentions / max(target_average, 1))
            meets_threshold = total_mentions >= min_threshold

            return FunctionResult(
                success=True,
                value={
                    "total_retaliation_mentions": total_mentions,
                    "term_breakdown": term_counts,
                    "causal_indicators": causal_count,
                    "strength_score": strength_score,
                    "meets_threshold": meets_threshold,
                    "recommendation": self._get_retaliation_recommendation(total_mentions, min_threshold)
                },
                metadata={"analysis_type": "retaliation_risk"}
            )

        except Exception as e:
            logger.error(f"Retaliation analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _get_retaliation_recommendation(self, current_count: int, min_threshold: int) -> str:
        """Get recommendation based on current retaliation mention count."""
        if current_count < min_threshold:
            return f"Increase retaliation mentions from {current_count} to at least {min_threshold}. Add causal connection language."
        elif current_count < min_threshold * 2:
            return f"Retaliation mentions adequate but could be strengthened. Add specific adverse action examples."
        else:
            return "Retaliation argument strength is good. Focus on concrete adverse action scenarios."
