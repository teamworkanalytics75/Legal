#!/usr/bin/env python3
"""
Privacy Harm Count Plugin - Atomic SK plugin for privacy harm count feature.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class PrivacyHarmCountPlugin(BaseFeaturePlugin):
    """Atomic plugin for privacy harm count feature analysis."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "privacy_harm_count", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("PrivacyHarmCountPlugin initialized")

    async def analyze_harm_diversity(self, draft_text: str) -> FunctionResult:
        """Analyze diversity of harm types mentioned."""
        try:
            text_lower = draft_text.lower()

            # Define harm types
            harm_types = {
                "privacy": ["privacy", "personal information", "confidential", "private"],
                "harassment": ["harassment", "harass", "intimidation", "intimidate"],
                "safety": ["safety", "danger", "threat", "harm", "risk"],
                "retaliation": ["retaliation", "retaliate", "reprisal", "adverse action"],
                "embarrassment": ["embarrassment", "embarrass", "humiliation", "shame"],
                "stigma": ["stigma", "stigmatize", "reputation", "standing"]
            }

            harm_counts = {}
            total_harm_mentions = 0

            for harm_type, terms in harm_types.items():
                count = sum(text_lower.count(term) for term in terms)
                harm_counts[harm_type] = count
                total_harm_mentions += count

            # Count distinct harm types
            distinct_harm_types = len([h for h in harm_counts.values() if h > 0])

            # Calculate strength score
            min_threshold = self.rules.get("minimum_threshold", 2)
            target_average = self.rules.get("successful_case_average", 3.0)

            diversity_score = min(1.0, distinct_harm_types / max(min_threshold, 1))
            meets_threshold = distinct_harm_types >= min_threshold

            return FunctionResult(
                success=True,
                value={
                    "distinct_harm_types": distinct_harm_types,
                    "harm_type_breakdown": harm_counts,
                    "total_harm_mentions": total_harm_mentions,
                    "diversity_score": diversity_score,
                    "meets_threshold": meets_threshold,
                    "recommendation": self._get_harm_diversity_recommendation(distinct_harm_types, min_threshold)
                },
                metadata={"analysis_type": "harm_diversity"}
            )

        except Exception as e:
            logger.error(f"Harm diversity analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _get_harm_diversity_recommendation(self, current_types: int, min_threshold: int) -> str:
        """Get recommendation based on current harm type diversity."""
        if current_types < min_threshold:
            return f"Increase harm type diversity from {current_types} to at least {min_threshold}. Add different types of harm."
        elif current_types < min_threshold + 1:
            return f"Harm diversity adequate but could be strengthened. Consider adding more specific harm examples."
        else:
            return "Harm diversity is good. Focus on quality and specificity of harm descriptions."
