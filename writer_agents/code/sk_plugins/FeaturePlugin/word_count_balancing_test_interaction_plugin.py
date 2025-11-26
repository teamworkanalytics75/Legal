#!/usr/bin/env python3
"""
Word Count × Balancing Test Position Interaction Plugin.

This is the #1 strongest interaction (0.337 strength) for pseudonym motions.
Shows how document length interacts with balancing test position.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class WordCountBalancingTestInteractionPlugin(BaseFeaturePlugin):
    """Plugin to validate word_count × balancing_test_position interaction."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "word_count_balancing_test_interaction", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("WordCountBalancingTestInteractionPlugin initialized")
        self.interaction_strength = 0.337  # #1 strongest interaction

    def _calculate_interaction(self, text: str) -> Dict[str, Any]:
        """Calculate word_count × balancing_test_position interaction."""
        # Calculate word count
        words = re.findall(r"[A-Za-z0-9']+", text)
        word_count = len(words)

        # Find balancing test position
        text_lower = text.lower()
        balancing_keywords = ['balancing', 'balance', 'weigh', 'weighing', 'test']
        balancing_pos = -1
        
        for keyword in balancing_keywords:
            pos = text_lower.find(keyword)
            if pos >= 0:
                balancing_pos = pos
                break

        # Normalize position (0-1 scale based on document length)
        normalized_position = (balancing_pos / len(text)) if balancing_pos >= 0 and len(text) > 0 else -1

        # Calculate interaction
        interaction_value = word_count * normalized_position if normalized_position >= 0 else 0

        return {
            "word_count": word_count,
            "balancing_test_position": balancing_pos,
            "normalized_position": normalized_position,
            "interaction_value": interaction_value,
            "has_balancing_test": balancing_pos >= 0
        }

    async def validate_word_count_balancing_test_interaction(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate word_count × balancing_test_position interaction.

        This is the #1 strongest interaction (0.337 strength) for pseudonym motions.
        """
        text = document.get_full_text()
        analysis = self._calculate_interaction(text)

        issues = []
        recommendations = []

        if not analysis["has_balancing_test"]:
            issues.append({
                "type": "missing_balancing_test",
                "severity": "high",
                "message": "Balancing test section not found. This interaction (word_count × balancing_test_position) is the #1 strongest predictor (0.337 strength).",
                "suggestion": "Include a balancing test section that weighs competing interests."
            })
            recommendations.append({
                "type": "add_balancing_test",
                "priority": "high",
                "action": "Add a balancing test section that weighs privacy/safety interests against public interest",
                "rationale": "word_count × balancing_test_position is the #1 strongest interaction (0.337 strength)"
            })
        else:
            # Interaction is present - validate that it's in a good position
            if analysis["normalized_position"] < 0.2:
                recommendations.append({
                    "type": "balancing_test_too_early",
                    "priority": "medium",
                    "action": f"Balancing test appears very early (position: {analysis['normalized_position']:.2%}). Consider placing it later in the document.",
                    "rationale": "Position relative to document length matters for this interaction"
                })
            elif analysis["normalized_position"] > 0.8:
                recommendations.append({
                    "type": "balancing_test_too_late",
                    "priority": "medium",
                    "action": f"Balancing test appears very late (position: {analysis['normalized_position']:.2%}). Consider placing it earlier.",
                    "rationale": "Position relative to document length matters for this interaction"
                })
            else:
                recommendations.append({
                    "type": "interaction_optimal",
                    "message": f"Word count × balancing test position interaction is optimal (word_count: {analysis['word_count']}, position: {analysis['normalized_position']:.2%}, interaction: {analysis['interaction_value']:.1f}).",
                    "interaction_value": analysis["interaction_value"]
                })

        return FunctionResult(
            success=analysis["has_balancing_test"],
            data={
                "interaction_value": analysis["interaction_value"],
                "word_count": analysis["word_count"],
                "balancing_test_position": analysis["balancing_test_position"],
                "normalized_position": analysis["normalized_position"],
                "has_balancing_test": analysis["has_balancing_test"],
                "interaction_strength": self.interaction_strength,
                "interaction_rank": 1,
                "issues": issues,
                "recommendations": recommendations
            },
            message="Word count × balancing test position interaction validation complete"
        )

