#!/usr/bin/env python3
"""
Word Count × Danger/Safety Position Interaction Plugin.

This interaction (0.093 strength) shows how document length (words) interacts with danger/safety section position.
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


class WordCountDangerSafetyInteractionPlugin(BaseFeaturePlugin):
    """Plugin to validate word_count × danger_safety_position interaction."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "word_count_danger_safety_interaction", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("WordCountDangerSafetyInteractionPlugin initialized")
        self.interaction_strength = 0.093

    def _calculate_interaction(self, text: str) -> Dict[str, Any]:
        """Calculate word_count × danger_safety_position interaction."""
        # Calculate word count
        words = re.findall(r"[A-Za-z0-9']+", text)
        word_count = len(words)

        # Find danger/safety section position
        text_lower = text.lower()
        danger_keywords = ['danger', 'safety', 'threat', 'harm', 'risk', 'vulnerable', 'vulnerability']
        danger_pos = -1
        
        for keyword in danger_keywords:
            pos = text_lower.find(keyword)
            if pos >= 0:
                danger_pos = pos
                break

        # Normalize position (0-1 scale based on document length)
        normalized_position = (danger_pos / len(text)) if danger_pos >= 0 and len(text) > 0 else -1

        # Calculate interaction
        interaction_value = word_count * normalized_position if normalized_position >= 0 else 0

        return {
            "word_count": word_count,
            "danger_safety_position": danger_pos,
            "normalized_position": normalized_position,
            "interaction_value": interaction_value,
            "has_danger_safety": danger_pos >= 0
        }

    async def validate_word_count_danger_safety_interaction(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate word_count × danger_safety_position interaction.
        """
        text = document.get_full_text()
        analysis = self._calculate_interaction(text)

        issues = []
        recommendations = []

        if not analysis["has_danger_safety"]:
            issues.append({
                "type": "missing_danger_safety",
                "severity": "medium",
                "message": "Danger/safety section not found. This interaction (word_count × danger_safety_position) has 0.093 strength.",
                "suggestion": "Include a section discussing danger, safety, threats, or risks."
            })
        else:
            recommendations.append({
                "type": "interaction_calculated",
                "message": f"Word count × danger/safety position interaction calculated (word_count: {analysis['word_count']}, position: {analysis['normalized_position']:.2%}, interaction: {analysis['interaction_value']:.1f}).",
                "interaction_value": analysis["interaction_value"]
            })

        return FunctionResult(
            success=analysis["has_danger_safety"],
            data={
                "interaction_value": analysis["interaction_value"],
                "word_count": analysis["word_count"],
                "danger_safety_position": analysis["danger_safety_position"],
                "normalized_position": analysis["normalized_position"],
                "has_danger_safety": analysis["has_danger_safety"],
                "interaction_strength": self.interaction_strength,
                "issues": issues,
                "recommendations": recommendations
            },
            message="Word count × danger/safety position interaction validation complete"
        )

