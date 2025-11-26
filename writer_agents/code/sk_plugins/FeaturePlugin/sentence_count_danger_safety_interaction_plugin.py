#!/usr/bin/env python3
"""
Sentence Count × Danger/Safety Position Interaction Plugin.

This is the #2 strongest interaction (0.332 strength) for pseudonym motions.
Shows how sentence structure interacts with danger/safety section position.
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


class SentenceCountDangerSafetyInteractionPlugin(BaseFeaturePlugin):
    """Plugin to validate sentence_count × danger_safety_position interaction."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "sentence_count_danger_safety_interaction", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("SentenceCountDangerSafetyInteractionPlugin initialized")
        self.interaction_strength = 0.332  # #2 strongest interaction

    def _calculate_interaction(self, text: str) -> Dict[str, Any]:
        """Calculate sentence_count × danger_safety_position interaction."""
        # Calculate sentence count
        sentence_count = text.count('.') + text.count('!') + text.count('?')

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
        interaction_value = sentence_count * normalized_position if normalized_position >= 0 else 0

        return {
            "sentence_count": sentence_count,
            "danger_safety_position": danger_pos,
            "normalized_position": normalized_position,
            "interaction_value": interaction_value,
            "has_danger_safety": danger_pos >= 0
        }

    async def validate_sentence_count_danger_safety_interaction(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate sentence_count × danger_safety_position interaction.

        This is the #2 strongest interaction (0.332 strength) for pseudonym motions.
        """
        text = document.get_full_text()
        analysis = self._calculate_interaction(text)

        issues = []
        recommendations = []

        if not analysis["has_danger_safety"]:
            issues.append({
                "type": "missing_danger_safety",
                "severity": "high",
                "message": "Danger/safety section not found. This interaction (sentence_count × danger_safety_position) is the #2 strongest predictor (0.332 strength).",
                "suggestion": "Include a section discussing danger, safety, threats, or risks."
            })
            recommendations.append({
                "type": "add_danger_safety_section",
                "priority": "high",
                "action": "Add a section discussing danger, safety, threats, or risks to the moving party",
                "rationale": "sentence_count × danger_safety_position is the #2 strongest interaction (0.332 strength)"
            })
        else:
            # Interaction is present - validate that it's in a good position
            if analysis["normalized_position"] < 0.2:
                recommendations.append({
                    "type": "danger_safety_too_early",
                    "priority": "medium",
                    "action": f"Danger/safety section appears very early (position: {analysis['normalized_position']:.2%}). Consider placing it later in the document.",
                    "rationale": "Position relative to sentence structure matters for this interaction"
                })
            elif analysis["normalized_position"] > 0.8:
                recommendations.append({
                    "type": "danger_safety_too_late",
                    "priority": "medium",
                    "action": f"Danger/safety section appears very late (position: {analysis['normalized_position']:.2%}). Consider placing it earlier.",
                    "rationale": "Position relative to sentence structure matters for this interaction"
                })
            else:
                recommendations.append({
                    "type": "interaction_optimal",
                    "message": f"Sentence count × danger/safety position interaction is optimal (sentence_count: {analysis['sentence_count']}, position: {analysis['normalized_position']:.2%}, interaction: {analysis['interaction_value']:.1f}).",
                    "interaction_value": analysis["interaction_value"]
                })

        return FunctionResult(
            success=analysis["has_danger_safety"],
            data={
                "interaction_value": analysis["interaction_value"],
                "sentence_count": analysis["sentence_count"],
                "danger_safety_position": analysis["danger_safety_position"],
                "normalized_position": analysis["normalized_position"],
                "has_danger_safety": analysis["has_danger_safety"],
                "interaction_strength": self.interaction_strength,
                "interaction_rank": 2,
                "issues": issues,
                "recommendations": recommendations
            },
            message="Sentence count × danger/safety position interaction validation complete"
        )

