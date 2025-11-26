#!/usr/bin/env python3
"""
Average Words Per Paragraph × Enumeration Density Interaction Plugin.

This interaction (2.61 importance) shows how paragraph structure interacts with enumeration patterns.
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


class AvgWordsEnumerationDensityInteractionPlugin(BaseFeaturePlugin):
    """Plugin to validate avg_words_per_paragraph × enumeration_density interaction."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "avg_words_enumeration_density_interaction", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("AvgWordsEnumerationDensityInteractionPlugin initialized")
        self.interaction_importance = 2.61

    def _calculate_interaction(self, text: str) -> Dict[str, Any]:
        """Calculate avg_words_per_paragraph × enumeration_density interaction."""
        # Split into paragraphs
        paragraphs = re.split(r'(?:\r?\n\s*){2,}', text.strip())
        if len(paragraphs) == 1:
            paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        else:
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Calculate average words per paragraph
        word_counts = []
        for para in paragraphs:
            words = re.findall(r"[A-Za-z0-9']+", para)
            word_counts.append(len(words))
        
        avg_words_per_paragraph = sum(word_counts) / len(word_counts) if word_counts else 0

        # Calculate enumeration density
        total_words = sum(word_counts) if word_counts else 0
        bullet_count = text.count('•') + text.count('- ') + text.count('* ')
        numbered_count = len(re.findall(r'\d+\.\s+[A-Z]', text))
        enumeration_count = bullet_count + numbered_count
        enumeration_density = (enumeration_count / total_words * 1000) if total_words > 0 else 0

        # Calculate interaction
        interaction_value = avg_words_per_paragraph * enumeration_density

        return {
            "avg_words_per_paragraph": avg_words_per_paragraph,
            "enumeration_density": enumeration_density,
            "enumeration_count": enumeration_count,
            "paragraph_count": len(paragraphs),
            "interaction_value": interaction_value
        }

    async def validate_avg_words_enumeration_density_interaction(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate avg_words_per_paragraph × enumeration_density interaction.

        This interaction (2.61 importance) shows how paragraph structure interacts with enumeration.
        """
        text = document.get_full_text()
        analysis = self._calculate_interaction(text)

        issues = []
        recommendations = []

        # Check if enumeration is present
        if analysis["enumeration_count"] == 0:
            recommendations.append({
                "type": "add_enumeration",
                "priority": "medium",
                "action": "Consider adding bullet points or numbered lists to improve enumeration density",
                "rationale": "Enumeration density interacts with paragraph structure (2.61 importance)"
            })
        else:
            recommendations.append({
                "type": "interaction_calculated",
                "message": f"Average words per paragraph × enumeration density interaction calculated (avg_words: {analysis['avg_words_per_paragraph']:.1f}, enum_density: {analysis['enumeration_density']:.2f}, interaction: {analysis['interaction_value']:.1f}).",
                "interaction_value": analysis["interaction_value"]
            })

        return FunctionResult(
            success=True,
            data={
                "interaction_value": analysis["interaction_value"],
                "avg_words_per_paragraph": analysis["avg_words_per_paragraph"],
                "enumeration_density": analysis["enumeration_density"],
                "enumeration_count": analysis["enumeration_count"],
                "paragraph_count": analysis["paragraph_count"],
                "interaction_importance": self.interaction_importance,
                "issues": issues,
                "recommendations": recommendations
            },
            message="Average words per paragraph × enumeration density interaction validation complete"
        )

