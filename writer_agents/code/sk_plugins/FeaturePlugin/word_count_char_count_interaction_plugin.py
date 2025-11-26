#!/usr/bin/env python3
"""
Word Count × Character Count Interaction Plugin.

This interaction shows consistency between word count and character count metrics.
Important for pseudonym motions.
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


class WordCountCharCountInteractionPlugin(BaseFeaturePlugin):
    """Plugin to validate word_count × char_count interaction."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "word_count_char_count_interaction", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("WordCountCharCountInteractionPlugin initialized")

    def _calculate_interaction(self, text: str) -> Dict[str, Any]:
        """Calculate word_count × char_count interaction."""
        # Calculate word count
        words = re.findall(r"[A-Za-z0-9']+", text)
        word_count = len(words)

        # Calculate character count
        char_count = len(text)

        # Calculate interaction
        interaction_value = word_count * char_count

        # Calculate average characters per word (consistency metric)
        avg_chars_per_word = char_count / word_count if word_count > 0 else 0

        return {
            "word_count": word_count,
            "char_count": char_count,
            "interaction_value": interaction_value,
            "avg_chars_per_word": avg_chars_per_word
        }

    async def validate_word_count_char_count_interaction(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate word_count × char_count interaction.

        This interaction shows consistency between length metrics.
        """
        text = document.get_full_text()
        analysis = self._calculate_interaction(text)

        issues = []
        recommendations = []

        # Check for reasonable consistency (avg chars per word should be in reasonable range)
        if analysis["avg_chars_per_word"] < 3.0:
            issues.append({
                "type": "low_chars_per_word",
                "severity": "low",
                "message": f"Average characters per word ({analysis['avg_chars_per_word']:.2f}) is unusually low. This may indicate formatting issues.",
                "suggestion": "Check for excessive whitespace or formatting problems."
            })
        elif analysis["avg_chars_per_word"] > 8.0:
            issues.append({
                "type": "high_chars_per_word",
                "severity": "low",
                "message": f"Average characters per word ({analysis['avg_chars_per_word']:.2f}) is unusually high. This may indicate very long words or formatting issues.",
                "suggestion": "Check for formatting issues or consider using shorter words where appropriate."
            })
        else:
            recommendations.append({
                "type": "interaction_optimal",
                "message": f"Word count × character count interaction is consistent (word_count: {analysis['word_count']}, char_count: {analysis['char_count']}, interaction: {analysis['interaction_value']:.1f}, avg_chars_per_word: {analysis['avg_chars_per_word']:.2f}).",
                "interaction_value": analysis["interaction_value"]
            })

        return FunctionResult(
            success=True,
            data={
                "interaction_value": analysis["interaction_value"],
                "word_count": analysis["word_count"],
                "char_count": analysis["char_count"],
                "avg_chars_per_word": analysis["avg_chars_per_word"],
                "issues": issues,
                "recommendations": recommendations
            },
            message="Word count × character count interaction validation complete"
        )

