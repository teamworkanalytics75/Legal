#!/usr/bin/env python3
"""
Document Length Interaction Plugin - Validates document length metrics.

This plugin handles document length validation.
Interaction calculations are handled by individual interaction plugins:
- WordCountCharCountInteractionPlugin (word_count × char_count)
- WordCountBalancingTestInteractionPlugin (word_count × balancing_test_position)
- SentenceCountDangerSafetyInteractionPlugin (sentence_count × danger_safety_position)
- etc.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult, EditRequest, DocumentLocation

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class DocumentLengthInteractionPlugin(BaseFeaturePlugin):
    """Plugin to enforce optimal document length and interactions."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "document_length_interaction", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("DocumentLengthInteractionPlugin initialized")

        # Target ranges (from analysis)
        self.target_word_count = self.rules.get("targets", {}).get("word_count", 2000)
        self.target_char_count = self.rules.get("targets", {}).get("char_count", 10000)
        self.target_sentence_count = self.rules.get("targets", {}).get("sentence_count", 100)

    def _calculate_document_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate document length metrics."""
        word_count = len(re.findall(r"[A-Za-z0-9']+", text))
        char_count = len(text)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        paragraph_count = text.count('\n\n') + 1

        # Calculate ratios
        words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        chars_per_word = char_count / word_count if word_count > 0 else 0

        # Check if in target ranges
        word_count_optimal = word_count >= self.target_word_count * 0.8
        char_count_optimal = char_count >= self.target_char_count * 0.8
        sentence_count_optimal = sentence_count >= self.target_sentence_count * 0.8

        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "words_per_sentence": words_per_sentence,
            "chars_per_word": chars_per_word,
            "word_count_optimal": word_count_optimal,
            "char_count_optimal": char_count_optimal,
            "sentence_count_optimal": sentence_count_optimal,
            "target_word_count": self.target_word_count,
            "target_char_count": self.target_char_count,
            "target_sentence_count": self.target_sentence_count
        }

    # Note: Interaction calculations are now handled by individual interaction plugins:
    # - WordCountCharCountInteractionPlugin
    # - WordCountBalancingTestInteractionPlugin
    # - SentenceCountDangerSafetyInteractionPlugin
    # - CharCountDangerSafetyInteractionPlugin
    # - WordCountDangerSafetyInteractionPlugin

    async def validate_document_length(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate document length metrics.

        This plugin validates document length (word count, char count, sentence count).
        Interaction calculations are handled by individual interaction plugins.
        """
        text = document.get_full_text()
        metrics = self._calculate_document_metrics(text)

        issues = []
        recommendations = []

        # Check word count
        if not metrics["word_count_optimal"]:
            issues.append({
                "type": "insufficient_word_count",
                "severity": "medium",
                "message": f"Word count ({metrics['word_count']}) is below target ({self.target_word_count}). Document length interactions are important.",
                "suggestion": f"Increase word count to at least {self.target_word_count * 0.8:.0f} words."
            })

        # Check char count
        if not metrics["char_count_optimal"]:
            issues.append({
                "type": "insufficient_char_count",
                "severity": "medium",
                "message": f"Character count ({metrics['char_count']}) is below target ({self.target_char_count}).",
                "suggestion": f"Increase character count to at least {self.target_char_count * 0.8:.0f} characters."
            })

        # Check sentence count
        if not metrics["sentence_count_optimal"]:
            issues.append({
                "type": "insufficient_sentence_count",
                "severity": "low",
                "message": f"Sentence count ({metrics['sentence_count']}) is below target ({self.target_sentence_count}).",
                "suggestion": f"Increase sentence count to at least {self.target_sentence_count * 0.8:.0f} sentences."
            })

        if metrics["word_count_optimal"] and metrics["char_count_optimal"]:
            recommendations.append({
                "type": "document_length_optimal",
                "message": "Document length is optimal. These metrics enable strong interactions (see individual interaction plugins).",
                "metrics": metrics
            })

        return FunctionResult(
            success=metrics["word_count_optimal"] and metrics["char_count_optimal"],
            data={
                "word_count": metrics["word_count"],
                "char_count": metrics["char_count"],
                "sentence_count": metrics["sentence_count"],
                "paragraph_count": metrics["paragraph_count"],
                "words_per_sentence": metrics["words_per_sentence"],
                "chars_per_word": metrics["chars_per_word"],
                "word_count_optimal": metrics["word_count_optimal"],
                "char_count_optimal": metrics["char_count_optimal"],
                "sentence_count_optimal": metrics["sentence_count_optimal"],
                "issues": issues,
                "recommendations": recommendations,
                "note": "Interaction calculations are handled by individual interaction plugins"
            },
            message="Document length validation complete"
        )

    async def query_document_length(self, case_context: str, **kwargs) -> FunctionResult:
        """Query for examples of optimal document length."""
        results = await self.query_chroma(
            case_context=f"document length word count {case_context}",
            n_results=kwargs.get('n_results', 10)
        )

        return FunctionResult(
            success=True,
            data={"examples": results},
            message=f"Found {len(results)} examples of document length"
        )

    async def generate_document_length_argument(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Generate argument about document length."""
        text = document.get_full_text()
        metrics = self._calculate_document_metrics(text)

        if metrics["word_count_optimal"] and metrics["char_count_optimal"]:
            argument = f"""The motion has optimal document length:

- Word count: {metrics['word_count']} (target: {self.target_word_count})
- Character count: {metrics['char_count']} (target: {self.target_char_count})
- Sentence count: {metrics['sentence_count']} (target: {self.target_sentence_count})

These metrics enable strong interactions with other features (see individual interaction plugins for interaction calculations)."""
        else:
            argument = f"""The motion should optimize document length:

Current:
- Word count: {metrics['word_count']} (target: {self.target_word_count})
- Character count: {metrics['char_count']} (target: {self.target_char_count})
- Sentence count: {metrics['sentence_count']} (target: {self.target_sentence_count})

Recommendations:
1. Increase word count to at least {self.target_word_count * 0.8:.0f} words
2. Increase character count to at least {self.target_char_count * 0.8:.0f} characters
3. Ensure sufficient sentence count

Document length enables strong interactions with section positions and other features (see individual interaction plugins)."""

        return FunctionResult(
            success=True,
            data={
                "argument": argument,
                "metrics": metrics
            },
            message="Document length argument generated"
        )

