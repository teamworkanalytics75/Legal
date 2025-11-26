#!/usr/bin/env python3
"""
Word Frequency Plugin - Word-level enforcement

Tracks word frequency and keyword usage (key terms, repetition, density)
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class WordFrequencyPlugin(BaseFeaturePlugin):
    """Plugin for tracking word frequency and keyword usage."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "word_frequency", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("WordFrequencyPlugin initialized")

    def _count_words(self, text: str) -> Counter:
        """Count word frequencies."""
        words = re.findall(r'\b\w+\b', text.lower())
        return Counter(words)

    def _analyze_word_frequency(self, text: str) -> Dict[str, Any]:
        """Analyze word frequency patterns."""
        criteria = self.rules.get("validation_criteria", {})
        key_terms = criteria.get("key_terms", [])
        max_repetition_ratio = criteria.get("max_repetition_ratio", 0.05)

        word_counts = self._count_words(text)
        total_words = sum(word_counts.values())

        # Check key terms
        key_term_mentions = {}
        for term in key_terms:
            # Search for term (case-insensitive, whole word)
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            key_term_mentions[term] = len(matches)

        # Find most repeated words (potential excessive repetition)
        most_common = word_counts.most_common(20)
        excessive_repetition = []
        for word, count in most_common:
            ratio = count / total_words if total_words > 0 else 0
            if ratio > max_repetition_ratio and len(word) > 3:  # Ignore short words like "the", "and"
                excessive_repetition.append({
                    "word": word,
                    "count": count,
                    "ratio": ratio
                })

        # Check key term mention counts
        key_term_issues = []
        min_mentions = criteria.get("min_key_term_mentions", 1)
        max_mentions = criteria.get("max_key_term_mentions", 50)

        for term, count in key_term_mentions.items():
            if count < min_mentions:
                key_term_issues.append(f"Key term '{term}' mentioned {count} times (minimum: {min_mentions})")
            elif count > max_mentions:
                key_term_issues.append(f"Key term '{term}' mentioned {count} times (maximum: {max_mentions})")

        return {
            "total_words": total_words,
            "unique_words": len(word_counts),
            "key_term_mentions": key_term_mentions,
            "excessive_repetition": excessive_repetition[:10],
            "most_common_words": most_common[:10],
            "key_term_issues": key_term_issues,
            "max_repetition_ratio": max_repetition_ratio
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate word frequency patterns."""
        freq_check = self._analyze_word_frequency(text)

        # Score calculation
        score = 1.0

        # Penalty for excessive repetition
        if freq_check["excessive_repetition"]:
            penalty = min(0.3, len(freq_check["excessive_repetition"]) * 0.05)
            score -= penalty

        # Penalty for key term issues
        if freq_check["key_term_issues"]:
            penalty = min(0.2, len(freq_check["key_term_issues"]) * 0.03)
            score -= penalty

        score = max(0.0, min(1.0, score))

        meets_threshold = (
            len(freq_check["excessive_repetition"]) == 0 and
            len(freq_check["key_term_issues"]) == 0
        )

        issues = []
        if freq_check["excessive_repetition"]:
            issues.append(f"{len(freq_check['excessive_repetition'])} words with excessive repetition")
            for item in freq_check["excessive_repetition"][:3]:
                issues.append(f"  - '{item['word']}': {item['count']} times ({item['ratio']:.1%} of text)")
        issues.extend(freq_check["key_term_issues"])

        return FunctionResult(
            success=meets_threshold,
            score=score,
            message=f"Word Frequency: {freq_check['unique_words']} unique words, {len(freq_check['excessive_repetition'])} excessive repetitions - {'PASS' if meets_threshold else 'WARN'}",
            metadata={
                "word_frequency": freq_check,
                "issues": issues,
                "threshold": self.rules.get("threshold", 0.85)
            }
        )

    async def _execute_native(self, text: str, **kwargs) -> FunctionResult:
        """Execute native validation."""
        return await self.validate(text)

    async def _execute_semantic(self, text: str, **kwargs) -> FunctionResult:
        """Execute semantic validation."""
        case_context = text[:500]
        similar_cases = await self.query_chroma(case_context, n_results=5)

        native_result = await self.validate(text)
        native_result.metadata["similar_cases"] = similar_cases[:3]

        return native_result

