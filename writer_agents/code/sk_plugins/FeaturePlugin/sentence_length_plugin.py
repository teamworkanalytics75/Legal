#!/usr/bin/env python3
"""
Sentence Length Plugin - Sentence-level enforcement

Enforces sentence length constraints for legal writing
Average: 15-25 words, Max: 40 words, Min: 5 words, Long sentence ratio < 20%
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class SentenceLengthPlugin(BaseFeaturePlugin):
    """Plugin for enforcing sentence length constraints."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "sentence_length", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("SentenceLengthPlugin initialized")

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        words = re.findall(r'\b\w+\b', text)
        return len(words)

    def _analyze_sentence_lengths(self, text: str) -> Dict[str, Any]:
        """Analyze sentence lengths."""
        sentences = self._split_sentences(text)

        criteria = self.rules.get("validation_criteria", {})
        min_words = criteria.get("min_words_per_sentence", 5)
        max_words = criteria.get("max_words_per_sentence", 40)
        target_avg_min = criteria.get("target_avg_min", 15)
        target_avg_max = criteria.get("target_avg_max", 25)
        target_avg = criteria.get("target_avg_words", 20)
        max_long_ratio = criteria.get("max_long_sentence_ratio", 0.2)

        sentence_lengths = []
        too_short = []
        too_long = []
        long_sentences = []

        for i, sentence in enumerate(sentences):
            word_count = self._count_words(sentence)
            sentence_lengths.append(word_count)

            if word_count < min_words:
                too_short.append({
                    "index": i,
                    "word_count": word_count,
                    "preview": sentence[:80] + "..." if len(sentence) > 80 else sentence
                })
            elif word_count > max_words:
                too_long.append({
                    "index": i,
                    "word_count": word_count,
                    "preview": sentence[:80] + "..." if len(sentence) > 80 else sentence
                })

            if word_count > target_avg_max:
                long_sentences.append(word_count)

        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            median_length = sorted(sentence_lengths)[len(sentence_lengths) // 2]
        else:
            avg_length = 0
            median_length = 0

        long_sentence_ratio = len(long_sentences) / len(sentence_lengths) if sentence_lengths else 0
        avg_within_target = target_avg_min <= avg_length <= target_avg_max

        return {
            "total_sentences": len(sentence_lengths),
            "avg_words_per_sentence": avg_length,
            "median_words_per_sentence": median_length,
            "min_words": min(sentence_lengths) if sentence_lengths else 0,
            "max_words": max(sentence_lengths) if sentence_lengths else 0,
            "too_short_count": len(too_short),
            "too_long_count": len(too_long),
            "long_sentence_ratio": long_sentence_ratio,
            "avg_within_target": avg_within_target,
            "target_avg_min": target_avg_min,
            "target_avg_max": target_avg_max,
            "target_avg": target_avg,
            "max_words_threshold": max_words,
            "min_words_threshold": min_words,
            "max_long_ratio": max_long_ratio,
            "too_short_sentences": too_short[:5],
            "too_long_sentences": too_long[:5]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate sentence lengths."""
        length_check = self._analyze_sentence_lengths(text)

        # Score based on multiple factors
        score = 0.0

        # Average length within target (40%)
        if length_check["avg_within_target"]:
            score += 0.4
        else:
            # Partial credit if close
            avg = length_check["avg_words_per_sentence"]
            if avg < length_check["target_avg_min"]:
                score += 0.4 * (avg / length_check["target_avg_min"])
            elif avg > length_check["target_avg_max"]:
                score += 0.4 * (1 - (avg - length_check["target_avg_max"]) / length_check["target_avg_max"])

        # Long sentence ratio acceptable (30%)
        if length_check["long_sentence_ratio"] <= length_check["max_long_ratio"]:
            score += 0.3
        else:
            score += 0.3 * (1 - (length_check["long_sentence_ratio"] - length_check["max_long_ratio"]))

        # No sentences too short or too long (30%)
        total_issues = length_check["too_short_count"] + length_check["too_long_count"]
        total_sentences = length_check["total_sentences"]
        if total_sentences > 0:
            issue_ratio = total_issues / total_sentences
            score += 0.3 * (1 - min(issue_ratio, 1.0))

        meets_threshold = (
            length_check["avg_within_target"] and
            length_check["long_sentence_ratio"] <= length_check["max_long_ratio"] and
            (length_check["too_short_count"] + length_check["too_long_count"]) / max(length_check["total_sentences"], 1) < 0.1
        )

        issues = []
        if not length_check["avg_within_target"]:
            issues.append(f"Average sentence length ({length_check['avg_words_per_sentence']:.1f} words) outside target range ({length_check['target_avg_min']}-{length_check['target_avg_max']} words)")
        if length_check["long_sentence_ratio"] > length_check["max_long_ratio"]:
            issues.append(f"Long sentence ratio ({length_check['long_sentence_ratio']:.1%}) exceeds maximum ({length_check['max_long_ratio']:.1%})")
        if length_check["too_short_count"] > 0:
            issues.append(f"{length_check['too_short_count']} sentences too short (< {length_check['min_words_threshold']} words)")
        if length_check["too_long_count"] > 0:
            issues.append(f"{length_check['too_long_count']} sentences too long (> {length_check['max_words_threshold']} words)")

        return FunctionResult(
            success=meets_threshold,
            score=score,
            message=f"Sentence Length: {length_check['avg_words_per_sentence']:.1f} avg words (target: {length_check['target_avg_min']}-{length_check['target_avg_max']}) - {'PASS' if meets_threshold else 'WARN'}",
            metadata={
                "sentence_lengths": length_check,
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

