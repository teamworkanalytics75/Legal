#!/usr/bin/env python3
"""
Sentence Count Plugin - Sentence-level enforcement

Enforces sentence count per paragraph (3-8 sentences typical for legal writing)
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class SentenceCountPlugin(BaseFeaturePlugin):
    """Plugin for enforcing sentence count per paragraph."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "sentence_count", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("SentenceCountPlugin initialized")

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines or single newline if followed by blank line
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - look for sentence-ending punctuation
        sentence_pattern = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _count_sentences_per_paragraph(self, text: str) -> Dict[str, Any]:
        """Count sentences per paragraph."""
        paragraphs = self._split_paragraphs(text)

        min_sentences = self.rules.get("validation_criteria", {}).get("min_sentences_per_paragraph", 3)
        max_sentences = self.rules.get("validation_criteria", {}).get("max_sentences_per_paragraph", 8)
        target_avg = self.rules.get("validation_criteria", {}).get("target_average", 5)

        paragraph_sentence_counts = []
        paragraphs_outside_range = []

        for i, para in enumerate(paragraphs):
            sentences = self._split_sentences(para)
            sentence_count = len(sentences)
            paragraph_sentence_counts.append(sentence_count)

            if sentence_count < min_sentences or sentence_count > max_sentences:
                paragraphs_outside_range.append({
                    "paragraph_index": i,
                    "sentence_count": sentence_count,
                    "preview": para[:100] + "..." if len(para) > 100 else para
                })

        if paragraph_sentence_counts:
            avg_sentences = sum(paragraph_sentence_counts) / len(paragraph_sentence_counts)
            median_sentences = sorted(paragraph_sentence_counts)[len(paragraph_sentence_counts) // 2]
        else:
            avg_sentences = 0
            median_sentences = 0

        paragraphs_within_range = len([c for c in paragraph_sentence_counts if min_sentences <= c <= max_sentences])
        total_paragraphs = len(paragraph_sentence_counts)
        within_range_ratio = paragraphs_within_range / total_paragraphs if total_paragraphs > 0 else 0

        return {
            "total_paragraphs": total_paragraphs,
            "avg_sentences_per_paragraph": avg_sentences,
            "median_sentences_per_paragraph": median_sentences,
            "min_sentences": min(paragraph_sentence_counts) if paragraph_sentence_counts else 0,
            "max_sentences": max(paragraph_sentence_counts) if paragraph_sentence_counts else 0,
            "paragraphs_within_range": paragraphs_within_range,
            "paragraphs_outside_range": len(paragraphs_outside_range),
            "within_range_ratio": within_range_ratio,
            "target_min": min_sentences,
            "target_max": max_sentences,
            "target_avg": target_avg,
            "out_of_range_paragraphs": paragraphs_outside_range[:10]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate sentence count per paragraph."""
        count_check = self._count_sentences_per_paragraph(text)

        within_range_ratio = count_check["within_range_ratio"]
        meets_threshold = within_range_ratio >= 0.8  # 80% of paragraphs should be within range

        score = within_range_ratio

        issues = []
        if count_check["paragraphs_outside_range"] > 0:
            issues.append(f"{count_check['paragraphs_outside_range']} paragraphs outside target range ({count_check['target_min']}-{count_check['target_max']} sentences)")
        if count_check["avg_sentences_per_paragraph"] < count_check["target_min"]:
            issues.append(f"Average sentence count ({count_check['avg_sentences_per_paragraph']:.1f}) below target minimum ({count_check['target_min']})")
        elif count_check["avg_sentences_per_paragraph"] > count_check["target_max"]:
            issues.append(f"Average sentence count ({count_check['avg_sentences_per_paragraph']:.1f}) above target maximum ({count_check['target_max']})")

        return FunctionResult(
            success=meets_threshold,
            score=score,
            message=f"Sentence Count: {count_check['avg_sentences_per_paragraph']:.1f} avg per paragraph (target: {count_check['target_min']}-{count_check['target_max']}) - {'PASS' if meets_threshold else 'WARN'}",
            metadata={
                "sentence_counts": count_check,
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

