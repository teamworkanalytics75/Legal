#!/usr/bin/env python3
"""
Word Count Plugin - Word-level enforcement

Enforces word count at all levels (document, section, paragraph, sentence)
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class WordCountPlugin(BaseFeaturePlugin):
    """Plugin for enforcing word count at all levels."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "word_count", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("WordCountPlugin initialized")

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        words = re.findall(r'\b\w+\b', text)
        return len(words)

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_sections(self, text: str) -> List[str]:
        """Split text into sections (by headers)."""
        # Look for section headers (##, ###, numbered sections, etc.)
        section_pattern = r'(?:^|\n)(?:#{1,3}\s+|(?:Section|Part|Chapter)\s+\d+\.?\s+|[IVX]+\.\s+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:)\s*'
        sections = re.split(section_pattern, text, flags=re.MULTILINE)
        return [s.strip() for s in sections if s.strip() and len(s.strip()) > 100]

    def _analyze_word_counts(self, text: str) -> Dict[str, Any]:
        """Analyze word counts at all levels."""
        criteria = self.rules.get("validation_criteria", {})

        # Document level
        doc_criteria = criteria.get("document", {})
        doc_word_count = self._count_words(text)

        # Section level
        sections = self._split_sections(text)
        section_word_counts = [self._count_words(s) for s in sections]
        section_criteria = criteria.get("section", {})

        # Paragraph level
        paragraphs = self._split_paragraphs(text)
        paragraph_word_counts = [self._count_words(p) for p in paragraphs]
        para_criteria = criteria.get("paragraph", {})

        # Sentence level
        sentences = self._split_sentences(text)
        sentence_word_counts = [self._count_words(s) for s in sentences]
        sent_criteria = criteria.get("sentence", {})

        return {
            "document": {
                "word_count": doc_word_count,
                "within_range": doc_criteria.get("target_range", [0, float('inf')])[0] <= doc_word_count <= doc_criteria.get("target_range", [0, float('inf')])[1],
                "criteria": doc_criteria
            },
            "sections": {
                "count": len(sections),
                "word_counts": section_word_counts,
                "avg_words": sum(section_word_counts) / len(section_word_counts) if section_word_counts else 0,
                "within_range_count": len([w for w in section_word_counts if para_criteria.get("target_range", [0, float('inf')])[0] <= w <= para_criteria.get("target_range", [0, float('inf')])[1]]),
                "criteria": section_criteria
            },
            "paragraphs": {
                "count": len(paragraphs),
                "word_counts": paragraph_word_counts,
                "avg_words": sum(paragraph_word_counts) / len(paragraph_word_counts) if paragraph_word_counts else 0,
                "within_range_count": len([w for w in paragraph_word_counts if para_criteria.get("target_range", [0, float('inf')])[0] <= w <= para_criteria.get("target_range", [0, float('inf')])[1]]),
                "criteria": para_criteria
            },
            "sentences": {
                "count": len(sentences),
                "word_counts": sentence_word_counts,
                "avg_words": sum(sentence_word_counts) / len(sentence_word_counts) if sentence_word_counts else 0,
                "within_range_count": len([w for w in sentence_word_counts if sent_criteria.get("target_range", [0, float('inf')])[0] <= w <= sent_criteria.get("target_range", [0, float('inf')])[1]]),
                "criteria": sent_criteria
            }
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate word counts at all levels."""
        count_check = self._analyze_word_counts(text)

        # Calculate overall score
        score = 0.0
        issues = []

        # Document level (40%)
        doc = count_check["document"]
        if doc["within_range"]:
            score += 0.4
        else:
            target_range = doc["criteria"].get("target_range", [0, float('inf')])
            if doc["word_count"] < target_range[0]:
                issues.append(f"Document word count ({doc['word_count']}) below target minimum ({target_range[0]})")
            elif doc["word_count"] > target_range[1]:
                issues.append(f"Document word count ({doc['word_count']}) above target maximum ({target_range[1]})")

        # Paragraph level (30%)
        paras = count_check["paragraphs"]
        if paras["count"] > 0:
            para_ratio = paras["within_range_count"] / paras["count"]
            score += 0.3 * para_ratio
            if para_ratio < 0.8:
                issues.append(f"Only {paras['within_range_count']}/{paras['count']} paragraphs within word count range")

        # Sentence level (20%)
        sents = count_check["sentences"]
        if sents["count"] > 0:
            sent_ratio = sents["within_range_count"] / sents["count"]
            score += 0.2 * sent_ratio
            if sent_ratio < 0.8:
                issues.append(f"Only {sents['within_range_count']}/{sents['count']} sentences within word count range")

        # Section level (10%)
        secs = count_check["sections"]
        if secs["count"] > 0:
            sec_ratio = secs["within_range_count"] / secs["count"]
            score += 0.1 * sec_ratio

        meets_threshold = score >= 0.85

        return FunctionResult(
            success=meets_threshold,
            score=score,
            message=f"Word Count: Doc={count_check['document']['word_count']} words, Para avg={count_check['paragraphs']['avg_words']:.1f}, Sent avg={count_check['sentences']['avg_words']:.1f} - {'PASS' if meets_threshold else 'WARN'}",
            metadata={
                "word_counts": count_check,
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

