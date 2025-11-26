#!/usr/bin/env python3
"""
Character Count Plugin - Character-level enforcement

Enforces character count limits at document, paragraph, and line levels
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class CharacterCountPlugin(BaseFeaturePlugin):
    """Plugin for enforcing character count limits."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "character_count", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("CharacterCountPlugin initialized")

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_lines(self, text: str) -> List[str]:
        """Split text into lines."""
        lines = text.split('\n')
        return [l.strip() for l in lines if l.strip()]

    def _analyze_character_counts(self, text: str) -> Dict[str, Any]:
        """Analyze character counts at all levels."""
        criteria = self.rules.get("validation_criteria", {})

        # Document level
        doc_char_count = len(text)
        doc_criteria = criteria.get("document", {})

        # Paragraph level
        paragraphs = self._split_paragraphs(text)
        para_char_counts = [len(p) for p in paragraphs]
        para_criteria = criteria.get("paragraph", {})

        # Line level
        lines = self._split_lines(text)
        line_char_counts = [len(l) for l in lines]
        line_criteria = criteria.get("line", {})

        # Check long lines
        max_line_chars = line_criteria.get("target_max", 80)
        long_lines = [i for i, count in enumerate(line_char_counts) if count > max_line_chars]

        return {
            "document": {
                "char_count": doc_char_count,
                "within_range": para_criteria.get("target_range", [0, float('inf')])[0] <= doc_char_count <= para_criteria.get("target_range", [0, float('inf')])[1],
                "criteria": doc_criteria
            },
            "paragraphs": {
                "count": len(paragraphs),
                "char_counts": para_char_counts,
                "avg_chars": sum(para_char_counts) / len(para_char_counts) if para_char_counts else 0,
                "within_range_count": len([c for c in para_char_counts if para_criteria.get("target_range", [0, float('inf')])[0] <= c <= para_criteria.get("target_range", [0, float('inf')])[1]]),
                "criteria": para_criteria
            },
            "lines": {
                "count": len(lines),
                "char_counts": line_char_counts,
                "avg_chars": sum(line_char_counts) / len(line_char_counts) if line_char_counts else 0,
                "max_chars": max(line_char_counts) if line_char_counts else 0,
                "long_lines_count": len(long_lines),
                "target_max": max_line_chars,
                "criteria": line_criteria
            }
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate character counts."""
        char_check = self._analyze_character_counts(text)

        score = 0.0
        issues = []

        # Document level (40%)
        doc = char_check["document"]
        if doc["within_range"]:
            score += 0.4
        else:
            target_range = doc["criteria"].get("target_range", [0, float('inf')])
            if doc["char_count"] < target_range[0]:
                issues.append(f"Document character count ({doc['char_count']:,}) below target minimum ({target_range[0]:,})")
            elif doc["char_count"] > target_range[1]:
                issues.append(f"Document character count ({doc['char_count']:,}) above target maximum ({target_range[1]:,})")

        # Paragraph level (40%)
        paras = char_check["paragraphs"]
        if paras["count"] > 0:
            para_ratio = paras["within_range_count"] / paras["count"]
            score += 0.4 * para_ratio
            if para_ratio < 0.8:
                issues.append(f"Only {paras['within_range_count']}/{paras['count']} paragraphs within character count range")

        # Line level (20%)
        lines = char_check["lines"]
        if lines["count"] > 0:
            long_line_ratio = lines["long_lines_count"] / lines["count"]
            if long_line_ratio < 0.1:  # Less than 10% long lines
                score += 0.2
            else:
                score += 0.2 * (1 - long_line_ratio)
                issues.append(f"{lines['long_lines_count']} lines exceed target maximum ({lines['target_max']} chars)")

        meets_threshold = score >= 0.85

        return FunctionResult(
            success=meets_threshold,
            score=score,
            message=f"Character Count: Doc={char_check['document']['char_count']:,} chars, Para avg={char_check['paragraphs']['avg_chars']:.0f}, Line max={char_check['lines']['max_chars']} - {'PASS' if meets_threshold else 'WARN'}",
            metadata={
                "character_counts": char_check,
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

