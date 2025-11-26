#!/usr/bin/env python3
"""
Formatting Plugin - Character-level enforcement

Enforces document formatting standards (headers, footers, margins, spacing, font consistency)
Note: This plugin primarily validates text-based formatting indicators since we're working with plain text
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class FormattingPlugin(BaseFeaturePlugin):
    """Plugin for enforcing document formatting standards."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "formatting", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("FormattingPlugin initialized")

    def _check_headers(self, text: str) -> Dict[str, Any]:
        """Check for proper header formatting."""
        # Look for section headers (##, ###, numbered sections, etc.)
        header_patterns = [
            r'^#{1,3}\s+.+$',  # Markdown headers
            r'^(?:Section|Part|Chapter)\s+\d+\.?\s+',  # Numbered sections
            r'^[IVX]+\.\s+',  # Roman numerals
            r'^[A-Z][A-Z\s]+:?\s*$'  # ALL CAPS headers
        ]

        headers_found = []
        for pattern in header_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            headers_found.extend([m.group() for m in matches])

        return {
            "headers_found": len(headers_found),
            "headers": headers_found[:10]
        }

    def _check_consistent_spacing(self, text: str) -> Dict[str, Any]:
        """Check for consistent spacing (double newlines between paragraphs, etc.)."""
        # Check for double newlines (paragraph breaks)
        double_newlines = len(re.findall(r'\n\s*\n', text))

        # Check for inconsistent spacing (triple+ newlines, single newlines where double expected)
        triple_plus_newlines = len(re.findall(r'\n\s*\n\s*\n+', text))

        # Check for inconsistent line spacing
        lines = text.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        avg_line_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0

        return {
            "double_newlines": double_newlines,
            "triple_plus_newlines": triple_plus_newlines,
            "avg_line_length": avg_line_length,
            "consistent": triple_plus_newlines == 0
        }

    def _check_structure(self, text: str) -> Dict[str, Any]:
        """Check for document structure elements."""
        # Check for title/caption
        has_title = bool(re.search(r'^(?:In\s+re|Case\s+No\.|Civil\s+Action)', text, re.MULTILINE | re.IGNORECASE))

        # Check for numbered sections
        has_numbered_sections = bool(re.search(r'^(?:Section|Part|Chapter)\s+\d+', text, re.MULTILINE))

        # Check for table of contents indicators
        has_toc = bool(re.search(r'Table\s+of\s+Contents', text, re.IGNORECASE))

        return {
            "has_title": has_title,
            "has_numbered_sections": has_numbered_sections,
            "has_toc": has_toc
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate formatting."""
        header_check = self._check_headers(text)
        spacing_check = self._check_consistent_spacing(text)
        structure_check = self._check_structure(text)

        score = 0.0
        issues = []

        # Headers (30%)
        if header_check["headers_found"] > 0:
            score += 0.3
        else:
            issues.append("No section headers found")

        # Consistent spacing (40%)
        if spacing_check["consistent"]:
            score += 0.4
        else:
            issues.append(f"Inconsistent spacing: {spacing_check['triple_plus_newlines']} instances of triple+ newlines")

        # Structure (30%)
        structure_score = 0.0
        if structure_check["has_title"]:
            structure_score += 0.1
        if structure_check["has_numbered_sections"]:
            structure_score += 0.1
        if structure_check["has_toc"]:
            structure_score += 0.1
        score += structure_score

        if not structure_check["has_title"]:
            issues.append("Title/caption not found")

        meets_threshold = score >= 0.85

        return FunctionResult(
            success=meets_threshold,
            score=score,
            message=f"Formatting: {header_check['headers_found']} headers, spacing {'consistent' if spacing_check['consistent'] else 'inconsistent'} - {'PASS' if meets_threshold else 'WARN'}",
            metadata={
                "headers": header_check,
                "spacing": spacing_check,
                "structure": structure_check,
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

