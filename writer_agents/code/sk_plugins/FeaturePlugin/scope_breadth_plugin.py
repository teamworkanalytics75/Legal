#!/usr/bin/env python3
"""
Scope Breadth Plugin - CatBoost Feature Enforcement

Enforces narrow scope (negative signal - minimize breadth)
Broad scope correlates with denial, narrow scope with favorable outcomes
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class ScopeBreadthPlugin(BaseFeaturePlugin):
    """Plugin for enforcing narrow scope (CatBoost negative signal)."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "scope_breadth", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("ScopeBreadthPlugin initialized")

    def _check_time_period_specificity(self, text: str) -> Dict[str, Any]:
        """Check if time period is narrow and specific."""
        positive_patterns = [
            r'between.*?\d{4}.*?and.*?\d{4}',
            r'from.*?\d{4}.*?to.*?\d{4}',
            r'January.*?\d{4}.*?December.*?\d{4}',
            r'specific\s+time\s+period',
            r'limited\s+time\s+period',
            r'date\s+range.*?\d{4}',
            r'from.*?date.*?to.*?date'
        ]

        negative_patterns = [
            r'all\s+time',
            r'any\s+time',
            r'unlimited',
            r'no\s+time\s+limit',
            r'all\s+years',
            r'any\s+date'
        ]

        positive_matches = []
        for pattern in positive_patterns:
            positive_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        negative_matches = []
        for pattern in negative_patterns:
            negative_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        # Check if negative matches are negated
        negated_negative = []
        for match in negative_matches:
            start = max(0, match.start() - 20)
            context = text[start:match.end()]
            if re.search(r'\b(not|no|neither|nor)\b', context, re.IGNORECASE):
                negated_negative.append(match)

        is_specific = len(positive_matches) > 0 and (len(negative_matches) == 0 or len(negated_negative) == len(negative_matches))

        return {
            "time_period_specific": is_specific,
            "positive_match_count": len(positive_matches),
            "negative_match_count": len(negative_matches),
            "negated_negative_count": len(negated_negative),
            "matches": [m.group() for m in positive_matches[:5]]
        }

    def _check_document_categories_specificity(self, text: str) -> Dict[str, Any]:
        """Check if document categories are specific."""
        positive_patterns = [
            r'specific\s+document\s+categories',
            r'particular\s+documents',
            r'specific\s+types\s+of\s+documents',
            r'document\s+category\s+\d+',
            r'category\s+\d+',
            r'request\s+for\s+production\s+\d+',
            r'RFP\s+\d+'
        ]

        negative_patterns = [
            r'all\s+documents',
            r'any\s+documents',
            r'all\s+records',
            r'any\s+records',
            r'all\s+files',
            r'any\s+files',
            r'all\s+information',
            r'any\s+information'
        ]

        positive_matches = []
        for pattern in positive_patterns:
            positive_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        negative_matches = []
        for pattern in negative_patterns:
            negative_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        # Check if negative matches are negated
        negated_negative = []
        for match in negative_matches:
            start = max(0, match.start() - 20)
            context = text[start:match.end()]
            if re.search(r'\b(not|no|neither|nor)\b', context, re.IGNORECASE):
                negated_negative.append(match)

        is_specific = len(positive_matches) > 0 and (len(negative_matches) == 0 or len(negated_negative) == len(negative_matches))

        return {
            "categories_specific": is_specific,
            "positive_match_count": len(positive_matches),
            "negative_match_count": len(negative_matches),
            "negated_negative_count": len(negated_negative),
            "matches": [m.group() for m in positive_matches[:5]]
        }

    def _check_search_terms_specificity(self, text: str) -> Dict[str, Any]:
        """Check if search terms are specific."""
        positive_patterns = [
            r'specific\s+search\s+terms',
            r'particular\s+keywords',
            r'specific\s+keywords',
            r'search\s+terms.*?including',
            r'keywords.*?including',
            r'search\s+for.*?term'
        ]

        matches = []
        for pattern in positive_patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "search_terms_specific": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate that scope is narrow and specific."""
        time_check = self._check_time_period_specificity(text)
        category_check = self._check_document_categories_specificity(text)
        search_check = self._check_search_terms_specificity(text)

        # Scope is narrow if time is specific and categories are specific
        is_narrow = time_check["time_period_specific"] and category_check["categories_specific"]

        score = 0.0
        if time_check["time_period_specific"]:
            score += 0.4
        if category_check["categories_specific"]:
            score += 0.4
        if search_check["search_terms_specific"]:
            score += 0.2

        issues = []
        if not time_check["time_period_specific"]:
            issues.append("Time period not limited or specific")
        if not category_check["categories_specific"]:
            issues.append("Document categories not specific (may contain 'all documents' language)")
        if not search_check["search_terms_specific"]:
            issues.append("Search terms not specified (optional but recommended)")

        return FunctionResult(
            success=is_narrow,
            score=min(score, 1.0),
            message=f"Scope Breadth: {'NARROW' if is_narrow else 'BROAD'} - {'PASS' if is_narrow else 'FAIL'}",
            metadata={
                "time_period_specificity": time_check,
                "category_specificity": category_check,
                "search_term_specificity": search_check,
                "issues": issues,
                "threshold": self.rules.get("threshold", 0.9)
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

