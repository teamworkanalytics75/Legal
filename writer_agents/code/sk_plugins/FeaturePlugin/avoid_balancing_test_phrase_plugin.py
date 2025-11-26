#!/usr/bin/env python3
"""
Avoid Balancing Test Phrase Plugin - Detects explicit 'balancing test' phrase.
Use balance concepts instead.

Impact: -16.4pp (18.1% vs 34.5% success)
Importance: 1.37
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


class AvoidBalancingTestPhrasePlugin(BaseFeaturePlugin):
    """Plugin to detect and flag explicit 'balancing test' phrase usage."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "avoid_balancing_test_phrase", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("AvoidBalancingTestPhrasePlugin initialized")

    async def validate_avoid_balancing_test_phrase(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate that explicit 'balancing test' phrase is avoided.

        CRITICAL: Avoid explicit 'balancing test' phrase.
        Use balance concepts ('weigh', 'balance of equities') instead.
        """
        text = document.get_full_text()
        text_lower = text.lower()

        issues = []
        recommendations = []

        # Bad phrases to avoid
        bad_phrases = [
            r'\bbalancing\s+test\b',
            r'\bunder\s+the\s+balancing\s+test\b',
            r'\bapplying\s+the\s+balancing\s+test\b',
            r'\bthe\s+balancing\s+test\s+requires\b',
        ]

        for pattern in bad_phrases:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            for match in matches:
                issues.append({
                    "type": "balancing_test_phrase_detected",
                    "severity": "CRITICAL",
                    "message": f"Found explicit 'balancing test' phrase at position {match.start()}",
                    "location": DocumentLocation(
                        start_line=text[:match.start()].count('\n') + 1,
                        end_line=text[:match.start()].count('\n') + 1,
                        start_char=match.start(),
                        end_char=match.end()
                    ),
                    "suggestion": "Replace with balance concepts: 'weigh the interests', 'balance of equities', 'competing interests'"
                })

        # Check for good alternatives
        good_phrases = [
            "weigh the interests",
            "balance of equities",
            "competing interests",
            "weighing the",
        ]

        found_good = []
        for phrase in good_phrases:
            if phrase.lower() in text_lower:
                found_good.append(phrase)

        if issues:
            recommendations.append({
                "type": "replace_balancing_test",
                "message": "Remove explicit 'balancing test' phrase. Use balance concepts instead: 'weigh the interests', 'balance of equities', 'competing interests'",
                "priority": "CRITICAL"
            })
        elif not found_good:
            recommendations.append({
                "type": "add_balance_concepts",
                "message": "Consider using balance concepts: 'weigh the interests', 'balance of equities', 'competing interests'",
                "priority": "MEDIUM"
            })

        return FunctionResult(
            success=True,
            value={
                "issues_found": len(issues),
                "good_phrases_found": len(found_good),
                "issues": issues,
                "recommendations": recommendations,
                "impact": "-16.4pp (18.1% vs 34.5% success)"
            },
            metadata={
                "feature_name": "avoid_balancing_test_phrase",
                "priority": "CRITICAL",
                "importance": 1.37
            }
        )

