#!/usr/bin/env python3
"""
Avoid Compelling Interest Phrase Plugin - Detects explicit 'compelling interest' phrase.
Show it, don't say it.

Impact: -27.8 percentage points (7.7% success WITH vs 35.5% WITHOUT)
Importance: 0.23
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


class AvoidCompellingInterestPhrasePlugin(BaseFeaturePlugin):
    """Plugin to detect explicit 'compelling interest' phrase usage."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "avoid_compelling_interest_phrase", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("AvoidCompellingInterestPhrasePlugin initialized")

    async def validate_avoid_compelling_interest_phrase(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate that explicit 'compelling interest' phrase is avoided.

        MEDIUM: Show compelling interest through facts, don't explicitly say 'compelling interest'.
        Strong negative predictor (-27.8pp).
        """
        text = document.get_full_text()
        text_lower = text.lower()

        issues = []
        recommendations = []

        # Bad phrases to avoid
        bad_phrases = [
            r'\bcompelling\s+interest\b',
            r'\bcompelling\s+governmental?\s+interest\b',
        ]

        for pattern in bad_phrases:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            for match in matches:
                issues.append({
                    "type": "compelling_interest_phrase_detected",
                    "severity": "MEDIUM",
                    "message": f"Found explicit 'compelling interest' phrase at position {match.start()}",
                    "location": DocumentLocation(
                        start_line=text[:match.start()].count('\n') + 1,
                        end_line=text[:match.start()].count('\n') + 1,
                        start_char=match.start(),
                        end_char=match.end()
                    ),
                    "suggestion": "Show compelling interest through facts and evidence, rather than stating it explicitly."
                })

        if issues:
            recommendations.append({
                "type": "show_compelling_interest",
                "message": "Show compelling interest through facts and evidence, don't explicitly say 'compelling interest'. Strong negative predictor (-27.8pp).",
                "priority": "MEDIUM"
            })

        return FunctionResult(
            success=True,
            value={
                "issues_found": len(issues),
                "issues": issues,
                "recommendations": recommendations,
                "impact": "-27.8pp (7.7% vs 35.5% success)"
            },
            metadata={
                "feature_name": "avoid_compelling_interest_phrase",
                "priority": "MEDIUM",
                "importance": 0.23
            }
        )

