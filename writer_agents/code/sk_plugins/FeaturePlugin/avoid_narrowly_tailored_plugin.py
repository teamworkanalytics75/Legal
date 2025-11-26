#!/usr/bin/env python3
"""
Avoid Narrowly Tailored Plugin - Detects and flags 'narrowly tailored' language.
Strong negative predictor (-28.4pp).

Impact: -28.4 percentage points (7.1% success WITH vs 35.5% WITHOUT)
Importance: 0.73
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


class AvoidNarrowlyTailoredPlugin(BaseFeaturePlugin):
    """Plugin to detect and flag 'narrowly tailored' language."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "avoid_narrowly_tailored", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("AvoidNarrowlyTailoredPlugin initialized")

    async def validate_avoid_narrowly_tailored(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate that 'narrowly tailored' language is avoided.

        MEDIUM: Avoid 'narrowly tailored' language unless you can prove it.
        Strong negative predictor (-28.4pp).
        """
        text = document.get_full_text()
        text_lower = text.lower()

        issues = []
        recommendations = []

        # Bad phrases to avoid
        bad_phrases = [
            r'\bnarrowly\s+tailored\b',
            r'\bleast\s+restrictive\s+alternative\b',
            r'\bnarrow\s+tailoring\b',
        ]

        for pattern in bad_phrases:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            for match in matches:
                issues.append({
                    "type": "narrowly_tailored_detected",
                    "severity": "MEDIUM",
                    "message": f"Found 'narrowly tailored' language at position {match.start()}",
                    "location": DocumentLocation(
                        start_line=text[:match.start()].count('\n') + 1,
                        end_line=text[:match.start()].count('\n') + 1,
                        start_char=match.start(),
                        end_char=match.end()
                    ),
                    "suggestion": "Remove 'narrowly tailored' language unless you can definitively prove it. This language often appears in denied cases."
                })

        if issues:
            recommendations.append({
                "type": "remove_narrowly_tailored",
                "message": "Avoid 'narrowly tailored' language. Strong negative predictor (-28.4pp). Only use if you can definitively prove it.",
                "priority": "MEDIUM"
            })

        return FunctionResult(
            success=True,
            value={
                "issues_found": len(issues),
                "issues": issues,
                "recommendations": recommendations,
                "impact": "-28.4pp (7.1% vs 35.5% success)"
            },
            metadata={
                "feature_name": "avoid_narrowly_tailored",
                "priority": "MEDIUM",
                "importance": 0.73
            }
        )

