#!/usr/bin/env python3
"""
Avoid First Amendment Plugin - Detects and flags First Amendment language.
Second strongest negative predictor (-37.1pp).

Impact: -37.1 percentage points (7.5% success WITH vs 44.6% WITHOUT)
Importance: 35.84 (2nd most important)
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


class AvoidFirstAmendmentPlugin(BaseFeaturePlugin):
    """Plugin to detect and flag First Amendment language."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "avoid_first_amendment", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("AvoidFirstAmendmentPlugin initialized")

    async def validate_avoid_first_amendment(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate that First Amendment language is avoided.

        CRITICAL: Avoid First Amendment language entirely.
        Second strongest negative predictor (-37.1pp).
        """
        text = document.get_full_text()
        text_lower = text.lower()

        issues = []
        recommendations = []

        # Bad phrases to avoid
        bad_phrases = [
            "first amendment",
            "free speech",
            "freedom of speech",
            "freedom of the press",
            "expressive rights",
            "first amendment rights",
            "first amendment protection"
        ]

        for phrase in bad_phrases:
            if phrase.lower() in text_lower:
                matches = list(re.finditer(re.escape(phrase.lower()), text_lower))
                for match in matches:
                    issues.append({
                        "type": "first_amendment_detected",
                        "severity": "CRITICAL",
                        "message": f"Found '{phrase}' at position {match.start()}",
                        "location": DocumentLocation(
                            start_line=text[:match.start()].count('\n') + 1,
                            end_line=text[:match.start()].count('\n') + 1,
                            start_char=match.start(),
                            end_char=match.end()
                        ),
                        "suggestion": "Remove First Amendment language. Frame as privacy, safety, or national security instead."
                    })

        if issues:
            recommendations.append({
                "type": "remove_first_amendment",
                "message": "Remove all First Amendment references. Second strongest negative predictor (-37.1pp). Frame issues as privacy, safety, or national security instead.",
                "priority": "CRITICAL"
            })

        return FunctionResult(
            success=True,
            value={
                "issues_found": len(issues),
                "issues": issues,
                "recommendations": recommendations,
                "impact": "-37.1pp (7.5% vs 44.6% success) - 2ND MOST IMPORTANT"
            },
            metadata={
                "feature_name": "avoid_first_amendment",
                "priority": "CRITICAL",
                "importance": 35.84
            }
        )

