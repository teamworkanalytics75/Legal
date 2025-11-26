#!/usr/bin/env python3
"""
Public Interest Plugin - Atomic SK plugin for public interest feature.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class PublicInterestPlugin(BaseFeaturePlugin):
    """Atomic plugin for public interest feature analysis."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "mentions_public_interest", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("PublicInterestPlugin initialized")

    async def validate_avoid_public_interest(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate that public interest language is avoided.

        CRITICAL: Avoid 'public interest' language entirely.
        Strongest negative predictor (-40.9pp, 52.97 importance).
        """
        from ..base_plugin import DocumentLocation
        import re

        text = document.get_full_text()
        text_lower = text.lower()

        issues = []
        recommendations = []

        # Public interest terms to avoid
        public_interest_terms = [
            "public interest",
            "public access",
            "transparency",
            "open court",
            "public's right to know",
            "public right",
            "public concern",
            "public matter",
            "open courts",
        ]

        public_count = 0
        for term in public_interest_terms:
            matches = list(re.finditer(re.escape(term.lower()), text_lower))
            for match in matches:
                public_count += 1
                issues.append({
                    "type": "public_interest_detected",
                    "severity": "CRITICAL",
                    "message": f"Found '{term}' at position {match.start()}",
                    "location": DocumentLocation(
                        start_line=text[:match.start()].count('\n') + 1,
                        end_line=text[:match.start()].count('\n') + 1,
                        start_char=match.start(),
                        end_char=match.end()
                    ),
                    "suggestion": "Remove 'public interest' language. Strongest negative predictor (-40.9pp). DO NOT mention public interest at all."
                })

        if public_count > 0:
            recommendations.append({
                "type": "remove_public_interest",
                "message": "CRITICAL: Remove all 'public interest' language. Strongest negative predictor (-40.9pp, 52.97 importance). DO NOT mention public interest at all. Focus on your compelling interest instead.",
                "priority": "CRITICAL"
            })

        return FunctionResult(
            success=True,
            value={
                "public_interest_mentions": public_count,
                "issues_found": len(issues),
                "issues": issues,
                "recommendations": recommendations,
                "impact": "-40.9pp (8.2% vs 49.1% success) - MOST IMPORTANT"
            },
            metadata={
                "analysis_type": "avoid_public_interest",
                "importance": 52.97,
                "priority": "CRITICAL"
            }
        )

    async def analyze_public_interest_balance(self, draft_text: str) -> FunctionResult:
        """DEPRECATED: Use validate_avoid_public_interest instead. This method kept for backwards compatibility."""
        # Redirect to new validation method
        return await self.validate_avoid_public_interest(None, {"draft_text": draft_text})
