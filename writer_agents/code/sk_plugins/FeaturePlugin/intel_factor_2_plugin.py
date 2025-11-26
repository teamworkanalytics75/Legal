#!/usr/bin/env python3
"""
Intel Factor 2 Plugin - Receptivity of Foreign Tribunal

Enforces Intel Factor 2:
The nature of the foreign tribunal, the character of the proceedings underway abroad,
and the receptivity of the foreign government or the court or agency abroad to U.S.
federal-court judicial assistance
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class IntelFactor2Plugin(BaseFeaturePlugin):
    """Plugin for enforcing Intel Factor 2: receptivity of foreign tribunal."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "intel_factor_2_receptivity", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("IntelFactor2Plugin initialized")

    def _check_tribunal_identified(self, text: str) -> Dict[str, Any]:
        """Check if foreign tribunal is identified."""
        patterns = [
            r'foreign\s+tribunal',
            r'Hong\s+Kong\s+(?:court|tribunal|judiciary)',
            r'Court.*?Hong\s+Kong',
            r'High\s+Court.*?Hong\s+Kong'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "tribunal_identified": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_receptivity_evidence(self, text: str) -> Dict[str, Any]:
        """Check if evidence of receptivity is provided."""
        patterns = [
            r'Hague\s+Evidence\s+Convention',
            r'receptive\s+to\s+U\.S\.',
            r'receptive\s+to\s+U\.S\.\s+judicial\s+assistance',
            r'receptivity',
            r'judicial\s+assistance',
            r'signatory\s+to.*?Hague',
            r'O\'Keeffe.*?receptive',
            r'Pishevar.*?receptive',
            r'Hong\s+Kong.*?receptive'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "receptivity_evidence_provided": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_hague_convention_mentioned(self, text: str) -> Dict[str, Any]:
        """Check if Hague Evidence Convention is mentioned (especially for Hong Kong)."""
        patterns = [
            r'Hague\s+Evidence\s+Convention',
            r'Hague.*?Convention.*?Evidence',
            r'Convention.*?Evidence.*?Hague'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        hong_kong_mentioned = bool(re.search(r'Hong\s+Kong', text, re.IGNORECASE))

        return {
            "hague_convention_mentioned": len(matches) > 0,
            "hong_kong_mentioned": hong_kong_mentioned,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate that Intel Factor 2 is addressed."""
        tribunal_check = self._check_tribunal_identified(text)
        receptivity_check = self._check_receptivity_evidence(text)
        hague_check = self._check_hague_convention_mentioned(text)

        # Factor 2 is addressed if receptivity is discussed or evidence provided
        addressed = (
            tribunal_check["tribunal_identified"] and
            (receptivity_check["receptivity_evidence_provided"] or hague_check["hague_convention_mentioned"])
        )

        score = 0.0
        if tribunal_check["tribunal_identified"]:
            score += 0.3
        if receptivity_check["receptivity_evidence_provided"]:
            score += 0.4
        if hague_check["hague_convention_mentioned"]:
            score += 0.3

        issues = []
        if not tribunal_check["tribunal_identified"]:
            issues.append("Foreign tribunal not clearly identified")
        if not receptivity_check["receptivity_evidence_provided"] and not hague_check["hague_convention_mentioned"]:
            issues.append("Evidence of receptivity not provided (should cite Hague Evidence Convention for Hong Kong)")

        return FunctionResult(
            success=addressed,
            score=min(score, 1.0),
            message=f"Intel Factor 2: Receptivity - {'PASS' if addressed else 'FAIL'}",
            metadata={
                "tribunal_identified": tribunal_check,
                "receptivity_evidence": receptivity_check,
                "hague_convention": hague_check,
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

