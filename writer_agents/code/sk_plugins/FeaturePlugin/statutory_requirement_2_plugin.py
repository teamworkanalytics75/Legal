#!/usr/bin/env python3
"""
Statutory Requirement 2 Plugin - Foreign Proceeding Exists

Enforces Section 1782 statutory requirement:
(2) the discovery is for use in a proceeding before a foreign tribunal
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class StatutoryRequirement2Plugin(BaseFeaturePlugin):
    """Plugin for enforcing Section 1782 statutory requirement 2: foreign proceeding exists."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "statutory_requirement_2_foreign_proceeding", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("StatutoryRequirement2Plugin initialized")

    def _check_foreign_proceeding_identified(self, text: str) -> Dict[str, Any]:
        """Check if foreign proceeding is identified."""
        patterns = [
            r'foreign\s+proceeding',
            r'proceeding\s+before\s+a\s+foreign\s+tribunal',
            r'foreign\s+litigation',
            r'Hong\s+Kong\s+(?:litigation|proceeding|action|case)',
            r'use\s+in\s+a\s+proceeding'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "proceeding_identified": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_foreign_tribunal_specified(self, text: str) -> Dict[str, Any]:
        """Check if foreign tribunal is specified."""
        patterns = [
            r'foreign\s+tribunal',
            r'Hong\s+Kong\s+(?:court|tribunal|judiciary)',
            r'Court.*?Hong\s+Kong',
            r'High\s+Court.*?Hong\s+Kong',
            r'District\s+Court.*?Hong\s+Kong'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "tribunal_specified": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_proceeding_status(self, text: str) -> Dict[str, Any]:
        """Check if proceeding status is specified (pending or reasonably contemplated)."""
        patterns = [
            r'pending',
            r'reasonably\s+contemplated',
            r'contemplated',
            r'proceeding\s+is\s+pending',
            r'proceeding\s+.*?\s+pending',
            r'currently\s+pending',
            r'active\s+proceeding'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "status_specified": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_evidence_provided(self, text: str) -> Dict[str, Any]:
        """Check if evidence of proceeding is provided."""
        evidence_indicators = [
            r'case\s+number',
            r'docket\s+number',
            r'filed\s+in',
            r'filed\s+with',
            r'statement\s+of\s+claim',
            r'complaint',
            r'exhibit',
            r'evidence\s+of\s+proceeding',
            r'proof\s+of\s+proceeding'
        ]

        matches = []
        for pattern in evidence_indicators:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "evidence_provided": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate that statutory requirement 2 is met."""
        proceeding_check = self._check_foreign_proceeding_identified(text)
        tribunal_check = self._check_foreign_tribunal_specified(text)
        status_check = self._check_proceeding_status(text)
        evidence_check = self._check_evidence_provided(text)

        all_passed = (
            proceeding_check["proceeding_identified"] and
            tribunal_check["tribunal_specified"] and
            (status_check["status_specified"] or evidence_check["evidence_provided"])
        )

        score = 0.0
        if proceeding_check["proceeding_identified"]:
            score += 0.3
        if tribunal_check["tribunal_specified"]:
            score += 0.3
        if status_check["status_specified"]:
            score += 0.2
        if evidence_check["evidence_provided"]:
            score += 0.2

        issues = []
        if not proceeding_check["proceeding_identified"]:
            issues.append("Foreign proceeding not clearly identified")
        if not tribunal_check["tribunal_specified"]:
            issues.append("Foreign tribunal not specified")
        if not status_check["status_specified"] and not evidence_check["evidence_provided"]:
            issues.append("Proceeding status (pending or reasonably contemplated) not specified, and no evidence provided")

        return FunctionResult(
            success=all_passed,
            score=score,
            message=f"Statutory Requirement 2: Foreign Proceeding - {'PASS' if all_passed else 'FAIL'}",
            metadata={
                "proceeding_identified": proceeding_check,
                "tribunal_specified": tribunal_check,
                "status_specified": status_check,
                "evidence_provided": evidence_check,
                "issues": issues,
                "threshold": self.rules.get("threshold", 1.0)
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

