#!/usr/bin/env python3
"""
Statutory Requirement 3 Plugin - Interested Person/Applicant

Enforces Section 1782 statutory requirement:
(3) the application is made by a foreign or international tribunal or 'any interested person'
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class StatutoryRequirement3Plugin(BaseFeaturePlugin):
    """Plugin for enforcing Section 1782 statutory requirement 3: interested person/applicant."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "statutory_requirement_3_interested_person", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("StatutoryRequirement3Plugin initialized")

    def _check_applicant_identified(self, text: str) -> Dict[str, Any]:
        """Check if applicant is identified."""
        patterns = [
            r'applicant',
            r'petitioner',
            r'plaintiff',
            r'undersigned',
            r'I\s+.*?apply',
            r'application\s+is\s+made\s+by'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "applicant_identified": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_interest_established(self, text: str) -> Dict[str, Any]:
        """Check if applicant's interest in foreign proceeding is established."""
        patterns = [
            r'interested\s+person',
            r'party\s+to\s+the\s+foreign\s+proceeding',
            r'party\s+to\s+.*?Hong\s+Kong',
            r'plaintiff\s+in\s+.*?proceeding',
            r'defendant\s+in\s+.*?proceeding',
            r'reasonable\s+interest',
            r'interest\s+in\s+.*?proceeding'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "interest_established": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_party_status(self, text: str) -> Dict[str, Any]:
        """Check if applicant is identified as party or has right to submit information."""
        patterns = [
            r'party\s+to\s+.*?proceeding',
            r'plaintiff\s+in',
            r'defendant\s+in',
            r'right\s+to\s+submit\s+information',
            r'entitled\s+to\s+submit',
            r'authorized\s+to\s+submit'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "party_status_established": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate that statutory requirement 3 is met."""
        applicant_check = self._check_applicant_identified(text)
        interest_check = self._check_interest_established(text)
        party_check = self._check_party_status(text)

        all_passed = (
            applicant_check["applicant_identified"] and
            (interest_check["interest_established"] or party_check["party_status_established"])
        )

        score = 0.0
        if applicant_check["applicant_identified"]:
            score += 0.4
        if interest_check["interest_established"]:
            score += 0.3
        if party_check["party_status_established"]:
            score += 0.3

        issues = []
        if not applicant_check["applicant_identified"]:
            issues.append("Applicant not clearly identified")
        if not interest_check["interest_established"] and not party_check["party_status_established"]:
            issues.append("Applicant's interest in foreign proceeding not established, and party status not demonstrated")

        return FunctionResult(
            success=all_passed,
            score=score,
            message=f"Statutory Requirement 3: Interested Person - {'PASS' if all_passed else 'FAIL'}",
            metadata={
                "applicant_identified": applicant_check,
                "interest_established": interest_check,
                "party_status_established": party_check,
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

