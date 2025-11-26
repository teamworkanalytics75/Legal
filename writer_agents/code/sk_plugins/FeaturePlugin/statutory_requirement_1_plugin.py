#!/usr/bin/env python3
"""
Statutory Requirement 1 Plugin - Person Found/Resides in District

Enforces Section 1782 statutory requirement:
(1) the person from whom discovery is sought resides (or is found) in the district
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class StatutoryRequirement1Plugin(BaseFeaturePlugin):
    """Plugin for enforcing Section 1782 statutory requirement 1: person found/resides in district."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "statutory_requirement_1_person_found", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("StatutoryRequirement1Plugin initialized")

    def _check_person_identified(self, text: str) -> Dict[str, Any]:
        """Check if discovery target is identified."""
        patterns = [
            r'respondent\s+.*?Harvard',
            r'discovery\s+target.*?Harvard',
            r'person\s+from\s+whom\s+discovery\s+is\s+sought.*?Harvard',
            r'President\s+and\s+Fellows\s+of\s+Harvard\s+College',
            r'Harvard\s+University',
            r'Harvard\s+College'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "person_identified": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_district_specified(self, text: str) -> Dict[str, Any]:
        """Check if district is specified."""
        patterns = [
            r'District\s+of\s+Massachusetts',
            r'D\.\s*Mass\.',
            r'D\.\s*Mass',
            r'District\s+Court.*?Massachusetts',
            r'United\s+States\s+District\s+Court.*?Massachusetts',
            r'resides\s+in\s+the\s+district',
            r'found\s+in\s+the\s+district',
            r'resides\s+.*?\s+district',
            r'found\s+.*?\s+district'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "district_specified": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_evidence_provided(self, text: str) -> Dict[str, Any]:
        """Check if evidence of residency/location is provided."""
        evidence_indicators = [
            r'located\s+at',
            r'principal\s+place\s+of\s+business',
            r'headquarters',
            r'address',
            r'Cambridge.*?Massachusetts',
            r'Massachusetts',
            r'evidence\s+of\s+residency',
            r'evidence\s+of\s+location',
            r'jurisdictional\s+basis'
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
        """Validate that statutory requirement 1 is met."""
        person_check = self._check_person_identified(text)
        district_check = self._check_district_specified(text)
        evidence_check = self._check_evidence_provided(text)

        all_passed = (
            person_check["person_identified"] and
            district_check["district_specified"] and
            evidence_check["evidence_provided"]
        )

        score = 0.0
        if person_check["person_identified"]:
            score += 0.4
        if district_check["district_specified"]:
            score += 0.3
        if evidence_check["evidence_provided"]:
            score += 0.3

        issues = []
        if not person_check["person_identified"]:
            issues.append("Discovery target not clearly identified")
        if not district_check["district_specified"]:
            issues.append("District where target is found/resides not specified")
        if not evidence_check["evidence_provided"]:
            issues.append("Evidence of residency/location not provided")

        return FunctionResult(
            success=all_passed,
            score=score,
            message=f"Statutory Requirement 1: Person Found/Resides - {'PASS' if all_passed else 'FAIL'}",
            metadata={
                "person_identified": person_check,
                "district_specified": district_check,
                "evidence_provided": evidence_check,
                "issues": issues,
                "threshold": self.rules.get("threshold", 1.0)
            }
        )

    async def _execute_native(self, text: str, **kwargs) -> FunctionResult:
        """Execute native validation."""
        return await self.validate(text)

    async def _execute_semantic(self, text: str, **kwargs) -> FunctionResult:
        """Execute semantic validation (can query Chroma for similar cases)."""
        # Use Chroma to find similar cases that met this requirement
        case_context = text[:500]  # First 500 chars for context
        similar_cases = await self.query_chroma(case_context, n_results=5)

        # Also run native validation
        native_result = await self.validate(text)

        # Enhance with similar case examples
        native_result.metadata["similar_cases"] = similar_cases[:3]

        return native_result

