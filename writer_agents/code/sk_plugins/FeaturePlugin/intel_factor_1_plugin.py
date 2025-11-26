#!/usr/bin/env python3
"""
Intel Factor 1 Plugin - Participant Status

Enforces Intel Factor 1:
Whether 'the person from whom discovery is sought is a participant in the foreign proceeding'
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class IntelFactor1Plugin(BaseFeaturePlugin):
    """Plugin for enforcing Intel Factor 1: participant status analysis."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "intel_factor_1_participant", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("IntelFactor1Plugin initialized")

    def _check_non_participant_status(self, text: str) -> Dict[str, Any]:
        """Check if discovery target is identified as non-participant."""
        positive_patterns = [
            r'not\s+a\s+party\s+to\s+.*?proceeding',
            r'non-participant',
            r'nonparticipant',
            r'not\s+a\s+participant',
            r'separate\s+entity',
            r'separate\s+legal\s+entity',
            r'Harvard\s+University\s+is\s+not\s+a\s+party',
            r'Harvard\s+is\s+not\s+a\s+party',
            r'not\s+.*?party\s+to\s+.*?Hong\s+Kong'
        ]

        negative_patterns = [
            r'is\s+a\s+party\s+to',
            r'participant\s+in\s+the\s+foreign\s+proceeding',
            r'party\s+to\s+.*?Hong\s+Kong\s+litigation'
        ]

        positive_matches = []
        for pattern in positive_patterns:
            positive_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        negative_matches = []
        for pattern in negative_patterns:
            negative_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        # Check if negative matches are negated (e.g., "Harvard is NOT a party")
        negated_negative = []
        for match in negative_matches:
            start = max(0, match.start() - 20)
            context = text[start:match.end()]
            if re.search(r'\b(not|no|neither|nor)\b', context, re.IGNORECASE):
                negated_negative.append(match)

        is_non_participant = len(positive_matches) > 0 or (len(negative_matches) > 0 and len(negated_negative) == len(negative_matches))

        return {
            "non_participant_identified": is_non_participant,
            "positive_match_count": len(positive_matches),
            "negative_match_count": len(negative_matches),
            "negated_negative_count": len(negated_negative),
            "matches": [m.group() for m in positive_matches[:5]]
        }

    def _check_evidence_provided(self, text: str) -> Dict[str, Any]:
        """Check if evidence that target is NOT a party is provided."""
        evidence_patterns = [
            r'separate\s+legal\s+entities',
            r'separate\s+entity',
            r'not\s+.*?defendant',
            r'not\s+.*?plaintiff',
            r'only\s+.*?clubs\s+are\s+parties',
            r'Harvard\s+Clubs.*?parties',
            r'Harvard\s+University.*?not.*?party'
        ]

        matches = []
        for pattern in evidence_patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "evidence_provided": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_justification_if_party(self, text: str) -> Dict[str, Any]:
        """Check if justification is provided in case target IS a party."""
        justification_patterns = [
            r'even\s+if.*?party',
            r'although.*?party',
            r'justification.*?party',
            r'participation.*?does\s+not.*?foreclose',
            r'Gorsoan.*?party'
        ]

        matches = []
        for pattern in justification_patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "justification_provided": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate that Intel Factor 1 is addressed."""
        non_participant_check = self._check_non_participant_status(text)
        evidence_check = self._check_evidence_provided(text)
        justification_check = self._check_justification_if_party(text)

        # Factor 1 favors non-participants, but can be addressed even if participant
        addressed = (
            non_participant_check["non_participant_identified"] or
            justification_check["justification_provided"]
        )

        score = 0.0
        if non_participant_check["non_participant_identified"]:
            score += 0.6
        if evidence_check["evidence_provided"]:
            score += 0.2
        if justification_check["justification_provided"]:
            score += 0.2

        issues = []
        if not addressed:
            issues.append("Intel Factor 1 not addressed: target's participant status not discussed")
        elif non_participant_check["non_participant_identified"] and not evidence_check["evidence_provided"]:
            issues.append("Non-participant status claimed but evidence not provided")

        return FunctionResult(
            success=addressed,
            score=min(score, 1.0),
            message=f"Intel Factor 1: Participant Status - {'PASS' if addressed else 'FAIL'}",
            metadata={
                "non_participant_status": non_participant_check,
                "evidence_provided": evidence_check,
                "justification_if_party": justification_check,
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

