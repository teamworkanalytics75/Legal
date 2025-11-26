#!/usr/bin/env python3
"""
Intel Factor 3 Plugin - Circumvention Concerns

Enforces Intel Factor 3:
Whether the § 1782(a) request conceals an attempt to circumvent foreign
proof-gathering restrictions or other policies of a foreign country or the United States
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class IntelFactor3Plugin(BaseFeaturePlugin):
    """Plugin for enforcing Intel Factor 3: circumvention concerns."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "intel_factor_3_circumvention", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("IntelFactor3Plugin initialized")

    def _check_no_bad_faith(self, text: str) -> Dict[str, Any]:
        """Check that there's no evidence of bad faith or circumvention."""
        negative_patterns = [
            r'circumvent.*?foreign',
            r'evade.*?foreign',
            r'bad\s+faith',
            r'sham\s+proceeding',
            r'fraudulent\s+proceeding'
        ]

        positive_patterns = [
            r'not.*?circumvent',
            r'no.*?circumvention',
            r'no\s+evidence\s+of\s+bad\s+faith',
            r'valid\s+proceeding',
            r'genuine\s+proceeding',
            r'separate\s+entity',
            r'Brandi-Dohrn.*?no.*?requirement'
        ]

        negative_matches = []
        for pattern in negative_patterns:
            negative_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        positive_matches = []
        for pattern in positive_patterns:
            positive_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        # Check if negative matches are negated
        negated_negative = []
        for match in negative_matches:
            start = max(0, match.start() - 20)
            context = text[start:match.end()]
            if re.search(r'\b(not|no|neither|nor)\b', context, re.IGNORECASE):
                negated_negative.append(match)

        no_bad_faith = len(positive_matches) > 0 or (len(negative_matches) == 0) or (len(negated_negative) == len(negative_matches))

        return {
            "no_bad_faith": no_bad_faith,
            "positive_match_count": len(positive_matches),
            "negative_match_count": len(negative_matches),
            "negated_negative_count": len(negated_negative),
            "matches": [m.group() for m in positive_matches[:5]]
        }

    def _check_valid_proceeding(self, text: str) -> Dict[str, Any]:
        """Check that valid foreign proceeding exists."""
        patterns = [
            r'valid\s+foreign\s+proceeding',
            r'genuine\s+proceeding',
            r'pending\s+proceeding',
            r'active\s+proceeding',
            r'legitimate\s+proceeding'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "valid_proceeding_identified": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_separate_entity(self, text: str) -> Dict[str, Any]:
        """Check that discovery target is separate entity (not sham proceeding)."""
        patterns = [
            r'separate\s+legal\s+entity',
            r'separate\s+entity',
            r'distinct\s+entity',
            r'independent\s+entity',
            r'not\s+.*?sham'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "separate_entity_identified": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate that Intel Factor 3 is addressed (no circumvention concerns)."""
        bad_faith_check = self._check_no_bad_faith(text)
        proceeding_check = self._check_valid_proceeding(text)
        entity_check = self._check_separate_entity(text)

        # Factor 3 is addressed if no bad faith and valid proceeding exists
        addressed = bad_faith_check["no_bad_faith"] and (proceeding_check["valid_proceeding_identified"] or entity_check["separate_entity_identified"])

        score = 0.0
        if bad_faith_check["no_bad_faith"]:
            score += 0.5
        if proceeding_check["valid_proceeding_identified"]:
            score += 0.25
        if entity_check["separate_entity_identified"]:
            score += 0.25

        issues = []
        if not bad_faith_check["no_bad_faith"]:
            issues.append("Evidence of bad faith or circumvention may exist")
        if not proceeding_check["valid_proceeding_identified"] and not entity_check["separate_entity_identified"]:
            issues.append("Valid proceeding or separate entity status not clearly established")

        return FunctionResult(
            success=addressed,
            score=min(score, 1.0),
            message=f"Intel Factor 3: Circumvention - {'PASS' if addressed else 'FAIL'}",
            metadata={
                "no_bad_faith": bad_faith_check,
                "valid_proceeding": proceeding_check,
                "separate_entity": entity_check,
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

