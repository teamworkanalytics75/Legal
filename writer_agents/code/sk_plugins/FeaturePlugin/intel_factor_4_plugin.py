#!/usr/bin/env python3
"""
Intel Factor 4 Plugin - Undue Burden/Intrusiveness

Enforces Intel Factor 4:
Whether the request is 'undue intrusive or burdensome'
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class IntelFactor4Plugin(BaseFeaturePlugin):
    """Plugin for enforcing Intel Factor 4: undue burden/intrusiveness."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "intel_factor_4_burden", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("IntelFactor4Plugin initialized")

    def _check_narrowly_tailored(self, text: str) -> Dict[str, Any]:
        """Check if request is narrowly tailored."""
        positive_patterns = [
            r'narrowly\s+tailored',
            r'narrow\s+request',
            r'specific\s+request',
            r'limited\s+request',
            r'targeted\s+request',
            r'proportional',
            r'specific\s+custodian',
            r'limited\s+time\s+period',
            r'specific\s+time\s+period',
            r'specific\s+document\s+categories'
        ]

        negative_patterns = [
            r'all\s+documents',
            r'any\s+documents',
            r'overly\s+broad',
            r'unlimited',
            r'fishing\s+expedition',
            r'wide-ranging'
        ]

        positive_matches = []
        for pattern in positive_patterns:
            positive_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        negative_matches = []
        for pattern in negative_patterns:
            negative_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        # Check if negative matches are negated
        negated_negative = []
        for match in negative_matches:
            start = max(0, match.start() - 20)
            context = text[start:match.end()]
            if re.search(r'\b(not|no|neither|nor)\b', context, re.IGNORECASE):
                negated_negative.append(match)

        is_narrow = len(positive_matches) > 0 and (len(negative_matches) == 0 or len(negated_negative) == len(negative_matches))

        return {
            "narrowly_tailored": is_narrow,
            "positive_match_count": len(positive_matches),
            "negative_match_count": len(negative_matches),
            "negated_negative_count": len(negated_negative),
            "matches": [m.group() for m in positive_matches[:5]]
        }

    def _check_custodian_specified(self, text: str) -> Dict[str, Any]:
        """Check if specific custodian(s) are identified."""
        patterns = [
            r'custodian',
            r'Marlyn\s+McGrath',
            r'specific\s+.*?custodian',
            r'limited\s+.*?custodian',
            r'individual.*?custodian'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        # Count number of custodians (should be 1-3 ideally)
        custodian_count = len(set(m.group().lower() for m in matches if 'custodian' in m.group().lower()))

        return {
            "custodian_specified": len(matches) > 0,
            "custodian_count": custodian_count,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_time_period_limited(self, text: str) -> Dict[str, Any]:
        """Check if time period is limited."""
        patterns = [
            r'limited\s+time\s+period',
            r'specific\s+time\s+period',
            r'between.*?and.*?\d{4}',
            r'from.*?\d{4}.*?to.*?\d{4}',
            r'January.*?\d{4}.*?December.*?\d{4}',
            r'date\s+range',
            r'limited\s+date\s+range'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        unlimited_patterns = [
            r'all\s+time',
            r'any\s+time',
            r'unlimited\s+time',
            r'no\s+time\s+limit'
        ]

        unlimited_matches = []
        for pattern in unlimited_patterns:
            unlimited_matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        is_limited = len(matches) > 0 and len(unlimited_matches) == 0

        return {
            "time_period_limited": is_limited,
            "match_count": len(matches),
            "unlimited_match_count": len(unlimited_matches),
            "matches": [m.group() for m in matches[:5]]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate that Intel Factor 4 is addressed (request is not unduly burdensome)."""
        narrow_check = self._check_narrowly_tailored(text)
        custodian_check = self._check_custodian_specified(text)
        time_check = self._check_time_period_limited(text)

        # Factor 4 is addressed if request is narrow and specific
        addressed = narrow_check["narrowly_tailored"] and (custodian_check["custodian_specified"] or time_check["time_period_limited"])

        score = 0.0
        if narrow_check["narrowly_tailored"]:
            score += 0.4
        if custodian_check["custodian_specified"]:
            score += 0.3
            # Bonus for low custodian count (1-3 ideal)
            if 1 <= custodian_check["custodian_count"] <= 3:
                score += 0.1
            # Penalty for high custodian count (20+)
            elif custodian_check["custodian_count"] >= 20:
                score -= 0.2
        if time_check["time_period_limited"]:
            score += 0.2

        issues = []
        if not narrow_check["narrowly_tailored"]:
            issues.append("Request may not be narrowly tailored")
        if not custodian_check["custodian_specified"]:
            issues.append("Specific custodian(s) not identified")
        elif custodian_check["custodian_count"] >= 20:
            issues.append(f"Too many custodians ({custodian_check['custodian_count']}) - should be 1-3 for narrow request")
        if not time_check["time_period_limited"]:
            issues.append("Time period not limited")

        return FunctionResult(
            success=addressed,
            score=max(0.0, min(score, 1.0)),
            message=f"Intel Factor 4: Burden - {'PASS' if addressed else 'FAIL'}",
            metadata={
                "narrowly_tailored": narrow_check,
                "custodian_specified": custodian_check,
                "time_period_limited": time_check,
                "issues": issues,
                "threshold": self.rules.get("threshold", 0.85)
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

