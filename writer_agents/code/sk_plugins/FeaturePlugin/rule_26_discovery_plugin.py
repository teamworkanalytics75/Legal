#!/usr/bin/env python3
"""
FRCP Rule 26 Discovery Plugin - Federal Rules enforcement

Enforces FRCP Rule 26 (discovery scope) requirements:
- Relevance requirement
- Proportionality requirement
- Privilege issues
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class Rule26DiscoveryPlugin(BaseFeaturePlugin):
    """Plugin for enforcing FRCP Rule 26 discovery scope requirements."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "rule_26_discovery", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("Rule26DiscoveryPlugin initialized")

    def _check_relevance(self, text: str) -> Dict[str, Any]:
        """Check if relevance requirement is addressed."""
        patterns = [
            r'relevant\s+to\s+any\s+party\'s\s+claim\s+or\s+defense',
            r'relevant\s+to',
            r'relates\s+to',
            r'relating\s+to',
            r'Rule\s+26.*?relevant',
            r'relevance'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "relevance_addressed": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_proportionality(self, text: str) -> Dict[str, Any]:
        """Check if proportionality requirement is addressed."""
        patterns = [
            r'proportional\s+to\s+the\s+needs\s+of\s+the\s+case',
            r'proportionality',
            r'proportional',
            r'Rule\s+26.*?proportional',
            r'burden.*?proportional',
            r'benefit.*?proportional'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "proportionality_addressed": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    def _check_privilege(self, text: str) -> Dict[str, Any]:
        """Check if privilege issues are addressed."""
        patterns = [
            r'privilege',
            r'attorney-client\s+privilege',
            r'work\s+product',
            r'work-product',
            r'privileged\s+information',
            r'not\s+seeking.*?privileged'
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, text, re.IGNORECASE))

        return {
            "privilege_addressed": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.group() for m in matches[:5]]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate Rule 26 requirements."""
        relevance_check = self._check_relevance(text)
        proportionality_check = self._check_proportionality(text)
        privilege_check = self._check_privilege(text)

        # Rule 26 is addressed if relevance and proportionality are mentioned
        addressed = relevance_check["relevance_addressed"] and proportionality_check["proportionality_addressed"]

        score = 0.0
        if relevance_check["relevance_addressed"]:
            score += 0.4
        if proportionality_check["proportionality_addressed"]:
            score += 0.4
        if privilege_check["privilege_addressed"]:
            score += 0.2  # Optional but good practice

        issues = []
        if not relevance_check["relevance_addressed"]:
            issues.append("Relevance requirement (Rule 26) not addressed")
        if not proportionality_check["proportionality_addressed"]:
            issues.append("Proportionality requirement (Rule 26) not addressed")

        return FunctionResult(
            success=addressed,
            score=score,
            message=f"Rule 26 Discovery: Relevance={relevance_check['relevance_addressed']}, Proportionality={proportionality_check['proportionality_addressed']} - {'PASS' if addressed else 'WARN'}",
            metadata={
                "relevance": relevance_check,
                "proportionality": proportionality_check,
                "privilege": privilege_check,
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

