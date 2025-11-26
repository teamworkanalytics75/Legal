#!/usr/bin/env python3
"""
Word Choice Plugin - Word-level enforcement

Enforces legal terminology and word choice (avoid informal language, contractions, etc.)
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class WordChoicePlugin(BaseFeaturePlugin):
    """Plugin for enforcing legal terminology and word choice."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "word_choice", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("WordChoicePlugin initialized")

    def _check_contractions(self, text: str) -> Dict[str, Any]:
        """Check for contractions (should be avoided)."""
        criteria = self.rules.get("validation_criteria", {})
        if not criteria.get("avoid_contractions", True):
            return {"contractions_found": [], "count": 0}

        informal_patterns = criteria.get("informal_patterns", [])
        contractions = []

        for pattern in informal_patterns:
            matches = list(re.finditer(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE))
            for match in matches:
                contractions.append({
                    "text": match.group(),
                    "position": match.start(),
                    "context": text[max(0, match.start()-30):match.end()+30]
                })

        return {
            "contractions_found": contractions[:10],
            "count": len(contractions)
        }

    def _check_informal_language(self, text: str) -> Dict[str, Any]:
        """Check for informal language."""
        criteria = self.rules.get("validation_criteria", {})
        if not criteria.get("avoid_informal", True):
            return {"informal_found": [], "count": 0}

        informal_patterns = [
            r'\b(yeah|yep|nope|ok|okay|hey|hi|sup|gonna|wanna|gotta)\b',
            r'\b(awesome|cool|nice|great|super|fantastic)\b',
            r'\b(just|really|very|totally|absolutely)\s+(so|too|much|many)\b',
            r'\b(like|literally|basically|actually)\s+',
            r'\b(um|uh|er|hmm)\b'
        ]

        informal_found = []
        for pattern in informal_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                informal_found.append({
                    "text": match.group(),
                    "position": match.start(),
                    "context": text[max(0, match.start()-30):match.end()+30]
                })

        return {
            "informal_found": informal_found[:10],
            "count": len(informal_found)
        }

    def _check_legal_terminology(self, text: str) -> Dict[str, Any]:
        """Check for appropriate legal terminology."""
        criteria = self.rules.get("validation_criteria", {})
        legal_terms = criteria.get("legal_terms", [])

        found_terms = []
        for term in legal_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                found_terms.append({
                    "term": term,
                    "count": len(matches)
                })

        return {
            "legal_terms_found": found_terms,
            "count": len(found_terms),
            "total_mentions": sum(t["count"] for t in found_terms)
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate word choice."""
        contraction_check = self._check_contractions(text)
        informal_check = self._check_informal_language(text)
        legal_check = self._check_legal_terminology(text)

        # Score calculation
        score = 1.0

        # Penalty for contractions (each contraction reduces score)
        if contraction_check["count"] > 0:
            penalty = min(0.3, contraction_check["count"] * 0.05)
            score -= penalty

        # Penalty for informal language (each instance reduces score)
        if informal_check["count"] > 0:
            penalty = min(0.3, informal_check["count"] * 0.05)
            score -= penalty

        # Bonus for legal terminology (optional)
        if legal_check["count"] > 0:
            bonus = min(0.1, legal_check["count"] * 0.02)
            score += bonus

        score = max(0.0, min(1.0, score))

        meets_threshold = score >= 0.85 and contraction_check["count"] == 0 and informal_check["count"] == 0

        issues = []
        if contraction_check["count"] > 0:
            issues.append(f"{contraction_check['count']} contractions found (should be avoided in legal writing)")
        if informal_check["count"] > 0:
            issues.append(f"{informal_check['count']} instances of informal language found")

        return FunctionResult(
            success=meets_threshold,
            score=score,
            message=f"Word Choice: {contraction_check['count']} contractions, {informal_check['count']} informal - {'PASS' if meets_threshold else 'WARN'}",
            metadata={
                "contractions": contraction_check,
                "informal_language": informal_check,
                "legal_terminology": legal_check,
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

