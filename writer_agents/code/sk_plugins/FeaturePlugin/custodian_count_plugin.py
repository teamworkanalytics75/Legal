#!/usr/bin/env python3
"""
Custodian Count Plugin - CatBoost Feature Enforcement

Enforces low custodian count (negative signal from CatBoost analysis)
SHAP Value: -0.309 (negative - decreases success probability)
Target: 1-3 custodians (narrow request)
Avoid: 20+ custodians (overly broad)
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class CustodianCountPlugin(BaseFeaturePlugin):
    """Plugin for enforcing low custodian count (CatBoost negative signal)."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "custodian_count", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("CustodianCountPlugin initialized")

    def _count_custodians(self, text: str) -> Dict[str, Any]:
        """Count number of custodians mentioned."""
        # Patterns to identify custodian mentions
        custodian_patterns = [
            r'custodian',
            r'document\s+custodian',
            r'records\s+custodian',
            r'individual.*?possesses',
            r'person.*?maintains',
            r'individual.*?maintains',
            r'person.*?possesses'
        ]

        # Also look for specific names (like "Marlyn McGrath")
        name_patterns = [
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Basic name pattern
            r'Marlyn\s+McGrath',
            r'individual\s+named\s+[A-Z]',
            r'person\s+named\s+[A-Z]'
        ]

        # Find all custodian mentions
        custodian_matches = []
        for pattern in custodian_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            custodian_matches.extend(matches)

        # Find named individuals who might be custodians
        name_matches = []
        for pattern in name_patterns:
            matches = re.finditer(pattern, text)
            name_matches.extend(matches)

        # Extract unique custodian references
        # Look for patterns like "custodian: [name]" or "[name], custodian"
        custodian_contexts = []
        for match in custodian_matches:
            # Look around the match for names
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            # Check if there's a name nearby
            for name_match in name_patterns:
                if re.search(name_match, context):
                    custodian_contexts.append(context.strip())
                    break

        # Count unique custodians
        # Simple heuristic: count unique names mentioned near custodian keywords
        unique_custodians = set()
        for match in custodian_matches:
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]

            # Extract potential names
            names = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z]+)', context)
            unique_custodians.update(names)

        # If no names found, estimate from custodian mentions
        if len(unique_custodians) == 0:
            # Count distinct custodian mentions (heuristic: assume each mention is a different custodian unless context suggests otherwise)
            custodian_count = len(set(m.group().lower() for m in custodian_matches))
        else:
            custodian_count = len(unique_custodians)

        # Get target thresholds from rules
        target_ideal_min = self.rules.get("target_count", {}).get("ideal_min", 1)
        target_ideal_max = self.rules.get("target_count", {}).get("ideal_max", 3)
        target_warning = self.rules.get("target_count", {}).get("warning_threshold", 10)
        target_critical = self.rules.get("target_count", {}).get("critical_threshold", 20)

        within_ideal = target_ideal_min <= custodian_count <= target_ideal_max
        above_warning = custodian_count > target_warning
        above_critical = custodian_count > target_critical

        return {
            "custodian_count": custodian_count,
            "within_ideal_range": within_ideal,
            "above_warning_threshold": above_warning,
            "above_critical_threshold": above_critical,
            "unique_custodians": list(unique_custodians),
            "target_ideal_min": target_ideal_min,
            "target_ideal_max": target_ideal_max,
            "target_warning": target_warning,
            "target_critical": target_critical
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate custodian count is within ideal range."""
        count_check = self._count_custodians(text)

        within_ideal = count_check["within_ideal_range"]
        above_critical = count_check["above_critical_threshold"]

        score = 1.0
        if within_ideal:
            score = 1.0
        elif count_check["custodian_count"] <= count_check["target_warning"]:
            # Linear penalty from ideal to warning
            excess = count_check["custodian_count"] - count_check["target_ideal_max"]
            max_excess = count_check["target_warning"] - count_check["target_ideal_max"]
            score = max(0.7, 1.0 - (excess / max_excess) * 0.3)
        elif above_critical:
            score = 0.3  # Critical penalty
        else:
            score = 0.5  # Warning penalty

        issues = []
        if not within_ideal:
            if count_check["custodian_count"] < count_check["target_ideal_min"]:
                issues.append(f"Custodian count ({count_check['custodian_count']}) below ideal minimum ({count_check['target_ideal_min']})")
            elif above_critical:
                issues.append(f"Custodian count ({count_check['custodian_count']}) exceeds critical threshold ({count_check['target_critical']}) - overly broad request")
            elif count_check["above_warning_threshold"]:
                issues.append(f"Custodian count ({count_check['custodian_count']}) exceeds warning threshold ({count_check['target_warning']}) - consider narrowing")
            else:
                issues.append(f"Custodian count ({count_check['custodian_count']}) exceeds ideal maximum ({count_check['target_ideal_max']})")

        return FunctionResult(
            success=within_ideal,
            score=score,
            message=f"Custodian Count: {count_check['custodian_count']} custodians (ideal: {count_check['target_ideal_min']}-{count_check['target_ideal_max']}) - {'PASS' if within_ideal else 'FAIL'}",
            metadata={
                "custodian_count": count_check,
                "shap_value": self.rules.get("shap_value", -0.309),
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

