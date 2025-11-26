#!/usr/bin/env python3
"""
Enumeration in Danger/Safety Plugin - Enforces enumeration in Danger/Safety section.

Using enumeration (numbered lists, bullets) in Danger/Safety section is predictive.
90.0% grant rate when enumeration is used in Danger/Safety.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult, EditRequest, DocumentLocation

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class EnumerationDangerSafetyPlugin(BaseFeaturePlugin):
    """Plugin to enforce enumeration in Danger/Safety section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "enumeration_danger_safety", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("EnumerationDangerSafetyPlugin initialized")

    def _detect_enumeration_in_danger_safety(self, text: str) -> Dict[str, Any]:
        """Detect enumeration in Danger/Safety section."""
        # Patterns for danger/safety sections
        danger_section_patterns = [
            r'(?i)(?:danger|safety|threat|harassment|harass|intimidation).{0,500}',
        ]

        # Patterns for enumeration
        numbered_patterns = [
            r'\n\s*\d+\.\s+',  # 1. 2. 3.
            r'\n\s*\(\d+\)\s+',  # (1) (2) (3)
        ]
        bullet_patterns = [
            r'\n\s*[-*•]\s+',  # - * •
        ]

        enumeration_count = 0
        enumeration_positions = []

        # Find danger/safety sections
        for pattern in danger_section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section_text = match.group()

                # Count enumeration in this section
                for enum_pattern in numbered_patterns + bullet_patterns:
                    enum_matches = re.finditer(enum_pattern, section_text, re.MULTILINE)
                    for enum_match in enum_matches:
                        enumeration_count += 1
                        enumeration_positions.append({
                            "position": match.start() + enum_match.start(),
                            "pattern": enum_match.group(),
                            "context": section_text[max(0, enum_match.start()-50):enum_match.end()+50]
                        })

        return {
            "has_enumeration": enumeration_count > 0,
            "enumeration_count": enumeration_count,
            "enumeration_positions": enumeration_positions[:10],  # Limit for performance
            "target_count": 1.0,  # From granted motions average
            "grant_rate_with": 0.90,
            "grant_rate_without": 0.85
        }

    async def validate_enumeration_danger_safety(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate enumeration in Danger/Safety section.

        Feature importance: 0.05 (rank #5)
        90.0% grant rate when enumeration is used in Danger/Safety
        """
        text = document.get_full_text()
        detection = self._detect_enumeration_in_danger_safety(text)

        issues = []
        recommendations = []

        if not detection["has_enumeration"]:
            issues.append({
                "type": "no_enumeration_in_danger_safety",
                "severity": "medium",
                "message": "No enumeration detected in Danger/Safety section. Enumeration in this section has 90.0% grant rate.",
                "suggestion": "Use enumeration (numbered lists or bullets) in your Danger/Safety section to list threats systematically."
            })
            recommendations.append({
                "type": "add_enumeration",
                "priority": "medium",
                "action": "Add enumeration to Danger/Safety section to list threats systematically.",
                "rationale": "Enumeration in key argument sections signals thoroughness and is strongly correlated with grants.",
                "example": "1. Threat 1: [description]\n2. Threat 2: [description]\n3. Threat 3: [description]"
            })
        else:
            if detection["enumeration_count"] < detection["target_count"]:
                recommendations.append({
                    "type": "add_more_enumeration",
                    "priority": "low",
                    "message": f"Enumeration found ({detection['enumeration_count']} items). Consider adding more for thoroughness.",
                    "current_count": detection["enumeration_count"],
                    "target_count": detection["target_count"]
                })
            else:
                recommendations.append({
                    "type": "enumeration_optimal",
                    "message": f"Enumeration in Danger/Safety section is optimal ({detection['enumeration_count']} items).",
                    "count": detection["enumeration_count"]
                })

        return FunctionResult(
            success=detection["has_enumeration"] and detection["enumeration_count"] >= detection["target_count"],
            data={
                "has_enumeration": detection["has_enumeration"],
                "enumeration_count": detection["enumeration_count"],
                "target_count": detection["target_count"],
                "grant_rate_with": detection["grant_rate_with"],
                "grant_rate_without": detection["grant_rate_without"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 0.05,
                "feature_rank": 5
            },
            message="Enumeration in Danger/Safety validation complete"
        )

    async def query_enumeration_danger_safety(self, case_context: str, **kwargs) -> FunctionResult:
        """Query for examples of enumeration in Danger/Safety sections."""
        results = await self.query_chroma(
            case_context=f"enumeration danger safety threat list {case_context}",
            n_results=kwargs.get('n_results', 10)
        )

        return FunctionResult(
            success=True,
            data={"examples": results},
            message=f"Found {len(results)} examples of enumeration in Danger/Safety sections"
        )

