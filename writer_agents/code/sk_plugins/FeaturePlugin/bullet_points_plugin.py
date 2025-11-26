#!/usr/bin/env python3
"""
Bullet Points Plugin - Enforces use of bullet points for enumeration.

Bullet points are the #2 most important feature (31.56 importance) and have
88.6% grant rate vs 84.9% without bullets.
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


class BulletPointsPlugin(BaseFeaturePlugin):
    """Plugin to enforce bullet point usage for enumeration."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "bullet_points", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("BulletPointsPlugin initialized")

    def _detect_bullet_points(self, text: str) -> Dict[str, Any]:
        """Detect bullet points in text."""
        # Patterns for bullet points
        bullet_patterns = [
            r'\n\s*[-*•]\s+',  # - * •
            r'\n\s*[o]\s+',  # o (lowercase o as bullet)
        ]

        bullet_count = 0
        bullet_positions = []

        for pattern in bullet_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                bullet_count += 1
                bullet_positions.append({
                    "position": match.start(),
                    "pattern": match.group(),
                    "line": text[:match.start()].count('\n') + 1
                })

        # Calculate density (bullets per 1000 words)
        word_count = len(text.split())
        density = (bullet_count / word_count * 1000) if word_count > 0 else 0.0

        # Check key sections
        key_sections = {
            'privacy_harm': ['privacy', 'privacy harm'],
            'danger_safety': ['danger', 'safety', 'threat'],
            'good_cause': ['good cause', 'cause'],
            'protective_measures': ['protective', 'measures', 'proposed']
        }

        bullets_by_section = {}
        for section_name, keywords in key_sections.items():
            section_bullets = 0
            # Simple keyword-based section detection
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # Count bullets in context around keyword
                    pattern = f'.{{0,500}}{keyword}.{{0,500}}'
                    matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        section_text = match.group()
                        for bp in bullet_patterns:
                            section_bullets += len(re.findall(bp, section_text))
                    break
            bullets_by_section[section_name] = section_bullets

        return {
            "bullet_count": bullet_count,
            "has_bullet_points": bullet_count > 0,
            "bullet_density": density,
            "bullet_positions": bullet_positions[:20],  # Limit for performance
            "bullets_by_section": bullets_by_section,
            "target_density": 1.68,  # From granted motions average
            "target_count": 11.75  # From granted motions average
        }

    async def validate_bullet_points(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate bullet point usage.

        Bullet points are the #2 most important feature (31.56 importance).
        Motions with bullets: 88.6% grant rate
        Motions without: 84.9% grant rate
        """
        text = document.get_full_text()
        detection = self._detect_bullet_points(text)

        issues = []
        recommendations = []

        if not detection["has_bullet_points"]:
            issues.append({
                "type": "no_bullet_points",
                "severity": "high",
                "message": "No bullet points detected. Bullet points are the #2 most important feature (31.56 importance).",
                "suggestion": "Add bullet points throughout the motion, especially in: Good Cause, Privacy Harm, Danger/Safety, and Protective Measures sections."
            })
        elif detection["bullet_count"] < detection["target_count"]:
            issues.append({
                "type": "insufficient_bullets",
                "severity": "medium",
                "message": f"Only {detection['bullet_count']} bullet points found. Target: {detection['target_count']} (based on granted motions average).",
                "suggestion": "Add more bullet points to key argument sections."
            })

        # Check key sections
        for section_name, count in detection["bullets_by_section"].items():
            if count == 0 and section_name in ['privacy_harm', 'danger_safety', 'good_cause', 'protective_measures']:
                recommendations.append({
                    "type": "add_bullets_to_section",
                    "section": section_name,
                    "priority": "high",
                    "action": f"Add bullet points to {section_name.replace('_', ' ').title()} section.",
                    "rationale": "Bullet points in key argument sections are strongly correlated with grants."
                })

        if detection["bullet_count"] >= detection["target_count"]:
            recommendations.append({
                "type": "bullet_points_optimal",
                "message": f"Bullet point usage is optimal ({detection['bullet_count']} bullets).",
                "density": detection["bullet_density"]
            })

        return FunctionResult(
            success=detection["has_bullet_points"] and detection["bullet_count"] >= 5,
            data={
                "has_bullet_points": detection["has_bullet_points"],
                "bullet_count": detection["bullet_count"],
                "bullet_density": detection["bullet_density"],
                "target_count": detection["target_count"],
                "target_density": detection["target_density"],
                "bullets_by_section": detection["bullets_by_section"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 31.56,
                "feature_rank": 2,
                "grant_rate_with": 0.886,
                "grant_rate_without": 0.849
            },
            message="Bullet points validation complete"
        )

    async def query_bullet_points(self, case_context: str, **kwargs) -> FunctionResult:
        """Query for examples of effective bullet point usage."""
        results = await self.query_chroma(
            case_context=f"bullet points enumeration list {case_context}",
            n_results=kwargs.get('n_results', 10)
        )

        return FunctionResult(
            success=True,
            data={"examples": results},
            message=f"Found {len(results)} examples of bullet point usage"
        )

    async def generate_bullet_points_argument(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Generate argument about bullet point usage."""
        detection = self._detect_bullet_points(document.get_full_text())

        if detection["has_bullet_points"]:
            argument = f"""The motion uses bullet points effectively ({detection['bullet_count']} bullets). Bullet points are the #2 most important structural feature (31.56 importance) because they:

1. Signal organization and systematic thinking
2. Make arguments scannable and easy to digest
3. Show thoroughness and professionalism
4. Help judges quickly identify key points

Motions with bullet points have an 88.6% grant rate vs 84.9% without."""
        else:
            argument = """The motion should use bullet points extensively. This is the #2 most important structural feature for motion success. Add bullet points in:
- Good Cause sections (list factors)
- Privacy Harm arguments (enumerate harms)
- Danger/Safety arguments (list threats)
- Proposed Protective Measures (list measures)"""

        return FunctionResult(
            success=True,
            data={"argument": argument, "bullet_count": detection["bullet_count"]},
            message="Bullet points argument generated"
        )

