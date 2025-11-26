#!/usr/bin/env python3
"""
Paragraph Count Plugin - Enforces paragraph_count feature.

For pseudonym motions:
- paragraph_count is the #1 most important feature (24.78 importance)

This plugin validates paragraph count independently.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class ParagraphCountPlugin(BaseFeaturePlugin):
    """Plugin to enforce paragraph count (critical for pseudonym motions)."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "paragraph_count", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("ParagraphCountPlugin initialized")

        # Target from analysis (pseudonym motions)
        # paragraph_count is #1 feature (24.78 importance)
        self.target_paragraph_count = self.rules.get("targets", {}).get("paragraph_count", 15.0)

    def _analyze_paragraph_count(self, text: str) -> Dict[str, Any]:
        """Analyze paragraph count in text."""
        # Split into paragraphs
        paragraphs = re.split(r'(?:\r?\n\s*){2,}', text.strip())
        if len(paragraphs) == 1:
            paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        else:
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

        paragraph_count = len(paragraphs) if paragraphs else 0
        meets_target = paragraph_count >= self.target_paragraph_count * 0.8

        return {
            "paragraph_count": paragraph_count,
            "target_paragraph_count": self.target_paragraph_count,
            "meets_target": meets_target,
            "gap": max(0, self.target_paragraph_count - paragraph_count)
        }

    async def validate_paragraph_count(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate paragraph count.

        This is CRITICAL: paragraph_count is the #1 most important feature (24.78 importance) for pseudonym motions.
        Target: paragraph_count ≥ 15
        """
        text = document.get_full_text()
        analysis = self._analyze_paragraph_count(text)

        issues = []
        recommendations = []

        if not analysis["meets_target"]:
            issues.append({
                "type": "insufficient_paragraphs",
                "severity": "critical",
                "message": f"Paragraph count ({analysis['paragraph_count']}) is below target ({self.target_paragraph_count}). This is the #1 most important feature (24.78 importance) for pseudonym motions.",
                "suggestion": f"Increase paragraph count to at least {self.target_paragraph_count:.0f} paragraphs. Break up long paragraphs."
            })
            recommendations.append({
                "type": "increase_paragraph_count",
                "priority": "critical",
                "action": f"Break up paragraphs to reach target count of {self.target_paragraph_count:.0f}. Current: {analysis['paragraph_count']}",
                "rationale": "Paragraph count is the #1 predictor for pseudonym motion success (24.78 importance)"
            })
        else:
            recommendations.append({
                "type": "paragraph_count_optimal",
                "message": f"Paragraph count is optimal ({analysis['paragraph_count']} paragraphs, target: {self.target_paragraph_count:.0f}).",
                "paragraph_count": analysis["paragraph_count"]
            })

        return FunctionResult(
            success=analysis["meets_target"],
            data={
                "paragraph_count": analysis["paragraph_count"],
                "target_paragraph_count": analysis["target_paragraph_count"],
                "meets_target": analysis["meets_target"],
                "gap": analysis["gap"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 24.78,
                "feature_rank": 1,
                "motion_type": "pseudonym"
            },
            message="Paragraph count validation complete"
        )

    async def query_paragraph_count(self, case_context: str, **kwargs) -> FunctionResult:
        """Query for examples of documents with high paragraph count."""
        results = await self.query_chroma(
            case_context=f"multiple paragraphs organization structure {case_context}",
            n_results=kwargs.get('n_results', 10)
        )

        return FunctionResult(
            success=True,
            data={"examples": results},
            message=f"Found {len(results)} examples of high paragraph count"
        )

    async def generate_paragraph_count_argument(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Generate argument about paragraph count."""
        analysis = self._analyze_paragraph_count(document.get_full_text())

        if analysis["meets_target"]:
            argument = f"""The motion uses an optimal paragraph count ({analysis['paragraph_count']} paragraphs). Paragraph count is the #1 most important feature (24.78 importance) for pseudonym motions because:

1. Shows systematic organization and thoroughness
2. Demonstrates attention to detail and structure
3. Makes complex arguments easier to follow
4. Signals professional drafting quality

Target: {self.target_paragraph_count:.0f} paragraphs. Current: {analysis['paragraph_count']} paragraphs."""
        else:
            argument = f"""The motion should increase paragraph count. This is the #1 most important feature (24.78 importance) for pseudonym motion success.

Current: {analysis['paragraph_count']} paragraphs
Target: {self.target_paragraph_count:.0f} paragraphs
Gap: {analysis['gap']:.0f} paragraphs

Recommendations:
1. Break up long paragraphs into smaller, focused paragraphs
2. Ensure each paragraph addresses a single point or argument
3. Use paragraph breaks to improve readability and organization
4. Aim for {self.target_paragraph_count:.0f}+ paragraphs to meet the target"""

        return FunctionResult(
            success=True,
            data={"argument": argument, "paragraph_count": analysis["paragraph_count"]},
            message="Paragraph count argument generated"
        )

