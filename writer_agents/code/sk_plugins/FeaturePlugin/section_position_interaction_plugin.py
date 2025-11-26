#!/usr/bin/env python3
"""
Section Position Interaction Plugin - Detects and validates section positions.

This plugin handles section position detection and validation.
Interaction calculations are handled by individual interaction plugins:
- WordCountBalancingTestInteractionPlugin (word_count × balancing_test_position)
- SentenceCountDangerSafetyInteractionPlugin (sentence_count × danger_safety_position)
- CharCountDangerSafetyInteractionPlugin (char_count × danger_safety_position)
- WordCountDangerSafetyInteractionPlugin (word_count × danger_safety_position)
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


class SectionPositionInteractionPlugin(BaseFeaturePlugin):
    """Plugin to enforce optimal section positioning and interactions."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "section_position_interaction", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("SectionPositionInteractionPlugin initialized")

        # Optimal positions (earlier is better, normalized to 0-1)
        self.optimal_balancing_test_position = 0.3  # Should appear in first 30% of document
        self.optimal_danger_safety_position = 0.4  # Should appear in first 40% of document

    def _detect_section_positions(self, text: str) -> Dict[str, Any]:
        """Detect positions of key sections."""
        text_lower = text.lower()
        text_length = len(text)

        # Section keywords
        balancing_keywords = ['balancing', 'balance of interests', 'weigh', 'weighing', 'balancing test']
        danger_safety_keywords = ['danger', 'safety', 'endangerment', 'threat', 'retaliation', 'harm']

        # Find positions
        balancing_positions = []
        danger_safety_positions = []

        for keyword in balancing_keywords:
            pos = text_lower.find(keyword)
            if pos >= 0:
                balancing_positions.append(pos)

        for keyword in danger_safety_keywords:
            pos = text_lower.find(keyword)
            if pos >= 0:
                danger_safety_positions.append(pos)

        # Get earliest positions (normalized to 0-1)
        balancing_pos = min(balancing_positions) / text_length if balancing_positions else -1
        danger_safety_pos = min(danger_safety_positions) / text_length if danger_safety_positions else -1

        # Check if in optimal range
        balancing_optimal = (0 <= balancing_pos <= self.optimal_balancing_test_position) if balancing_pos >= 0 else False
        danger_safety_optimal = (0 <= danger_safety_pos <= self.optimal_danger_safety_position) if danger_safety_pos >= 0 else False

        return {
            "balancing_test_position": balancing_pos,
            "danger_safety_position": danger_safety_pos,
            "has_balancing_test": balancing_pos >= 0,
            "has_danger_safety": danger_safety_pos >= 0,
            "balancing_optimal": balancing_optimal,
            "danger_safety_optimal": danger_safety_optimal,
            "text_length": text_length
        }

    # Note: Interaction calculations are now handled by individual interaction plugins:
    # - WordCountBalancingTestInteractionPlugin
    # - SentenceCountDangerSafetyInteractionPlugin
    # - CharCountDangerSafetyInteractionPlugin
    # - WordCountDangerSafetyInteractionPlugin

    async def validate_section_positions(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate section positions.

        This plugin detects and validates section positions.
        Interaction calculations are handled by individual interaction plugins.
        """
        text = document.get_full_text()
        positions = self._detect_section_positions(text)

        issues = []
        recommendations = []

        # Check balancing test position
        if not positions["has_balancing_test"]:
            issues.append({
                "type": "missing_balancing_test",
                "severity": "high",
                "message": "Balancing test section is missing. This is critical for interaction features.",
                "suggestion": "Add a balancing test section early in the document (within first 30%)."
            })
        elif not positions["balancing_optimal"]:
            issues.append({
                "type": "balancing_test_too_late",
                "severity": "medium",
                "message": f"Balancing test appears at {positions['balancing_test_position']*100:.1f}% of document. Should be within first {self.optimal_balancing_test_position*100:.0f}%.",
                "suggestion": f"Move balancing test section earlier (within first {self.optimal_balancing_test_position*100:.0f}% of document)."
            })
            recommendations.append({
                "type": "reposition_balancing_test",
                "priority": "high",
                "action": f"Move balancing test section to appear within first {self.optimal_balancing_test_position*100:.0f}% of document",
                "rationale": "word_count × balancing_test_position is the #1 strongest interaction (0.337 strength) for pseudonym motions"
            })

        # Check danger/safety position
        if not positions["has_danger_safety"]:
            issues.append({
                "type": "missing_danger_safety",
                "severity": "high",
                "message": "Danger/safety section is missing. This is critical for interaction features.",
                "suggestion": "Add a danger/safety section early in the document (within first 40%)."
            })
        elif not positions["danger_safety_optimal"]:
            issues.append({
                "type": "danger_safety_too_late",
                "severity": "medium",
                "message": f"Danger/safety section appears at {positions['danger_safety_position']*100:.1f}% of document. Should be within first {self.optimal_danger_safety_position*100:.0f}%.",
                "suggestion": f"Move danger/safety section earlier (within first {self.optimal_danger_safety_position*100:.0f}% of document)."
            })
            recommendations.append({
                "type": "reposition_danger_safety",
                "priority": "high",
                "action": f"Move danger/safety section to appear within first {self.optimal_danger_safety_position*100:.0f}% of document",
                "rationale": "sentence_count × danger_safety_position is the #2 strongest interaction (0.332 strength) for pseudonym motions"
            })

        # Section positions are optimal
        if positions["balancing_optimal"] and positions["danger_safety_optimal"]:
            recommendations.append({
                "type": "section_positions_optimal",
                "message": "Section positions are optimal. These positions enable strong interactions (see individual interaction plugins)."
            })

        return FunctionResult(
            success=positions["balancing_optimal"] and positions["danger_safety_optimal"],
            data={
                "balancing_test_position": positions["balancing_test_position"],
                "danger_safety_position": positions["danger_safety_position"],
                "has_balancing_test": positions["has_balancing_test"],
                "has_danger_safety": positions["has_danger_safety"],
                "balancing_optimal": positions["balancing_optimal"],
                "danger_safety_optimal": positions["danger_safety_optimal"],
                "issues": issues,
                "recommendations": recommendations,
                "note": "Interaction calculations are handled by individual interaction plugins"
            },
            message="Section position validation complete"
        )

    async def query_section_positions(self, case_context: str, **kwargs) -> FunctionResult:
        """Query for examples of optimal section positioning."""
        results = await self.query_chroma(
            case_context=f"balancing test danger safety section position {case_context}",
            n_results=kwargs.get('n_results', 10)
        )

        return FunctionResult(
            success=True,
            data={"examples": results},
            message=f"Found {len(results)} examples of section positioning"
        )

    async def generate_section_position_argument(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Generate argument about section positioning."""
        text = document.get_full_text()
        positions = self._detect_section_positions(text)

        if positions["balancing_optimal"] and positions["danger_safety_optimal"]:
            argument = f"""The motion uses optimal section positioning:

1. Balancing test position: {positions['balancing_test_position']*100:.1f}% (optimal: <{self.optimal_balancing_test_position*100:.0f}%)
2. Danger/safety position: {positions['danger_safety_position']*100:.1f}% (optimal: <{self.optimal_danger_safety_position*100:.0f}%)

These positions enable strong interactions with document length (see individual interaction plugins for interaction calculations)."""
        else:
            argument = f"""The motion should optimize section positioning:

Current:
- Balancing test: {'Present' if positions['has_balancing_test'] else 'Missing'} at {positions['balancing_test_position']*100:.1f}% (optimal: <{self.optimal_balancing_test_position*100:.0f}%)
- Danger/safety: {'Present' if positions['has_danger_safety'] else 'Missing'} at {positions['danger_safety_position']*100:.1f}% (optimal: <{self.optimal_danger_safety_position*100:.0f}%)

Recommendations:
1. Move balancing test section to appear within first {self.optimal_balancing_test_position*100:.0f}% of document
2. Move danger/safety section to appear within first {self.optimal_danger_safety_position*100:.0f}% of document

These positions enable strong interactions with document length (see individual interaction plugins)."""

        return FunctionResult(
            success=True,
            data={
                "argument": argument,
                "positions": positions
            },
            message="Section position argument generated"
        )

