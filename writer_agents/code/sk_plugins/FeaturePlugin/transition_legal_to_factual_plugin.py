#!/usr/bin/env python3
"""
Transition Legal Standard to Factual Background Plugin.

Enforces the critical transition: Legal Standard section -> Factual Background section.
This is the #1 most important feature (64.80 importance) for motion success.
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


class TransitionLegalToFactualPlugin(BaseFeaturePlugin):
    """Plugin to enforce Legal Standard -> Factual Background transition."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "transition_legal_to_factual", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("TransitionLegalToFactualPlugin initialized")

    def _detect_sections(self, text: str) -> Dict[str, Any]:
        """Detect Legal Standard and Factual Background sections."""
        # Patterns for section headers
        legal_standard_patterns = [
            r'(?i)^\s*(?:legal\s+standard|legal\s+framework|applicable\s+law|governing\s+law)',
            r'(?i)^\s*(?:standard\s+of\s+review|legal\s+test|test\s+for)',
            r'(?i)^\s*(?:I\.?\s+)?(?:legal\s+standard|legal\s+framework)',
        ]

        factual_background_patterns = [
            r'(?i)^\s*(?:factual\s+background|statement\s+of\s+facts|facts|background)',
            r'(?i)^\s*(?:II\.?\s+)?(?:factual\s+background|statement\s+of\s+facts)',
        ]

        lines = text.split('\n')
        legal_standard_positions = []
        factual_background_positions = []

        for i, line in enumerate(lines):
            # Check for Legal Standard
            for pattern in legal_standard_patterns:
                if re.search(pattern, line):
                    legal_standard_positions.append(i)
                    break

            # Check for Factual Background
            for pattern in factual_background_patterns:
                if re.search(pattern, line):
                    factual_background_positions.append(i)
                    break

        # Find if Legal Standard immediately precedes Factual Background
        transition_present = False
        transition_position = None

        if legal_standard_positions and factual_background_positions:
            for ls_pos in legal_standard_positions:
                # Check if there's a Factual Background within 10 lines after
                for fb_pos in factual_background_positions:
                    if 0 < fb_pos - ls_pos <= 10:
                        transition_present = True
                        transition_position = ls_pos
                        break
                if transition_present:
                    break

        return {
            "legal_standard_positions": legal_standard_positions,
            "factual_background_positions": factual_background_positions,
            "transition_present": transition_present,
            "transition_position": transition_position,
            "has_legal_standard": len(legal_standard_positions) > 0,
            "has_factual_background": len(factual_background_positions) > 0,
        }

    async def validate_transition(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate that Legal Standard -> Factual Background transition exists.

        This is CRITICAL: It's the #1 most important feature (64.80 importance).
        """
        text = document.get_full_text()
        detection = self._detect_sections(text)

        issues = []
        recommendations = []

        if not detection["has_legal_standard"]:
            issues.append({
                "type": "missing_legal_standard",
                "severity": "critical",
                "message": "Legal Standard section is missing. This is required before Factual Background.",
                "suggestion": "Add a 'Legal Standard' or 'Legal Framework' section before the Factual Background."
            })
        elif not detection["has_factual_background"]:
            issues.append({
                "type": "missing_factual_background",
                "severity": "critical",
                "message": "Factual Background section is missing. It should immediately follow Legal Standard.",
                "suggestion": "Add a 'Factual Background' section immediately after the Legal Standard section."
            })
        elif not detection["transition_present"]:
            issues.append({
                "type": "transition_not_immediate",
                "severity": "critical",
                "message": "Factual Background does not immediately follow Legal Standard. This transition is THE MOST IMPORTANT feature (64.80 importance).",
                "suggestion": "Move Factual Background to immediately follow Legal Standard. Do not put any other sections between them."
            })
            recommendations.append({
                "type": "reorder_sections",
                "priority": "critical",
                "action": "Move Factual Background section to immediately follow Legal Standard section.",
                "rationale": "This transition is the #1 most predictive feature for motion success according to CatBoost analysis."
            })
        else:
            recommendations.append({
                "type": "transition_confirmed",
                "message": "Legal Standard -> Factual Background transition is present. This is optimal.",
                "position": detection["transition_position"]
            })

        return FunctionResult(
            success=len(issues) == 0,
            data={
                "transition_present": detection["transition_present"],
                "has_legal_standard": detection["has_legal_standard"],
                "has_factual_background": detection["has_factual_background"],
                "legal_standard_positions": detection["legal_standard_positions"],
                "factual_background_positions": detection["factual_background_positions"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 64.80,
                "feature_rank": 1
            },
            message="Transition validation complete"
        )

    async def query_transition(self, case_context: str, **kwargs) -> FunctionResult:
        """Query for examples of Legal Standard -> Factual Background transitions."""
        results = await self.query_chroma(
            case_context=f"legal standard factual background transition {case_context}",
            n_results=kwargs.get('n_results', 10)
        )

        return FunctionResult(
            success=True,
            data={"examples": results},
            message=f"Found {len(results)} examples of Legal Standard -> Factual Background transitions"
        )

    async def generate_transition_argument(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Generate argument explaining why Legal Standard -> Factual Background transition is optimal."""
        detection = self._detect_sections(document.get_full_text())

        if detection["transition_present"]:
            argument = """The motion follows the optimal structure: Legal Standard immediately followed by Factual Background. This transition is the most important structural feature (64.80 importance) for motion success because it:

1. Establishes the legal framework BEFORE presenting facts
2. Signals strategic thinking: "I know the test, here's how facts meet it"
3. Creates cognitive efficiency: the judge knows what to look for in the facts
4. Shows a precedent-driven approach, not narrative-driven

This structure is associated with a 71.4% grant rate in motions with this transition pattern."""
        else:
            argument = """The motion should be restructured to place Factual Background immediately after Legal Standard. This is the #1 most important structural feature for motion success."""

        return FunctionResult(
            success=True,
            data={"argument": argument, "transition_present": detection["transition_present"]},
            message="Transition argument generated"
        )

