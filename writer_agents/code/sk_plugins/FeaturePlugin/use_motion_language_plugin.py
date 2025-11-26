#!/usr/bin/env python3
"""
Use Motion Language Plugin - Encourages use of motion-specific language.
Only positive predictor (+9.8pp).

Impact: +9.8 percentage points (40.4% success WITH vs 30.6% WITHOUT)
Importance: 3.89 (3rd most important, ONLY POSITIVE)
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


class UseMotionLanguagePlugin(BaseFeaturePlugin):
    """Plugin to encourage use of motion-specific language."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "use_motion_language", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("UseMotionLanguagePlugin initialized")

    async def validate_use_motion_language(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate that motion-specific language is used throughout.

        HIGH: Use motion-specific language throughout.
        Only positive predictor (+9.8pp).
        """
        text = document.get_full_text()
        text_lower = text.lower()

        recommendations = []

        # Good phrases to use
        good_phrases = [
            "motion to seal",
            "protective order",
            "under seal",
            "impound",
            "seal",
            "confidential",
            "motion for protective order",
            "file under seal",
            "proceed under seal",
        ]

        found_good = []
        for phrase in good_phrases:
            if phrase.lower() in text_lower:
                found_good.append(phrase)

        if not found_good:
            recommendations.append({
                "type": "add_motion_language",
                "message": "Use motion-specific language throughout: 'motion to seal', 'protective order', 'under seal', 'impound', 'confidential'",
                "priority": "HIGH"
            })
        elif len(found_good) < 3:
            recommendations.append({
                "type": "increase_motion_language",
                "message": f"Found {len(found_good)} motion-specific phrases. Consider using more: 'motion to seal', 'protective order', 'under seal'",
                "priority": "MEDIUM"
            })

        return FunctionResult(
            success=True,
            value={
                "good_phrases_found": len(found_good),
                "found_phrases": found_good,
                "recommendations": recommendations,
                "impact": "+9.8pp (40.4% vs 30.6% success) - ONLY POSITIVE"
            },
            metadata={
                "feature_name": "use_motion_language",
                "priority": "HIGH",
                "importance": 3.89
            }
        )

