#!/usr/bin/env python3
"""
Use Competing Interests Plugin - Encourages 'competing interests' phrase.
Positive signal in successful motions.
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


class UseCompetingInterestsPlugin(BaseFeaturePlugin):
    """Plugin to encourage use of 'competing interests' phrase."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "use_competing_interests", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("UseCompetingInterestsPlugin initialized")

    async def validate_use_competing_interests(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate that 'competing interests' phrase is used.

        MEDIUM: Use 'competing interests' phrase. Positive signal in successful motions.
        """
        text = document.get_full_text()
        text_lower = text.lower()

        recommendations = []

        # Good phrase to use
        pattern = r'\bcompeting\s+interests?\b'
        matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))

        if not matches:
            recommendations.append({
                "type": "add_competing_interests",
                "message": "Consider using 'competing interests' phrase. Positive signal in successful motions.",
                "priority": "MEDIUM"
            })

        return FunctionResult(
            success=True,
            value={
                "phrase_found": len(matches) > 0,
                "occurrences": len(matches),
                "recommendations": recommendations,
                "impact": "Positive - appears in successful motions"
            },
            metadata={
                "feature_name": "use_competing_interests",
                "priority": "MEDIUM"
            }
        )

