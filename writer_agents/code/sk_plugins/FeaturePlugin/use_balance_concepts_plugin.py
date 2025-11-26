#!/usr/bin/env python3
"""
Use Balance Concepts Plugin - Encourages use of balance concepts without the formal phrase.
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


class UseBalanceConceptsPlugin(BaseFeaturePlugin):
    """Plugin to encourage use of balance concepts without the formal phrase."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "use_balance_concepts", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("UseBalanceConceptsPlugin initialized")

    async def validate_use_balance_concepts(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate that balance concepts are used (without the formal 'balancing test' phrase).

        Use balance concepts ('weigh', 'balance of equities', 'competing interests')
        but avoid the formal 'balancing test' phrase.
        """
        text = document.get_full_text()
        text_lower = text.lower()

        recommendations = []

        # Good phrases to use
        good_phrases = [
            r'\bweigh\s+the\s+interests?\b',
            r'\bbalance\s+of\s+equities?\b',
            r'\bcompeting\s+interests?\b',
            r'\bweighing\s+the\s+interests?\b',
        ]

        found_good = []
        for pattern in good_phrases:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if matches:
                found_good.append(pattern)

        # Check if explicit "balancing test" is present
        has_balancing_test = bool(re.search(r'\bbalancing\s+test\b', text_lower, re.IGNORECASE))

        if has_balancing_test:
            recommendations.append({
                "type": "avoid_balancing_test_phrase",
                "message": "Remove explicit 'balancing test' phrase. Use balance concepts instead.",
                "priority": "CRITICAL"
            })

        if not found_good:
            recommendations.append({
                "type": "add_balance_concepts",
                "message": "Consider using balance concepts: 'weigh the interests', 'balance of equities', 'competing interests'",
                "priority": "MEDIUM"
            })

        return FunctionResult(
            success=True,
            value={
                "good_phrases_found": len(found_good),
                "has_balancing_test_phrase": has_balancing_test,
                "recommendations": recommendations,
                "impact": "Positive - successful movants use these concepts"
            },
            metadata={
                "feature_name": "use_balance_concepts",
                "priority": "HIGH"
            }
        )

