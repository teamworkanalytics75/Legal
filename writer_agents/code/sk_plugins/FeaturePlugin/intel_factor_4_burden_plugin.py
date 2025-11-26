#!/usr/bin/env python3
"""
Enforces Intel Factor 4: undue burden/intrusiveness

Intel, 542 U.S. at 264-65
Key cases: Pishevar v. Fusion GPS, O'Keeffe v. Adelson
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


class IntelFactor4BurdenPlugin(BaseFeaturePlugin):
    """Plugin: Enforces Intel Factor 4: undue burden/intrusiveness"""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "intel_factor_4_burden", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("IntelFactor4BurdenPlugin initialized")

    async def validate_intel_factor_4_burden(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate intel_factor_4_burden in document.

        Enforces Intel Factor 4: undue burden/intrusiveness
        - Request is narrowly tailored
        - Not overly broad
        - Proportional to foreign proceeding
        - Specific custodian(s) identified
        - Time period limited
        - ... and 1 more items
        """
        text = document.get_full_text()
        text_lower = text.lower()

        issues = []
        recommendations = []

        # Check checklist items
        checklist_items = ['Request is narrowly tailored', 'Not overly broad', 'Proportional to foreign proceeding', 'Specific custodian(s) identified', 'Time period limited', 'Document categories specific']
        for item in checklist_items:
            # Simple keyword-based validation
            keywords = self._extract_keywords(item)
            found = any(keyword.lower() in text_lower for keyword in keywords if len(keyword) > 3)
            
            if not found:
                issues.append({
                    "type": "missing_checklist_item",
                    "severity": "medium",
                    "message": f"Checklist item not found: {item}",
                    "suggestion": "Ensure this requirement is addressed in the document"
                })

        # Check for required phrases/concepts (simple lowercase string contains)
        required_phrases = [
            "request is narrowly",
            "proportional to foreign",
            "specific custodian",
            "time period",
        ]
        for phrase in required_phrases:
            if phrase not in text_lower:
                recommendations.append({
                    "type": "add_required_phrase",
                    "message": f"Consider adding: {phrase}",
                    "priority": "medium"
                })

        # Check for problematic phrases
        bad_phrases = []
        for phrase in bad_phrases:
            if phrase.lower() in text_lower:
                matches = list(re.finditer(re.escape(phrase.lower()), text_lower))
                for match in matches:
                    issues.append({
                        "type": "problematic_phrase",
                        "severity": "medium",
                        "message": f"Found '{phrase}' at position {match.start()}",
                        "location": DocumentLocation(
                            start_line=text[:match.start()].count('\n') + 1,
                            end_line=text[:match.start()].count('\n') + 1,
                            start_char=match.start(),
                            end_char=match.end()
                        ),
                        "suggestion": "Review and potentially revise this language"
                    })

        return FunctionResult(
            success=len(issues) == 0,
            value={
                "issues_found": len(issues),
                "checklist_items_checked": len(checklist_items),
                "issues": issues,
                "recommendations": recommendations
            },
            metadata={
                "feature_name": "intel_factor_4_burden",
                "priority": "medium",
                "threshold": 0.85,
                "legal_standard": "Intel, 542 U.S. at 264-65"
            }
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from checklist item."""
        # Simple extraction - remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 3]
