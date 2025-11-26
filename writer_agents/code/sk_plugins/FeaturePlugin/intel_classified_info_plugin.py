#!/usr/bin/env python3
"""
Intel/Classified Information Plugin

Enforces mention of intelligence/classified information features.
This is the #1 most predictive feature (importance: 17.01) according to CatBoost analysis.
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


class IntelClassifiedInfoPlugin(BaseFeaturePlugin):
    """Plugin to enforce intelligence/classified information mentions."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "intel_classified_info", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("IntelClassifiedInfoPlugin initialized")

    def _detect_intel_indicators(self, text: str) -> Dict[str, Any]:
        """Detect intelligence/classified information indicators."""
        text_lower = text.lower()

        # Intelligence/classified indicators (from CatBoost analysis - #1 feature)
        intel_indicators = [
            "classified information",
            "classified material",
            "state secret",
            "intelligence",
            "intelligence source",
            "intelligence method",
            "sensitive compartmented",
            "top secret",
            "secret",
            "confidential",
        ]

        found_indicators = []
        indicator_locations = []

        for indicator in intel_indicators:
            pattern = r'\b' + re.escape(indicator) + r'\b'
            matches = list(re.finditer(pattern, text_lower))

            if matches:
                found_indicators.append(indicator)
                for match in matches:
                    # Check if it's in national security context
                    context = text[max(0, match.start()-100):min(len(text), match.end()+100)].lower()
                    has_ns_context = any(
                        ns_term in context for ns_term in [
                            "national security", "security", "intelligence", "classified"
                        ]
                    )

                    indicator_locations.append({
                        "indicator": indicator,
                        "start": match.start(),
                        "end": match.end(),
                        "has_ns_context": has_ns_context,
                        "context": context[:200]
                    })

        # Count total intel indicators (this is the #1 predictive feature)
        total_intel_count = len(indicator_locations)

        # Target: At least 2-3 intel indicators (based on successful cases)
        target_count = 2
        min_count = 1

        return {
            "found_indicators": found_indicators,
            "unique_indicators_count": len(found_indicators),
            "total_intel_count": total_intel_count,
            "indicator_locations": indicator_locations[:10],
            "target_count": target_count,
            "min_count": min_count,
            "meets_target": total_intel_count >= target_count,
            "meets_minimum": total_intel_count >= min_count
        }

    async def generate_edit_requests(
        self,
        text: str,
        structure: "DocumentStructure",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """Generate edit requests to add intel/classified information mentions."""
        detection = self._detect_intel_indicators(text)
        edit_requests = []

        if not detection["meets_target"]:
            # Find best location (typically in factual background or national security section)
            best_location = self._find_national_security_section(structure) or self._find_factual_background_section(structure)

            if best_location:
                # Suggest adding intel indicators if applicable
                all_indicators = [
                    "classified information",
                    "intelligence",
                    "state secret",
                    "confidential"
                ]

                missing_indicators = [ind for ind in all_indicators if ind not in detection["found_indicators"]]

                if missing_indicators:
                    suggested_text = f"If applicable, consider mentioning: {', '.join(missing_indicators[:2])} in the context of national security."

                    priority_int = 85 if not detection["meets_minimum"] else 60
                    edit_requests.append(EditRequest(
                        plugin_name="intel_classified_info",
                        location=best_location,
                        edit_type="insert",
                        content=suggested_text,
                        priority=priority_int,
                        affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                        metadata={
                            "detection": detection,
                            "feature": "intel_classified_info",
                            "target_count": detection.get("target_count", 0),
                            "current_count": detection.get("total_intel_count", 0),
                            "missing_indicators": missing_indicators[:2]
                        },
                        reason=f"Add intelligence/classified information mentions (found {detection['total_intel_count']}, target: {detection['target_count']}) - This is the #1 most predictive feature"
                    ))

        return edit_requests

    def _find_national_security_section(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find national security section."""
        sections = structure.get_sections()

        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in ["national security", "security", "intelligence"]):
                return DocumentLocation(
                    section_name=section.get("title", ""),
                    paragraph_index=0,
                    character_offset=section.get("start_char", 0),
                    position_type="after"
                )

        return None

    def _find_factual_background_section(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find factual background section."""
        sections = structure.get_sections()

        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in ["background", "facts", "factual"]):
                return DocumentLocation(
                    section_name=section.get("title", ""),
                    paragraph_index=0,
                    character_offset=section.get("start_char", 0),
                    position_type="after"
                )

        return None

    async def validate_intel_classified_info(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate that intelligence/classified information is mentioned."""
        text = document.get_full_text()
        detection = self._detect_intel_indicators(text)

        issues = []
        recommendations = []

        if not detection["meets_minimum"]:
            issues.append({
                "type": "missing_intel_indicators",
                "severity": "high",
                "message": f"No intelligence/classified information mentioned. This is the #1 most predictive feature (importance: 17.01). Found: {detection['total_intel_count']}, Minimum: {detection['min_count']}",
                "suggestion": "If applicable, mention intelligence sources, classified information, state secrets, or confidential materials in the context of national security."
            })
        elif not detection["meets_target"]:
            recommendations.append({
                "type": "insufficient_intel_indicators",
                "priority": "high",
                "message": f"Found {detection['total_intel_count']} intel/classified mentions, but target is {detection['target_count']} for optimal strength (this is the #1 predictive feature).",
                "suggestion": "Add additional intelligence/classified information mentions if applicable."
            })

        return FunctionResult(
            success=detection["meets_minimum"],
            data={
                "found_indicators": detection["found_indicators"],
                "unique_indicators_count": detection["unique_indicators_count"],
                "total_intel_count": detection["total_intel_count"],
                "target_count": detection["target_count"],
                "meets_target": detection["meets_target"],
                "meets_minimum": detection["meets_minimum"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 17.01,  # From CatBoost analysis - #1 feature
                "feature_rank": 1
            },
            message="Intel/classified information validation complete"
        )

