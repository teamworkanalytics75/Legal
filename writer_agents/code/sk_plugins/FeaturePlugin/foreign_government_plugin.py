#!/usr/bin/env python3
"""
Foreign Government Plugin

Enforces mention of foreign government involvement in national security context.
This is predictive (importance: 3.40) according to CatBoost analysis.
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


class ForeignGovernmentPlugin(BaseFeaturePlugin):
    """Plugin to enforce foreign government involvement language."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "foreign_government", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("ForeignGovernmentPlugin initialized")

    def _detect_foreign_government(self, text: str) -> Dict[str, Any]:
        """Detect foreign government involvement indicators."""
        text_lower = text.lower()

        # Foreign government indicators (from CatBoost analysis)
        foreign_gov_indicators = [
            "foreign government",
            "foreign state",
            "foreign agent",
            "foreign interference",
            "foreign influence",
            "sovereign immunity",
            "foreign sovereign",
            "foreign entity",
        ]

        found_indicators = []
        indicator_locations = []

        for indicator in foreign_gov_indicators:
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

        # Check if foreign government is linked to national security
        has_ns_link = any(loc["has_ns_context"] for loc in indicator_locations)

        # Target: At least 1-2 foreign government indicators with NS context
        target_count = 1
        min_count = 1

        return {
            "found_indicators": found_indicators,
            "unique_indicators_count": len(found_indicators),
            "total_mentions": len(indicator_locations),
            "has_ns_link": has_ns_link,
            "indicator_locations": indicator_locations[:10],
            "target_count": target_count,
            "min_count": min_count,
            "meets_target": len(found_indicators) >= target_count and has_ns_link,
            "meets_minimum": len(found_indicators) >= min_count
        }

    async def generate_edit_requests(
        self,
        text: str,
        structure: "DocumentStructure",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """Generate edit requests to add foreign government involvement language."""
        detection = self._detect_foreign_government(text)
        edit_requests = []

        if not detection["meets_target"]:
            if not detection["meets_minimum"]:
                # No foreign government mention at all
                best_location = self._find_factual_background_section(structure)
                if best_location:
                    suggested_text = "This case involves foreign government interference/influence with national security implications."
                    edit_requests.append(EditRequest(
                        plugin_name="foreign_government",
                        location=best_location,
                        edit_type="insert",
                        content=suggested_text,
                        priority=60,
                        affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                        metadata={
                            "detection": detection,
                            "feature": "foreign_government",
                            "meets_target": detection.get("meets_target", False),
                            "meets_minimum": detection.get("meets_minimum", False),
                            "has_ns_link": detection.get("has_ns_link", False)
                        },
                        reason="Add foreign government involvement language (predictive feature)"
                    ))
            elif not detection["has_ns_link"]:
                # Foreign government mentioned but not linked to NS
                best_location = self._find_national_security_section(structure)
                if best_location:
                    suggested_text = "The foreign government involvement creates significant national security concerns."
                    edit_requests.append(EditRequest(
                        plugin_name="foreign_government",
                        location=best_location,
                        edit_type="insert",
                        content=suggested_text,
                        priority=60,
                        affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                        metadata={
                            "detection": detection,
                            "feature": "foreign_government",
                            "meets_target": detection.get("meets_target", False),
                            "meets_minimum": detection.get("meets_minimum", False),
                            "has_ns_link": detection.get("has_ns_link", False)
                        },
                        reason="Link foreign government involvement to national security"
                    ))

        return edit_requests

    def _find_factual_background_section(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find factual background section."""
        sections = structure.get_sections()

        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in ["background", "facts", "factual", "statement"]):
                return DocumentLocation(
                    section_name=section.get("title", ""),
                    paragraph_index=0,
                    character_offset=section.get("start_char", 0),
                    position_type="after"
                )

        return None

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

    async def validate_foreign_government(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate that foreign government involvement is mentioned with NS context."""
        text = document.get_full_text()
        detection = self._detect_foreign_government(text)

        issues = []
        recommendations = []

        if not detection["meets_minimum"]:
            issues.append({
                "type": "missing_foreign_government",
                "severity": "medium",
                "message": "No foreign government involvement mentioned.",
                "suggestion": "If applicable, mention foreign government involvement and link it to national security concerns."
            })
        elif not detection["has_ns_link"]:
            recommendations.append({
                "type": "foreign_gov_not_linked_to_ns",
                "priority": "medium",
                "message": "Foreign government mentioned but not explicitly linked to national security.",
                "suggestion": "Link foreign government involvement to national security concerns: 'The foreign government involvement creates significant national security concerns.'"
            })

        return FunctionResult(
            success=detection["meets_target"],
            data={
                "found_indicators": detection["found_indicators"],
                "unique_indicators_count": detection["unique_indicators_count"],
                "has_ns_link": detection["has_ns_link"],
                "meets_target": detection["meets_target"],
                "meets_minimum": detection["meets_minimum"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 3.40,  # From CatBoost analysis
                "feature_rank": 9
            },
            message="Foreign government validation complete"
        )

