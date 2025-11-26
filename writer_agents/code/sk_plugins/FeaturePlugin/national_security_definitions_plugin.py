#!/usr/bin/env python3
"""
National Security Definitions Plugin

Enforces inclusion of national security definition phrases that are predictive
of successful sealing motions based on CatBoost analysis.

Top feature: total_ns_definitions (importance: 3.67)
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


class NationalSecurityDefinitionsPlugin(BaseFeaturePlugin):
    """Plugin to enforce national security definition phrases."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "national_security_definitions", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("NationalSecurityDefinitionsPlugin initialized")

    def _detect_ns_definitions(self, text: str) -> Dict[str, Any]:
        """Detect national security definition phrases in text."""
        text_lower = text.lower()

        # National Security Definition Phrases (from CatBoost analysis)
        ns_definitions = [
            "matter of national security",
            "national security interest",
            "national security concern",
            "national security implications",
            "national security risk",
            "national security threat",
            "national security classification",
            "national security exception",
            "national security exemption",
            "national security privilege",
        ]

        found_phrases = []
        phrase_locations = []

        for phrase in ns_definitions:
            if phrase in text_lower:
                found_phrases.append(phrase)
                # Find all occurrences
                for match in re.finditer(re.escape(phrase), text_lower):
                    phrase_locations.append({
                        "phrase": phrase,
                        "start": match.start(),
                        "end": match.end(),
                        "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    })

        # Target: At least 2-3 different NS definition phrases (based on successful cases)
        target_count = 2
        min_count = 1

        return {
            "found_phrases": found_phrases,
            "unique_phrases_count": len(found_phrases),
            "total_mentions": len(phrase_locations),
            "phrase_locations": phrase_locations[:10],  # Limit for reporting
            "target_count": target_count,
            "min_count": min_count,
            "meets_target": len(found_phrases) >= target_count,
            "meets_minimum": len(found_phrases) >= min_count
        }

    async def generate_edit_requests(
        self,
        text: str,
        structure: "DocumentStructure",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """Generate edit requests to add missing NS definition phrases."""
        detection = self._detect_ns_definitions(text)
        edit_requests = []

        # Check if we need more NS definition phrases
        if not detection["meets_target"]:
            # Find best location to add NS definitions (typically in introduction or legal standard section)
            best_location = self._find_best_location_for_ns_definitions(structure)

            if best_location:
                # Generate suggested text with NS definition phrases
                suggested_text = self._generate_ns_definition_text(detection)

                priority_int = 85 if not detection["meets_minimum"] else 60
                edit_requests.append(EditRequest(
                    plugin_name="national_security_definitions",
                    location=best_location,
                    edit_type="insert",
                    content=suggested_text,
                    priority=priority_int,
                    affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                    metadata={
                        "detection": detection,
                        "feature": "national_security_definitions",
                        "target_count": detection.get("target_count", 0),
                        "current_count": detection.get("unique_phrases_count", 0),
                        "meets_minimum": detection.get("meets_minimum", False)
                    },
                    reason=f"Add national security definition phrases (found {detection['unique_phrases_count']}, target: {detection['target_count']})"
                ))

        return edit_requests

    def _find_best_location_for_ns_definitions(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find best location to add NS definitions."""
        # Prefer Introduction or Legal Standard sections
        sections = structure.get_sections()

        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in ["introduction", "legal standard", "background", "statement"]):
                return DocumentLocation(
                    section_name=section.get("title", ""),
                    paragraph_index=0,
                    character_offset=section.get("start_char", 0),
                    position_type="after"
                )

        # Fallback: First section
        if sections:
            return DocumentLocation(
                section_name=sections[0].get("title", ""),
                paragraph_index=0,
                character_offset=sections[0].get("start_char", 0),
                position_type="after"
            )

        return None

    def _generate_ns_definition_text(self, detection: Dict[str, Any]) -> str:
        """Generate suggested text with NS definition phrases."""
        # Phrases not yet used
        all_phrases = [
            "matter of national security",
            "national security interest",
            "national security concern",
            "national security implications",
            "national security risk",
        ]

        used_phrases = set(phrase.lower() for phrase in detection["found_phrases"])
        missing_phrases = [p for p in all_phrases if p.lower() not in used_phrases]

        if not missing_phrases:
            return ""

        # Generate natural text incorporating missing phrases
        phrases_to_add = missing_phrases[:2]  # Add 2 most important

        suggestions = []
        if "matter of national security" in phrases_to_add:
            suggestions.append("This case presents a matter of national security")
        if "national security interest" in phrases_to_add:
            suggestions.append("the national security interests at stake")
        if "national security concern" in phrases_to_add:
            suggestions.append("genuine national security concerns")
        if "national security implications" in phrases_to_add:
            suggestions.append("significant national security implications")

        if suggestions:
            return f"Consider adding language such as: {', '.join(suggestions[:2])}."

        return ""

    async def validate_national_security_definitions(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate that national security definition phrases are present."""
        text = document.get_full_text()
        detection = self._detect_ns_definitions(text)

        issues = []
        recommendations = []

        if not detection["meets_minimum"]:
            issues.append({
                "type": "missing_ns_definitions",
                "severity": "high",
                "message": f"No national security definition phrases found. Found: {detection['unique_phrases_count']}, Minimum: {detection['min_count']}",
                "suggestion": "Add explicit national security definition phrases such as 'matter of national security', 'national security interest', or 'national security concern'."
            })
        elif not detection["meets_target"]:
            recommendations.append({
                "type": "insufficient_ns_definitions",
                "priority": "medium",
                "message": f"Found {detection['unique_phrases_count']} NS definition phrases, but target is {detection['target_count']} for optimal strength.",
                "suggestion": "Add additional national security definition phrases to strengthen the argument."
            })

        return FunctionResult(
            success=detection["meets_minimum"],
            data={
                "found_phrases": detection["found_phrases"],
                "unique_phrases_count": detection["unique_phrases_count"],
                "total_mentions": detection["total_mentions"],
                "target_count": detection["target_count"],
                "meets_target": detection["meets_target"],
                "meets_minimum": detection["meets_minimum"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 3.67,  # From CatBoost analysis
                "feature_rank": 8
            },
            message="National security definitions validation complete"
        )

