#!/usr/bin/env python3
"""
Harvard Lawsuit Plugin

Enforces mention of Harvard's lawsuit in response to the June 4th proclamation,
which further demonstrates that Harvard's ties to the CCP are a matter of
national security concern.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult, EditRequest, DocumentLocation
from .CaseFactsProvider import get_case_facts_provider

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class HarvardLawsuitPlugin(BaseFeaturePlugin):
    """Plugin to enforce mention of Harvard's lawsuit in response to the proclamation."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "harvard_lawsuit", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("HarvardLawsuitPlugin initialized")

    def _get_provider(self):
        """Get CaseFactsProvider lazily (fetches fresh each time since it's set after plugin init)."""
        return get_case_facts_provider()

    def _detect_harvard_lawsuit(self, text: str) -> Dict[str, Any]:
        """Detect mention of Harvard's lawsuit in response to the proclamation."""
        text_lower = text.lower()

        # Harvard lawsuit indicators
        lawsuit_indicators = [
            r"harvard.*lawsuit",
            r"harvard.*sued",
            r"harvard.*suing",
            r"harvard.*filed.*suit",
            r"harvard.*litigation",
            r"harvard.*challenged",
            r"harvard.*proclamation",
            r"harvard.*june\s+4",
            r"harvard.*trump",
        ]

        # Connection to proclamation
        proclamation_connection = [
            r"in\s+response\s+to.*proclamation",
            r"response.*june\s+4",
            r"challenged.*proclamation",
            r"challenged.*june\s+4",
            r"sued.*proclamation",
            r"lawsuit.*proclamation",
        ]

        # National security context
        ns_context = [
            r"national\s+security",
            r"ccp",
            r"chinese\s+communist\s+party",
            r"ties.*china",
        ]

        found_lawsuit = []
        found_connection = []
        found_ns_context = []

        # Check for lawsuit mentions
        for pattern in lawsuit_indicators:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_lawsuit.append(pattern)

        # Check for connection to proclamation
        for pattern in proclamation_connection:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_connection.append(pattern)

        # Check for national security context
        for pattern in ns_context:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_ns_context.append(pattern)

        # Check if there's explicit connection between lawsuit, proclamation, and NS
        has_explicit_connection = False
        if found_lawsuit:
            # Look for sentences/paragraphs that connect lawsuit to proclamation and NS
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                has_lawsuit = any(term in sentence_lower for term in ["harvard", "lawsuit", "sued", "challenged"])
                has_proclamation = any(term in sentence_lower for term in ["june 4", "proclamation", "trump"])
                has_ns = any(term in sentence_lower for term in ["national security", "ccp", "chinese communist party"])
                if has_lawsuit and (has_proclamation or has_ns):
                    has_explicit_connection = True
                    break

        return {
            "has_lawsuit": len(found_lawsuit) > 0,
            "has_proclamation_connection": len(found_connection) > 0,
            "has_ns_context": len(found_ns_context) > 0,
            "has_explicit_connection": has_explicit_connection,
            "found_lawsuit": found_lawsuit,
            "found_connection": found_connection,
            "found_ns_context": found_ns_context,
            "meets_target": has_explicit_connection,
            "meets_minimum": len(found_lawsuit) > 0
        }

    async def generate_edit_requests(
        self,
        text: str,
        structure: "DocumentStructure",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """Generate edit requests to add Harvard lawsuit context."""
        detection = self._detect_harvard_lawsuit(text)
        edit_requests = []

        if not detection["has_explicit_connection"]:
            # Find best location (typically in factual background or timeline)
            best_location = self._find_best_location_for_lawsuit(structure)

            if best_location:
                # Get actual lawsuit facts from CaseFactsProvider if available
                lawsuit_context = ""
                provider = self._get_provider()
                if provider:
                    # Try to get relevant facts about Harvard lawsuit and proclamation
                    hk_facts = provider.get_fact_block('hk_allegation_ccp_family')
                    if hk_facts:
                        lawsuit_context = f" {hk_facts}"

                if not detection["has_lawsuit"]:
                    if lawsuit_context:
                        suggested_text = f"In response to the June 4th proclamation, Harvard filed a lawsuit challenging the proclamation, further demonstrating that Harvard's ties to the CCP are a matter of national security concern.{lawsuit_context}"
                    else:
                        suggested_text = "In response to the June 4th proclamation, Harvard filed a lawsuit challenging the proclamation, further demonstrating that Harvard's ties to the CCP are a matter of national security concern."
                elif not detection["has_proclamation_connection"]:
                    suggested_text = "Harvard's lawsuit was filed in response to the June 4th proclamation that identified Harvard's ties to the CCP as a national security risk."
                else:
                    suggested_text = "Harvard's lawsuit in response to the June 4th proclamation further demonstrates that Harvard's ties to the CCP are a recognized matter of national security concern."

                edit_requests.append(EditRequest(
                    plugin_name="harvard_lawsuit",
                    location=best_location,
                    edit_type="insert",
                    content=suggested_text,
                    priority=60,
                    affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                    metadata={
                        "detection": detection,
                        "feature": "harvard_lawsuit",
                        "has_lawsuit": detection.get("has_lawsuit", False),
                        "has_proclamation_connection": detection.get("has_proclamation_connection", False)
                    },
                    reason="Add Harvard lawsuit context showing it was in response to the June 4th proclamation"
                ))

        return edit_requests

    def _find_best_location_for_lawsuit(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find best location to add lawsuit context."""
        sections = structure.get_sections()

        # Prefer sections that discuss factual background, timeline, or procedural history
        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in [
                "background", "facts", "factual", "statement", "timeline",
                "procedural history", "national security", "related litigation"
            ]):
                return DocumentLocation(
                    section_name=section.get("title", ""),
                    paragraph_index=0,
                    character_offset=section.get("start_char", 0),
                    position_type="after"
                )

        # Fallback: first section
        if sections:
            return DocumentLocation(
                section_name=sections[0].get("title", ""),
                paragraph_index=0,
                character_offset=sections[0].get("start_char", 0),
                position_type="after"
            )

        return None

    async def validate_harvard_lawsuit(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate that Harvard's lawsuit is mentioned."""
        text = document.get_full_text()
        detection = self._detect_harvard_lawsuit(text)

        issues = []
        recommendations = []

        if not detection["meets_minimum"]:
            # Get actual lawsuit context from CaseFactsProvider if available
            lawsuit_suggestion = "Add mention that Harvard filed a lawsuit in response to the June 4th proclamation, further demonstrating that Harvard's ties to the CCP are a matter of national security concern."
            provider = self._get_provider()
            if provider:
                hk_facts = provider.get_fact_block('hk_allegation_ccp_family')
                if hk_facts:
                    lawsuit_suggestion = f"Add mention that Harvard filed a lawsuit in response to the June 4th proclamation. {hk_facts}"

            issues.append({
                "type": "missing_harvard_lawsuit",
                "severity": "medium",
                "message": "No mention of Harvard's lawsuit in response to the June 4th proclamation found.",
                "suggestion": lawsuit_suggestion
            })
        elif not detection["has_explicit_connection"]:
            recommendations.append({
                "type": "lawsuit_not_connected",
                "priority": "medium",
                "message": "Harvard lawsuit mentioned, but not explicitly connected to the proclamation or national security context.",
                "suggestion": "Explicitly connect Harvard's lawsuit to the June 4th proclamation and national security: 'Harvard filed a lawsuit in response to the June 4th proclamation that identified Harvard's ties to the CCP as a national security risk.'"
            })

        return FunctionResult(
            success=detection["meets_target"],
            data={
                "has_lawsuit": detection["has_lawsuit"],
                "has_proclamation_connection": detection["has_proclamation_connection"],
                "has_ns_context": detection["has_ns_context"],
                "has_explicit_connection": detection["has_explicit_connection"],
                "meets_target": detection["meets_target"],
                "meets_minimum": detection["meets_minimum"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 3.5,  # Important supporting evidence
                "feature_rank": 13
            },
            message="Harvard lawsuit validation complete"
        )

