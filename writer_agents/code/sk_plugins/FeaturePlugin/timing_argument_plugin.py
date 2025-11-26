#!/usr/bin/env python3
"""
Timing Argument Plugin

Enforces the critical timing argument:
- User first wrote Harvard on April 7th
- User filed HK claim in court (date)
- Trump's June 4th proclamation came into effect 2 days after HK claim filed
- Therefore, Harvard's ties to Xi Jinping's daughter became a recognized matter
  of national security since April 7th

This timing argument is crucial for showing the national security issue was
present from the beginning of the case.
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


class TimingArgumentPlugin(BaseFeaturePlugin):
    """Plugin to enforce the timing argument connecting events to establish NS since April 7th."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "timing_argument", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("TimingArgumentPlugin initialized")

    def _get_provider(self):
        """Get CaseFactsProvider lazily (fetches fresh each time since it's set after plugin init)."""
        return get_case_facts_provider()

    def _detect_timing_argument(self, text: str) -> Dict[str, Any]:
        """Detect the timing argument connecting April 7th, HK claim, June 4th proclamation."""
        text_lower = text.lower()

        # Key dates and events
        april_7_indicators = [
            r"april\s+7",
            r"april\s+7th",
            r"first\s+wrote\s+harvard",
            r"initially\s+contacted\s+harvard",
            r"first\s+contacted\s+harvard",
        ]

        hk_claim_indicators = [
            r"hong\s+kong\s+claim",
            r"hk\s+claim",
            r"filed.*hong\s+kong",
            r"filed.*hk",
            r"hong\s+kong.*filed",
        ]

        june_4_indicators = [
            r"june\s+4",
            r"june\s+4th",
            r"proclamation.*june",
            r"june.*proclamation",
        ]

        timing_connection_indicators = [
            r"two\s+days\s+after",
            r"2\s+days\s+after",
            r"came\s+into\s+effect.*days\s+after",
            r"effective.*days\s+after",
            r"since.*april\s+7",
            r"since.*first\s+wrote",
            r"became.*national\s+security",
            r"recognized.*national\s+security.*since",
        ]

        found_april_7 = []
        found_hk_claim = []
        found_june_4 = []
        found_timing_connection = []

        # Check for April 7th
        for pattern in april_7_indicators:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_april_7.append(pattern)

        # Check for HK claim
        for pattern in hk_claim_indicators:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_hk_claim.append(pattern)

        # Check for June 4th
        for pattern in june_4_indicators:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_june_4.append(pattern)

        # Check for timing connections
        for pattern in timing_connection_indicators:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_timing_connection.append(pattern)

        # Check if there's explicit timing argument connecting all pieces
        has_complete_timing_argument = False
        if found_april_7 and found_june_4:
            # Look for sentences/paragraphs that connect the timing
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                has_april_7 = any(term in sentence_lower for term in ["april 7", "first wrote", "initially contacted"])
                has_june_4 = any(term in sentence_lower for term in ["june 4", "proclamation"])
                has_timing = any(term in sentence_lower for term in ["days after", "since", "became", "recognized"])
                has_ns = "national security" in sentence_lower
                if has_april_7 and has_june_4 and (has_timing or has_ns):
                    has_complete_timing_argument = True
                    break

        return {
            "has_april_7": len(found_april_7) > 0,
            "has_hk_claim": len(found_hk_claim) > 0,
            "has_june_4": len(found_june_4) > 0,
            "has_timing_connection": len(found_timing_connection) > 0,
            "has_complete_timing_argument": has_complete_timing_argument,
            "found_april_7": found_april_7,
            "found_hk_claim": found_hk_claim,
            "found_june_4": found_june_4,
            "found_timing_connection": found_timing_connection,
            "meets_target": has_complete_timing_argument,
            "meets_minimum": len(found_april_7) > 0 and len(found_june_4) > 0
        }

    async def generate_edit_requests(
        self,
        text: str,
        structure: "DocumentStructure",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """Generate edit requests to add timing argument."""
        detection = self._detect_timing_argument(text)
        edit_requests = []

        if not detection["has_complete_timing_argument"]:
            # Find best location (typically in factual background or timeline)
            best_location = self._find_best_location_for_timing(structure)

            if best_location:
                # Get actual timing facts from CaseFactsProvider if available
                timing_context = ""
                provider = self._get_provider()
                if provider:
                    # Get OGC email facts which contain April 7th reference
                    ogc_facts = provider.get_fact_block('ogc_email_1_threat')
                    hk_facts = provider.get_fact_block('hk_allegation_ccp_family')
                    if ogc_facts:
                        timing_context = f" {ogc_facts}"
                    elif hk_facts:
                        timing_context = f" {hk_facts}"

                if not detection["has_april_7"]:
                    if timing_context:
                        suggested_text = f"Plaintiff first wrote Harvard on April 7th regarding these matters. The June 4th proclamation came into effect 2 days after Plaintiff filed the Hong Kong claim in this court, making Harvard's ties to Xi Jinping's daughter a recognized matter of national security since April 7th.{timing_context}"
                    else:
                        suggested_text = "Plaintiff first wrote Harvard on April 7th regarding these matters. The June 4th proclamation came into effect 2 days after Plaintiff filed the Hong Kong claim in this court, making Harvard's ties to Xi Jinping's daughter a recognized matter of national security since April 7th."
                elif not detection["has_june_4"]:
                    suggested_text = "The June 4th proclamation came into effect 2 days after Plaintiff filed the Hong Kong claim, making this a recognized matter of national security since Plaintiff first wrote Harvard on April 7th."
                elif not detection["has_timing_connection"]:
                    suggested_text = "The June 4th proclamation came into effect 2 days after Plaintiff filed the Hong Kong claim in this court, making Harvard's ties to Xi Jinping's daughter a recognized matter of national security since the time Plaintiff first wrote Harvard on April 7th."
                else:
                    suggested_text = "Because the June 4th proclamation came into effect 2 days after Plaintiff filed the HK claim, Harvard's ties to Xi Jinping's daughter became a recognized matter of national security since April 7th, when Plaintiff first wrote Harvard."

                edit_requests.append(EditRequest(
                    plugin_name="timing_argument",
                    location=best_location,
                    edit_type="insert",
                    content=suggested_text,
                    priority=85,
                    affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                    metadata={
                        "detection": detection,
                        "feature": "timing_argument",
                        "has_timing_argument": detection.get("has_timing_argument", False)
                    },
                    reason="Add timing argument connecting April 7th, HK claim filing, and June 4th proclamation to establish NS since beginning"
                ))

        return edit_requests

    def _find_best_location_for_timing(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find best location to add timing argument."""
        sections = structure.get_sections()

        # Prefer sections that discuss factual background, timeline, or procedural history
        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in [
                "background", "facts", "factual", "statement", "timeline",
                "procedural history", "national security", "chronology"
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

    async def validate_timing_argument(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate that timing argument is present."""
        text = document.get_full_text()
        detection = self._detect_timing_argument(text)

        issues = []
        recommendations = []

        if not detection["meets_minimum"]:
            # Get actual timing facts from CaseFactsProvider if available
            timing_suggestion = "Add timing argument: 'The June 4th proclamation came into effect 2 days after Plaintiff filed the Hong Kong claim in this court, making Harvard's ties to Xi Jinping's daughter a recognized matter of national security since the time Plaintiff first wrote Harvard on April 7th.'"
            provider = self._get_provider()
            if provider:
                ogc_facts = provider.get_fact_block('ogc_email_1_threat')
                if ogc_facts and "April 7" in ogc_facts:
                    timing_suggestion = f"Add timing argument connecting the events. {ogc_facts} The June 4th proclamation came into effect 2 days after Plaintiff filed the Hong Kong claim, establishing this as a recognized matter of national security since April 7th."

            issues.append({
                "type": "missing_timing_argument",
                "severity": "high",
                "message": "Timing argument not fully present. Missing key dates (April 7th or June 4th).",
                "suggestion": timing_suggestion
            })
        elif not detection["has_complete_timing_argument"]:
            recommendations.append({
                "type": "incomplete_timing_argument",
                "priority": "high",
                "message": "Key dates present but not explicitly connected in timing argument.",
                "suggestion": "Explicitly connect the timing: explain that the June 4th proclamation (2 days after HK claim filed) made this a recognized matter of national security since April 7th, when Plaintiff first wrote Harvard."
            })

        return FunctionResult(
            success=detection["has_complete_timing_argument"],
            data={
                "has_april_7": detection["has_april_7"],
                "has_hk_claim": detection["has_hk_claim"],
                "has_june_4": detection["has_june_4"],
                "has_timing_connection": detection["has_timing_connection"],
                "has_complete_timing_argument": detection["has_complete_timing_argument"],
                "meets_target": detection["meets_target"],
                "meets_minimum": detection["meets_minimum"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 4.5,  # Very high - establishes NS from beginning
                "feature_rank": 14
            },
            message="Timing argument validation complete"
        )

