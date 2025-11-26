#!/usr/bin/env python3
"""
Trump June 4th Proclamation Plugin

Enforces mention of President Trump's June 4, 2020 proclamation and fact sheet
that explicitly identified Harvard's ties to the Chinese Communist Party (CCP)
as a national security risk.

This is critical timing context - the proclamation came into effect 2 days after
the user filed their HK claim in court, making Harvard's ties to Xi Jinping's
daughter a recognized matter of national security since the time the user first
wrote Harvard (April 7th).
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


class TrumpJune4ProclamationPlugin(BaseFeaturePlugin):
    """Plugin to enforce mention of Trump's June 4th proclamation on Harvard-CCP ties."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "trump_june4_proclamation", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("TrumpJune4ProclamationPlugin initialized")

    def _get_provider(self):
        """Get CaseFactsProvider lazily (fetches fresh each time since it's set after plugin init)."""
        return get_case_facts_provider()

    def _detect_proclamation_mention(self, text: str) -> Dict[str, Any]:
        """Detect mention of Trump's June 4th proclamation and fact sheet."""
        text_lower = text.lower()

        # Proclamation indicators
        proclamation_indicators = [
            r"trump.*june\s+4",
            r"june\s+4.*proclamation",
            r"proclamation.*june\s+4",
            r"president.*trump.*june\s+4",
            r"june\s+4.*2020.*proclamation",
            r"proclamation.*harvard",
            r"harvard.*proclamation",
        ]

        # Fact sheet indicators
        fact_sheet_indicators = [
            r"fact\s+sheet.*harvard",
            r"harvard.*fact\s+sheet",
            r"fact\s+sheet.*june\s+4",
            r"june\s+4.*fact\s+sheet",
            r"fact\s+sheet.*ccp",
            r"fact\s+sheet.*national\s+security",
        ]

        # Harvard-CCP tie indicators
        harvard_ccp_indicators = [
            r"harvard.*ccp",
            r"harvard.*chinese\s+communist\s+party",
            r"harvard.*ties.*ccp",
            r"harvard.*ties.*china",
            r"harvard.*ties.*chinese",
            r"harvard.*national\s+security\s+risk",
            r"harvard.*national\s+security",
        ]

        # National security risk language
        ns_risk_indicators = [
            r"national\s+security\s+risk",
            r"national\s+security\s+threat",
            r"national\s+security\s+concern",
            r"identified.*national\s+security",
            r"declared.*national\s+security",
        ]

        found_proclamation = []
        found_fact_sheet = []
        found_harvard_ccp = []
        found_ns_risk = []

        # Check for proclamation mentions
        for pattern in proclamation_indicators:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_proclamation.append(pattern)

        # Check for fact sheet mentions
        for pattern in fact_sheet_indicators:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_fact_sheet.append(pattern)

        # Check for Harvard-CCP ties
        for pattern in harvard_ccp_indicators:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_harvard_ccp.append(pattern)

        # Check for national security risk language
        for pattern in ns_risk_indicators:
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_ns_risk.append(pattern)

        # Check if there's explicit connection between proclamation and Harvard-CCP ties as NS risk
        has_explicit_connection = False
        if found_proclamation or found_fact_sheet:
            # Look for sentences/paragraphs that connect proclamation to Harvard-CCP-NS risk
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                has_proclamation = any(term in sentence_lower for term in ["june 4", "proclamation", "trump", "fact sheet"])
                has_harvard_ccp = any(term in sentence_lower for term in ["harvard", "ccp", "chinese communist party"])
                has_ns_risk = any(term in sentence_lower for term in ["national security risk", "national security threat", "national security concern"])
                if has_proclamation and has_harvard_ccp and has_ns_risk:
                    has_explicit_connection = True
                    break

        return {
            "has_proclamation": len(found_proclamation) > 0,
            "has_fact_sheet": len(found_fact_sheet) > 0,
            "has_harvard_ccp": len(found_harvard_ccp) > 0,
            "has_ns_risk": len(found_ns_risk) > 0,
            "has_explicit_connection": has_explicit_connection,
            "found_proclamation": found_proclamation,
            "found_fact_sheet": found_fact_sheet,
            "found_harvard_ccp": found_harvard_ccp,
            "found_ns_risk": found_ns_risk,
            "meets_target": has_explicit_connection,
            "meets_minimum": (len(found_proclamation) > 0 or len(found_fact_sheet) > 0) and len(found_harvard_ccp) > 0
        }

    async def generate_edit_requests(
        self,
        text: str,
        structure: "DocumentStructure",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """Generate edit requests to add Trump June 4th proclamation context."""
        detection = self._detect_proclamation_mention(text)
        edit_requests = []

        if not detection["has_explicit_connection"]:
            # Find best location (typically in factual background or legal context)
            best_location = self._find_best_location_for_proclamation(structure)

            if best_location:
                # Get actual case facts from CaseFactsProvider if available
                proclamation_context = ""
                provider = self._get_provider()
                if provider:
                    hk_facts = provider.get_fact_block('hk_allegation_ccp_family')
                    ogc_facts = provider.get_fact_block('ogc_email_1_threat')
                    if hk_facts and ("CCP" in hk_facts or "Xi Mingze" in hk_facts):
                        proclamation_context = f" {hk_facts}"
                    elif ogc_facts and "April 7" in ogc_facts:
                        proclamation_context = f" {ogc_facts}"

                if not detection["has_proclamation"] and not detection["has_fact_sheet"]:
                    if proclamation_context:
                        suggested_text = f"On June 4, 2020, President Trump issued a proclamation and fact sheet that explicitly identified Harvard's ties to the Chinese Communist Party (CCP) as a national security risk. This proclamation came into effect 2 days after Plaintiff filed the Hong Kong claim in this court, making Harvard's ties to Xi Jinping's daughter a recognized matter of national security since the time Plaintiff first wrote Harvard on April 7th.{proclamation_context}"
                    else:
                        suggested_text = "On June 4, 2020, President Trump issued a proclamation and fact sheet that explicitly identified Harvard's ties to the Chinese Communist Party (CCP) as a national security risk. This proclamation came into effect 2 days after Plaintiff filed the Hong Kong claim in this court, making Harvard's ties to Xi Jinping's daughter a recognized matter of national security since the time Plaintiff first wrote Harvard on April 7th."
                elif not detection["has_harvard_ccp"]:
                    suggested_text = "The June 4, 2020 proclamation and fact sheet explicitly identified Harvard's ties to the Chinese Communist Party (CCP) as a national security risk."
                elif not detection["has_ns_risk"]:
                    suggested_text = "The June 4, 2020 proclamation and fact sheet explicitly identified Harvard's ties to the CCP as a national security risk."
                else:
                    suggested_text = "The June 4, 2020 proclamation explicitly identified Harvard's ties to the CCP as a national security risk, providing official recognition that this matter concerns national security."

                edit_requests.append(EditRequest(
                    plugin_name="trump_june4_proclamation",
                    location=best_location,
                    edit_type="insert",
                    content=suggested_text,
                    priority=85,
                    affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                    metadata={
                        "detection": detection,
                        "feature": "trump_june4_proclamation",
                        "has_proclamation": detection.get("has_proclamation", False),
                        "has_fact_sheet": detection.get("has_fact_sheet", False),
                        "has_harvard_ccp": detection.get("has_harvard_ccp", False),
                        "has_ns_risk": detection.get("has_ns_risk", False)
                    },
                    reason="Add Trump June 4th proclamation context showing Harvard-CCP ties were officially recognized as national security risk"
                ))

        return edit_requests

    def _find_best_location_for_proclamation(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find best location to add proclamation context."""
        sections = structure.get_sections()

        # Prefer sections that discuss factual background, timeline, or legal context
        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in [
                "background", "facts", "factual", "statement", "timeline",
                "national security", "legal context", "procedural history"
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

    async def validate_trump_june4_proclamation(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate that Trump June 4th proclamation is mentioned."""
        text = document.get_full_text()
        detection = self._detect_proclamation_mention(text)

        issues = []
        recommendations = []

        if not detection["meets_minimum"]:
            # Get actual proclamation context from CaseFactsProvider if available
            proclamation_suggestion = "Add mention that on June 4, 2020, President Trump issued a proclamation and fact sheet explicitly identifying Harvard's ties to the CCP as a national security risk. This proclamation came into effect 2 days after Plaintiff filed the HK claim, making Harvard's ties to Xi Jinping's daughter a recognized matter of national security."
            provider = self._get_provider()
            if provider:
                hk_facts = provider.get_fact_block('hk_allegation_ccp_family')
                if hk_facts:
                    proclamation_suggestion = f"Add mention that on June 4, 2020, President Trump issued a proclamation and fact sheet explicitly identifying Harvard's ties to the CCP as a national security risk. {hk_facts} This proclamation came into effect 2 days after Plaintiff filed the HK claim."

            issues.append({
                "type": "missing_june4_proclamation",
                "severity": "high",
                "message": "No mention of President Trump's June 4, 2020 proclamation and fact sheet found.",
                "suggestion": proclamation_suggestion
            })
        elif not detection["has_explicit_connection"]:
            recommendations.append({
                "type": "proclamation_not_connected",
                "priority": "high",
                "message": "Proclamation or fact sheet mentioned, but not explicitly connected to Harvard-CCP ties as national security risk.",
                "suggestion": "Explicitly connect the June 4th proclamation to Harvard's CCP ties as a national security risk: 'The June 4, 2020 proclamation and fact sheet explicitly identified Harvard's ties to the Chinese Communist Party (CCP) as a national security risk.'"
            })

        return FunctionResult(
            success=detection["meets_target"],
            data={
                "has_proclamation": detection["has_proclamation"],
                "has_fact_sheet": detection["has_fact_sheet"],
                "has_harvard_ccp": detection["has_harvard_ccp"],
                "has_ns_risk": detection["has_ns_risk"],
                "has_explicit_connection": detection["has_explicit_connection"],
                "meets_target": detection["meets_target"],
                "meets_minimum": detection["meets_minimum"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 4.0,  # High importance - official recognition
                "feature_rank": 12
            },
            message="Trump June 4th proclamation validation complete"
        )

