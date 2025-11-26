#!/usr/bin/env python3
"""
PRC National Security Plugin

Enforces mention that leadership family identification is explicitly defined
as a national security issue under PRC (People's Republic of China) national security law.

This is important context for US courts - even if it doesn't automatically make
it US national security, it's worth noting that other jurisdictions explicitly
treat this as a national security matter.
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


class PRCNationalSecurityPlugin(BaseFeaturePlugin):
    """Plugin to enforce mention of PRC National Security Law context."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "prc_national_security", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("PRCNationalSecurityPlugin initialized")

    def _get_provider(self):
        """Get CaseFactsProvider lazily (fetches fresh each time since it's set after plugin init)."""
        return get_case_facts_provider()

    def _detect_prc_ns_context(self, text: str) -> Dict[str, Any]:
        """Detect PRC National Security Law context and leadership family references."""
        text_lower = text.lower()

        # PRC National Security indicators
        prc_ns_indicators = [
            "prc national security",
            "people's republic of china.*national security",
            "china.*national security law",
            "chinese national security law",
            "prc.*national security",
            "mainland china.*national security",
            "national security law.*china",
            "national security law.*prc",
        ]

        # Leadership family identification indicators
        leadership_family_indicators = [
            "leadership family",
            "family member.*leadership",
            "political leader.*family",
            "leader.*family member",
            "political family",
            "leadership.*relative",
            "xi.*family",
            "xi mingze",
            "president.*daughter",
            "president.*family",
            "ccp.*family",
            "communist party.*family",
        ]

        # Context linking them together
        linking_phrases = [
            "national security issue",
            "national security matter",
            "national security concern",
            "defined as.*national security",
            "treated as.*national security",
            "considered.*national security",
            "state secret",
            "state security",
        ]

        found_prc_ns = []
        found_leadership_family = []
        found_linking = []

        # Check for PRC NS Law mentions
        for indicator in prc_ns_indicators:
            pattern = re.escape(indicator).replace(r'\*', r'.*')
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_prc_ns.append(indicator)

        # Check for leadership family mentions
        for indicator in leadership_family_indicators:
            pattern = re.escape(indicator).replace(r'\*', r'.*')
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                found_leadership_family.append(indicator)

        # Check for linking language
        for phrase in linking_phrases:
            pattern = re.escape(phrase).replace(r'\*', r'.*')
            matches = list(re.finditer(pattern, text_lower))
            if matches:
                # Check if it's in context of PRC or leadership family
                for match in matches:
                    context = text[max(0, match.start()-200):min(len(text), match.end()+200)].lower()
                    has_prc_context = any(prc_term in context for prc_term in ["china", "chinese", "prc", "people's republic", "mainland"])
                    has_leadership_context = any(lead_term in context for lead_term in ["family", "leadership", "xi", "president", "ccp", "communist party"])
                    if has_prc_context or has_leadership_context:
                        found_linking.append(phrase)
                        break

        # Check if there's explicit connection between PRC NS Law and leadership family
        has_explicit_connection = False
        if found_prc_ns and found_leadership_family:
            # Look for sentences/paragraphs that mention both
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence_lower = sentence.lower()
                has_prc = any(prc_term in sentence_lower for prc_term in ["china", "chinese", "prc", "people's republic"]) and any(ns_term in sentence_lower for ns_term in ["national security", "state security", "state secret"])
                has_leadership = any(lead_term in sentence_lower for lead_term in ["family", "leadership", "xi", "president", "daughter", "ccp"])
                if has_prc and has_leadership:
                    has_explicit_connection = True
                    break

        return {
            "has_prc_ns_law": len(found_prc_ns) > 0,
            "has_leadership_family": len(found_leadership_family) > 0,
            "has_linking_language": len(found_linking) > 0,
            "has_explicit_connection": has_explicit_connection,
            "found_prc_ns": found_prc_ns,
            "found_leadership_family": found_leadership_family,
            "found_linking": found_linking,
            "meets_target": has_explicit_connection or (len(found_prc_ns) > 0 and len(found_leadership_family) > 0),
            "meets_minimum": len(found_prc_ns) > 0 or len(found_leadership_family) > 0
        }

    async def generate_edit_requests(
        self,
        text: str,
        structure: "DocumentStructure",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """Generate edit requests to add PRC NS Law context."""
        detection = self._detect_prc_ns_context(text)
        edit_requests = []

        if not detection["has_explicit_connection"]:
            # Find best location (typically in factual background or legal context)
            best_location = self._find_best_location_for_prc_context(structure)

            if best_location:
                # Get actual case facts from CaseFactsProvider if available
                leadership_family_context = ""
                provider = self._get_provider()
                if provider:
                    hk_facts = provider.get_fact_block('hk_allegation_ccp_family')
                    if hk_facts:
                        # Extract leadership family reference from facts
                        if "Xi Mingze" in hk_facts or "leadership family" in hk_facts.lower() or "CCP" in hk_facts:
                            leadership_family_context = f" {hk_facts}"

                if not detection["has_prc_ns_law"]:
                    suggested_text = "Under PRC (People's Republic of China) national security law, identification of leadership family members is explicitly defined as a national security issue and state secret. While this does not automatically make it US national security, it is relevant context that other jurisdictions explicitly treat this matter as a national security concern."
                    if leadership_family_context:
                        suggested_text += leadership_family_context
                elif not detection["has_leadership_family"]:
                    if leadership_family_context:
                        suggested_text = f"The identification of leadership family members is explicitly treated as a national security issue and state secret under PRC national security law.{leadership_family_context}"
                    else:
                        suggested_text = "The identification of leadership family members is explicitly treated as a national security issue and state secret under PRC national security law."
                else:
                    suggested_text = "It is worth noting that PRC national security law explicitly defines identification of leadership family members as a national security issue and state secret, providing relevant context for this Court's consideration."

                edit_requests.append(EditRequest(
                    plugin_name="prc_national_security",
                    location=best_location,
                    edit_type="insert",
                    content=suggested_text,
                    priority=60,
                    affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                    metadata={
                        "detection": detection,
                        "feature": "prc_national_security",
                        "has_explicit_connection": detection.get("has_explicit_connection", False),
                        "has_prc_ns_law": detection.get("has_prc_ns_law", False),
                        "has_leadership_family": detection.get("has_leadership_family", False)
                    },
                    reason="Add PRC National Security Law context showing leadership family identification is explicitly a national security issue"
                ))

        return edit_requests

    def _find_best_location_for_prc_context(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find best location to add PRC NS Law context."""
        sections = structure.get_sections()

        # Prefer sections that discuss legal context or factual background
        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in [
                "legal context", "background", "facts", "factual", "statement",
                "national security", "jurisdiction", "applicable law", "foreign law"
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

    async def validate_prc_national_security(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate that PRC NS Law context is mentioned."""
        text = document.get_full_text()
        detection = self._detect_prc_ns_context(text)

        issues = []
        recommendations = []

        if not detection["meets_minimum"]:
            issues.append({
                "type": "missing_prc_ns_context",
                "severity": "medium",
                "message": "No mention of PRC National Security Law context found.",
                "suggestion": "Add context that under PRC national security law, identification of leadership family members is explicitly defined as a national security issue and state secret. This provides relevant context for the Court even if it doesn't automatically make it US national security."
            })
        elif not detection["has_explicit_connection"]:
            recommendations.append({
                "type": "prc_context_not_connected",
                "priority": "medium",
                "message": "PRC NS Law or leadership family mentioned, but not explicitly connected.",
                "suggestion": "Explicitly connect PRC national security law to leadership family identification: 'Under PRC national security law, identification of leadership family members is explicitly defined as a national security issue and state secret.'"
            })

        return FunctionResult(
            success=detection["meets_target"],
            data={
                "has_prc_ns_law": detection["has_prc_ns_law"],
                "has_leadership_family": detection["has_leadership_family"],
                "has_explicit_connection": detection["has_explicit_connection"],
                "meets_target": detection["meets_target"],
                "meets_minimum": detection["meets_minimum"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 3.0,  # Contextual importance
                "feature_rank": 11
            },
            message="PRC National Security context validation complete"
        )

