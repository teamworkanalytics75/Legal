#!/usr/bin/env python3
"""
Balancing Outweigh Plugin

Enforces explicit language showing national security interests "outweigh" public access.
This is the #2 most predictive feature (importance: 12.75) according to CatBoost analysis.
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


class BalancingOutweighPlugin(BaseFeaturePlugin):
    """Plugin to enforce explicit 'outweigh' language in balancing test."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "balancing_outweigh", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("BalancingOutweighPlugin initialized")

    def _detect_outweigh_language(self, text: str) -> Dict[str, Any]:
        """Detect explicit 'outweigh' language showing NS interests outweigh public access."""
        text_lower = text.lower()

        # Patterns for explicit outweigh language
        outweigh_patterns = [
            r'national\s+security\s+.*?outweigh',
            r'outweigh.*?national\s+security',
            r'outweigh.*?public\s+access',
            r'outweigh.*?presumption.*?public',
            r'national\s+security\s+interest.*?outweigh',
            r'privacy\s+interest.*?outweigh',
            r'outweigh.*?disclosure',
            r'outweigh.*?transparency',
            r'outweigh.*?first\s+amendment',
        ]

        found_matches = []
        for pattern in outweigh_patterns:
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                found_matches.append({
                    "pattern": pattern,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "context": text[max(0, match.start()-100):min(len(text), match.end()+100)]
                })

        # Check if it's explicit about national security outweighing public access
        is_explicit_ns = any(
            "national security" in match["context"].lower() and
            ("public" in match["context"].lower() or "presumption" in match["context"].lower())
            for match in found_matches
        )

        return {
            "has_outweigh_language": len(found_matches) > 0,
            "is_explicit_ns_outweigh": is_explicit_ns,
            "match_count": len(found_matches),
            "matches": found_matches[:5]  # Top 5 matches
        }

    async def generate_edit_requests(
        self,
        text: str,
        structure: "DocumentStructure",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """Generate edit requests to add explicit outweigh language."""
        detection = self._detect_outweigh_language(text)
        edit_requests = []

        if not detection["has_outweigh_language"] or not detection["is_explicit_ns_outweigh"]:
            # Find balancing test section
            best_location = self._find_balancing_test_section(structure)

            if best_location:
                suggested_text = "The national security interests at stake clearly outweigh the presumption of public access to court records."

                edit_requests.append(EditRequest(
                    plugin_name="balancing_outweigh",
                    location=best_location,
                    edit_type="insert",
                    content=suggested_text,
                    priority=85,
                    affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                    metadata={
                        "detection": detection,
                        "feature": "balancing_outweigh",
                        "has_outweigh_language": detection.get("has_outweigh_language", False),
                        "is_explicit_ns_outweigh": detection.get("is_explicit_ns_outweigh", False)
                    },
                    reason="Add explicit language showing national security interests outweigh public access (critical predictive feature)"
                ))

        return edit_requests

    def _find_balancing_test_section(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find balancing test section or create one."""
        sections = structure.get_sections()

        # Look for existing balancing test section
        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in ["balancing", "balance", "weigh", "competing"]):
                return DocumentLocation(
                    section_name=section.get("title", ""),
                    paragraph_index=0,
                    character_offset=section.get("start_char", 0),
                    position_type="after"
                )

        # Fallback: look for sections that might contain balancing
        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in ["legal standard", "analysis", "discussion"]):
                return DocumentLocation(
                    section_name=section.get("title", ""),
                    paragraph_index=0,
                    character_offset=section.get("start_char", 0),
                    position_type="after"
                )

        return None

    async def validate_balancing_outweigh(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate that explicit outweigh language is present."""
        text = document.get_full_text()
        detection = self._detect_outweigh_language(text)

        issues = []
        recommendations = []

        if not detection["has_outweigh_language"]:
            issues.append({
                "type": "missing_outweigh_language",
                "severity": "high",
                "message": "No explicit 'outweigh' language found. This is the #2 most predictive feature (importance: 12.75).",
                "suggestion": "Add explicit language stating that national security interests 'outweigh' the presumption of public access."
            })
        elif not detection["is_explicit_ns_outweigh"]:
            recommendations.append({
                "type": "outweigh_not_explicit_ns",
                "priority": "high",
                "message": "Outweigh language found but not explicitly linking national security to public access.",
                "suggestion": "Make the outweigh language more explicit: 'The national security interests at stake clearly outweigh the presumption of public access.'"
            })

        return FunctionResult(
            success=detection["is_explicit_ns_outweigh"],
            data={
                "has_outweigh_language": detection["has_outweigh_language"],
                "is_explicit_ns_outweigh": detection["is_explicit_ns_outweigh"],
                "match_count": detection["match_count"],
                "matches": detection["matches"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 12.75,  # From CatBoost analysis
                "feature_rank": 2
            },
            message="Balancing outweigh validation complete"
        )

