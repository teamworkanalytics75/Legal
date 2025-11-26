#!/usr/bin/env python3
"""
Presumption Acknowledgment Plugin

Enforces explicit acknowledgment of the presumption of public access before showing it's overcome.
This is highly predictive (importance: 8.65) according to CatBoost analysis.
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


class PresumptionAcknowledgmentPlugin(BaseFeaturePlugin):
    """Plugin to enforce explicit acknowledgment of presumption of public access."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "presumption_acknowledgment", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("PresumptionAcknowledgmentPlugin initialized")

    def _detect_presumption_acknowledgment(self, text: str) -> Dict[str, Any]:
        """Detect explicit acknowledgment of presumption of public access."""
        text_lower = text.lower()

        # Patterns for presumption acknowledgment
        presumption_patterns = [
            r'presumption\s+of\s+public\s+access',
            r'presumption.*?public\s+access',
            r'public\s+access.*?presumption',
            r'presumption\s+.*?open\s+court',
            r'presumption\s+.*?transparency',
            r'strong\s+presumption.*?public',
            r'common\s+law\s+presumption.*?public',
        ]

        found_matches = []
        for pattern in presumption_patterns:
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                # Check if it's acknowledged (not just mentioned negatively)
                context = text[max(0, match.start()-100):min(len(text), match.end()+100)].lower()
                is_acknowledged = any(
                    keyword in context for keyword in [
                        "acknowledge", "recognize", "understand", "note", "respect",
                        "strong", "common law", "well-established"
                    ]
                )

                found_matches.append({
                    "pattern": pattern,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "is_acknowledged": is_acknowledged,
                    "context": context
                })

        # Check if presumption is acknowledged and then overcome
        has_acknowledgment = any(match["is_acknowledged"] for match in found_matches)
        has_overcome_language = any(
            keyword in text_lower for keyword in [
                "outweigh", "overcome", "overrides", "exceeds", "greater than",
                "outweighs the presumption", "overcome the presumption"
            ]
        )

        return {
            "has_presumption_mention": len(found_matches) > 0,
            "has_acknowledgment": has_acknowledgment,
            "has_overcome_language": has_overcome_language,
            "is_complete": has_acknowledgment and has_overcome_language,
            "match_count": len(found_matches),
            "matches": found_matches[:5]
        }

    async def generate_edit_requests(
        self,
        text: str,
        structure: "DocumentStructure",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """Generate edit requests to add presumption acknowledgment."""
        detection = self._detect_presumption_acknowledgment(text)
        edit_requests = []

        if not detection["has_acknowledgment"]:
            best_location = self._find_balancing_test_section(structure)

            if best_location:
                suggested_text = "The Court recognizes the strong presumption of public access to court records. However, the national security interests at stake in this case clearly outweigh that presumption."

                edit_requests.append(EditRequest(
                    plugin_name="presumption_acknowledgment",
                    location=best_location,
                    edit_type="insert",
                    content=suggested_text,
                    priority=85,
                    affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                    metadata={
                        "detection": detection,
                        "feature": "presumption_acknowledgment",
                        "has_acknowledgment": detection.get("has_acknowledgment", False),
                        "has_overcome_language": detection.get("has_overcome_language", False)
                    },
                    reason="Add explicit acknowledgment of presumption of public access (highly predictive feature)"
                ))
        elif not detection["has_overcome_language"]:
            best_location = self._find_balancing_test_section(structure)
            if best_location:
                suggested_text = "However, the national security interests at stake clearly outweigh this presumption."
                edit_requests.append(EditRequest(
                    plugin_name="presumption_acknowledgment",
                    location=best_location,
                    edit_type="insert",
                    content=suggested_text,
                    priority=60,
                    affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                    metadata={
                        "detection": detection,
                        "feature": "presumption_acknowledgment",
                        "has_acknowledgment": detection.get("has_acknowledgment", False),
                        "has_overcome_language": detection.get("has_overcome_language", False)
                    },
                    reason="Add language showing presumption is overcome"
                ))

        return edit_requests

    def _find_balancing_test_section(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find balancing test section."""
        sections = structure.get_sections()

        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in ["balancing", "balance", "legal standard"]):
                return DocumentLocation(
                    section_name=section.get("title", ""),
                    paragraph_index=0,
                    character_offset=section.get("start_char", 0),
                    position_type="after"
                )

        return None

    async def validate_presumption_acknowledgment(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate that presumption is explicitly acknowledged."""
        text = document.get_full_text()
        detection = self._detect_presumption_acknowledgment(text)

        issues = []
        recommendations = []

        if not detection["has_presumption_mention"]:
            issues.append({
                "type": "missing_presumption_mention",
                "severity": "high",
                "message": "No mention of 'presumption of public access' found. This is highly predictive (importance: 8.65).",
                "suggestion": "Explicitly acknowledge the presumption of public access before showing it's overcome."
            })
        elif not detection["has_acknowledgment"]:
            issues.append({
                "type": "presumption_not_acknowledged",
                "severity": "medium",
                "message": "Presumption mentioned but not explicitly acknowledged.",
                "suggestion": "Use language like 'The Court recognizes the strong presumption of public access...'"
            })
        elif not detection["has_overcome_language"]:
            recommendations.append({
                "type": "missing_overcome_language",
                "priority": "high",
                "message": "Presumption acknowledged but not explicitly shown to be overcome.",
                "suggestion": "Add language showing how national security interests outweigh the presumption."
            })

        return FunctionResult(
            success=detection["is_complete"],
            data={
                "has_presumption_mention": detection["has_presumption_mention"],
                "has_acknowledgment": detection["has_acknowledgment"],
                "has_overcome_language": detection["has_overcome_language"],
                "is_complete": detection["is_complete"],
                "match_count": detection["match_count"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 8.65,  # From CatBoost analysis
                "feature_rank": 5
            },
            message="Presumption acknowledgment validation complete"
        )

