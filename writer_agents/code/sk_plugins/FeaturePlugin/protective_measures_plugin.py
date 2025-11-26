#!/usr/bin/env python3
"""
Protective Measures Plugin

Enforces explicit mention of specific protective measures being requested.
This is predictive (importance: 4.32) according to CatBoost analysis.
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


class ProtectiveMeasuresPlugin(BaseFeaturePlugin):
    """Plugin to enforce explicit mention of protective measures."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "protective_measures", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("ProtectiveMeasuresPlugin initialized")

    def _detect_protective_measures(self, text: str) -> Dict[str, Any]:
        """Detect specific protective measures mentioned."""
        text_lower = text.lower()

        # Protective measures (from CatBoost analysis)
        protective_measures = [
            "protective order",
            "confidentiality order",
            "file under seal",
            "impound",
            "proceed under pseudonym",
            "redact",
            "sealed filing",
            "ex parte",
            "seal",
            "sealing",
            "confidential",
        ]

        found_measures = []
        measure_locations = []

        for measure in protective_measures:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(measure) + r'\b'
            matches = list(re.finditer(pattern, text_lower))

            if matches:
                found_measures.append(measure)
                for match in matches:
                    measure_locations.append({
                        "measure": measure,
                        "start": match.start(),
                        "end": match.end(),
                        "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    })

        # Target: At least 2-3 different protective measures
        target_count = 2
        min_count = 1

        return {
            "found_measures": found_measures,
            "unique_measures_count": len(found_measures),
            "total_mentions": len(measure_locations),
            "measure_locations": measure_locations[:10],
            "target_count": target_count,
            "min_count": min_count,
            "meets_target": len(found_measures) >= target_count,
            "meets_minimum": len(found_measures) >= min_count
        }

    async def generate_edit_requests(
        self,
        text: str,
        structure: "DocumentStructure",
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """Generate edit requests to add missing protective measures."""
        detection = self._detect_protective_measures(text)
        edit_requests = []

        if not detection["meets_target"]:
            # Find best location (typically in requested relief or introduction)
            best_location = self._find_best_location_for_protective_measures(structure)

            if best_location:
                # Suggest missing protective measures
                all_measures = [
                    "protective order",
                    "file under seal",
                    "confidentiality order",
                    "impound",
                    "redact"
                ]

                missing_measures = [m for m in all_measures if m not in detection["found_measures"]]

                if missing_measures:
                    suggested_text = f"Consider requesting: {', '.join(missing_measures[:2])}."

                    edit_requests.append(EditRequest(
                        plugin_name="protective_measures",
                        location=best_location,
                        edit_type="insert",
                        content=suggested_text,
                        priority=60,
                        affected_plugins=["word_count", "sentence_count", "character_count", "citation_format"],
                        metadata={
                            "detection": detection,
                            "feature": "protective_measures",
                            "target_count": detection.get("target_count", 0),
                            "current_count": detection.get("unique_measures_count", 0),
                            "meets_target": detection.get("meets_target", False),
                            "missing_measures": missing_measures[:2]
                        },
                        reason=f"Specify protective measures (found {detection['unique_measures_count']}, target: {detection['target_count']})"
                    ))

        return edit_requests

    def _find_best_location_for_protective_measures(self, structure: "DocumentStructure") -> Optional[DocumentLocation]:
        """Find best location to add protective measures."""
        sections = structure.get_sections()

        # Prefer "Requested Relief" or "Conclusion" sections
        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if any(keyword in section_title_lower for keyword in ["relief", "request", "conclusion", "prayer"]):
                return DocumentLocation(
                    section_name=section.get("title", ""),
                    paragraph_index=0,
                    character_offset=section.get("start_char", 0),
                    position_type="after"
                )

        # Fallback: Introduction
        for section in sections:
            section_title_lower = section.get("title", "").lower()
            if "introduction" in section_title_lower:
                return DocumentLocation(
                    section_name=section.get("title", ""),
                    paragraph_index=0,
                    character_offset=section.get("start_char", 0),
                    position_type="after"
                )

        return None

    async def validate_protective_measures(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate that specific protective measures are mentioned."""
        text = document.get_full_text()
        detection = self._detect_protective_measures(text)

        issues = []
        recommendations = []

        if not detection["meets_minimum"]:
            issues.append({
                "type": "missing_protective_measures",
                "severity": "medium",
                "message": f"No specific protective measures mentioned. Found: {detection['unique_measures_count']}, Minimum: {detection['min_count']}",
                "suggestion": "Explicitly mention specific protective measures such as 'protective order', 'file under seal', 'confidentiality order', or 'impound'."
            })
        elif not detection["meets_target"]:
            recommendations.append({
                "type": "insufficient_protective_measures",
                "priority": "medium",
                "message": f"Found {detection['unique_measures_count']} protective measures, but target is {detection['target_count']} for optimal strength.",
                "suggestion": "Add additional specific protective measures to strengthen the request."
            })

        return FunctionResult(
            success=detection["meets_minimum"],
            data={
                "found_measures": detection["found_measures"],
                "unique_measures_count": detection["unique_measures_count"],
                "total_mentions": detection["total_mentions"],
                "target_count": detection["target_count"],
                "meets_target": detection["meets_target"],
                "meets_minimum": detection["meets_minimum"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 4.32,  # From CatBoost analysis
                "feature_rank": 7
            },
            message="Protective measures validation complete"
        )

