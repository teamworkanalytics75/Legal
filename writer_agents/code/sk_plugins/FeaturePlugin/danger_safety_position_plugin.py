#!/usr/bin/env python3
"""
Danger/Safety Position Plugin - Enforces early placement of Danger/Safety arguments.

Position matters: earlier placement (lower position number) = more emphasis.
Average position in granted motions: 6.8
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


class DangerSafetyPositionPlugin(BaseFeaturePlugin):
    """Plugin to enforce optimal placement of Danger/Safety arguments."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "danger_safety_position", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("DangerSafetyPositionPlugin initialized")

    def _detect_danger_safety_section(self, text: str) -> Dict[str, Any]:
        """Detect Danger/Safety section and its position."""
        # Patterns for danger/safety sections
        danger_patterns = [
            r'(?i)^\s*(?:danger|safety|threat|harassment|harass|intimidation)',
            r'(?i)^\s*(?:risk\s+of\s+harm|potential\s+harm|safety\s+concern)',
            r'(?i)^\s*(?:danger\s+to\s+petitioner|safety\s+of\s+petitioner)',
        ]

        lines = text.split('\n')
        section_positions = []

        for i, line in enumerate(lines):
            for pattern in danger_patterns:
                if re.search(pattern, line):
                    # Check if this looks like a section header (all caps, short line, etc.)
                    if len(line.strip()) < 100 and (line.isupper() or ':' in line or i == 0):
                        section_positions.append({
                            "position": i + 1,  # 1-indexed
                            "line_number": i + 1,
                            "header": line.strip()[:100]
                        })
                        break

        # Calculate position relative to document
        total_lines = len(lines)
        position_score = None
        if section_positions:
            earliest_position = min(s["position"] for s in section_positions)
            position_score = earliest_position / total_lines if total_lines > 0 else 1.0

        # Target position: 6.8 (average in granted motions)
        target_position = 6.8
        optimal_range = (4, 10)  # Optimal range for position

        return {
            "has_danger_safety": len(section_positions) > 0,
            "section_positions": section_positions,
            "earliest_position": min(s["position"] for s in section_positions) if section_positions else None,
            "position_score": position_score,
            "target_position": target_position,
            "optimal_range": optimal_range,
            "is_optimal": earliest_position is not None and optimal_range[0] <= earliest_position <= optimal_range[1]
        }

    async def validate_danger_safety_position(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate Danger/Safety section position.

        Position matters: earlier placement = more emphasis.
        Feature importance: 3.36 (rank #3)
        """
        text = document.get_full_text()
        detection = self._detect_danger_safety_section(text)

        issues = []
        recommendations = []

        if not detection["has_danger_safety"]:
            issues.append({
                "type": "missing_danger_safety",
                "severity": "high",
                "message": "Danger/Safety arguments section is missing.",
                "suggestion": "Add a Danger/Safety section to address safety concerns and threats."
            })
        else:
            position = detection["earliest_position"]
            if position is not None:
                if position > detection["optimal_range"][1]:
                    issues.append({
                        "type": "danger_safety_too_late",
                        "severity": "medium",
                        "message": f"Danger/Safety section appears at position {position}, which is late. Optimal range: {detection['optimal_range'][0]}-{detection['optimal_range'][1]}.",
                        "suggestion": "Move Danger/Safety arguments earlier in the motion for more emphasis."
                    })
                    recommendations.append({
                        "type": "reorder_section",
                        "priority": "medium",
                        "action": f"Move Danger/Safety section from position {position} to earlier in the motion (target: position {detection['target_position']:.1f}).",
                        "rationale": "Section order matters for persuasive impact. Earlier placement = more emphasis."
                    })
                elif position < detection["optimal_range"][0]:
                    recommendations.append({
                        "type": "danger_safety_very_early",
                        "message": f"Danger/Safety section is very early (position {position}). This may be optimal if it's your strongest argument.",
                        "position": position
                    })
                else:
                    recommendations.append({
                        "type": "danger_safety_optimal",
                        "message": f"Danger/Safety section position is optimal (position {position}).",
                        "position": position
                    })

        return FunctionResult(
            success=detection["has_danger_safety"] and detection["is_optimal"],
            data={
                "has_danger_safety": detection["has_danger_safety"],
                "earliest_position": detection["earliest_position"],
                "position_score": detection["position_score"],
                "target_position": detection["target_position"],
                "optimal_range": detection["optimal_range"],
                "is_optimal": detection["is_optimal"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 3.36,
                "feature_rank": 3
            },
            message="Danger/Safety position validation complete"
        )

    async def query_danger_safety_position(self, case_context: str, **kwargs) -> FunctionResult:
        """Query for examples of optimal Danger/Safety section placement."""
        results = await self.query_chroma(
            case_context=f"danger safety threat section position placement {case_context}",
            n_results=kwargs.get('n_results', 10)
        )

        return FunctionResult(
            success=True,
            data={"examples": results},
            message=f"Found {len(results)} examples of Danger/Safety section placement"
        )

