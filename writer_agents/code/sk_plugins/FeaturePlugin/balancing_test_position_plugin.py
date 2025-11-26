#!/usr/bin/env python3
"""
Balancing Test Position Plugin - Enforces prominent placement of Balancing Test section.

Balancing test is a key legal requirement. Position indicates emphasis.
Average position in granted motions: 5.3
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


class BalancingTestPositionPlugin(BaseFeaturePlugin):
    """Plugin to enforce optimal placement of Balancing Test section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "balancing_test_position", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("BalancingTestPositionPlugin initialized")

    def _detect_balancing_test_section(self, text: str) -> Dict[str, Any]:
        """
        Detect balance/weighing section and its position.

        UPDATED: Avoids checking for explicit "balancing test" phrase.
        Checks for concept-based sections instead (weigh, balance of, competing interests).
        """
        # Patterns for balance concept sections (NOT the explicit phrase "balancing test")
        # Based on findings: avoid "balancing test" phrase, use concepts instead
        balance_concept_patterns = [
            r'(?i)^\s*(?:weighing\s+the\s+interests|weighing\s+factors|weigh\s+the)',
            r'(?i)^\s*(?:balance\s+of\s+equities|balance\s+of\s+interests)',
            r'(?i)^\s*(?:competing\s+interests|weighing\s+competing)',
            r'(?i)^\s*(?:privacy\s+interest\s+vs\s+.*?interest)',
        ]

        # Check if explicit "balancing test" phrase is used (BAD - should be flagged)
        has_explicit_phrase = bool(re.search(r'(?i)\bbalancing\s+test\b', text))

        lines = text.split('\n')
        section_positions = []

        for i, line in enumerate(lines):
            for pattern in balance_concept_patterns:
                if re.search(pattern, line):
                    # Check if this looks like a section header
                    if len(line.strip()) < 100 and (line.isupper() or ':' in line or i == 0):
                        section_positions.append({
                            "position": i + 1,  # 1-indexed
                            "line_number": i + 1,
                            "header": line.strip()[:100]
                        })
                        break

        # Calculate position
        total_lines = len(lines)
        position_score = None
        if section_positions:
            earliest_position = min(s["position"] for s in section_positions)
            position_score = earliest_position / total_lines if total_lines > 0 else 1.0

        # Target position: 5.3 (average in granted motions)
        target_position = 5.3
        optimal_range = (3, 8)  # Optimal range for position
        max_position = 12  # Don't bury it

        earliest_position = min(s["position"] for s in section_positions) if section_positions else None

        return {
            "has_balance_section": len(section_positions) > 0,
            "has_explicit_phrase": has_explicit_phrase,  # Flag if "balancing test" phrase used
            "section_positions": section_positions,
            "earliest_position": earliest_position,
            "position_score": position_score,
            "target_position": target_position,
            "optimal_range": optimal_range,
            "max_position": max_position,
            "is_optimal": earliest_position is not None and optimal_range[0] <= earliest_position <= optimal_range[1],
            "is_buried": earliest_position is not None and earliest_position > max_position
        }

    async def validate_balancing_test_position(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate balance/weighing section position.

        UPDATED: Checks for concept-based sections (weigh, balance of, competing interests),
        NOT explicit "balancing test" phrase. Position matters: earlier placement = more prominence.
        Feature importance: 0.23 (rank #4)
        """
        text = document.get_full_text()
        detection = self._detect_balancing_test_section(text)

        issues = []
        recommendations = []

        # Flag if explicit "balancing test" phrase is used (BAD)
        if detection["has_explicit_phrase"]:
            issues.append({
                "type": "explicit_balancing_test_phrase",
                "severity": "high",
                "message": "Document uses explicit 'balancing test' phrase. Research shows this correlates with denial (-16.4pp).",
                "suggestion": "Replace with concept-based language: 'weigh the interests', 'balance of equities', or 'competing interests'."
            })

        if not detection["has_balance_section"]:
            issues.append({
                "type": "missing_balance_section",
                "severity": "high",
                "message": "Balance/weighing section is missing. This is a key legal requirement for sealing/pseudonym motions.",
                "suggestion": "Add a section using concept-based language (e.g., 'Weighing the Interests' or 'Balance of Equities') showing how privacy interests outweigh public interest."
            })
        else:
            position = detection["earliest_position"]
            if position is not None:
                if detection["is_buried"]:
                    issues.append({
                        "type": "balancing_test_buried",
                        "severity": "medium",
                        "message": f"Balancing Test section appears at position {position}, which is too late (max: {detection['max_position']}).",
                        "suggestion": "Move Balancing Test section earlier in the motion for more prominence."
                    })
                    recommendations.append({
                        "type": "reorder_section",
                        "priority": "medium",
                        "action": f"Move balance/weighing section from position {position} to earlier in the motion (target: position {detection['target_position']:.1f}).",
                        "rationale": "Balance/weighing analysis is a key legal requirement. Earlier placement = more prominence."
                    })
                elif not detection["is_optimal"]:
                    recommendations.append({
                        "type": "balance_section_suboptimal",
                        "priority": "low",
                        "message": f"Balance/weighing section at position {position}. Optimal range: {detection['optimal_range'][0]}-{detection['optimal_range'][1]}.",
                        "suggestion": "Consider moving to more prominent position."
                    })
                else:
                    recommendations.append({
                        "type": "balance_section_optimal",
                        "message": f"Balance/weighing section position is optimal (position {position}).",
                        "position": position
                    })

        return FunctionResult(
            success=detection["has_balance_section"] and not detection["is_buried"] and not detection["has_explicit_phrase"],
            data={
                "has_balance_section": detection["has_balance_section"],
                "has_explicit_phrase": detection["has_explicit_phrase"],
                "earliest_position": detection["earliest_position"],
                "position_score": detection["position_score"],
                "target_position": detection["target_position"],
                "optimal_range": detection["optimal_range"],
                "is_optimal": detection["is_optimal"],
                "is_buried": detection["is_buried"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 0.23,
                "feature_rank": 4
            },
            message="Balancing Test position validation complete"
        )

