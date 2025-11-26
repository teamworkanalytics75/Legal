#!/usr/bin/env python3
"""
Max Enumeration Depth Plugin - Enforces nested enumeration structure.

Max enumeration depth is the #1 most important feature (27.27 importance) for seal motions.
This plugin enforces the use of nested lists (e.g., 1. a. i.) to maximize enumeration depth.
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


class MaxEnumerationDepthPlugin(BaseFeaturePlugin):
    """Plugin to enforce maximum enumeration depth (nested lists)."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "max_enumeration_depth", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("MaxEnumerationDepthPlugin initialized")

        # Target depth from analysis: max_enumeration_depth ≥ 10 for pseudonym, ≥ 13 for both
        # For seal motions, this is the #1 feature (27.27 importance)
        self.target_depth = self.rules.get("targets", {}).get("max_enumeration_depth", 10)

    def _calculate_max_enumeration_depth(self, text: str) -> Dict[str, Any]:
        """Calculate maximum enumeration depth in text."""
        lines = text.split('\n')
        max_depth = 0
        depth_positions = []

        # Patterns for different enumeration levels
        patterns = [
            r'^\s*\d+[\.\)]\s+',  # Level 1: "1. " or "1) "
            r'^\s+[a-z][\.\)]\s+',  # Level 2: "a. " or "a) "
            r'^\s+[ivxlcdm]+[\.\)]\s+',  # Level 3: "i. " or "ii. "
            r'^\s+[A-Z][\.\)]\s+',  # Level 4: "A. " or "A) "
            r'^\s+\([a-z]\)\s+',  # Level 2 alt: "(a) "
            r'^\s+\(\d+\)\s+',  # Level 1 alt: "(1) "
        ]

        current_depth = 0
        depth_stack = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check for enumeration markers
            found_level = None
            for level, pattern in enumerate(patterns, 1):
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    found_level = level
                    break

            if found_level:
                # Adjust depth based on level
                if not depth_stack or found_level > depth_stack[-1]:
                    # Going deeper
                    depth_stack.append(found_level)
                    current_depth = len(depth_stack)
                elif found_level <= depth_stack[-1]:
                    # Same or shallower level - reset stack
                    depth_stack = [found_level]
                    current_depth = len(depth_stack)

                if current_depth > max_depth:
                    max_depth = current_depth
                    depth_positions.append({
                        "line": i + 1,
                        "depth": current_depth,
                        "text": line_stripped[:100]
                    })
            else:
                # No enumeration - reset if we hit a paragraph break
                if line_stripped and not any(re.match(p, line_stripped) for p in patterns):
                    if len(line_stripped) > 50:  # Likely a paragraph break
                        depth_stack = []
                        current_depth = 0

        # Count nested structures
        nested_count = sum(1 for pos in depth_positions if pos["depth"] >= 3)

        return {
            "max_depth": max_depth,
            "target_depth": self.target_depth,
            "meets_target": max_depth >= self.target_depth,
            "depth_positions": depth_positions[:20],  # Limit for performance
            "nested_count": nested_count,
            "has_nested_lists": max_depth >= 3
        }

    async def validate_enumeration_depth(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate maximum enumeration depth.

        This is CRITICAL: It's the #1 most important feature (27.27 importance) for seal motions.
        Target: max_enumeration_depth ≥ 10 (for pseudonym), ≥ 13 (for both)
        """
        text = document.get_full_text()
        analysis = self._calculate_max_enumeration_depth(text)

        issues = []
        recommendations = []

        if not analysis["has_nested_lists"]:
            issues.append({
                "type": "no_nested_lists",
                "severity": "critical",
                "message": f"No nested enumeration detected. Max depth: {analysis['max_depth']}. This is the #1 most important feature (27.27 importance) for seal motions.",
                "suggestion": "Add nested lists using multiple enumeration levels (e.g., 1. a. i. or 1) a) i))"
            })
        elif analysis["max_depth"] < self.target_depth:
            issues.append({
                "type": "insufficient_depth",
                "severity": "high",
                "message": f"Enumeration depth ({analysis['max_depth']}) is below target ({self.target_depth}). This is the #1 most important feature for seal motions.",
                "suggestion": f"Increase enumeration depth to at least {self.target_depth} levels using nested lists."
            })
            recommendations.append({
                "type": "increase_depth",
                "priority": "critical",
                "action": f"Add more nested enumeration levels. Current: {analysis['max_depth']}, Target: {self.target_depth}",
                "rationale": "Max enumeration depth is the #1 predictor for seal motion success (27.27 importance). Use nested lists like: 1. a. i. or 1) a) i)"
            })
        else:
            recommendations.append({
                "type": "depth_optimal",
                "message": f"Enumeration depth is optimal ({analysis['max_depth']} levels).",
                "depth": analysis["max_depth"]
            })

        return FunctionResult(
            success=analysis["meets_target"],
            data={
                "max_depth": analysis["max_depth"],
                "target_depth": analysis["target_depth"],
                "meets_target": analysis["meets_target"],
                "has_nested_lists": analysis["has_nested_lists"],
                "nested_count": analysis["nested_count"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 27.27,
                "feature_rank": 1,
                "motion_type": "seal"
            },
            message="Enumeration depth validation complete"
        )

    async def query_enumeration_depth(self, case_context: str, **kwargs) -> FunctionResult:
        """Query for examples of high enumeration depth."""
        results = await self.query_chroma(
            case_context=f"nested enumeration lists numbered items {case_context}",
            n_results=kwargs.get('n_results', 10)
        )

        return FunctionResult(
            success=True,
            data={"examples": results},
            message=f"Found {len(results)} examples of nested enumeration"
        )

    async def generate_enumeration_depth_argument(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Generate argument about enumeration depth."""
        analysis = self._calculate_max_enumeration_depth(document.get_full_text())

        if analysis["meets_target"]:
            argument = f"""The motion uses nested enumeration effectively (max depth: {analysis['max_depth']} levels). Maximum enumeration depth is the #1 most important structural feature (27.27 importance) for seal motions because it:

1. Shows systematic organization and thoroughness
2. Demonstrates attention to detail and structure
3. Makes complex arguments easier to follow
4. Signals professional drafting quality

Target depth: {self.target_depth} levels. Current: {analysis['max_depth']} levels."""
        else:
            argument = f"""The motion should increase enumeration depth. This is the #1 most important feature (27.27 importance) for seal motion success.

Current depth: {analysis['max_depth']} levels
Target depth: {self.target_depth} levels

Add nested lists using multiple enumeration levels:
- Level 1: 1. 2. 3.
- Level 2: a. b. c.
- Level 3: i. ii. iii.
- Level 4: A. B. C.

Example structure:
1. First main point
   a. First sub-point
      i. First sub-sub-point
      ii. Second sub-sub-point
   b. Second sub-point
2. Second main point"""

        return FunctionResult(
            success=True,
            data={"argument": argument, "max_depth": analysis["max_depth"]},
            message="Enumeration depth argument generated"
        )

    async def generate_edit_requests(
        self,
        text: str,
        structure: 'DocumentStructure',
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """
        Generate edit requests to increase enumeration depth.

        Converts flat lists into nested lists to increase max_enumeration_depth.
        """
        try:
            analysis = self._calculate_max_enumeration_depth(text)

            # If already meets target, return empty
            if analysis["meets_target"]:
                return []

            requests = []

            # Find sections that could benefit from nested enumeration
            # Look for sections with multiple flat lists
            paragraphs = structure.paragraphs if hasattr(structure, 'paragraphs') else []

            for para_idx, paragraph in enumerate(paragraphs):
                para_text = paragraph.text if hasattr(paragraph, 'text') else str(paragraph)

                # Check if paragraph has flat enumeration that could be nested
                flat_items = re.findall(r'^\s*\d+[\.\)]\s+(.+)$', para_text, re.MULTILINE)

                if len(flat_items) >= 3:  # At least 3 items to nest
                    # Convert to nested structure
                    nested_content = ""
                    for i, item in enumerate(flat_items, 1):
                        nested_content += f"{i}. {item}\n"
                        # Add sub-items if item is long enough
                        if len(item) > 50:
                            nested_content += f"   a. Detail point 1\n"
                            nested_content += f"   b. Detail point 2\n"

                    location = DocumentLocation(
                        paragraph_index=para_idx,
                        position_type="replace"
                    )

                    request = EditRequest(
                        plugin_name="max_enumeration_depth",
                        location=location,
                        edit_type="replace",
                        content=nested_content,
                        priority=90,  # Very high priority (this is #1 feature)
                        affected_plugins=["enumeration_density", "max_enumeration_depth", "bullet_points"],
                        metadata={
                            "original_depth": analysis["max_depth"],
                            "target_depth": self.target_depth,
                            "conversion_type": "flat_to_nested"
                        },
                        reason=f"Increase enumeration depth from {analysis['max_depth']} to meet target {self.target_depth} (this is the #1 most important feature)"
                    )
                    requests.append(request)

                    # Limit to avoid too many edits
                    if len(requests) >= 3:
                        break

            logger.info(f"Generated {len(requests)} enumeration depth edit requests")
            return requests

        except Exception as e:
            logger.error(f"Failed to generate enumeration depth edit requests: {e}")
            return []

