#!/usr/bin/env python3
"""
Legal Standard Sub-Outline Plugin - Validates nested enumeration structure for Legal Standard section.

Enforces optimal sub-outline hierarchy based on CatBoost analysis of successful motions.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult
from .section_utils import extract_section_text

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class LegalStandardSubOutlinePlugin(BaseFeaturePlugin):
    """Plugin to validate sub-outline structure (nested enumeration patterns) for Legal Standard section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "legal_standard_sub_outline", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("LegalStandardSubOutlinePlugin initialized")

        # Sub-outline structure requirements (from analysis)
        # Example: Level 1: Legal framework, Level 2: Cases, Level 3: Holdings
        self.required_levels = self.rules.get("sub_outline", {}).get("required_levels", [1, 2, 3])
        self.min_depth = self.rules.get("sub_outline", {}).get("min_depth", 3)

    def _analyze_sub_outline_structure(self, section_text: str) -> Dict[str, Any]:
        """Analyze sub-outline structure (nested enumeration patterns) in section."""
        if not section_text:
            return {"has_sub_outline": False, "max_depth": 0, "levels_present": [], "meets_target": False}
        
        lines = section_text.split('\n')
        levels_present = set()
        max_depth = 0
        depth_positions = []
        
        patterns = [
            r'^\s*\d+[\.\)]\s+',  # Level 1
            r'^\s+[a-z][\.\)]\s+',  # Level 2
            r'^\s+[ivxlcdm]+[\.\)]\s+',  # Level 3
            r'^\s+[A-Z][\.\)]\s+',  # Level 4
        ]
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            for level, pattern in enumerate(patterns, 1):
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    leading_spaces = len(line) - len(line.lstrip())
                    depth = (leading_spaces // 2) + 1
                    levels_present.add(level)
                    if depth > max_depth:
                        max_depth = depth
                        depth_positions.append({"line": i + 1, "depth": depth, "level": level})
                    break
        
        # Check if required levels are present
        has_required_levels = all(level in levels_present for level in self.required_levels)
        meets_depth_target = max_depth >= self.min_depth
        
        return {
            "has_sub_outline": len(levels_present) > 0,
            "max_depth": max_depth,
            "levels_present": sorted(list(levels_present)),
            "required_levels": self.required_levels,
            "has_required_levels": has_required_levels,
            "meets_depth_target": meets_depth_target,
            "meets_target": has_required_levels and meets_depth_target,
            "depth_positions": depth_positions[:10]
        }

    async def validate_legal_standard_sub_outline(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate sub-outline structure for Legal Standard section."""
        text = document.get_full_text()
        section_text = extract_section_text(text, 'legal_standard')
        
        if not section_text:
            return FunctionResult(success=False, data={"section_found": False, "issues": [{"type": "missing_section", "severity": "high", "message": "Legal Standard section not found"}]}, message="Legal Standard section not found")
        
        analysis = self._analyze_sub_outline_structure(section_text)
        issues = []
        recommendations = []
        
        if not analysis["has_sub_outline"]:
            issues.append({
                "type": "no_sub_outline",
                "severity": "high",
                "message": "Legal Standard section lacks nested enumeration structure",
                "suggestion": f"Add nested enumeration with levels {self.required_levels}"
            })
        elif not analysis["has_required_levels"]:
            missing_levels = [l for l in self.required_levels if l not in analysis["levels_present"]]
            issues.append({
                "type": "missing_enumeration_levels",
                "severity": "medium",
                "message": f"Legal Standard section missing enumeration levels: {missing_levels}",
                "suggestion": f"Add enumeration at levels {missing_levels}"
            })
        
        if not analysis["meets_depth_target"]:
            issues.append({
                "type": "insufficient_depth",
                "severity": "medium",
                "message": f"Legal Standard section enumeration depth ({analysis['max_depth']}) is below target ({self.min_depth})",
                "suggestion": f"Increase enumeration depth to at least {self.min_depth} levels"
            })
        
        if analysis["meets_target"]:
            recommendations.append({
                "type": "sub_outline_optimal",
                "message": f"Legal Standard section sub-outline structure is optimal (depth: {analysis['max_depth']}, levels: {analysis['levels_present']})"
            })
        
        return FunctionResult(
            success=analysis["meets_target"],
            data={**analysis, "section_found": True, "issues": issues, "recommendations": recommendations},
            message="Legal Standard section sub-outline validation complete"
        )

