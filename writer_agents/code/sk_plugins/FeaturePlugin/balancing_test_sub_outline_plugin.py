#!/usr/bin/env python3
"""
Balancing Test Sub-Outline Plugin - Validates nested enumeration structure for Balancing Test section.
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


class BalancingTestSubOutlinePlugin(BaseFeaturePlugin):
    """Plugin to validate sub-outline structure for Balancing Test section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "balancing_test_sub_outline", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("BalancingTestSubOutlinePlugin initialized")

        self.required_levels = self.rules.get("sub_outline", {}).get("required_levels", [1, 2])
        self.min_depth = self.rules.get("sub_outline", {}).get("min_depth", 2)

    def _analyze_sub_outline_structure(self, section_text: str) -> Dict[str, Any]:
        """Analyze sub-outline structure in section."""
        if not section_text:
            return {"has_sub_outline": False, "max_depth": 0, "levels_present": [], "meets_target": False}
        
        lines = section_text.split('\n')
        levels_present = set()
        max_depth = 0
        
        patterns = [
            r'^\s*\d+[\.\)]\s+', r'^\s+[a-z][\.\)]\s+', r'^\s+[ivxlcdm]+[\.\)]\s+', r'^\s+[A-Z][\.\)]\s+',
        ]
        
        for line in lines:
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
                    break
        
        has_required_levels = all(level in levels_present for level in self.required_levels)
        meets_depth_target = max_depth >= self.min_depth
        
        return {
            "has_sub_outline": len(levels_present) > 0,
            "max_depth": max_depth,
            "levels_present": sorted(list(levels_present)),
            "required_levels": self.required_levels,
            "has_required_levels": has_required_levels,
            "meets_depth_target": meets_depth_target,
            "meets_target": has_required_levels and meets_depth_target
        }

    async def validate_balancing_test_sub_outline(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate sub-outline structure for Balancing Test section."""
        text = document.get_full_text()
        section_text = extract_section_text(text, 'balancing_test')
        
        if not section_text:
            return FunctionResult(success=False, data={"section_found": False, "issues": [{"type": "missing_section", "severity": "high", "message": "Balancing Test section not found"}]}, message="Balancing Test section not found")
        
        analysis = self._analyze_sub_outline_structure(section_text)
        issues = []
        recommendations = []
        
        if not analysis["has_sub_outline"]:
            issues.append({"type": "no_sub_outline", "severity": "medium", "message": "Balancing Test section lacks nested enumeration structure"})
        elif not analysis["has_required_levels"]:
            missing_levels = [l for l in self.required_levels if l not in analysis["levels_present"]]
            issues.append({"type": "missing_enumeration_levels", "severity": "medium", "message": f"Balancing Test section missing enumeration levels: {missing_levels}"})
        
        if analysis["meets_target"]:
            recommendations.append({"type": "sub_outline_optimal", "message": f"Balancing Test section sub-outline structure is optimal"})
        
        return FunctionResult(success=analysis["meets_target"], data={**analysis, "section_found": True, "issues": issues, "recommendations": recommendations}, message="Balancing Test section sub-outline validation complete")

