#!/usr/bin/env python3
"""
Danger/Safety Section Enumeration Depth Plugin - Enforces enumeration depth for Danger/Safety section.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult
from .section_utils import extract_section_text, find_section_examples_from_corpus
from .constraint_data_loader import get_loader

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class DangerSafetyEnumerationDepthPlugin(BaseFeaturePlugin):
    """Plugin to validate enumeration depth for Danger/Safety section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "danger_safety_enumeration_depth", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("DangerSafetyEnumerationDepthPlugin initialized")

        targets = self.rules.get("targets", {})
        self.target_depth = targets.get("optimal_enumeration_depth", 2)

    def _calculate_max_enumeration_depth(self, text: str) -> Dict[str, Any]:
        """Calculate maximum enumeration depth in section text."""
        lines = text.split('\n')
        max_depth = 0
        depth_positions = []
        patterns = [
            r'^\s*\d+[\.\)]\s+', r'^\s+[a-z][\.\)]\s+', r'^\s+[ivxlcdm]+[\.\)]\s+',
            r'^\s+[A-Z][\.\)]\s+', r'^\s+\([a-z]\)\s+', r'^\s+\(\d+\)\s+',
        ]

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            for level, pattern in enumerate(patterns, 1):
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    leading_spaces = len(line) - len(line.lstrip())
                    depth = (leading_spaces // 2) + 1
                    if depth > max_depth:
                        max_depth = depth
                        depth_positions.append({"line": i + 1, "depth": depth, "text": line_stripped[:100]})
                    break

        nested_count = sum(1 for pos in depth_positions if pos["depth"] >= 3)
        return {
            "max_depth": max_depth,
            "target_depth": self.target_depth,
            "meets_target": max_depth >= self.target_depth,
            "depth_positions": depth_positions[:20],
            "nested_count": nested_count,
            "has_nested_lists": max_depth >= 3
        }

    async def validate_danger_safety_enumeration_depth(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate enumeration depth for Danger/Safety section."""
        text = document.get_full_text()
        section_text = extract_section_text(text, 'danger_safety')
        
        if not section_text:
            return FunctionResult(success=False, data={"section_found": False, "issues": [{"type": "missing_section", "severity": "high", "message": "Danger/Safety section not found"}]}, message="Danger/Safety section not found")
        
        analysis = self._calculate_max_enumeration_depth(section_text)
        issues = []
        recommendations = []
        corpus_examples = []
        
        # Search corpus for examples if validation fails
        if not analysis["meets_target"]:
            try:
                corpus_examples = await find_section_examples_from_corpus(
                    plugin_instance=self,
                    section_name='danger_safety',
                    feature_type='enumeration_depth',
                    optimal_range=(self.target_depth, self.target_depth + 2),
                    limit=5
                )
            except Exception as e:
                logger.debug(f"Corpus search failed for danger_safety enumeration_depth: {e}")
        
        if not analysis["meets_target"]:
            issues.append({
                "type": "insufficient_enumeration_depth",
                "severity": "medium",
                "message": f"Danger/Safety section enumeration depth ({analysis['max_depth']}) is below target ({self.target_depth})",
                "suggestion": f"Increase enumeration depth to at least {self.target_depth} levels"
            })
            recommendation = {
                "type": "increase_enumeration_depth",
                "priority": "medium",
                "action": f"Add nested enumeration structures to Danger/Safety section to reach depth {self.target_depth}",
                "rationale": "Optimal enumeration depth for Danger/Safety section"
            }
            # Add corpus examples to recommendation
            if corpus_examples:
                recommendation["corpus_examples"] = [
                    {
                        "case_name": ex.get("case_name", "Unknown"),
                        "citation": ex.get("citation", ""),
                        "enumeration_depth": ex.get("feature_value", 0),
                        "snippet": ex.get("section_text", "")[:200]
                    }
                    for ex in corpus_examples[:3]
                ]
                recommendation["action"] += f". See examples from successful motions: {', '.join([ex.get('case_name', 'Unknown') for ex in corpus_examples[:3]])}"
            recommendations.append(recommendation)
        else:
            recommendations.append({
                "type": "enumeration_depth_optimal",
                "message": f"Danger/Safety section enumeration depth is optimal ({analysis['max_depth']} levels)"
            })
        
        return FunctionResult(
            success=analysis["meets_target"],
            data={
                "max_depth": analysis["max_depth"],
                "target_depth": self.target_depth,
                "meets_target": analysis["meets_target"],
                "nested_count": analysis["nested_count"],
                "has_nested_lists": analysis["has_nested_lists"],
                "section_found": True,
                "issues": issues,
                "recommendations": recommendations,
                "corpus_examples": corpus_examples
            },
            message="Danger/Safety section enumeration depth validation complete"
        )

