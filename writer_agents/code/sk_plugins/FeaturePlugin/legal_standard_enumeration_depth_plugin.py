#!/usr/bin/env python3
"""
Legal Standard Section Enumeration Depth Plugin - Enforces enumeration depth for Legal Standard section.

Validates that the Legal Standard section has optimal enumeration depth based on CatBoost analysis.
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


class LegalStandardEnumerationDepthPlugin(BaseFeaturePlugin):
    """Plugin to validate enumeration depth for Legal Standard section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "legal_standard_enumeration_depth", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("LegalStandardEnumerationDepthPlugin initialized")

        # Load optimal enumeration depth from analysis data
        loader = get_loader()
        enum_depth_threshold = loader.get_enumeration_depth_threshold('legal_standard')
        
        if enum_depth_threshold is not None:
            self.target_depth = int(enum_depth_threshold)
            logger.debug(f"Loaded enumeration depth threshold from analysis: {self.target_depth}")
        else:
            # Since median is 0 for most sections, enumeration is primarily document-level
            targets = self.rules.get("targets", {})
            self.target_depth = targets.get("optimal_enumeration_depth", 0)
            logger.warning(f"Using fallback enumeration depth: {self.target_depth}")

    def _calculate_max_enumeration_depth(self, text: str) -> Dict[str, Any]:
        """Calculate maximum enumeration depth in section text."""
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
                # Calculate depth based on indentation
                leading_spaces = len(line) - len(line.lstrip())
                # Depth = (leading spaces / 2) + 1 (assuming 2 spaces per level)
                depth = (leading_spaces // 2) + 1
                
                if depth > max_depth:
                    max_depth = depth
                    depth_positions.append({
                        "line": i + 1,
                        "depth": depth,
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

    async def validate_legal_standard_enumeration_depth(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate enumeration depth for Legal Standard section.
        """
        text = document.get_full_text()
        section_text = extract_section_text(text, 'legal_standard')
        
        if not section_text:
            return FunctionResult(
                success=False,
                data={
                    "section_found": False,
                    "issues": [{
                        "type": "missing_section",
                        "severity": "high",
                        "message": "Legal Standard section not found",
                        "suggestion": "Add a Legal Standard section"
                    }]
                },
                message="Legal Standard section not found"
            )
        
        analysis = self._calculate_max_enumeration_depth(section_text)
        
        issues = []
        recommendations = []
        corpus_examples = []
        
        # Search corpus for examples if validation fails
        if not analysis["meets_target"] and self.target_depth > 0:
            try:
                # For enumeration depth, we search for examples with similar depth
                corpus_examples = await find_section_examples_from_corpus(
                    plugin_instance=self,
                    section_name='legal_standard',
                    feature_type='enumeration_depth',
                    optimal_range=(self.target_depth, self.target_depth + 2),  # Allow some flexibility
                    limit=5
                )
            except Exception as e:
                logger.debug(f"Corpus search failed for legal_standard enumeration_depth: {e}")
        
        if not analysis["meets_target"]:
            issues.append({
                "type": "insufficient_enumeration_depth",
                "severity": "medium",
                "message": f"Legal Standard section enumeration depth ({analysis['max_depth']}) is below target ({self.target_depth})",
                "suggestion": f"Increase enumeration depth to at least {self.target_depth} levels"
            })
            recommendation = {
                "type": "increase_enumeration_depth",
                "priority": "medium",
                "action": f"Add nested enumeration structures to Legal Standard section to reach depth {self.target_depth}",
                "rationale": "Optimal enumeration depth for Legal Standard section"
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
                "message": f"Legal Standard section enumeration depth is optimal ({analysis['max_depth']} levels, target: {self.target_depth})"
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
            message="Legal Standard section enumeration depth validation complete"
        )

