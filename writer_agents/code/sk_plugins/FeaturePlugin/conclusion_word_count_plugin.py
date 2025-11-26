#!/usr/bin/env python3
"""
Conclusion Section Word Count Plugin - Enforces word count for Conclusion section.

Optimal ranges are based on Q25-Q75 percentiles from successful motions (1,252 sections analyzed).
Data source: case_law_data/analysis/section_optimal_thresholds.json
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


class ConclusionWordCountPlugin(BaseFeaturePlugin):
    """Plugin to validate word count for Conclusion section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "conclusion_word_count", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("ConclusionWordCountPlugin initialized")

        # Load optimal word count range from analysis data
        loader = get_loader()
        word_count_range = loader.get_word_count_range('conclusion')
        
        if word_count_range:
            self.optimal_min = int(word_count_range[0])
            self.optimal_max = int(word_count_range[1])
            logger.debug(f"Loaded optimal range from analysis: {self.optimal_min}-{self.optimal_max}")
        else:
            targets = self.rules.get("targets", {})
            self.optimal_min = targets.get("min_word_count", 403)
            self.optimal_max = targets.get("max_word_count", 3670)
            logger.warning(f"Using fallback range: {self.optimal_min}-{self.optimal_max}")

    def _analyze_section_word_count(self, section_text: str) -> Dict[str, Any]:
        """Analyze word count in Conclusion section."""
        if not section_text:
            return {
                "word_count": 0,
                "meets_target": False,
                "gap": self.optimal_min
            }
        
        words = re.findall(r"[A-Za-z0-9']+", section_text)
        word_count = len(words)
        
        meets_target = self.optimal_min <= word_count <= self.optimal_max
        gap = 0
        if word_count < self.optimal_min:
            gap = self.optimal_min - word_count
        elif word_count > self.optimal_max:
            gap = word_count - self.optimal_max
        
        return {
            "word_count": word_count,
            "optimal_min": self.optimal_min,
            "optimal_max": self.optimal_max,
            "meets_target": meets_target,
            "gap": gap
        }

    async def validate_conclusion_word_count(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate word count for Conclusion section.
        """
        text = document.get_full_text()
        section_text = extract_section_text(text, 'conclusion')
        
        if not section_text:
            return FunctionResult(
                success=False,
                data={
                    "section_found": False,
                    "issues": [{
                        "type": "missing_section",
                        "severity": "medium",
                        "message": "Conclusion section not found",
                        "suggestion": "Add a Conclusion section"
                    }]
                },
                message="Conclusion section not found"
            )
        
        analysis = self._analyze_section_word_count(section_text)
        
        issues = []
        recommendations = []
        corpus_examples = []
        
        # Search corpus for examples if validation fails
        if not analysis["meets_target"]:
            try:
                optimal_range = (self.optimal_min, self.optimal_max)
                corpus_examples = await find_section_examples_from_corpus(
                    plugin_instance=self,
                    section_name='conclusion',
                    feature_type='word_count',
                    optimal_range=optimal_range,
                    limit=5
                )
            except Exception as e:
                logger.debug(f"Corpus search failed for conclusion word_count: {e}")
        
        if not analysis["meets_target"]:
            if analysis["word_count"] < self.optimal_min:
                issues.append({
                    "type": "insufficient_words",
                    "severity": "medium",
                    "message": f"Conclusion section word count ({analysis['word_count']}) is below optimal range ({self.optimal_min}-{self.optimal_max} words, Q25-Q75 from successful motions)",
                    "suggestion": f"Increase Conclusion section to at least {self.optimal_min} words"
                })
                recommendation = {
                    "type": "increase_section_length",
                    "priority": "medium",
                    "action": f"Increase Conclusion section from {analysis['word_count']} to at least {self.optimal_min} words",
                    "rationale": "Optimal word count range for Conclusion section"
                }
                # Add corpus examples to recommendation
                if corpus_examples:
                    recommendation["corpus_examples"] = [
                        {
                            "case_name": ex.get("case_name", "Unknown"),
                            "citation": ex.get("citation", ""),
                            "word_count": ex.get("feature_value", 0),
                            "snippet": ex.get("section_text", "")[:200]
                        }
                        for ex in corpus_examples[:3]
                    ]
                    recommendation["action"] += f". See examples from successful motions: {', '.join([ex.get('case_name', 'Unknown') for ex in corpus_examples[:3]])}"
                recommendations.append(recommendation)
            elif analysis["word_count"] > self.optimal_max:
                issues.append({
                    "type": "excessive_words",
                    "severity": "low",
                    "message": f"Conclusion section word count ({analysis['word_count']}) exceeds optimal range ({self.optimal_min}-{self.optimal_max} words, Q25-Q75 from successful motions)",
                    "suggestion": f"Consider condensing Conclusion section to {self.optimal_max} words or less"
                })
        else:
            recommendations.append({
                "type": "section_length_optimal",
                "message": f"Conclusion section word count is optimal ({analysis['word_count']} words, range: {self.optimal_min}-{self.optimal_max})"
            })
        
        return FunctionResult(
            success=analysis["meets_target"],
            data={
                "word_count": analysis["word_count"],
                "optimal_range": [self.optimal_min, self.optimal_max],
                "meets_target": analysis["meets_target"],
                "gap": analysis["gap"],
                "section_found": True,
                "issues": issues,
                "recommendations": recommendations,
                "corpus_examples": corpus_examples
            },
            message="Conclusion section word count validation complete"
        )

