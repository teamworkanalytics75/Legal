#!/usr/bin/env python3
"""
Introduction Section Paragraph Structure Plugin - Enforces paragraph structure for Introduction section.

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
from .section_utils import extract_section_text, extract_paragraphs_from_section, find_section_examples_from_corpus
from .constraint_data_loader import get_loader

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class IntroductionParagraphStructurePlugin(BaseFeaturePlugin):
    """Plugin to validate paragraph structure for Introduction section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "introduction_paragraph_structure", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("IntroductionParagraphStructurePlugin initialized")

        # Load optimal ranges from analysis data
        loader = get_loader()
        thresholds = loader.load_section_thresholds('introduction')
        
        para_count_range = loader.get_paragraph_count_range('introduction')
        avg_words_range = loader.get_avg_words_per_paragraph_range('introduction')
        
        if para_count_range:
            self.optimal_para_min = int(para_count_range[0])
            self.optimal_para_max = int(para_count_range[1])
        else:
            self.optimal_para_min = 3
            self.optimal_para_max = 30
        
        if avg_words_range:
            self.optimal_avg_words_min = int(avg_words_range[0])
            self.optimal_avg_words_max = int(avg_words_range[1])
            self.optimal_avg_words = (self.optimal_avg_words_min + self.optimal_avg_words_max) / 2
        else:
            self.optimal_avg_words = 28

    def _analyze_paragraph_structure(self, section_text: str) -> Dict[str, Any]:
        """Analyze paragraph structure in section."""
        if not section_text:
            return {"paragraph_count": 0, "avg_words_per_paragraph": 0, "meets_target": False}
        
        paragraphs = extract_paragraphs_from_section(section_text)
        paragraph_count = len(paragraphs) if paragraphs else 0
        para_word_counts = [len(re.findall(r"[A-Za-z0-9']+", p)) for p in paragraphs]
        avg_words = sum(para_word_counts) / len(para_word_counts) if para_word_counts else 0
        
        meets_para_target = self.optimal_para_min <= paragraph_count <= self.optimal_para_max
        meets_avg_target = abs(avg_words - self.optimal_avg_words) / self.optimal_avg_words < 0.3 if self.optimal_avg_words > 0 else True
        
        return {
            "paragraph_count": paragraph_count,
            "avg_words_per_paragraph": avg_words,
            "optimal_para_range": [self.optimal_para_min, self.optimal_para_max],
            "optimal_avg_words": self.optimal_avg_words,
            "meets_para_target": meets_para_target,
            "meets_avg_target": meets_avg_target,
            "meets_target": meets_para_target and meets_avg_target
        }

    async def validate_introduction_paragraph_structure(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate paragraph structure for Introduction section."""
        text = document.get_full_text()
        section_text = extract_section_text(text, 'introduction')
        
        if not section_text:
            return FunctionResult(success=False, data={"section_found": False, "issues": [{"type": "missing_section", "severity": "low", "message": "Introduction section not found"}]}, message="Introduction section not found")
        
        analysis = self._analyze_paragraph_structure(section_text)
        issues = []
        recommendations = []
        corpus_examples = []
        
        # Search corpus for examples if validation fails
        if not analysis["meets_target"]:
            try:
                optimal_range = (self.optimal_para_min, self.optimal_para_max)
                corpus_examples = await find_section_examples_from_corpus(
                    plugin_instance=self,
                    section_name='introduction',
                    feature_type='paragraph_structure',
                    optimal_range=optimal_range,
                    limit=5
                )
            except Exception as e:
                logger.debug(f"Corpus search failed for introduction paragraph_structure: {e}")
        
        if not analysis["meets_para_target"]:
            if analysis["paragraph_count"] < self.optimal_para_min:
                issues.append({"type": "insufficient_paragraphs", "severity": "low", "message": f"Introduction section has {analysis['paragraph_count']} paragraphs, below optimal range ({self.optimal_para_min}-{self.optimal_para_max})"})
            elif analysis["paragraph_count"] > self.optimal_para_max:
                issues.append({"type": "excessive_paragraphs", "severity": "low", "message": f"Introduction section has {analysis['paragraph_count']} paragraphs, above optimal range ({self.optimal_para_min}-{self.optimal_para_max})"})
        
        if not analysis["meets_avg_target"]:
            issues.append({"type": "suboptimal_avg_words", "severity": "low", "message": f"Introduction section average words per paragraph ({analysis['avg_words_per_paragraph']:.0f}) differs from optimal ({self.optimal_avg_words:.0f})"})
        
        if analysis["meets_target"]:
            recommendations.append({"type": "paragraph_structure_optimal", "message": f"Introduction section paragraph structure is optimal"})
        else:
            # Add corpus examples to recommendations if available
            if corpus_examples:
                for rec in recommendations:
                    if rec.get("type") in ["insufficient_paragraphs", "excessive_paragraphs", "suboptimal_avg_words"]:
                        rec["corpus_examples"] = [
                            {
                                "case_name": ex.get("case_name", "Unknown"),
                                "citation": ex.get("citation", ""),
                                "paragraph_count": ex.get("feature_value", 0),
                                "snippet": ex.get("section_text", "")[:200]
                            }
                            for ex in corpus_examples[:3]
                        ]
                        rec["action"] = rec.get("action", rec.get("message", "")) + f". See examples from successful motions: {', '.join([ex.get('case_name', 'Unknown') for ex in corpus_examples[:3]])}"
        
        return FunctionResult(
            success=analysis["meets_target"],
            data={
                **analysis,
                "section_found": True,
                "issues": issues,
                "recommendations": recommendations,
                "corpus_examples": corpus_examples
            },
            message="Introduction section paragraph structure validation complete"
        )

