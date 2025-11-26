#!/usr/bin/env python3
"""
Balancing Test Section Paragraph Structure Plugin - Enforces paragraph structure for Balancing Test section.
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


class BalancingTestParagraphStructurePlugin(BaseFeaturePlugin):
    """Plugin to validate paragraph structure for Balancing Test section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "balancing_test_paragraph_structure", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("BalancingTestParagraphStructurePlugin initialized")

        targets = self.rules.get("targets", {})
        self.optimal_para_min = targets.get("min_paragraph_count", 2)
        self.optimal_para_max = targets.get("max_paragraph_count", 6)
        self.optimal_avg_words = targets.get("avg_words_per_paragraph", 120)

    def _analyze_paragraph_structure(self, section_text: str) -> Dict[str, Any]:
        """Analyze paragraph structure in section."""
        if not section_text:
            return {"paragraph_count": 0, "avg_words_per_paragraph": 0, "meets_target": False}
        
        paragraphs = re.split(r'(?:\r?\n\s*){2,}', section_text.strip())
        if len(paragraphs) == 1:
            paragraphs = [p.strip() for p in section_text.splitlines() if p.strip()]
        else:
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        paragraph_count = len(paragraphs) if paragraphs else 0
        para_word_counts = [len(re.findall(r"[A-Za-z0-9']+", p)) for p in paragraphs]
        avg_words = sum(para_word_counts) / len(para_word_counts) if para_word_counts else 0
        
        meets_para_target = self.optimal_para_min <= paragraph_count <= self.optimal_para_max
        meets_avg_target = abs(avg_words - self.optimal_avg_words) / self.optimal_avg_words < 0.3
        
        return {
            "paragraph_count": paragraph_count,
            "avg_words_per_paragraph": avg_words,
            "optimal_para_range": [self.optimal_para_min, self.optimal_para_max],
            "optimal_avg_words": self.optimal_avg_words,
            "meets_para_target": meets_para_target,
            "meets_avg_target": meets_avg_target,
            "meets_target": meets_para_target and meets_avg_target
        }

    async def validate_balancing_test_paragraph_structure(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate paragraph structure for Balancing Test section."""
        text = document.get_full_text()
        section_text = extract_section_text(text, 'balancing_test')
        
        if not section_text:
            return FunctionResult(success=False, data={"section_found": False, "issues": [{"type": "missing_section", "severity": "high", "message": "Balancing Test section not found"}]}, message="Balancing Test section not found")
        
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
                    section_name='balancing_test',
                    feature_type='paragraph_structure',
                    optimal_range=optimal_range,
                    limit=5
                )
            except Exception as e:
                logger.debug(f"Corpus search failed for balancing_test paragraph_structure: {e}")
        
        if not analysis["meets_para_target"]:
            if analysis["paragraph_count"] < self.optimal_para_min:
                issues.append({"type": "insufficient_paragraphs", "severity": "medium", "message": f"Balancing Test section has {analysis['paragraph_count']} paragraphs, below optimal range ({self.optimal_para_min}-{self.optimal_para_max})"})
            elif analysis["paragraph_count"] > self.optimal_para_max:
                issues.append({"type": "excessive_paragraphs", "severity": "low", "message": f"Balancing Test section has {analysis['paragraph_count']} paragraphs, above optimal range ({self.optimal_para_min}-{self.optimal_para_max})"})
        
        if not analysis["meets_avg_target"]:
            issues.append({"type": "suboptimal_avg_words", "severity": "low", "message": f"Balancing Test section average words per paragraph ({analysis['avg_words_per_paragraph']:.0f}) differs from optimal ({self.optimal_avg_words:.0f})"})
        
        if analysis["meets_target"]:
            recommendations.append({"type": "paragraph_structure_optimal", "message": f"Balancing Test section paragraph structure is optimal"})
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
            message="Balancing Test section paragraph structure validation complete"
        )

