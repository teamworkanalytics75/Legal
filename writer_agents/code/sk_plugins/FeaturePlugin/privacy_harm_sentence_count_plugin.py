#!/usr/bin/env python3
"""
Privacy Harm Section Sentence Count Plugin - Enforces sentences per paragraph for Privacy Harm section.

Validates that paragraphs in the Privacy Harm section have optimal sentence counts based on analysis.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult
from .section_utils import extract_section_text, extract_paragraphs_from_section, extract_sentences_from_paragraph, find_section_examples_from_corpus
from .constraint_data_loader import get_loader

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class PrivacyHarmSentenceCountPlugin(BaseFeaturePlugin):
    """Plugin to validate sentences per paragraph for Privacy Harm section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "privacy_harm_sentence_count", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("PrivacyHarmSentenceCountPlugin initialized")

        loader = get_loader()
        sentences_range = loader.get_sentences_per_paragraph_range('privacy_harm')
        
        if sentences_range:
            self.optimal_min = int(sentences_range[0])
            self.optimal_max = int(sentences_range[1])
            logger.debug(f"Loaded optimal range from analysis: {self.optimal_min}-{self.optimal_max}")
        else:
            targets = self.rules.get("targets", {})
            self.optimal_min = targets.get("min_sentences_per_paragraph", 2)
            self.optimal_max = targets.get("max_sentences_per_paragraph", 8)
            logger.warning(f"Using fallback range: {self.optimal_min}-{self.optimal_max}")

    def _analyze_sentence_count(self, section_text: str) -> Dict[str, Any]:
        """Analyze sentences per paragraph in section."""
        if not section_text:
            return {"avg_sentences_per_paragraph": 0, "paragraph_sentence_counts": [], "meets_target": False}
        
        paragraphs = extract_paragraphs_from_section(section_text)
        paragraph_sentence_counts = []
        
        for para in paragraphs:
            sentences = extract_sentences_from_paragraph(para)
            paragraph_sentence_counts.append(len(sentences))
        
        if not paragraph_sentence_counts:
            return {"avg_sentences_per_paragraph": 0, "paragraph_sentence_counts": [], "meets_target": False}
        
        avg_sentences = sum(paragraph_sentence_counts) / len(paragraph_sentence_counts)
        min_sentences = min(paragraph_sentence_counts)
        max_sentences = max(paragraph_sentence_counts)
        
        paragraphs_in_range = sum(1 for count in paragraph_sentence_counts if self.optimal_min <= count <= self.optimal_max)
        meets_target = (paragraphs_in_range / len(paragraph_sentence_counts)) >= 0.7
        
        return {
            "avg_sentences_per_paragraph": avg_sentences,
            "min_sentences_per_paragraph": min_sentences,
            "max_sentences_per_paragraph": max_sentences,
            "paragraph_sentence_counts": paragraph_sentence_counts,
            "paragraphs_in_range": paragraphs_in_range,
            "total_paragraphs": len(paragraph_sentence_counts),
            "optimal_range": [self.optimal_min, self.optimal_max],
            "meets_target": meets_target
        }

    async def validate_privacy_harm_sentence_count(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate sentences per paragraph for Privacy Harm section."""
        text = document.get_full_text()
        section_text = extract_section_text(text, 'privacy_harm')
        
        if not section_text:
            return FunctionResult(success=False, data={"section_found": False, "issues": [{"type": "missing_section", "severity": "high", "message": "Privacy Harm section not found"}]}, message="Privacy Harm section not found")
        
        analysis = self._analyze_sentence_count(section_text)
        issues = []
        recommendations = []
        corpus_examples = []
        
        # Search corpus for examples if validation fails
        if not analysis["meets_target"]:
            try:
                optimal_range = (self.optimal_min, self.optimal_max)
                corpus_examples = await find_section_examples_from_corpus(
                    plugin_instance=self,
                    section_name='privacy_harm',
                    feature_type='sentence_count',
                    optimal_range=optimal_range,
                    limit=5
                )
            except Exception as e:
                logger.debug(f"Corpus search failed for privacy_harm sentence_count: {e}")
        
        if not analysis["meets_target"]:
            issues.append({
                "type": "suboptimal_sentence_count",
                "severity": "low",
                "message": f"Privacy Harm section has {analysis['paragraphs_in_range']}/{analysis['total_paragraphs']} paragraphs with optimal sentence count ({self.optimal_min}-{self.optimal_max} sentences)",
                "suggestion": f"Adjust paragraph structure to have {self.optimal_min}-{self.optimal_max} sentences per paragraph"
            })
            recommendation = {
                "type": "adjust_sentence_count",
                "priority": "low",
                "action": f"Adjust paragraph structure to have {self.optimal_min}-{self.optimal_max} sentences per paragraph",
                "rationale": "Optimal sentence count per paragraph for Privacy Harm section"
            }
            # Add corpus examples to recommendation
            if corpus_examples:
                recommendation["corpus_examples"] = [
                    {
                        "case_name": ex.get("case_name", "Unknown"),
                        "citation": ex.get("citation", ""),
                        "sentence_count": ex.get("feature_value", 0),
                        "snippet": ex.get("section_text", "")[:200]
                    }
                    for ex in corpus_examples[:3]
                ]
                recommendation["action"] += f". See examples from successful motions: {', '.join([ex.get('case_name', 'Unknown') for ex in corpus_examples[:3]])}"
            recommendations.append(recommendation)
        else:
            recommendations.append({
                "type": "sentence_count_optimal",
                "message": f"Privacy Harm section sentence count per paragraph is optimal (avg: {analysis['avg_sentences_per_paragraph']:.1f})"
            })
        
        return FunctionResult(
            success=analysis["meets_target"],
            data={
                **analysis,
                "section_found": True,
                "issues": issues,
                "recommendations": recommendations,
                "corpus_examples": corpus_examples
            },
            message="Privacy Harm section sentence count validation complete"
        )
