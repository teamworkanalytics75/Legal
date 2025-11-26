#!/usr/bin/env python3
"""
Conclusion Section Words Per Sentence Plugin - Enforces words per sentence for Conclusion section.

Validates that sentences in the Conclusion section have optimal word counts based on analysis.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult
from .section_utils import extract_section_text, extract_paragraphs_from_section, extract_sentences_from_paragraph, count_words_in_sentence, find_section_examples_from_corpus
from .constraint_data_loader import get_loader

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class ConclusionWordsPerSentencePlugin(BaseFeaturePlugin):
    """Plugin to validate words per sentence for Conclusion section."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "conclusion_words_per_sentence", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("ConclusionWordsPerSentencePlugin initialized")

        loader = get_loader()
        words_range = loader.get_words_per_sentence_range('conclusion')
        
        if words_range:
            self.optimal_min = int(words_range[0])
            self.optimal_max = int(words_range[1])
            logger.debug(f"Loaded optimal range from analysis: {self.optimal_min}-{self.optimal_max}")
        else:
            targets = self.rules.get("targets", {})
            self.optimal_min = targets.get("min_words_per_sentence", 10)
            self.optimal_max = targets.get("max_words_per_sentence", 30)
            logger.warning(f"Using fallback range: {self.optimal_min}-{self.optimal_max}")

    def _analyze_words_per_sentence(self, section_text: str) -> Dict[str, Any]:
        """Analyze words per sentence in section."""
        if not section_text:
            return {"avg_words_per_sentence": 0, "sentence_word_counts": [], "meets_target": False}
        
        paragraphs = extract_paragraphs_from_section(section_text)
        sentence_word_counts = []
        
        for para in paragraphs:
            sentences = extract_sentences_from_paragraph(para)
            for sent in sentences:
                word_count = count_words_in_sentence(sent)
                if word_count > 0:
                    sentence_word_counts.append(word_count)
        
        if not sentence_word_counts:
            return {"avg_words_per_sentence": 0, "sentence_word_counts": [], "meets_target": False}
        
        avg_words = sum(sentence_word_counts) / len(sentence_word_counts)
        min_words = min(sentence_word_counts)
        max_words = max(sentence_word_counts)
        
        sentences_in_range = sum(1 for count in sentence_word_counts if self.optimal_min <= count <= self.optimal_max)
        meets_target = (sentences_in_range / len(sentence_word_counts)) >= 0.7
        
        return {
            "avg_words_per_sentence": avg_words,
            "min_words_per_sentence": min_words,
            "max_words_per_sentence": max_words,
            "sentence_word_counts": sentence_word_counts,
            "sentences_in_range": sentences_in_range,
            "total_sentences": len(sentence_word_counts),
            "optimal_range": [self.optimal_min, self.optimal_max],
            "meets_target": meets_target
        }

    async def validate_conclusion_words_per_sentence(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Validate words per sentence for Conclusion section."""
        text = document.get_full_text()
        section_text = extract_section_text(text, 'conclusion')
        
        if not section_text:
            return FunctionResult(success=False, data={"section_found": False, "issues": [{"type": "missing_section", "severity": "medium", "message": "Conclusion section not found"}]}, message="Conclusion section not found")
        
        analysis = self._analyze_words_per_sentence(section_text)
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
                    feature_type='words_per_sentence',
                    optimal_range=optimal_range,
                    limit=5
                )
            except Exception as e:
                logger.debug(f"Corpus search failed for conclusion words_per_sentence: {e}")
        
        if not analysis["meets_target"]:
            issues.append({
                "type": "suboptimal_words_per_sentence",
                "severity": "low",
                "message": f"Conclusion section has {analysis['sentences_in_range']}/{analysis['total_sentences']} sentences with optimal word count ({self.optimal_min}-{self.optimal_max} words)",
                "suggestion": f"Adjust sentence length to have {self.optimal_min}-{self.optimal_max} words per sentence"
            })
            recommendation = {
                "type": "adjust_words_per_sentence",
                "priority": "low",
                "action": f"Adjust sentence length to have {self.optimal_min}-{self.optimal_max} words per sentence",
                "rationale": "Optimal words per sentence for Conclusion section"
            }
            # Add corpus examples to recommendation
            if corpus_examples:
                recommendation["corpus_examples"] = [
                    {
                        "case_name": ex.get("case_name", "Unknown"),
                        "citation": ex.get("citation", ""),
                        "words_per_sentence": ex.get("feature_value", 0),
                        "snippet": ex.get("section_text", "")[:200]
                    }
                    for ex in corpus_examples[:3]
                ]
                recommendation["action"] += f". See examples from successful motions: {', '.join([ex.get('case_name', 'Unknown') for ex in corpus_examples[:3]])}"
            recommendations.append(recommendation)
        else:
            recommendations.append({
                "type": "words_per_sentence_optimal",
                "message": f"Conclusion section words per sentence is optimal (avg: {analysis['avg_words_per_sentence']:.1f})"
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
            message="Conclusion section words per sentence validation complete"
        )
