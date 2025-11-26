#!/usr/bin/env python3
"""
Per Paragraph Plugin - Atomic SK plugin for individual paragraph analysis and enforcement.

Analyzes each paragraph individually for:
- Word count per paragraph
- Sentence count per paragraph
- Readability metrics
- Topic coherence
- Structural issues
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class PerParagraphPlugin(BaseFeaturePlugin):
    """Atomic plugin for per-paragraph analysis and enforcement."""

    def __init__(
        self,
        kernel: Kernel,
        chroma_store,
        rules_dir: Path,
        memory_store=None,
        db_paths=None,
        enable_langchain: bool = True,
        enable_courtlistener: bool = False,
        enable_storm: bool = False
    ):
        super().__init__(
            kernel, "per_paragraph", chroma_store, rules_dir,
            memory_store=memory_store,
            db_paths=db_paths,
            enable_langchain=enable_langchain,
            enable_courtlistener=enable_courtlistener,
            enable_storm=enable_storm
        )
        logger.info("PerParagraphPlugin initialized")

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'(?:\r?\n\s*){2,}', text.strip())
        if len(paragraphs) == 1:
            paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        else:
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        normalized = re.sub(r'\s+', ' ', text.strip())
        if not normalized:
            return []
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', normalized)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize words."""
        return re.findall(r"[A-Za-z0-9']+", text)

    def _analyze_individual_paragraph(self, paragraph: str, index: int, total_paras: int) -> Dict[str, Any]:
        """Analyze a single paragraph."""
        words = self._tokenize_words(paragraph)
        sentences = self._split_sentences(paragraph)

        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0

        # Check for issues
        issues = []
        target_min_words = self.rules.get("targets", {}).get("min_words_per_paragraph", 50)
        target_max_words = self.rules.get("targets", {}).get("max_words_per_paragraph", 180)
        target_min_sentences = self.rules.get("targets", {}).get("min_sentences_per_paragraph", 2)
        target_max_sentences = self.rules.get("targets", {}).get("max_sentences_per_paragraph", 8)

        if word_count < target_min_words:
            issues.append(f"Too short: {word_count} words (target: {target_min_words}+)")
        if word_count > target_max_words:
            issues.append(f"Too long: {word_count} words (target: <{target_max_words})")
        if sentence_count < target_min_sentences:
            issues.append(f"Too few sentences: {sentence_count} (target: {target_min_sentences}+)")
        if sentence_count > target_max_sentences:
            issues.append(f"Too many sentences: {sentence_count} (target: <{target_max_sentences})")

        # Check for very long sentences in paragraph
        long_sentences = []
        for i, sent in enumerate(sentences):
            sent_words = len(self._tokenize_words(sent))
            if sent_words > 45:
                long_sentences.append({"sentence_index": i, "word_count": sent_words})

        return {
            "paragraph_index": index,
            "position": "intro" if index == 0 else "conclusion" if index == total_paras - 1 else "body",
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": avg_words_per_sentence,
            "issues": issues,
            "long_sentences": long_sentences,
            "health_score": self._calculate_paragraph_health(word_count, sentence_count, len(issues)),
            "preview": paragraph[:150] + "..." if len(paragraph) > 150 else paragraph
        }

    def _calculate_paragraph_health(self, word_count: int, sentence_count: int, issue_count: int) -> float:
        """Calculate health score for a paragraph (0.0 to 1.0)."""
        score = 1.0

        # Word count penalty
        target_min = 50
        target_max = 180
        if word_count < target_min:
            score -= (target_min - word_count) / target_min * 0.3
        elif word_count > target_max:
            score -= (word_count - target_max) / target_max * 0.3

        # Sentence count penalty
        if sentence_count < 2:
            score -= 0.2
        elif sentence_count > 8:
            score -= (sentence_count - 8) / 8 * 0.2

        # Issue count penalty
        score -= issue_count * 0.1

        return max(0.0, min(1.0, score))

    async def analyze_all_paragraphs(self, draft_text: str) -> FunctionResult:
        """Analyze all paragraphs individually."""
        try:
            paragraphs = self._split_paragraphs(draft_text)

            if not paragraphs:
                return FunctionResult(
                    success=False,
                    value=None,
                    error="No paragraphs found in document"
                )

            paragraph_analyses = []
            for i, para in enumerate(paragraphs):
                analysis = self._analyze_individual_paragraph(para, i, len(paragraphs))
                paragraph_analyses.append(analysis)

            # Identify problematic paragraphs
            problematic = [p for p in paragraph_analyses if p["issues"]]
            healthy = [p for p in paragraph_analyses if not p["issues"]]

            # Calculate overall health
            avg_health = sum(p["health_score"] for p in paragraph_analyses) / len(paragraph_analyses) if paragraph_analyses else 0.0

            return FunctionResult(
                success=True,
                value={
                    "total_paragraphs": len(paragraphs),
                    "paragraph_analyses": paragraph_analyses,
                    "problematic_paragraphs": problematic,
                    "healthy_paragraphs": healthy,
                    "avg_health_score": avg_health,
                    "health_distribution": {
                        "excellent": len([p for p in paragraph_analyses if p["health_score"] >= 0.9]),
                        "good": len([p for p in paragraph_analyses if 0.7 <= p["health_score"] < 0.9]),
                        "fair": len([p for p in paragraph_analyses if 0.5 <= p["health_score"] < 0.7]),
                        "poor": len([p for p in paragraph_analyses if p["health_score"] < 0.5])
                    }
                },
                metadata={"method": "per_paragraph_analysis"}
            )

        except Exception as e:
            logger.error(f"Per-paragraph analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    async def fix_specific_paragraph(self, paragraph: str, paragraph_index: int, issues: List[str]) -> FunctionResult:
        """Generate fixes for a specific problematic paragraph."""
        try:
            suggestions = []

            analysis = self._analyze_individual_paragraph(paragraph, paragraph_index, 1)

            for issue in issues:
                if "Too short" in issue:
                    suggestions.append(
                        f"Expand paragraph {paragraph_index + 1}: Add more detail, examples, or explanation. "
                        f"Current: {analysis['word_count']} words, target: 50+ words."
                    )
                elif "Too long" in issue:
                    suggestions.append(
                        f"Split paragraph {paragraph_index + 1}: Break into multiple paragraphs at logical breaks. "
                        f"Current: {analysis['word_count']} words, target: <180 words."
                    )
                elif "Too few sentences" in issue:
                    suggestions.append(
                        f"Expand paragraph {paragraph_index + 1}: Add more sentences to develop the argument. "
                        f"Current: {analysis['sentence_count']} sentences, target: 2+ sentences."
                    )
                elif "Too many sentences" in issue:
                    suggestions.append(
                        f"Consolidate paragraph {paragraph_index + 1}: Combine related sentences or split into new paragraph. "
                        f"Current: {analysis['sentence_count']} sentences, target: <8 sentences."
                    )

            # Handle long sentences within paragraph
            if analysis["long_sentences"]:
                suggestions.append(
                    f"Paragraph {paragraph_index + 1} contains {len(analysis['long_sentences'])} long sentences. "
                    f"Split sentences exceeding 45 words for better readability."
                )

            return FunctionResult(
                success=True,
                value={
                    "paragraph_index": paragraph_index,
                    "current_analysis": analysis,
                    "suggestions": suggestions,
                    "target_metrics": self.rules.get("targets", {})
                },
                metadata={"method": "paragraph_fix"}
            )

        except Exception as e:
            logger.error(f"Paragraph fix generation failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    async def validate_draft(self, draft_text: str) -> FunctionResult:
        """Validate draft by analyzing all paragraphs."""
        return await self.analyze_all_paragraphs(draft_text)

