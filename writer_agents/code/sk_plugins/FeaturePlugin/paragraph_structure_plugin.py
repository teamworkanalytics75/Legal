#!/usr/bin/env python3
"""
Paragraph Structure Plugin - Atomic SK plugin for paragraph-level metrics enforcement.

Enforces:
- paragraph_count
- paragraph_avg_word_count
- paragraph_median_word_count
- paragraph_avg_sentence_count
- paragraph_sentence_std
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import statistics

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult, EditRequest, DocumentLocation

logger = logging.getLogger(__name__)

# Forward reference for DocumentStructure
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class ParagraphStructurePlugin(BaseFeaturePlugin):
    """Atomic plugin for paragraph structure metrics enforcement."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "paragraph_structure", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("ParagraphStructurePlugin initialized")

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

    def _analyze_paragraph_structure(self, text: str) -> Dict[str, Any]:
        """Analyze paragraph structure in draft text."""
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return {
                "paragraph_count": 0,
                "avg_word_count": 0,
                "median_word_count": 0,
                "avg_sentence_count": 0,
                "sentence_std": 0.0
            }

        paragraph_word_counts = []
        paragraph_sentence_counts = []

        for para in paragraphs:
            words = self._tokenize_words(para)
            paragraph_word_counts.append(len(words))

            sentences = self._split_sentences(para)
            paragraph_sentence_counts.append(len(sentences))

        word_counts_sorted = sorted(paragraph_word_counts)
        n = len(paragraph_word_counts)
        median_word = word_counts_sorted[n // 2] if n > 0 else 0

        avg_words = sum(paragraph_word_counts) / n if n > 0 else 0
        avg_sentences = sum(paragraph_sentence_counts) / n if n > 0 else 0
        sentence_std = float(statistics.pstdev(paragraph_sentence_counts)) if len(paragraph_sentence_counts) > 1 else 0.0

        return {
            "paragraph_count": n,
            "avg_word_count": avg_words,
            "median_word_count": median_word,
            "avg_sentence_count": avg_sentences,
            "sentence_std": sentence_std,
            "paragraph_word_counts": paragraph_word_counts
        }

    async def analyze_paragraph_strength(self, draft_text: str) -> FunctionResult:
        """Analyze paragraph structure and identify weak areas."""
        try:
            analysis = self._analyze_paragraph_structure(draft_text)

            # Get target thresholds from rules
            targets = self.rules.get("targets", {})
            count_target = targets.get("paragraph_count", 15.0)
            avg_word_target = targets.get("paragraph_avg_word_count", 120.0)
            median_word_target = targets.get("paragraph_median_word_count", 120.0)
            avg_sentence_target = targets.get("paragraph_avg_sentence_count", 5.0)

            issues = []
            recommendations = []

            # Check paragraph count
            if analysis["paragraph_count"] < count_target * 0.8:
                issues.append({
                    "feature": "paragraph_count",
                    "current": analysis["paragraph_count"],
                    "target": count_target,
                    "gap": count_target - analysis["paragraph_count"]
                })
                recommendations.append(
                    f"Increase paragraph count from {analysis['paragraph_count']:.0f} to around {count_target:.0f}. "
                    f"Break up long paragraphs to improve readability."
                )
            elif analysis["paragraph_count"] > count_target * 1.5:
                issues.append({
                    "feature": "paragraph_count",
                    "current": analysis["paragraph_count"],
                    "target": count_target,
                    "gap": analysis["paragraph_count"] - count_target
                })
                recommendations.append(
                    f"Consider consolidating some paragraphs. Current: {analysis['paragraph_count']:.0f}, "
                    f"target: {count_target:.0f}."
                )

            # Check average paragraph word count
            if analysis["avg_word_count"] > avg_word_target * 1.3:
                issues.append({
                    "feature": "paragraph_avg_word_count",
                    "current": analysis["avg_word_count"],
                    "target": avg_word_target,
                    "gap": analysis["avg_word_count"] - avg_word_target
                })
                recommendations.append(
                    f"Reduce average paragraph length from {analysis['avg_word_count']:.1f} to {avg_word_target:.0f} words. "
                    f"Break up paragraphs exceeding {avg_word_target * 1.2:.0f} words."
                )

            # Check median paragraph word count
            if analysis["median_word_count"] > median_word_target * 1.3:
                issues.append({
                    "feature": "paragraph_median_word_count",
                    "current": analysis["median_word_count"],
                    "target": median_word_target,
                    "gap": analysis["median_word_count"] - median_word_target
                })
                recommendations.append(
                    f"Reduce median paragraph length from {analysis['median_word_count']:.1f} to {median_word_target:.0f} words."
                )

            # Check sentence count per paragraph
            if analysis["avg_sentence_count"] < avg_sentence_target * 0.7:
                recommendations.append(
                    f"Average sentences per paragraph is {analysis['avg_sentence_count']:.1f} (target: {avg_sentence_target:.1f}). "
                    f"Some paragraphs may be too short."
                )

            success = len(issues) == 0

            return FunctionResult(
                success=success,
                value={
                    "analysis": analysis,
                    "issues": issues,
                    "recommendations": recommendations,
                    "targets": targets
                },
                error=None if success else f"Found {len(issues)} paragraph structure issues"
            )

        except Exception as e:
            logger.error(f"Paragraph structure analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    async def optimize_paragraphs(self, draft_text: str, target_metrics: Dict[str, float] = None) -> FunctionResult:
        """Generate optimized paragraph structure based on target metrics."""
        try:
            analysis = self._analyze_paragraph_structure(draft_text)
            targets = target_metrics or self.rules.get("targets", {})

            max_para_words = targets.get("paragraph_max_word_count", 180.0)
            paragraphs = self._split_paragraphs(draft_text)

            # Identify paragraphs to split
            paragraphs_to_split = []
            for i, para in enumerate(paragraphs):
                word_count = len(self._tokenize_words(para))
                if word_count > max_para_words:
                    paragraphs_to_split.append({
                        "index": i,
                        "word_count": word_count,
                        "excess": word_count - max_para_words,
                        "preview": para[:150] + "..." if len(para) > 150 else para
                    })

            optimization_guidance = []
            if paragraphs_to_split:
                optimization_guidance.append(
                    f"Split {len(paragraphs_to_split)} long paragraphs (exceeding {max_para_words:.0f} words). "
                    f"Look for topic shifts, logical breaks, or new argument points where paragraphs can be divided."
                )

            return FunctionResult(
                success=True,
                value={
                    "current_analysis": analysis,
                    "targets": targets,
                    "paragraphs_to_split": paragraphs_to_split,
                    "optimization_guidance": optimization_guidance
                },
                metadata={"method": "paragraph_optimization"}
            )

        except Exception as e:
            logger.error(f"Paragraph optimization failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    async def validate_draft(self, draft_text: str) -> FunctionResult:
        """Validate draft against paragraph structure criteria."""
        return await self.analyze_paragraph_strength(draft_text)

    async def generate_edit_requests(
        self,
        text: str,
        structure: 'DocumentStructure',
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """
        Generate edit requests to split overly long paragraphs.

        Args:
            text: Full document text
            structure: Parsed document structure

        Returns:
            List of EditRequest objects for splitting long paragraphs
        """
        try:
            # Get target thresholds
            targets = self.rules.get("targets", {})
            max_para_words = targets.get("paragraph_max_word_count", 180.0)
            avg_target = targets.get("paragraph_avg_word_count", 120.0)

            requests = []

            # Find long paragraphs that need splitting
            for para_idx, paragraph in enumerate(structure.paragraphs):
                word_count = len(self._tokenize_words(paragraph.text))

                # If paragraph is too long, split it
                if word_count > max_para_words:
                    # Find a good split point (after a sentence near the middle)
                    sentences = paragraph.sentences
                    if len(sentences) >= 2:
                        # Split at sentence near the middle
                        split_sent_idx = len(sentences) // 2

                        # Create two new paragraphs
                        first_part_sentences = sentences[:split_sent_idx]
                        second_part_sentences = sentences[split_sent_idx:]

                        first_part_text = ' '.join(s.text for s in first_part_sentences)
                        second_part_text = ' '.join(s.text for s in second_part_sentences)

                        # Ensure proper formatting
                        if not first_part_text.endswith(('.', '!', '?')):
                            first_part_text = first_part_text.rstrip() + '.'

                        # Replace paragraph with split version
                        replacement = f"{first_part_text}\n\n{second_part_text}"

                        location = DocumentLocation(
                            paragraph_index=para_idx,
                            position_type="replace"
                        )

                        request = EditRequest(
                            plugin_name="paragraph_structure",
                            location=location,
                            edit_type="replace",
                            content=replacement,
                            priority=60,  # Medium priority
                            affected_plugins=["word_count", "sentence_count", "character_count", "paragraph_structure", "paragraph_monitor"],
                            metadata={
                                "original_word_count": word_count,
                                "target_max": max_para_words,
                                "split_sentence_index": split_sent_idx,
                                "original_sentence_count": len(sentences)
                            },
                            reason=f"Split long paragraph ({word_count} words) to meet target max ({max_para_words:.0f} words)"
                        )
                        requests.append(request)

            logger.info(f"Generated {len(requests)} paragraph splitting edit requests")
            return requests

        except Exception as e:
            logger.error(f"Failed to generate paragraph structure edit requests: {e}")
            return []

