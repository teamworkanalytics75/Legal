#!/usr/bin/env python3
"""
Sentence Structure Plugin - Atomic SK plugin for sentence-level metrics enforcement.

Enforces:
- max_sentence_word_count
- median_sentence_word_count
- avg_sentence_word_count
- sentence_complex_ratio
- sentence_long_ratio
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult, EditRequest, DocumentLocation

logger = logging.getLogger(__name__)

# Forward reference for DocumentStructure
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class SentenceStructurePlugin(BaseFeaturePlugin):
    """Atomic plugin for sentence structure metrics enforcement."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "sentence_structure", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("SentenceStructurePlugin initialized")

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

    def _analyze_sentence_structure(self, text: str) -> Dict[str, Any]:
        """Analyze sentence structure in draft text."""
        sentences = self._split_sentences(text)
        if not sentences:
            return {
                "sentence_count": 0,
                "max_word_count": 0,
                "median_word_count": 0,
                "avg_word_count": 0,
                "complex_ratio": 0.0,
                "long_ratio": 0.0
            }

        word_counts = [len(self._tokenize_words(s)) for s in sentences]
        word_counts_sorted = sorted(word_counts)
        n = len(word_counts)

        complex_count = sum(1 for wc in word_counts if wc >= 30)
        long_count = sum(1 for wc in word_counts if wc >= 45)

        median = word_counts_sorted[n // 2] if n > 0 else 0

        return {
            "sentence_count": n,
            "max_word_count": max(word_counts) if word_counts else 0,
            "median_word_count": median,
            "avg_word_count": sum(word_counts) / n if n > 0 else 0,
            "complex_ratio": complex_count / n if n > 0 else 0.0,
            "long_ratio": long_count / n if n > 0 else 0.0
        }

    async def analyze_sentence_strength(self, draft_text: str) -> FunctionResult:
        """Analyze sentence structure and identify weak areas."""
        try:
            analysis = self._analyze_sentence_structure(draft_text)

            # Get target thresholds from rules
            targets = self.rules.get("targets", {})
            max_target = targets.get("max_sentence_word_count", 45.0)
            median_target = targets.get("median_sentence_word_count", 22.0)
            avg_target = targets.get("avg_sentence_word_count", 20.0)

            issues = []
            recommendations = []

            # Check max sentence length
            if analysis["max_word_count"] > max_target:
                issues.append({
                    "feature": "max_sentence_word_count",
                    "current": analysis["max_word_count"],
                    "target": max_target,
                    "gap": analysis["max_word_count"] - max_target
                })
                recommendations.append(
                    f"Reduce maximum sentence length from {analysis['max_word_count']:.1f} to {max_target:.1f} words. "
                    f"Split sentences exceeding {max_target:.0f} words."
                )

            # Check median sentence length
            if analysis["median_word_count"] > median_target * 1.2:
                issues.append({
                    "feature": "median_sentence_word_count",
                    "current": analysis["median_word_count"],
                    "target": median_target,
                    "gap": analysis["median_word_count"] - median_target
                })
                recommendations.append(
                    f"Reduce median sentence length from {analysis['median_word_count']:.1f} to {median_target:.1f} words. "
                    f"Keep sentences concise and clear."
                )

            # Check average sentence length
            if analysis["avg_word_count"] > avg_target * 1.2:
                issues.append({
                    "feature": "avg_sentence_word_count",
                    "current": analysis["avg_word_count"],
                    "target": avg_target,
                    "gap": analysis["avg_word_count"] - avg_target
                })
                recommendations.append(
                    f"Reduce average sentence length from {analysis['avg_word_count']:.1f} to {avg_target:.1f} words."
                )

            # Check complex/long sentence ratios
            complex_target = targets.get("sentence_complex_ratio", 0.15)
            if analysis["complex_ratio"] > complex_target * 1.5:
                recommendations.append(
                    f"Reduce complex sentences (≥30 words): current {analysis['complex_ratio']:.1%}, "
                    f"target {complex_target:.1%}."
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
                error=None if success else f"Found {len(issues)} sentence structure issues"
            )

        except Exception as e:
            logger.error(f"Sentence structure analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    async def optimize_sentences(self, draft_text: str, target_metrics: Dict[str, float] = None) -> FunctionResult:
        """Generate optimized sentences based on target metrics."""
        try:
            analysis = self._analyze_sentence_structure(draft_text)
            targets = target_metrics or self.rules.get("targets", {})

            max_target = targets.get("max_sentence_word_count", 45.0)
            sentences = self._split_sentences(draft_text)

            # Identify sentences to split
            sentences_to_split = []
            for i, sentence in enumerate(sentences):
                word_count = len(self._tokenize_words(sentence))
                if word_count > max_target:
                    sentences_to_split.append({
                        "index": i,
                        "sentence": sentence,
                        "word_count": word_count,
                        "excess": word_count - max_target
                    })

            optimization_guidance = []
            if sentences_to_split:
                optimization_guidance.append(
                    f"Split {len(sentences_to_split)} long sentences (exceeding {max_target:.0f} words). "
                    f"Look for conjunctions, clauses, or logical breaks where sentences can be divided."
                )

            # Generate specific recommendations
            specific_recommendations = []
            for item in sentences_to_split[:5]:  # Top 5 longest sentences
                specific_recommendations.append(
                    f"Sentence {item['index'] + 1}: {item['word_count']} words "
                    f"({item['excess']:.0f} over target). Consider splitting at: "
                    f"'{item['sentence'][:100]}...'"
                )

            return FunctionResult(
                success=True,
                value={
                    "current_analysis": analysis,
                    "targets": targets,
                    "sentences_to_split": sentences_to_split,
                    "optimization_guidance": optimization_guidance,
                    "specific_recommendations": specific_recommendations
                },
                metadata={"method": "sentence_optimization"}
            )

        except Exception as e:
            logger.error(f"Sentence optimization failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    async def validate_draft(self, draft_text: str) -> FunctionResult:
        """Validate draft against sentence structure criteria."""
        return await self.analyze_sentence_strength(draft_text)

    async def generate_edit_requests(
        self,
        text: str,
        structure: 'DocumentStructure',
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """
        Generate edit requests to split overly long sentences.

        Args:
            text: Full document text
            structure: Parsed document structure

        Returns:
            List of EditRequest objects for splitting long sentences
        """
        try:
            # Get target thresholds
            targets = self.rules.get("targets", {})
            max_target = targets.get("max_sentence_word_count", 45.0)

            requests = []

            # Find long sentences that need splitting
            for para_idx, paragraph in enumerate(structure.paragraphs):
                for sent_idx, sentence in enumerate(paragraph.sentences):
                    word_count = len(self._tokenize_words(sentence.text))

                    # If sentence is too long, mark for splitting
                    if word_count > max_target:
                        # Try to find a good split point (conjunction, clause marker)
                        # For now, we'll use a simple heuristic: split at first comma or conjunction
                        # after the midpoint
                        midpoint = len(sentence.text) // 2
                        split_markers = [', and ', ', but ', ', or ', ', which ', ', that ', '; ', ' — ', ' -- ']

                        split_pos = None
                        for marker in split_markers:
                            # Find marker after midpoint
                            pos = sentence.text.find(marker, midpoint)
                            if pos > 0 and pos < len(sentence.text) - 20:  # Ensure reasonable split
                                split_pos = pos + len(marker)
                                break

                        if split_pos:
                            # Split sentence
                            first_part = sentence.text[:split_pos].strip()
                            second_part = sentence.text[split_pos:].strip()

                            # Capitalize second part if needed
                            if second_part and second_part[0].islower():
                                second_part = second_part[0].upper() + second_part[1:]

                            # Ensure first part ends with period
                            if not first_part.endswith(('.', '!', '?')):
                                first_part = first_part.rstrip(',;') + '.'

                            # Create replacement content
                            replacement = f"{first_part} {second_part}"

                            location = DocumentLocation(
                                paragraph_index=para_idx,
                                sentence_index=sent_idx,
                                position_type="replace"
                            )

                            request = EditRequest(
                                plugin_name="sentence_structure",
                                location=location,
                                edit_type="replace",
                                content=replacement,
                                priority=65,  # Medium priority
                                affected_plugins=["word_count", "sentence_count", "character_count", "sentence_length", "paragraph_structure"],
                                metadata={
                                    "original_word_count": word_count,
                                    "target_max": max_target,
                                    "split_at": split_pos,
                                    "split_method": "conjunction"
                                },
                                reason=f"Split long sentence ({word_count} words) to meet target max ({max_target:.0f} words)"
                            )
                            requests.append(request)

            logger.info(f"Generated {len(requests)} sentence splitting edit requests")
            return requests

        except Exception as e:
            logger.error(f"Failed to generate sentence structure edit requests: {e}")
            return []

