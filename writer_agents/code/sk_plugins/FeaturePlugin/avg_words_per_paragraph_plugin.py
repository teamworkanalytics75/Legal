#!/usr/bin/env python3
"""
Average Words Per Paragraph Plugin - Enforces avg_words_per_paragraph feature.

For pseudonym motions:
- avg_words_per_paragraph is the #2 most important feature (8.13 importance)

This plugin validates average words per paragraph independently.
"""

import logging
import re
import statistics
from pathlib import Path
from typing import Dict, Any, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class AvgWordsPerParagraphPlugin(BaseFeaturePlugin):
    """Plugin to enforce average words per paragraph (critical for pseudonym motions)."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "avg_words_per_paragraph", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("AvgWordsPerParagraphPlugin initialized")

        # Target from analysis (pseudonym motions)
        # avg_words_per_paragraph is #2 feature (8.13 importance)
        self.target_avg_words = self.rules.get("targets", {}).get("avg_words_per_paragraph", 120.0)
        self.optimal_range = (100, 150)  # Optimal words per paragraph range

    def _analyze_avg_words(self, text: str) -> Dict[str, Any]:
        """Analyze average words per paragraph in text."""
        # Split into paragraphs
        paragraphs = re.split(r'(?:\r?\n\s*){2,}', text.strip())
        if len(paragraphs) == 1:
            paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        else:
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return {
                "avg_words_per_paragraph": 0,
                "median_words_per_paragraph": 0,
                "word_counts": [],
                "meets_target": False,
                "in_optimal_range": False
            }

        # Calculate word counts per paragraph
        word_counts = []
        for para in paragraphs:
            words = re.findall(r"[A-Za-z0-9']+", para)
            word_counts.append(len(words))

        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        median_words = statistics.median(word_counts) if word_counts else 0

        # Check if in optimal range
        in_optimal_range = self.optimal_range[0] <= avg_words <= self.optimal_range[1]
        paragraphs_in_range = sum(1 for wc in word_counts if self.optimal_range[0] <= wc <= self.optimal_range[1])
        pct_in_range = (paragraphs_in_range / len(word_counts) * 100) if word_counts else 0

        return {
            "avg_words_per_paragraph": avg_words,
            "median_words_per_paragraph": median_words,
            "word_counts": word_counts,
            "paragraphs_in_optimal_range": paragraphs_in_range,
            "pct_in_optimal_range": pct_in_range,
            "meets_target": in_optimal_range,
            "in_optimal_range": in_optimal_range,
            "target_avg_words": self.target_avg_words,
            "optimal_range": self.optimal_range
        }

    async def validate_avg_words_per_paragraph(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate average words per paragraph.

        This is CRITICAL: avg_words_per_paragraph is the #2 most important feature (8.13 importance) for pseudonym motions.
        Target: 100-150 words per paragraph (optimal range)
        """
        text = document.get_full_text()
        analysis = self._analyze_avg_words(text)

        issues = []
        recommendations = []

        if not analysis["meets_target"]:
            if analysis["avg_words_per_paragraph"] < self.optimal_range[0]:
                issues.append({
                    "type": "paragraphs_too_short",
                    "severity": "medium",
                    "message": f"Average words per paragraph ({analysis['avg_words_per_paragraph']:.1f}) is below optimal range ({self.optimal_range[0]}-{self.optimal_range[1]}). This is the #2 most important feature (8.13 importance).",
                    "suggestion": f"Combine short paragraphs or add more content to reach {self.optimal_range[0]}-{self.optimal_range[1]} words per paragraph."
                })
                recommendations.append({
                    "type": "increase_paragraph_length",
                    "priority": "medium",
                    "action": f"Increase average words per paragraph from {analysis['avg_words_per_paragraph']:.1f} to {self.optimal_range[0]}-{self.optimal_range[1]} words",
                    "rationale": "Average words per paragraph is the #2 predictor for pseudonym motion success (8.13 importance)"
                })
            elif analysis["avg_words_per_paragraph"] > self.optimal_range[1]:
                issues.append({
                    "type": "paragraphs_too_long",
                    "severity": "high",
                    "message": f"Average words per paragraph ({analysis['avg_words_per_paragraph']:.1f}) exceeds optimal range ({self.optimal_range[0]}-{self.optimal_range[1]}). This is the #2 most important feature (8.13 importance).",
                    "suggestion": f"Break up long paragraphs to reach {self.optimal_range[0]}-{self.optimal_range[1]} words per paragraph."
                })
                recommendations.append({
                    "type": "decrease_paragraph_length",
                    "priority": "high",
                    "action": f"Break up long paragraphs to reduce average from {analysis['avg_words_per_paragraph']:.1f} to {self.optimal_range[0]}-{self.optimal_range[1]} words",
                    "rationale": "Average words per paragraph is the #2 predictor for pseudonym motion success (8.13 importance)"
                })

        # Check percentage in optimal range
        if analysis["pct_in_optimal_range"] < 60:
            recommendations.append({
                "type": "optimize_paragraph_lengths",
                "priority": "high",
                "action": f"Only {analysis['pct_in_optimal_range']:.1f}% of paragraphs are in optimal range ({self.optimal_range[0]}-{self.optimal_range[1]} words). Adjust paragraph lengths.",
                "rationale": "Consistent paragraph length (100-150 words) is critical for pseudonym motions"
            })

        if analysis["meets_target"]:
            recommendations.append({
                "type": "avg_words_optimal",
                "message": f"Average words per paragraph is optimal ({analysis['avg_words_per_paragraph']:.1f} words, range: {self.optimal_range[0]}-{self.optimal_range[1]}).",
                "avg_words": analysis["avg_words_per_paragraph"]
            })

        return FunctionResult(
            success=analysis["meets_target"],
            data={
                "avg_words_per_paragraph": analysis["avg_words_per_paragraph"],
                "median_words_per_paragraph": analysis["median_words_per_paragraph"],
                "pct_in_optimal_range": analysis["pct_in_optimal_range"],
                "meets_target": analysis["meets_target"],
                "in_optimal_range": analysis["in_optimal_range"],
                "optimal_range": analysis["optimal_range"],
                "issues": issues,
                "recommendations": recommendations,
                "feature_importance": 8.13,
                "feature_rank": 2,
                "motion_type": "pseudonym"
            },
            message="Average words per paragraph validation complete"
        )

    async def query_avg_words_per_paragraph(self, case_context: str, **kwargs) -> FunctionResult:
        """Query for examples of documents with optimal paragraph length."""
        results = await self.query_chroma(
            case_context=f"paragraph length word count optimal {case_context}",
            n_results=kwargs.get('n_results', 10)
        )

        return FunctionResult(
            success=True,
            data={"examples": results},
            message=f"Found {len(results)} examples of optimal paragraph length"
        )

    async def generate_avg_words_per_paragraph_argument(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Generate argument about average words per paragraph."""
        analysis = self._analyze_avg_words(document.get_full_text())

        if analysis["meets_target"]:
            argument = f"""The motion uses optimal paragraph length (average: {analysis['avg_words_per_paragraph']:.1f} words per paragraph). Average words per paragraph is the #2 most important feature (8.13 importance) for pseudonym motions because:

1. Consistent paragraph length improves readability
2. Optimal length (100-150 words) balances detail with clarity
3. Shows professional drafting quality
4. Interacts with other features to create strong signals

Optimal range: {self.optimal_range[0]}-{self.optimal_range[1]} words. Current: {analysis['avg_words_per_paragraph']:.1f} words.
{analysis['pct_in_optimal_range']:.1f}% of paragraphs are in optimal range."""
        else:
            if analysis["avg_words_per_paragraph"] < self.optimal_range[0]:
                action = "increase"
                reason = "paragraphs are too short"
            else:
                action = "decrease"
                reason = "paragraphs are too long"

            argument = f"""The motion should {action} average words per paragraph. This is the #2 most important feature (8.13 importance) for pseudonym motion success.

Current: {analysis['avg_words_per_paragraph']:.1f} words per paragraph
Optimal range: {self.optimal_range[0]}-{self.optimal_range[1]} words per paragraph
Issue: {reason}

Recommendations:
1. {'Combine short paragraphs or add more content' if action == 'increase' else 'Break up long paragraphs'}
2. Aim for {self.optimal_range[0]}-{self.optimal_range[1]} words per paragraph
3. Ensure 60%+ of paragraphs are in optimal range (currently {analysis['pct_in_optimal_range']:.1f}%)"""

        return FunctionResult(
            success=True,
            data={"argument": argument, "avg_words_per_paragraph": analysis["avg_words_per_paragraph"]},
            message="Average words per paragraph argument generated"
        )

