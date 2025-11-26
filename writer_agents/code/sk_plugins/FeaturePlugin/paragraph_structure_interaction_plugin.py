#!/usr/bin/env python3
"""
Paragraph Structure Interaction Plugin - Enforces paragraph structure interactions.

This plugin handles interaction features related to paragraph structure:
- word_count × paragraph_count
- avg_words_per_paragraph × enumeration_density
- sentence_count × avg_words_per_paragraph
- paragraph_count × enumeration_density

Note: Individual features (paragraph_count, avg_words_per_paragraph) are handled by:
- ParagraphCountPlugin (paragraph_count)
- AvgWordsPerParagraphPlugin (avg_words_per_paragraph)
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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class ParagraphStructureInteractionPlugin(BaseFeaturePlugin):
    """Plugin to enforce paragraph structure interactions (interaction features only)."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "paragraph_structure_interaction", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("ParagraphStructureInteractionPlugin initialized (interactions only)")

    def _get_paragraph_metrics(self, text: str) -> Dict[str, Any]:
        """Get paragraph metrics needed for interaction calculations."""
        # Split into paragraphs
        paragraphs = re.split(r'(?:\r?\n\s*){2,}', text.strip())
        if len(paragraphs) == 1:
            paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        else:
            paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return {
                "paragraph_count": 0,
                "avg_words_per_paragraph": 0
            }

        # Calculate word counts per paragraph
        word_counts = []
        for para in paragraphs:
            words = re.findall(r"[A-Za-z0-9']+", para)
            word_counts.append(len(words))

        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

        return {
            "paragraph_count": len(paragraphs),
            "avg_words_per_paragraph": avg_words
        }

    def _calculate_interactions(self, paragraph_metrics: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Calculate interaction features with paragraph structure."""
        # Basic features for interactions
        word_count = len(re.findall(r"[A-Za-z0-9']+", text))
        sentence_count = text.count('.') + text.count('!') + text.count('?')

        # Check for enumeration density
        bullet_count = text.count('•') + text.count('- ') + text.count('* ')
        numbered_count = len(re.findall(r'\d+\.\s+[A-Z]', text))
        enumeration_count = bullet_count + numbered_count
        enumeration_density = (enumeration_count / word_count * 1000) if word_count > 0 else 0

        # Calculate interaction features (from analysis)
        interactions = {
            # Top interactions for pseudonym motions
            "word_count_x_paragraph_count": word_count * paragraph_metrics["paragraph_count"],
            "avg_words_per_paragraph_x_enumeration_density": paragraph_metrics["avg_words_per_paragraph"] * enumeration_density,
            "sentence_count_x_avg_words_per_paragraph": sentence_count * paragraph_metrics["avg_words_per_paragraph"],
            "paragraph_count_x_enumeration_density": paragraph_metrics["paragraph_count"] * enumeration_density,
        }

        return interactions

    async def validate_paragraph_interactions(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate paragraph structure interactions.

        This plugin handles interaction features only. Individual features are validated by:
        - ParagraphCountPlugin (paragraph_count)
        - AvgWordsPerParagraphPlugin (avg_words_per_paragraph)
        """
        text = document.get_full_text()
        paragraph_metrics = self._get_paragraph_metrics(text)
        interactions = self._calculate_interactions(paragraph_metrics, text)

        issues = []
        recommendations = []

        # Validate interaction values
        # Note: Interaction validation thresholds would come from analysis
        # For now, we report the interaction values

        if interactions:
            recommendations.append({
                "type": "paragraph_interactions_calculated",
                "message": "Paragraph structure interactions calculated successfully.",
                "interactions": interactions
            })

        return FunctionResult(
            success=True,
            data={
                "interactions": interactions,
                "paragraph_metrics": paragraph_metrics,
                "issues": issues,
                "recommendations": recommendations
            },
            message="Paragraph structure interaction validation complete"
        )

    async def query_paragraph_interactions(self, case_context: str, **kwargs) -> FunctionResult:
        """Query for examples of paragraph structure interactions."""
        results = await self.query_chroma(
            case_context=f"paragraph structure interactions word count enumeration {case_context}",
            n_results=kwargs.get('n_results', 10)
        )

        return FunctionResult(
            success=True,
            data={"examples": results},
            message=f"Found {len(results)} examples of paragraph structure interactions"
        )

    async def generate_paragraph_interaction_argument(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Generate argument about paragraph structure interactions."""
        text = document.get_full_text()
        paragraph_metrics = self._get_paragraph_metrics(text)
        interactions = self._calculate_interactions(paragraph_metrics, text)

        argument = f"""Paragraph structure interactions are important for pseudonym motions:

Key Interactions:
- word_count × paragraph_count: {interactions.get('word_count_x_paragraph_count', 0):.1f}
- avg_words_per_paragraph × enumeration_density: {interactions.get('avg_words_per_paragraph_x_enumeration_density', 0):.2f}
- sentence_count × avg_words_per_paragraph: {interactions.get('sentence_count_x_avg_words_per_paragraph', 0):.1f}
- paragraph_count × enumeration_density: {interactions.get('paragraph_count_x_enumeration_density', 0):.2f}

These interactions show how paragraph structure works together with document length, sentence structure, and enumeration to create strong predictive signals for motion success."""

        return FunctionResult(
            success=True,
            data={
                "argument": argument,
                "interactions": interactions,
                "paragraph_metrics": paragraph_metrics
            },
            message="Paragraph structure interaction argument generated"
        )

