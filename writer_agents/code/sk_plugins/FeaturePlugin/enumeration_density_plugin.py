#!/usr/bin/env python3
"""
Enumeration Density Plugin - Atomic SK plugin for structured lists and enumeration enforcement.

Enforces:
- enumeration_density (ratio of enumerated items to total content)
- Structured list usage
- Numbered/bulleted item formatting
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


class EnumerationDensityPlugin(BaseFeaturePlugin):
    """Atomic plugin for enumeration density enforcement."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "enumeration_density", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("EnumerationDensityPlugin initialized")

    def _detect_enumerations(self, text: str) -> Dict[str, Any]:
        """Detect enumerated items in text."""
        # Patterns for various enumeration formats
        numbered_patterns = [
            r'\d+[\.\)]\s+',  # "1. " or "1) "
            r'[\(]\d+[\)]\s+',  # "(1) "
            r'[A-Z]\.\s+',  # "A. "
            r'[a-z]\.\s+',  # "a. "
            r'[ivxlcdm]+\.\s+',  # Roman numerals "i. ", "ii. ", etc.
        ]

        bullet_patterns = [
            r'[•\-\*]\s+',  # Bullet points
            r'─+\s+',  # Em dashes
        ]

        # Find all enumerated items
        enumerated_items = []
        total_matches = 0

        for pattern in numbered_patterns + bullet_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                total_matches += 1
                # Extract the item text (next 100 chars or until next enumeration/newline)
                start_pos = match.end()
                next_enum = None
                for p in numbered_patterns + bullet_patterns:
                    next_match = re.search(p, text[start_pos:start_pos + 200], re.MULTILINE)
                    if next_match:
                        if next_enum is None or next_match.start() < next_enum:
                            next_enum = next_match.start()

                if next_enum:
                    item_text = text[start_pos:start_pos + next_enum].strip()
                else:
                    # Take next line or 200 chars
                    next_newline = text.find('\n', start_pos, start_pos + 200)
                    item_text = text[start_pos:next_newline if next_newline > 0 else start_pos + 200].strip()

                if item_text and len(item_text) > 10:  # Minimum item length
                    enumerated_items.append({
                        "pattern": pattern,
                        "text": item_text[:100],
                        "position": match.start()
                    })

        # Calculate density
        total_chars = len(text)
        enum_chars = sum(len(item["text"]) for item in enumerated_items)
        density = enum_chars / total_chars if total_chars > 0 else 0.0

        # Count structured sections (RFPs, categories, etc.)
        structured_patterns = [
            r'Request\s+for\s+Production\s+\d+',
            r'RFP\s+\d+',
            r'Category\s+\d+',
            r'Document\s+Category\s+\d+',
            r'Item\s+\d+',
        ]
        structured_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in structured_patterns)

        return {
            "enumerated_items_count": len(enumerated_items),
            "total_matches": total_matches,
            "enumeration_density": density,
            "structured_sections": structured_count,
            "enum_chars": enum_chars,
            "total_chars": total_chars
        }

    async def analyze_enumeration_strength(self, draft_text: str) -> FunctionResult:
        """Analyze enumeration density and identify weak areas."""
        try:
            analysis = self._detect_enumerations(draft_text)

            # Get target from rules
            target_density = self.rules.get("targets", {}).get("enumeration_density", 0.20)
            min_items = self.rules.get("targets", {}).get("min_enumerated_items", 5)

            issues = []
            recommendations = []

            # Check enumeration density
            if analysis["enumeration_density"] < target_density * 0.8:
                issues.append({
                    "feature": "enumeration_density",
                    "current": analysis["enumeration_density"],
                    "target": target_density,
                    "gap": target_density - analysis["enumeration_density"]
                })
                recommendations.append(
                    f"Increase enumeration density from {analysis['enumeration_density']:.2%} to {target_density:.2%}. "
                    f"Add structured lists, numbered items, or bullet points to organize content."
                )

            # Check minimum enumerated items
            if analysis["enumerated_items_count"] < min_items:
                issues.append({
                    "feature": "enumerated_items_count",
                    "current": analysis["enumerated_items_count"],
                    "target": min_items,
                    "gap": min_items - analysis["enumerated_items_count"]
                })
                recommendations.append(
                    f"Add more enumerated items (current: {analysis['enumerated_items_count']}, "
                    f"target: {min_items}). Use numbered lists for requests, categories, or arguments."
                )

            # Check structured sections (RFPs, categories)
            min_structured = self.rules.get("targets", {}).get("min_structured_sections", 3)
            if analysis["structured_sections"] < min_structured:
                recommendations.append(
                    f"Consider adding structured sections (e.g., RFP numbers, document categories). "
                    f"Current: {analysis['structured_sections']}, target: {min_structured}."
                )

            success = len(issues) == 0

            return FunctionResult(
                success=success,
                value={
                    "analysis": analysis,
                    "issues": issues,
                    "recommendations": recommendations,
                    "targets": {
                        "enumeration_density": target_density,
                        "min_enumerated_items": min_items
                    }
                },
                error=None if success else f"Found {len(issues)} enumeration issues"
            )

        except Exception as e:
            logger.error(f"Enumeration analysis failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    async def generate_enumeration_suggestions(self, draft_text: str) -> FunctionResult:
        """Generate suggestions for adding enumerated items."""
        try:
            analysis = self._detect_enumerations(draft_text)
            target_density = self.rules.get("targets", {}).get("enumeration_density", 0.20)

            suggestions = []

            # Identify sections that could benefit from enumeration
            paragraphs = re.split(r'\n\s*\n', draft_text)
            sections_needing_enum = []

            for i, para in enumerate(paragraphs):
                # Check if paragraph has multiple sentences but no enumeration
                sentences = re.split(r'[.!?]+', para)
                if len(sentences) >= 3 and not any(re.search(r'^\d+[\.\)]', s) for s in sentences):
                    sections_needing_enum.append({
                        "paragraph_index": i,
                        "preview": para[:200] + "..." if len(para) > 200 else para,
                        "sentence_count": len(sentences)
                    })

            if sections_needing_enum:
                suggestions.append(
                    f"Convert {len(sections_needing_enum)} paragraphs into numbered lists. "
                    f"Each paragraph contains multiple related points that would benefit from enumeration."
                )

            # Specific formatting suggestions
            formatting_suggestions = [
                "Use numbered lists (1., 2., 3.) for sequential requests or arguments",
                "Use lettered lists (A., B., C.) for categories or sub-points",
                "Use bullet points (• or -) for non-sequential items",
                "Consider using RFP numbering format for document requests (RFP 1, RFP 2, etc.)",
                "Add category headers with numbered items for document types"
            ]

            return FunctionResult(
                success=True,
                value={
                    "current_density": analysis["enumeration_density"],
                    "target_density": target_density,
                    "sections_needing_enum": sections_needing_enum[:5],  # Top 5
                    "suggestions": suggestions,
                    "formatting_suggestions": formatting_suggestions
                },
                metadata={"method": "enumeration_suggestions"}
            )

        except Exception as e:
            logger.error(f"Enumeration suggestions failed: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    async def validate_draft(self, draft_text: str) -> FunctionResult:
        """Validate draft against enumeration density criteria."""
        return await self.analyze_enumeration_strength(draft_text)

    async def generate_edit_requests(
        self,
        text: str,
        structure: 'DocumentStructure',
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """
        Generate edit requests to convert paragraphs into enumerated lists.

        Args:
            text: Full document text
            structure: Parsed document structure

        Returns:
            List of EditRequest objects for converting paragraphs to enumerated lists
        """
        try:
            # Check current enumeration density
            analysis = self._detect_enumerations(text)
            target_density = self.rules.get("targets", {}).get("enumeration_density", 0.20)
            min_items = self.rules.get("targets", {}).get("min_enumerated_items", 5)

            # If density is already good, return empty
            if analysis["enumeration_density"] >= target_density * 0.9:
                return []

            requests = []

            # Find paragraphs that could be converted to enumerated lists
            for para_idx, paragraph in enumerate(structure.paragraphs):
                # Skip if paragraph already has enumeration
                if any(re.search(r'^\d+[\.\)]|^[•\-\*]', line.strip())
                       for line in paragraph.text.split('\n')):
                    continue

                # Check if paragraph has multiple sentences (good candidate for enumeration)
                if len(paragraph.sentences) >= 3:
                    # Convert sentences to numbered list
                    enumerated_content = "\n"
                    for i, sentence in enumerate(paragraph.sentences, 1):
                        # Remove leading/trailing whitespace and add enumeration
                        sent_text = sentence.text.strip()
                        if sent_text:
                            enumerated_content += f"{i}. {sent_text}\n"

                    # Create location for replacing the paragraph
                    location = DocumentLocation(
                        paragraph_index=para_idx,
                        position_type="replace"
                    )

                    request = EditRequest(
                        plugin_name="enumeration_density",
                        location=location,
                        edit_type="replace",
                        content=enumerated_content,
                        priority=70,  # Medium-high priority (enumeration is positive signal)
                        affected_plugins=["word_count", "sentence_count", "character_count", "paragraph_structure", "enumeration_density"],
                        metadata={
                            "original_sentence_count": len(paragraph.sentences),
                            "conversion_type": "paragraph_to_numbered_list",
                            "enumeration_style": "numbered"
                        },
                        reason=f"Increase enumeration density by converting paragraph with {len(paragraph.sentences)} sentences to numbered list"
                    )
                    requests.append(request)

                    # Stop when we have enough enumerations
                    if len(requests) >= min_items - analysis["enumerated_items_count"]:
                        break

            logger.info(f"Generated {len(requests)} enumeration edit requests")
            return requests

        except Exception as e:
            logger.error(f"Failed to generate enumeration edit requests: {e}")
            return []

