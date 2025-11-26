#!/usr/bin/env python3
"""
Rule 45 Mentions Plugin - CatBoost Feature Enforcement

Enforces Rule 45 mentions (positive signal from CatBoost analysis)
SHAP Value: 0.144 (positive)
Target: 10-20 mentions (not excessive)
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


class Rule45MentionsPlugin(BaseFeaturePlugin):
    """Plugin for enforcing Rule 45 mentions (CatBoost positive signal)."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "rule_45_mentions", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("Rule45MentionsPlugin initialized")

    def _count_rule_45_mentions(self, text: str) -> Dict[str, Any]:
        """Count Rule 45 mentions and related keywords."""
        patterns = [
            r'Rule\s+45',
            r'Rule\s+45\(d\)',
            r'Rule\s+45\(g\)',
            r'undue\s+burden',
            r'burden\s+or\s+expense',
            r'quash',
            r'quashing',
            r'motion\s+to\s+quash',
            r'subpoena'
        ]

        all_matches = []
        pattern_counts = {}

        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            pattern_counts[pattern] = len(matches)
            all_matches.extend(matches)

        # Deduplicate by position (same match might match multiple patterns)
        unique_positions = set()
        unique_matches = []
        for match in all_matches:
            pos_key = (match.start(), match.end())
            if pos_key not in unique_positions:
                unique_positions.add(pos_key)
                unique_matches.append(match)

        total_count = len(unique_matches)

        # Get target range from rules
        target_min = self.rules.get("target_mentions", {}).get("min", 10)
        target_max = self.rules.get("target_mentions", {}).get("max", 20)
        target_optimal = self.rules.get("target_mentions", {}).get("optimal", 15)

        within_range = target_min <= total_count <= target_max

        return {
            "total_mentions": total_count,
            "within_target_range": within_range,
            "below_min": total_count < target_min,
            "above_max": total_count > target_max,
            "pattern_counts": pattern_counts,
            "target_min": target_min,
            "target_max": target_max,
            "target_optimal": target_optimal,
            "matches": [m.group() for m in unique_matches[:10]]
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate Rule 45 mentions are within target range."""
        mention_check = self._count_rule_45_mentions(text)

        within_range = mention_check["within_target_range"]

        score = 0.0
        if within_range:
            score = 1.0
        elif mention_check["below_min"]:
            # Linear score from 0 to 1 as we approach min
            score = min(1.0, mention_check["total_mentions"] / mention_check["target_min"])
        elif mention_check["above_max"]:
            # Penalty for excessive mentions
            excess = mention_check["total_mentions"] - mention_check["target_max"]
            score = max(0.5, 1.0 - (excess / mention_check["target_max"]))

        issues = []
        if mention_check["below_min"]:
            issues.append(f"Rule 45 mentions ({mention_check['total_mentions']}) below target minimum ({mention_check['target_min']})")
        elif mention_check["above_max"]:
            issues.append(f"Rule 45 mentions ({mention_check['total_mentions']}) above target maximum ({mention_check['target_max']})")

        return FunctionResult(
            success=within_range,
            score=score,
            message=f"Rule 45 Mentions: {mention_check['total_mentions']} mentions (target: {mention_check['target_min']}-{mention_check['target_max']}) - {'PASS' if within_range else 'WARN'}",
            metadata={
                "mention_count": mention_check,
                "shap_value": self.rules.get("shap_value", 0.144),
                "issues": issues,
                "threshold": self.rules.get("threshold", 0.85)
            }
        )

    async def _execute_native(self, text: str, **kwargs) -> FunctionResult:
        """Execute native validation."""
        return await self.validate(text)

    async def _execute_semantic(self, text: str, **kwargs) -> FunctionResult:
        """Execute semantic validation."""
        case_context = text[:500]
        similar_cases = await self.query_chroma(case_context, n_results=5)

        native_result = await self.validate(text)
        native_result.metadata["similar_cases"] = similar_cases[:3]

        return native_result

    async def generate_edit_requests(
        self,
        text: str,
        structure: 'DocumentStructure',
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """
        Generate edit requests to insert Rule 45 mentions where needed.

        Args:
            text: Full document text
            structure: Parsed document structure

        Returns:
            List of EditRequest objects for inserting Rule 45 mentions
        """
        # Check current mention count
        mention_check = self._count_rule_45_mentions(text)
        current_count = mention_check["total_mentions"]
        target_min = mention_check["target_min"]
        target_max = mention_check["target_max"]

        # If we're within range, no edits needed
        if mention_check["within_target_range"]:
            return []

        requests = []

        # If below minimum, need to add mentions
        if mention_check["below_min"]:
            needed = target_min - current_count
            logger.info(f"Rule 45 mentions below target: {current_count} < {target_min}, need to add {needed}")

            # Rule 45 mention templates (varied to avoid repetition)
            mention_templates = [
                " This request complies with Rule 45 and minimizes undue burden.",
                " Rule 45(d) requires that the court quash or modify a subpoena that subjects a person to undue burden.",
                " The discovery request is narrowly tailored to avoid undue burden under Rule 45.",
                " Rule 45(g) provides for sanctions for failure to comply with a subpoena.",
                " This subpoena is properly scoped under Rule 45 to avoid undue burden or expense.",
                " The requested discovery is proportional and does not impose an undue burden under Rule 45.",
            ]

            # Find paragraphs that discuss discovery/subpoenas/requests
            discovery_keywords = ["discovery", "subpoena", "document", "request", "production", "deposition", "testimony"]

            for para_idx, paragraph in enumerate(structure.paragraphs):
                para_text_lower = paragraph.text.lower()

                # Check if paragraph is relevant to discovery/subpoena
                if any(keyword in para_text_lower for keyword in discovery_keywords):
                    # Find a good insertion point (after first sentence, before last)
                    if paragraph.sentences:
                        # Insert after first sentence if available
                        sent_idx = min(1, len(paragraph.sentences) - 1)

                        # Use varied templates
                        template_idx = len(requests) % len(mention_templates)
                        content = mention_templates[template_idx]

                        location = DocumentLocation(
                            paragraph_index=para_idx,
                            sentence_index=sent_idx,
                            position_type="after"
                        )

                        request = EditRequest(
                            plugin_name="rule_45_mentions",
                            location=location,
                            edit_type="insert",
                            content=content,
                            priority=80,  # High priority (positive CatBoost signal)
                            affected_plugins=["word_count", "sentence_count", "character_count", "paragraph_structure"],
                            metadata={
                                "mention_type": "rule_45",
                                "template_idx": template_idx,
                                "target_increase": 1
                            },
                            reason=f"Increase Rule 45 mentions to meet target threshold (current: {current_count}, target: {target_min})"
                        )

                        requests.append(request)

                        # Stop when we have enough
                        if len(requests) >= needed:
                            break

        # If above maximum, we could generate delete requests, but that's more complex
        # For now, just log a warning
        elif mention_check["above_max"]:
            logger.warning(f"Rule 45 mentions above maximum: {current_count} > {target_max}. "
                          f"Consider reducing mentions, but not implementing delete requests yet.")

        return requests

