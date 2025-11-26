#!/usr/bin/env python3
"""
CatBoost Word Monitor Plugin - Base class for monitoring specific success signal words.

Each word gets its own plugin instance to track frequency and usage patterns.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional

from semantic_kernel import Kernel
from ...base_feature_plugin import BaseFeaturePlugin
from ....base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class CatBoostWordMonitorPlugin(BaseFeaturePlugin):
    """
    Plugin for monitoring a specific CatBoost success signal word.

    Tracks word frequency, context, and usage patterns to help calibrate
    with other plugins (sentences, paragraphs, arguments, etc.).
    """

    def __init__(
        self,
        kernel: Kernel,
        word: str,
        word_category: str,
        success_rate: str,
        chroma_store=None,
        rules_dir: Optional[Path] = None,
        memory_store=None,
        **kwargs
    ):
        """
        Initialize word monitor plugin.

        Args:
            kernel: Semantic Kernel instance
            word: The word to monitor (e.g., "safety", "harm", "citizen")
            word_category: Category of success signal (e.g., "Endangerment Implicit", "Motion Language")
            success_rate: Success rate description (e.g., "83.3% success", "50% success")
            chroma_store: Optional ChromaDB store
            rules_dir: Optional rules directory
            memory_store: Optional memory store
        """
        self.monitored_word = word.lower()
        self.word_category = word_category
        self.success_rate = success_rate

        # Set feature_name for base class
        feature_name = f"catboost_word_{word.lower().replace(' ', '_')}"

        super().__init__(
            kernel=kernel,
            feature_name=feature_name,
            chroma_store=chroma_store,
            rules_dir=rules_dir or Path(__file__).parent.parent / "rules",
            memory_store=memory_store,
            **kwargs
        )

        logger.info(f"CatBoostWordMonitorPlugin initialized for word: '{word}' ({word_category})")

    def _count_word_occurrences(self, text: str) -> Dict[str, Any]:
        """
        Count occurrences of the monitored word in text.

        Returns:
            Dictionary with count, positions, context, and statistics
        """
        text_lower = text.lower()
        word_lower = self.monitored_word.lower()

        # Find all occurrences (whole word matches only)
        pattern = r'\b' + re.escape(word_lower) + r'\b'
        matches = list(re.finditer(pattern, text_lower))

        # Extract context around each occurrence
        contexts = []
        for match in matches:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()
            contexts.append({
                "position": match.start(),
                "context": context,
                "sentence_context": self._extract_sentence_context(text, match.start())
            })

        # Calculate statistics
        total_words = len(re.findall(r'\b\w+\b', text_lower))
        word_frequency = len(matches) / total_words if total_words > 0 else 0.0

        return {
            "word": self.monitored_word,
            "count": len(matches),
            "total_words": total_words,
            "frequency": word_frequency,
            "frequency_percentage": word_frequency * 100,
            "contexts": contexts[:10],  # Limit to first 10 for performance
            "category": self.word_category,
            "success_rate": self.success_rate
        }

    def _extract_sentence_context(self, text: str, position: int) -> str:
        """Extract the sentence containing the word at the given position."""
        # Find sentence boundaries
        sentence_start = max(0, text.rfind('.', 0, position))
        sentence_start = max(0, text.rfind('!', 0, position), sentence_start)
        sentence_start = max(0, text.rfind('?', 0, position), sentence_start)
        if sentence_start > 0:
            sentence_start += 1

        sentence_end = text.find('.', position)
        sentence_end = min(sentence_end, text.find('!', position)) if sentence_end > 0 else text.find('!', position)
        sentence_end = min(sentence_end, text.find('?', position)) if sentence_end > 0 else text.find('?', position)
        if sentence_end < 0:
            sentence_end = len(text)
        else:
            sentence_end += 1

        return text[sentence_start:sentence_end].strip()

    async def validate(self, text: str) -> FunctionResult:
        """
        Validate word usage in text.

        Returns:
            FunctionResult with word count, frequency, and validation status
        """
        word_stats = self._count_word_occurrences(text)

        # Get thresholds from rules
        min_mentions = self.rules.get("validation_criteria", {}).get("min_mentions", 1)
        max_mentions = self.rules.get("validation_criteria", {}).get("max_mentions", 100)

        count = word_stats["count"]

        # Calculate score based on whether word appears
        if count < min_mentions:
            score = count / min_mentions if min_mentions > 0 else 0.0
            success = False
            message = f"Word '{self.monitored_word}' appears {count} times (minimum: {min_mentions})"
        elif count > max_mentions:
            score = max_mentions / count if count > 0 else 0.0
            success = False
            message = f"Word '{self.monitored_word}' appears {count} times (maximum: {max_mentions})"
        else:
            score = 1.0
            success = True
            message = f"Word '{self.monitored_word}' appears {count} times (acceptable range: {min_mentions}-{max_mentions})"

        return FunctionResult(
            success=success,
            score=score,
            value=word_stats,
            message=message,
            metadata={
                "word": self.monitored_word,
                "category": self.word_category,
                "success_rate": self.success_rate,
                "word_stats": word_stats,
                "threshold": self.rules.get("threshold", 0.7)
            }
        )

    async def _execute_native(self, text: str, **kwargs) -> FunctionResult:
        """Execute native word monitoring."""
        return await self.validate(text)

    async def _execute_semantic(self, text: str, **kwargs) -> FunctionResult:
        """Execute semantic word monitoring with context analysis."""
        native_result = await self.validate(text)

        # Add semantic context if ChromaDB is available
        if self.chroma_store and native_result.value:
            try:
                # Query for similar usage patterns
                query_text = f"{self.monitored_word} {self.word_category} motion to seal"
                similar_cases = await self.query_chroma(query_text, n_results=3)
                native_result.metadata["similar_usage_patterns"] = similar_cases[:3]
            except Exception as e:
                logger.debug(f"Chroma query failed for {self.monitored_word}: {e}")

        return native_result

    def _get_metadata(self):
        """Get plugin metadata."""
        from ..base_plugin import PluginMetadata
        return PluginMetadata(
            name=f"CatBoostWordMonitor_{self.monitored_word.title().replace('_', '')}Plugin",
            description=f"Monitor usage of CatBoost success signal word: '{self.monitored_word}' ({self.word_category}, {self.success_rate})",
            version="1.0.0",
            functions=[
                f"monitor_{self.monitored_word}",
                f"validate_{self.monitored_word}",
                f"count_{self.monitored_word}"
            ]
        )

    async def _register_functions(self) -> None:
        """Register plugin functions with the kernel."""
        # Functions are registered via the base class
        # Additional registration can be done here if needed
        pass
