"""Text preprocessing pipeline for ML system.

This module provides text preprocessing utilities for:
1. Legal document cleaning and normalization
2. Text tokenization and filtering
3. Preprocessing pipeline configuration
"""

import re
import string
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing pipeline for legal documents."""

    def __init__(self,
                 remove_citations: bool = True,
                 remove_case_names: bool = False,
                 normalize_whitespace: bool = True,
                 remove_special_chars: bool = False,
                 min_length: int = 10):
        """Initialize preprocessor with configuration.

        Args:
            remove_citations: Whether to remove legal citations
            remove_case_names: Whether to remove case names
            normalize_whitespace: Whether to normalize whitespace
            remove_special_chars: Whether to remove special characters
            min_length: Minimum text length to keep
        """
        self.remove_citations = remove_citations
        self.remove_case_names = remove_case_names
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars
        self.min_length = min_length

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for text processing."""
        # Legal citation patterns
        self.citation_patterns = [
            r'\b[A-Z][a-z]+ v\. [A-Z][a-z]+\b',  # Case names
            r'\b\d+ [A-Z]\.?\d+[a-z]? \d+\b',    # Volume citations
            r'\b\d+ U\.S\.C\. \d+\b',            # USC citations
            r'\b\d+ F\.\d+[a-z]? \d+\b',         # Federal citations
            r'\b\d+ S\.Ct\. \d+\b',              # Supreme Court citations
        ]

        # Case name patterns
        self.case_name_patterns = [
            r'\b[A-Z][a-z]+ v\. [A-Z][a-z]+\b',
            r'\b[A-Z][a-z]+ v\. [A-Z][a-z]+ [A-Z][a-z]+\b',
        ]

        # Special character patterns
        self.special_char_pattern = r'[^\w\s\.\,\;\:\!\?\-\(\)]'

    def preprocess(self, text: str) -> str:
        """Preprocess a single text document.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        # Apply preprocessing steps
        processed_text = text

        if self.remove_citations:
            processed_text = self._remove_citations(processed_text)

        if self.remove_case_names:
            processed_text = self._remove_case_names(processed_text)

        if self.normalize_whitespace:
            processed_text = self._normalize_whitespace(processed_text)

        if self.remove_special_chars:
            processed_text = self._remove_special_chars(processed_text)

        # Filter by minimum length
        if len(processed_text.strip()) < self.min_length:
            return ""

        return processed_text.strip()

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts.

        Args:
            texts: List of texts to preprocess

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]

    def _remove_citations(self, text: str) -> str:
        """Remove legal citations from text."""
        for pattern in self.citation_patterns:
            text = re.sub(pattern, '', text)
        return text

    def _remove_case_names(self, text: str) -> str:
        """Remove case names from text."""
        for pattern in self.case_name_patterns:
            text = re.sub(pattern, '', text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters from text."""
        return re.sub(self.special_char_pattern, '', text)


class LegalTextPreprocessor(TextPreprocessor):
    """Specialized preprocessor for legal documents."""

    def __init__(self, **kwargs):
        """Initialize with legal-specific defaults."""
        super().__init__(
            remove_citations=True,
            remove_case_names=False,  # Keep case names for legal analysis
            normalize_whitespace=True,
            remove_special_chars=False,  # Keep legal punctuation
            min_length=50,  # Longer minimum for legal documents
            **kwargs
        )

    def preprocess(self, text: str) -> str:
        """Preprocess legal text with additional legal-specific steps."""
        # Start with base preprocessing
        processed_text = super().preprocess(text)

        if not processed_text:
            return ""

        # Legal-specific preprocessing
        processed_text = self._clean_legal_formatting(processed_text)
        processed_text = self._normalize_legal_terms(processed_text)

        return processed_text

    def _clean_legal_formatting(self, text: str) -> str:
        """Clean legal document formatting."""
        # Remove page numbers and headers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

        return text

    def _normalize_legal_terms(self, text: str) -> str:
        """Normalize common legal terms."""
        # Common legal term normalizations
        normalizations = {
            r'\bplaintiffs?\b': 'plaintiff',
            r'\bdefendants?\b': 'defendant',
            r'\bappellants?\b': 'appellant',
            r'\bappellees?\b': 'appellee',
            r'\bpetitioners?\b': 'petitioner',
            r'\brespondents?\b': 'respondent',
        }

        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text


class AgentTextPreprocessor(TextPreprocessor):
    """Specialized preprocessor for agent-generated text."""

    def __init__(self, **kwargs):
        """Initialize with agent-specific defaults."""
        super().__init__(
            remove_citations=False,  # Keep citations in agent output
            remove_case_names=False,
            normalize_whitespace=True,
            remove_special_chars=True,  # Clean agent output
            min_length=5,  # Shorter minimum for agent text
            **kwargs
        )

    def preprocess(self, text: str) -> str:
        """Preprocess agent text with additional agent-specific steps."""
        # Start with base preprocessing
        processed_text = super().preprocess(text)

        if not processed_text:
            return ""

        # Agent-specific preprocessing
        processed_text = self._clean_agent_formatting(processed_text)
        processed_text = self._normalize_agent_terms(processed_text)

        return processed_text

    def _clean_agent_formatting(self, text: str) -> str:
        """Clean agent-generated text formatting."""
        # Remove agent-specific markers
        text = re.sub(r'\[Agent:.*?\]', '', text)
        text = re.sub(r'\[System:.*?\]', '', text)

        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)

        return text

    def _normalize_agent_terms(self, text: str) -> str:
        """Normalize agent-specific terms."""
        # Common agent term normalizations
        normalizations = {
            r'\bI\b': 'the agent',
            r'\bwe\b': 'the system',
            r'\bmy\b': 'the agent\'s',
        }

        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text)

        return text


def preprocess_legal_text(text: str, **kwargs) -> str:
    """Convenience function to preprocess legal text."""
    preprocessor = LegalTextPreprocessor(**kwargs)
    return preprocessor.preprocess(text)

def preprocess_agent_text(text: str, **kwargs) -> str:
    """Convenience function to preprocess agent text."""
    preprocessor = AgentTextPreprocessor(**kwargs)
    return preprocessor.preprocess(text)

def preprocess_text_batch(texts: List[str], text_type: str = 'legal', **kwargs) -> List[str]:
    """Convenience function to preprocess a batch of texts.

    Args:
        texts: List of texts to preprocess
        text_type: Type of text ('legal' or 'agent')
        **kwargs: Additional preprocessing options

    Returns:
        List of preprocessed texts
    """
    if text_type == 'legal':
        preprocessor = LegalTextPreprocessor(**kwargs)
    else:
        preprocessor = AgentTextPreprocessor(**kwargs)

    return preprocessor.preprocess_batch(texts)
