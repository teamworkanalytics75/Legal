#!/usr/bin/env python3
"""
Document Structure Parser - Consistent document parsing for edit coordination.

Provides shared parsing logic for paragraphs and sentences, with location tracking
for edit requests.
"""

import re
import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from ..base_plugin import DocumentLocation

logger = logging.getLogger(__name__)

SECTION_HEADER_PATTERNS = [
    # Roman numeral or numbered headings (e.g., "I. INTRODUCTION", "2. LEGAL STANDARD")
    re.compile(r'^(?:[IVXLCDM]+|\d+)(?:\.\d+)?\s*[).:-]?\s+[A-Z][A-Za-z0-9 ,/&\'()-]{3,}$'),
    # SECTION/ARTICLE style headings
    re.compile(r'^(?:SECTION|ARTICLE)\s+[A-Z0-9.\-]+\s*[).:-]?\s+[A-Z][A-Za-z0-9 ,/&\'()-]{3,}$', re.IGNORECASE),
]

COMMON_SECTION_TITLES = {
    "INTRODUCTION",
    "BACKGROUND",
    "FACTUAL BACKGROUND",
    "PROCEDURAL HISTORY",
    "LEGAL STANDARD",
    "ARGUMENT",
    "DISCUSSION",
    "CONCLUSION",
    "PRAYER FOR RELIEF",
    "RELIEF REQUESTED",
    "REQUESTED RELIEF",
    "SUMMARY OF ARGUMENT",
}


@dataclass
class Sentence:
    """Represents a sentence with position information."""
    text: str
    paragraph_index: int
    sentence_index: int  # Within paragraph
    start_char: int  # Character position in full document
    end_char: int


@dataclass
class Paragraph:
    """Represents a paragraph with position information."""
    text: str
    paragraph_index: int
    start_char: int  # Character position in full document
    end_char: int
    sentences: List[Sentence] = field(default_factory=list)


@dataclass
class DocumentStructure:
    """Complete document structure with paragraphs and sentences."""
    original_text: str
    paragraphs: List[Paragraph] = field(default_factory=list)
    _sections_cache: Optional[List[Dict[str, Any]]] = field(default=None, init=False, repr=False)

    def get_paragraph(self, index: int) -> Optional[Paragraph]:
        """Get paragraph by index."""
        if 0 <= index < len(self.paragraphs):
            return self.paragraphs[index]
        return None

    def get_sentence(self, para_index: int, sent_index: int) -> Optional[Sentence]:
        """Get sentence by paragraph and sentence indices."""
        para = self.get_paragraph(para_index)
        if para and 0 <= sent_index < len(para.sentences):
            return para.sentences[sent_index]
        return None

    def find_character_position(self, location: DocumentLocation) -> Optional[int]:
        """Find character position for a DocumentLocation."""
        # If character_offset is provided, use it directly
        if location.character_offset is not None:
            if 0 <= location.character_offset < len(self.original_text):
                return location.character_offset
            logger.warning(f"Character offset {location.character_offset} out of bounds")
            return None

        # Use paragraph and sentence indices
        if location.paragraph_index is not None:
            para = self.get_paragraph(location.paragraph_index)
            if not para:
                logger.warning(f"Paragraph index {location.paragraph_index} out of bounds")
                return None

            # If sentence_index is specified
            if location.sentence_index is not None:
                sent = self.get_sentence(location.paragraph_index, location.sentence_index)
                if not sent:
                    logger.warning(f"Sentence index {location.sentence_index} out of bounds in paragraph {location.paragraph_index}")
                    return None

                # Return start or end of sentence based on position_type
                if location.position_type == "before":
                    return sent.start_char
                elif location.position_type == "after":
                    return sent.end_char
                else:  # replace
                    return sent.start_char
            else:
                # No sentence index - use paragraph boundaries
                if location.position_type == "before":
                    return para.start_char
                elif location.position_type == "after":
                    return para.end_char
                else:  # replace
                    return para.start_char

        logger.warning(f"Could not resolve location: {location}")
        return None

    def _normalize_section_title(self, text: str) -> str:
        """Normalize paragraph text for section comparisons."""
        if not text:
            return ""
        normalized = re.sub(r'\s+', ' ', text.strip())
        return normalized.rstrip(":;").strip()

    def _looks_like_section_header(self, text: str) -> bool:
        """Heuristic test to determine if text is a section header."""
        if not text:
            return False

        words = text.split()
        if len(words) == 0 or len(words) > 20:
            return False

        letters = [c for c in text if c.isalpha()]
        uppercase_ratio = (
            sum(1 for c in letters if c.isupper()) / len(letters)
            if letters else 0.0
        )

        if text.upper() in COMMON_SECTION_TITLES:
            return True

        for pattern in SECTION_HEADER_PATTERNS:
            if pattern.match(text):
                return True

        return uppercase_ratio >= 0.65 and len(words) <= 8

    def get_sections(self) -> List[Dict[str, Any]]:
        """
        Return detected section headers with positional metadata.

        Each entry includes:
            - title
            - start_char / end_char (spanning until next section or document end)
            - paragraph_index
        """
        if self._sections_cache is not None:
            return self._sections_cache

        sections: List[Dict[str, Any]] = []
        doc_end = len(self.original_text) if self.original_text else 0

        for paragraph in self.paragraphs:
            normalized = self._normalize_section_title(paragraph.text)
            if not normalized or not self._looks_like_section_header(normalized):
                continue

            if sections:
                sections[-1]["end_char"] = max(sections[-1]["start_char"], paragraph.start_char)

            sections.append({
                "title": normalized,
                "start_char": paragraph.start_char,
                "end_char": paragraph.end_char,
                "paragraph_index": paragraph.paragraph_index,
            })

        if sections:
            sections[-1]["end_char"] = doc_end

        self._sections_cache = sections
        logger.debug("Detected %d section headers", len(sections))
        return sections


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs consistently."""
    # Try splitting by double newlines first
    paragraphs = re.split(r'(?:\r?\n\s*){2,}', text.strip())

    # If only one paragraph found, try single newlines
    if len(paragraphs) == 1:
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    else:
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def split_sentences(text: str) -> List[str]:
    """Split text into sentences consistently."""
    if not text:
        return []

    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', text.strip())
    if not normalized:
        return []

    # Split on sentence-ending punctuation followed by space and capital letter/number
    # More sophisticated pattern to handle abbreviations, decimals, etc.
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z0-9])'
    sentences = re.split(sentence_pattern, normalized)

    # Filter out empty sentences and very short fragments (likely false positives)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    return sentences


def parse_document_structure(text: str) -> DocumentStructure:
    """
    Parse document into structured paragraphs and sentences.

    Args:
        text: Full document text

    Returns:
        DocumentStructure with parsed paragraphs and sentences
    """
    structure = DocumentStructure(original_text=text)

    if not text:
        return structure

    paragraphs_text = split_paragraphs(text)
    current_char_pos = 0

    for para_idx, para_text in enumerate(paragraphs_text):
        # Find paragraph position in original text
        para_start = text.find(para_text, current_char_pos)
        if para_start == -1:
            # Fallback: estimate position
            para_start = current_char_pos

        para_end = para_start + len(para_text)

        # Split paragraph into sentences
        sentences_text = split_sentences(para_text)
        sentences = []

        for sent_idx, sent_text in enumerate(sentences_text):
            # Find sentence position within paragraph
            sent_start_in_para = para_text.find(sent_text)
            if sent_start_in_para == -1:
                # Fallback: estimate position
                sent_start_in_para = sum(len(s) + 1 for s in sentences_text[:sent_idx])

            sent_start = para_start + sent_start_in_para
            sent_end = sent_start + len(sent_text)

            sentence = Sentence(
                text=sent_text,
                paragraph_index=para_idx,
                sentence_index=sent_idx,
                start_char=sent_start,
                end_char=sent_end
            )
            sentences.append(sentence)

        paragraph = Paragraph(
            text=para_text,
            paragraph_index=para_idx,
            start_char=para_start,
            end_char=para_end,
            sentences=sentences
        )

        structure.paragraphs.append(paragraph)
        current_char_pos = para_end + 1

    logger.debug(f"Parsed document: {len(structure.paragraphs)} paragraphs, "
                 f"{sum(len(p.sentences) for p in structure.paragraphs)} sentences")

    return structure


def apply_edit(text: str, structure: DocumentStructure, edit_request: 'EditRequest') -> Tuple[str, DocumentStructure]:
    """
    Apply a single edit request to the document.

    Args:
        text: Current document text
        structure: Current document structure
        edit_request: Edit request to apply

    Returns:
        Tuple of (modified_text, updated_structure)
    """
    # Import here to avoid circular dependency
    from ..base_plugin import EditRequest

    if not isinstance(edit_request, EditRequest):
        raise TypeError(f"Expected EditRequest, got {type(edit_request)}")

    # Find insertion/edition point
    char_pos = structure.find_character_position(edit_request.location)
    if char_pos is None:
        logger.error(f"Could not find location for edit: {edit_request.location}")
        return text, structure

    # Apply edit based on type
    if edit_request.edit_type == "insert":
        # Insert content at position
        if edit_request.location.position_type == "before":
            modified_text = text[:char_pos] + edit_request.content + text[char_pos:]
        else:  # after or replace (treat as after for insert)
            modified_text = text[:char_pos] + edit_request.content + text[char_pos:]

    elif edit_request.edit_type == "replace":
        # Replace content at position
        # For replace, we need to determine what to replace
        if edit_request.location.sentence_index is not None and edit_request.location.paragraph_index is not None:
            # Replace sentence
            sent = structure.get_sentence(
                edit_request.location.paragraph_index,
                edit_request.location.sentence_index
            )
            if sent:
                modified_text = text[:sent.start_char] + edit_request.content + text[sent.end_char:]
            else:
                logger.error(f"Could not find sentence to replace at para {edit_request.location.paragraph_index}, sent {edit_request.location.sentence_index}")
                return text, structure
        elif edit_request.location.paragraph_index is not None:
            # Replace paragraph (sentence_index is None, so replace entire paragraph)
            para = structure.get_paragraph(edit_request.location.paragraph_index)
            if para:
                modified_text = text[:para.start_char] + edit_request.content + text[para.end_char:]
            else:
                logger.error(f"Could not find paragraph to replace at index {edit_request.location.paragraph_index}")
                return text, structure
        else:
            # Replace at character position (use length from metadata if available)
            replace_length = edit_request.metadata.get('replace_length', 0)
            if replace_length == 0:
                # Try to find what to replace from context
                logger.warning(f"Replace operation at character {char_pos} but no replace_length specified")
                return text, structure
            modified_text = text[:char_pos] + edit_request.content + text[char_pos + replace_length:]

    elif edit_request.edit_type == "delete":
        # Delete content at position
        if edit_request.location.sentence_index is not None:
            # Delete sentence
            sent = structure.get_sentence(
                edit_request.location.paragraph_index,
                edit_request.location.sentence_index
            )
            if sent:
                modified_text = text[:sent.start_char] + text[sent.end_char:]
            else:
                logger.error(f"Could not find sentence to delete")
                return text, structure
        elif edit_request.location.paragraph_index is not None:
            # Delete paragraph
            para = structure.get_paragraph(edit_request.location.paragraph_index)
            if para:
                modified_text = text[:para.start_char] + text[para.end_char:]
            else:
                logger.error(f"Could not find paragraph to delete")
                return text, structure
        else:
            # Delete at character position (use length from metadata)
            delete_length = edit_request.metadata.get('delete_length', len(edit_request.content) if edit_request.content else 0)
            modified_text = text[:char_pos] + text[char_pos + delete_length:]

    else:
        logger.error(f"Unknown edit type: {edit_request.edit_type}")
        return text, structure

    # Re-parse structure after edit
    updated_structure = parse_document_structure(modified_text)

    return modified_text, updated_structure
