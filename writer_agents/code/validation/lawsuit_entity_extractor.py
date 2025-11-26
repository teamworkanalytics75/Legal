"""
Extended EntityRelationExtractor with lawsuit-specific patterns.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from nlp_analysis.code.EntityRelationExtractor import EntityRelationExtractor


class LawsuitEntityExtractor(EntityRelationExtractor):
    """Entity extractor augmented with lawsuit-specific patterns."""

    _DOCUMENT_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = [
        (re.compile(r"hong\s+kong\s+statement\s+of\s+claim", re.IGNORECASE), "DOCUMENT"),
        (re.compile(r"\bhk\s+statement\b", re.IGNORECASE), "DOCUMENT"),
        (re.compile(r"\bogc\b", re.IGNORECASE), "ORGANIZATION"),
        (re.compile(r"harvard\s+ogc", re.IGNORECASE), "ORGANIZATION"),
        (re.compile(r"office\s+of\s+general\s+counsel", re.IGNORECASE), "ORGANIZATION"),
        (re.compile(r"motion\s+to\s+seal", re.IGNORECASE), "LEGAL_DOCUMENT"),
        (re.compile(r"(motion|petition)\s+for\s+pseudonym", re.IGNORECASE), "LEGAL_DOCUMENT"),
        (re.compile(r"section\s+1782", re.IGNORECASE), "STATUTE"),
    ]

    _DATE_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = [
        (re.compile(r"(april|june)\s+\d{1,2}(st|nd|rd|th)?,\s*2025", re.IGNORECASE), "DATE"),
        (re.compile(r"\d{1,2}\s+(april|june)\s+2025", re.IGNORECASE), "DATE"),
        (re.compile(r"\d{1,2}/\d{1,2}/2025", re.IGNORECASE), "DATE"),
    ]

    _ALLEGATION_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = [
        (re.compile(r"privacy\s+(breach|violation)", re.IGNORECASE), "ALLEGATION"),
        (re.compile(r"data\s+breach", re.IGNORECASE), "ALLEGATION"),
        (re.compile(r"retaliation|retaliatory", re.IGNORECASE), "ALLEGATION"),
        (re.compile(r"harassment|harassing", re.IGNORECASE), "ALLEGATION"),
        (re.compile(r"defamation|defamatory", re.IGNORECASE), "ALLEGATION"),
    ]

    _CASE_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = [
        (re.compile(r"[A-Z][A-Za-z\.\s]+ v\.\s+[A-Z][A-Za-z\.\s]+", re.IGNORECASE), "LEGAL_CASE"),
        (re.compile(r"\d+\s+U\.S\.C\.\s+§?\s*\d+", re.IGNORECASE), "STATUTE"),
    ]

    def extract_entities(self, text: str) -> List[Dict]:
        base_entities = super().extract_entities(text)
        extra_entities: List[Dict] = []
        for patterns in (
            self._DOCUMENT_PATTERNS,
            self._DATE_PATTERNS,
            self._ALLEGATION_PATTERNS,
            self._CASE_PATTERNS,
        ):
            extra_entities.extend(self._extract_pattern_entities(text, patterns))
        return base_entities + extra_entities

    @staticmethod
    def _extract_pattern_entities(text: str, patterns: Iterable[Tuple[re.Pattern[str], str]]) -> List[Dict]:
        entities: List[Dict] = []
        for pattern, label in patterns:
            for match in pattern.finditer(text):
                entities.append(
                    {
                        "text": match.group(0),
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                        "lemma": match.group(0),
                    }
                )
        return entities
