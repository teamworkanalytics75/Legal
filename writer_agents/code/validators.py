"""Validation helpers to keep writer outputs consistent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from .tasks import DraftSection, ReviewFindings


CITATION_PATTERN = re.compile(r"\[Node:[A-Za-z0-9_]+(?:->[A-Za-z0-9_]+)?\]")


@dataclass(slots=True)
class CitationCheckConfig:
    """Configuration for citation validation."""

    required_nodes: Iterable[str]
    allow_missing: bool = False


class CitationValidator:
    """Ensures that drafted sections reference required BN nodes."""

    def __init__(self, config: CitationCheckConfig) -> None:
        self._config = config

    def run(self, section: DraftSection) -> List[ReviewFindings]:
        """Return findings if citations are missing or malformed."""
        findings: List[ReviewFindings] = []
        present = set(node.group(0) for node in CITATION_PATTERN.finditer(section.body))
        for required in self._config.required_nodes:
            token = f"[Node:{required}]"
            if token not in present and not self._config.allow_missing:
                findings.append(
                    ReviewFindings(
                        section_id=section.section_id,
                        severity="error",
                        message=f"Missing citation token for node {required}",
                        suggestions=f"Add {token} in the relevant paragraph.",
                    )
                )
        if not present:
            findings.append(
                ReviewFindings(
                    section_id=section.section_id,
                    severity="warning",
                    message="Section contains no citations.",
                    suggestions="Insert at least one [Node:Outcome] citation tied to evidence.",
                )
            )
        return findings


class StructureValidator:
    """Verifies that all expected sections are delivered."""

    def __init__(self, expected_ids: Iterable[str]) -> None:
        self._expected = list(expected_ids)

    def run(self, drafted_sections: List[DraftSection]) -> List[ReviewFindings]:
        """Return findings for missing sections."""
        delivered = {section.section_id for section in drafted_sections}
        findings: List[ReviewFindings] = []
        for expected in self._expected:
            if expected not in delivered:
                findings.append(
                    ReviewFindings(
                        section_id=expected,
                        severity="error",
                        message="Section missing from draft output.",
                    )
                )
        return findings


__all__ = [
    "CitationValidator",
    "CitationCheckConfig",
    "StructureValidator",
]
