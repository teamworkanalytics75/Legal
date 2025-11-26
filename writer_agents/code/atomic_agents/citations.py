"""Citation processing atomic agents.

All citation agents are deterministic (no LLM calls) for maximum cost efficiency.
They use regex patterns, templates, and database queries.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

try:
    from ..atomic_agent import DeterministicAgent, AgentFactory
except ImportError:  # Fallback when imported outside package context
    from atomic_agent import DeterministicAgent, AgentFactory


class CitationFinderAgent(DeterministicAgent):
    """Find raw citation strings in text using pattern matching.

    Single duty: Identify potential legal citations in text.
    Method: Regex patterns for common citation formats.
    Output: List of citation candidates with positions.
    """

    duty = "Find raw citation strings in text using regex patterns"

    # Citation patterns (Bluebook-style)
    PATTERNS = [
        # Case citations: "123 U.S. 456"
        (r'\b(\d+)\s+([A-Z][a-z]*\.?\s*(?:[A-Z][a-z\.]*\s*)*)\s+(\d+)', 'case_reporter'),

        # Case names: "Smith v. Jones"
        (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'case_name'),

        # Statute citations: "42 U.S.C. Section 1983"
        (r'\b(\d+)\s+([A-Z]\.?[A-Z]\.?[A-Z]?\.?)\s+Section\s*(\d+(?:\([a-z]\))?)', 'statute'),

        # Code citations: "Cal. Civ. Code Section 1234"
        (r'\b([A-Z][a-z]+\.)\s+([A-Z][a-z]+\.)\s+Code\s+Section\s*(\d+)', 'code'),

        # Federal rules: "Fed. R. Civ. P. 12(b)(6)"
        (r'\bFed\.\s+R\.\s+(Civ\.|Crim\.|App\.|Evid\.)\s+P\.\s+(\d+(?:\([a-z]\))*(?:\(\d+\))*)', 'federal_rule'),

        # Parenthetical years: "(2023)"
        (r'\((\d{4})\)', 'year'),
    ]

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find citations in text using regex patterns.

        Args:
            input_data: Must contain 'text' key with document text

        Returns:
            Dictionary with:
                - citations: List of found citations
                - count: Total count
                - by_type: Count by citation type
        """
        text = input_data.get('text', '')

        if not text:
            return {
                'citations': [],
                'count': 0,
                'by_type': {},
            }

        citations = []
        type_counts: Dict[str, int] = {}

        for pattern, citation_type in self.PATTERNS:
            for match in re.finditer(pattern, text):
                citation = {
                    'type': citation_type,
                    'text': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'groups': match.groups(),
                }
                citations.append(citation)
                type_counts[citation_type] = type_counts.get(citation_type, 0) + 1

        # Sort by position
        citations.sort(key=lambda c: c['start'])

        return {
            'citations': citations,
            'count': len(citations),
            'by_type': type_counts,
        }


class CitationNormalizerAgent(DeterministicAgent):
    """Normalize citations to Bluebook format using templates.

    Single duty: Convert raw citations to standardized Bluebook format.
    Method: Template-based transformation rules.
    Output: Normalized citation strings.
    """

    duty = "Normalize citations to Bluebook format using templates"

    # Reporter abbreviations (Bluebook Table 1)
    REPORTER_ABBREV = {
        'U.S.': 'U.S.',
        'S. Ct.': 'S. Ct.',
        'F.': 'F.',
        'F.2d': 'F.2d',
        'F.3d': 'F.3d',
        'F. Supp.': 'F. Supp.',
        'F. Supp. 2d': 'F. Supp. 2d',
        'F. Supp. 3d': 'F. Supp. 3d',
        'Cal.': 'Cal.',
        'Cal. 2d': 'Cal. 2d',
        'Cal. 3d': 'Cal. 3d',
        'Cal. 4th': 'Cal. 4th',
        'N.Y.': 'N.Y.',
        'N.E.': 'N.E.',
        'N.E.2d': 'N.E.2d',
        'Mass.': 'Mass.',
    }

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize citations to Bluebook format.

        Args:
            input_data: Must contain 'citations' from CitationFinderAgent

        Returns:
            Dictionary with:
                - normalized_citations: List of normalized citations
                - changes: List of normalization changes made
        """
        citations = input_data.get('citations', [])

        normalized = []
        changes = []

        for cite in citations:
            cite_type = cite.get('type')
            original = cite.get('text', '')

            if cite_type == 'case_reporter':
                # Normalize case reporter citation
                groups = cite.get('groups', ())
                if len(groups) >= 3:
                    volume, reporter, page = groups[0], groups[1], groups[2]

                    # Normalize reporter abbreviation
                    reporter_clean = reporter.strip()
                    normalized_reporter = self.REPORTER_ABBREV.get(
                        reporter_clean,
                        reporter_clean
                    )

                    normalized_text = f"{volume} {normalized_reporter} {page}"

                    if normalized_text != original:
                        changes.append({
                            'original': original,
                            'normalized': normalized_text,
                            'change_type': 'reporter_format',
                        })

                    normalized.append({
                        **cite,
                        'normalized_text': normalized_text,
                        'original_text': original,
                    })

            elif cite_type == 'case_name':
                # Normalize case name (italicize in bluebook)
                normalized_text = original # Keep as-is for now
                normalized.append({
                    **cite,
                    'normalized_text': normalized_text,
                    'original_text': original,
                    'format': 'italic', # Hint for formatter
                })

            elif cite_type == 'statute':
                # Normalize statute citation
                groups = cite.get('groups', ())
                if len(groups) >= 3:
                    title, code, section = groups[0], groups[1], groups[2]
                    normalized_text = f"{title} {code} Section {section}"

                    if normalized_text != original:
                        changes.append({
                            'original': original,
                            'normalized': normalized_text,
                            'change_type': 'statute_format',
                        })

                    normalized.append({
                        **cite,
                        'normalized_text': normalized_text,
                        'original_text': original,
                    })

            else:
                # Keep other types as-is
                normalized.append({
                    **cite,
                    'normalized_text': original,
                    'original_text': original,
                })

        return {
            'normalized_citations': normalized,
            'changes': changes,
            'change_count': len(changes),
        }


class CitationVerifierAgent(DeterministicAgent):
    """Verify citations against case law database.

    Single duty: Check if citations point to real cases.
    Method: SQL queries against lawsuit_docs database.
    Output: Verification status for each citation.
    """

    duty = "Verify citations against case law database using SQL queries"

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify citations against database.

        Args:
            input_data: Must contain 'normalized_citations' and optional 'db_connection'

        Returns:
            Dictionary with:
                - verified_citations: List with verification status
                - verified_count: Number of verified citations
                - unverified_count: Number that couldn't be verified
        """
        citations = input_data.get('normalized_citations', [])
        db_connection = input_data.get('db_connection')

        verified = []
        verified_count = 0
        unverified_count = 0

        for cite in citations:
            # For now, mark all as "pending_verification" since we may not have DB connection
            # In production, this would do actual SQL lookups

            if db_connection:
                # TODO: Actual database verification
                # SELECT * FROM case_law WHERE citation LIKE ?
                verification_status = 'pending_check'
            else:
                verification_status = 'no_db_connection'

            verified.append({
                **cite,
                'verified': verification_status == 'verified',
                'verification_status': verification_status,
            })

            if verification_status == 'verified':
                verified_count += 1
            else:
                unverified_count += 1

        return {
            'verified_citations': verified,
            'verified_count': verified_count,
            'unverified_count': unverified_count,
            'verification_rate': verified_count / len(citations) if citations else 0.0,
        }


class CitationLocatorAgent(DeterministicAgent):
    """Map citation tokens to file paths or URLs in database.

    Single duty: Find the source document for each citation.
    Method: Database queries to locate case files.
    Output: File paths or URLs for each citation.
    """

    duty = "Map citation tokens to file paths/URLs in database"

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Locate source files for citations.

        Args:
            input_data: Must contain 'verified_citations' and optional 'db_connection'

        Returns:
            Dictionary with:
                - located_citations: Citations with file paths
                - located_count: Number successfully located
        """
        citations = input_data.get('verified_citations', [])
        db_connection = input_data.get('db_connection')

        located = []
        located_count = 0

        for cite in citations:
            # In production, query database for file path
            # SELECT file_path FROM case_law WHERE citation = ?

            if db_connection:
                # TODO: Actual database lookup
                file_path = None
            else:
                file_path = None

            located.append({
                **cite,
                'file_path': file_path,
                'located': file_path is not None,
            })

            if file_path:
                located_count += 1

        return {
            'located_citations': located,
            'located_count': located_count,
            'location_rate': located_count / len(citations) if citations else 0.0,
        }


class CitationInserterAgent(DeterministicAgent):
    """Insert formatted citations into document text.

    Single duty: Place normalized citations at correct positions.
    Method: String manipulation based on position markers.
    Output: Document text with properly formatted citations.
    """

    duty = "Insert formatted citations into document text at correct positions"

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert citations into text.

        Args:
            input_data: Must contain 'text' and 'located_citations'

        Returns:
            Dictionary with:
                - output_text: Text with inserted citations
                - insertions: List of insertion operations performed
        """
        text = input_data.get('text', '')
        citations = input_data.get('located_citations', [])

        if not text or not citations:
            return {
                'output_text': text,
                'insertions': [],
                'insertion_count': 0,
            }

        # Sort citations by position (descending) to maintain positions during insertion
        sorted_citations = sorted(
            citations,
            key=lambda c: c.get('start', 0),
            reverse=True
        )

        output_text = text
        insertions = []

        for cite in sorted_citations:
            normalized_text = cite.get('normalized_text', '')
            start = cite.get('start', 0)
            end = cite.get('end', 0)

            if start < len(output_text):
                # Replace original citation with normalized version
                output_text = (
                    output_text[:start] +
                    normalized_text +
                    output_text[end:]
                )

                insertions.append({
                    'position': start,
                    'original': text[start:end],
                    'inserted': normalized_text,
                })

        return {
            'output_text': output_text,
            'insertions': insertions,
            'insertion_count': len(insertions),
        }
