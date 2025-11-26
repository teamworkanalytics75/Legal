#!/usr/bin/env python3
"""
Citation Format Plugin - Character-level enforcement

Enforces character-level citation formatting (Bluebook compliance)
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult

logger = logging.getLogger(__name__)


class CitationFormatPlugin(BaseFeaturePlugin):
    """Plugin for enforcing character-level citation formatting."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "citation_format", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("CitationFormatPlugin initialized")

    def _find_citations(self, text: str) -> List[Dict[str, Any]]:
        """Find all citations in text."""
        # Common citation patterns
        citation_patterns = [
            # Standard case citation: Case Name, Volume Reporter Page (Court Year)
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s+(\d+)\s+([A-Z\.\s]+)\s+(\d+)\s+\(([^)]+)\s+(\d{4})\)',
            # Citation with pin cite: Case Name, Volume Reporter Page (Court Year) (page number)
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s+(\d+)\s+([A-Z\.\s]+)\s+(\d+)\s+\(([^)]+)\s+(\d{4})\)\s+\((\d+)\)',
            # Short form: Volume Reporter Page
            r'(\d+)\s+([A-Z\.\s]+)\s+(\d+)',
            # Supra/Infra references
            r'(supra|infra)\s+note\s+(\d+)'
        ]

        citations = []
        for pattern in citation_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                citations.append({
                    "full_match": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "context": text[max(0, match.start()-30):match.end()+30]
                })

        return citations

    def _validate_citation_format(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual citation format."""
        citation_text = citation["full_match"]
        issues = []

        # Check for proper comma placement (after case name)
        if re.search(r'v\.\s+[A-Z]', citation_text):
            if not re.search(r'v\.\s+[A-Z][a-z]+,\s+', citation_text):
                issues.append("Missing comma after case name in 'v.' citation")

        # Check for proper parentheses placement (around court and year)
        if not re.search(r'\([^)]+\s+\d{4}\)', citation_text):
            if re.search(r'\d{4}', citation_text):
                issues.append("Year not properly parenthesized with court")

        # Check pin cite format (should be in parentheses)
        if re.search(r'\d{4}\)\s+\d+', citation_text):
            if not re.search(r'\d{4}\)\s+\(\d+\)', citation_text):
                issues.append("Pin cite not properly parenthesized")

        # Check for proper spacing
        if re.search(r'\d+\s+[A-Z]', citation_text):
            # Should have space between volume and reporter
            pass  # This is handled by the pattern

        return {
            "valid": len(issues) == 0,
            "issues": issues
        }

    async def validate(self, text: str) -> FunctionResult:
        """Validate citation formatting."""
        citations = self._find_citations(text)

        valid_citations = 0
        invalid_citations = []
        all_issues = []

        for citation in citations:
            validation = self._validate_citation_format(citation)
            if validation["valid"]:
                valid_citations += 1
            else:
                invalid_citations.append({
                    "citation": citation["full_match"],
                    "issues": validation["issues"]
                })
                all_issues.extend(validation["issues"])

        total_citations = len(citations)
        if total_citations > 0:
            valid_ratio = valid_citations / total_citations
        else:
            valid_ratio = 1.0  # No citations is valid (may not be required)

        score = valid_ratio
        meets_threshold = valid_ratio >= 1.0  # All citations must be valid

        issues = []
        if invalid_citations:
            issues.append(f"{len(invalid_citations)}/{total_citations} citations have format issues")
            for inv in invalid_citations[:5]:
                issues.append(f"  - '{inv['citation'][:50]}...': {', '.join(inv['issues'])}")
        issues.extend(set(all_issues))  # Add unique issues

        return FunctionResult(
            success=meets_threshold,
            score=score,
            message=f"Citation Format: {valid_citations}/{total_citations} citations valid - {'PASS' if meets_threshold else 'FAIL'}",
            metadata={
                "total_citations": total_citations,
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations[:10],
                "issues": issues,
                "threshold": self.rules.get("threshold", 1.0)
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

