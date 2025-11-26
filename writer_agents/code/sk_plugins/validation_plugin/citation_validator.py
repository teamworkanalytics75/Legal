"""
Regex-based citation validation.
"""

from __future__ import annotations

import re
from typing import Dict

from ..base_plugin import kernel_function

from ..base_plugin import ValidationResult


CITATION_PATTERN = re.compile(
    r"""
    (?P<cite>
        \b\d{1,4}                # volume
        \s+
        [A-Z][A-Za-z0-9\.\s]*    # reporter
        \s+
        \d{1,5}                  # page
    )
    """,
    re.VERBOSE,
)


@kernel_function(
    name="ValidateCitationFormat",
    description="Ensure the drafted section contains properly formatted legal citations.",
)
def validate_citation_format(context: Dict[str, str]) -> ValidationResult:
    document = context.get("document", "")
    matches = CITATION_PATTERN.findall(document)
    if not matches:
        return ValidationResult(
            passed=False,
            score=0.0,
            message="No recognised legal citations were found.",
        )
    # crude heuristic: assume all matches acceptable for now
    return ValidationResult(passed=True, score=1.0, message=f"{len(matches)} citations detected.")

