"""
Validator to ensure the document contains required headings.
"""

from __future__ import annotations

from typing import Dict, List

from ..base_plugin import kernel_function

from ..base_plugin import ValidationResult


REQUIRED_HEADINGS: List[str] = [
    "privacy harm",
    "retaliation",
]


@kernel_function(
    name="ValidateStructure",
    description="Check that required headings exist in the drafted section.",
)
def validate_structure(context: Dict[str, str]) -> ValidationResult:
    document = context.get("document", "").lower()
    missing = [heading for heading in REQUIRED_HEADINGS if heading not in document]
    if missing:
        return ValidationResult(
            passed=False,
            score=0.0,
            message=f"Missing required headings: {', '.join(missing)}",
        )
    return ValidationResult(passed=True, score=1.0, message="All required headings present.")

