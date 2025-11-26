"""
Native privacy harm drafting function.

This deterministic routine assembles an argument skeleton using the
evidence curated by upstream research modules. It is intentionally
straightforward – the semantic function layered on top provides the
high-fidelity prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..base_plugin import kernel_function


@dataclass
class EvidenceItem:
    description: str
    citation: Optional[str] = None


def _format_evidence(evidence_list: List[EvidenceItem]) -> str:
    lines = []
    for idx, item in enumerate(evidence_list, start=1):
        citation = f" ({item.citation})" if item.citation else ""
        lines.append(f"{idx}. {item.description}{citation}")
    return "\n".join(lines)


@kernel_function(
    name="PrivacyHarmNative",
    description="Generate a structured outline for the privacy harm section.",
)
def privacy_harm_native_function(
    context: Dict[str, str],
) -> str:
    """
    Build a deterministic outline based on structured evidence supplied via context.

    Expected context keys:
        - identity_risk: str
        - retaliation_risk: str
        - psychological_harm: str
        - evidence_items: list[dict]
    """
    identity_risk = context.get("identity_risk") or "Risk of identity disclosure leading to irreparable harm."
    retaliation_risk = context.get("retaliation_risk") or "Credible threat of retaliation or harassment."
    psychological_harm = context.get("psychological_harm") or "Documented psychological impact from public exposure."

    evidence_items_raw = context.get("evidence_items", [])
    evidence_items = [
        EvidenceItem(description=item.get("description", ""), citation=item.get("citation"))
        for item in evidence_items_raw
        if item.get("description")
    ]

    outline = [
        "# Privacy Harm Argument (Structured Outline)",
        "",
        "## Identity Security Risk",
        identity_risk,
        "",
        "## Retaliation Risk",
        retaliation_risk,
        "",
        "## Psychological / Reputational Harm",
        psychological_harm,
    ]

    if evidence_items:
        outline.extend(
            [
                "",
                "## Supporting Evidence",
                _format_evidence(evidence_items),
            ]
        )

    return "\n".join(outline)

