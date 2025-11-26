"""Utilities to transform Bayesian output into CaseInsights."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

from .insights import CaseInsights, EvidenceItem, Posterior


def parse_bn_posteriors(raw: Dict[str, Dict[str, float]]) -> List[Posterior]:
    """Convert a nested mapping into Posterior objects."""
    return [Posterior(node_id=node_id, probabilities=states) for node_id, states in raw.items()]


def build_insights(
    reference_id: str,
    summary: str,
    posterior_data: Dict[str, Dict[str, float]],
    evidence: Iterable[EvidenceItem],
    *,
    jurisdiction: str | None = None,
    case_style: str | None = None,
) -> CaseInsights:
    """Create a CaseInsights instance from BN primitives."""
    posteriors = parse_bn_posteriors(posterior_data)
    return CaseInsights(
        reference_id=reference_id,
        summary=summary,
        posteriors=posteriors,
        evidence=list(evidence),
        jurisdiction=jurisdiction,
        case_style=case_style,
    )


def from_json_payload(payload: str) -> CaseInsights:
    """Load insights from a serialized JSON payload."""
    obj = json.loads(payload)
    evidence_items = [
        EvidenceItem(
            node_id=item["node_id"],
            state=item["state"],
            weight=item.get("weight"),
            description=item.get("description"),
        )
        for item in obj.get("evidence", [])
    ]
    return build_insights(
        reference_id=obj.get("reference_id", "case"),
        summary=obj.get("summary", ""),
        posterior_data=obj.get("posteriors", {}),
        evidence=evidence_items,
        jurisdiction=obj.get("jurisdiction"),
        case_style=obj.get("case_style"),
    )


__all__ = ["parse_bn_posteriors", "build_insights", "from_json_payload"]
