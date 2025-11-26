"""Data structures for representing Bayesian network insights."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass(slots=True)
class EvidenceItem:
    """Represents an evidence node and chosen state."""

    node_id: str
    state: str
    weight: Optional[float] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialize the evidence item into a JSON-friendly dictionary."""
        payload: Dict[str, object] = {"node_id": self.node_id, "state": self.state}
        if self.weight is not None:
            payload["weight"] = self.weight
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass(slots=True)
class Posterior:
    """Represents the posterior distribution of a Bayesian network node."""

    node_id: str
    probabilities: Dict[str, float]
    interpretation: Optional[str] = None

    def normalized(self) -> "Posterior":
        """Return a posterior with probabilities normalized to sum to one."""
        total = sum(self.probabilities.values())
        if total <= 0:
            return self
        normalized_probs = {state: value / total for state, value in self.probabilities.items()}
        return Posterior(node_id=self.node_id, probabilities=normalized_probs, interpretation=self.interpretation)

    def top_outcome(self) -> Optional[str]:
        """Return the most probable outcome if available."""
        if not self.probabilities:
            return None
        return max(self.probabilities.items(), key=lambda item: item[1])[0]


@dataclass(slots=True)
class CaseInsights:
    """Container for BN results passed into the writing pipeline."""

    reference_id: str
    summary: str
    posteriors: List[Posterior]
    evidence: List[EvidenceItem] = field(default_factory=list)
    jurisdiction: Optional[str] = None
    case_style: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def posterior_lookup(self) -> Dict[str, Posterior]:
        """Return a dictionary keyed by node identifiers for quick access."""
        return {posterior.node_id: posterior for posterior in self.posteriors}

    def to_prompt_block(self) -> str:
        """Render the insights as a structured markdown segment for prompting."""
        lines = [f"Case Reference: {self.reference_id}"]
        if self.jurisdiction:
            lines.append(f"Jurisdiction: {self.jurisdiction}")
        if self.case_style:
            lines.append(f"Style: {self.case_style}")
        lines.append("\nKey Posteriors:")
        for posterior in self.posteriors:
            normalized = posterior.normalized()
            probs = ", ".join(f"{state}: {prob:.2%}" for state, prob in normalized.probabilities.items())
            interpretation = f" - {posterior.interpretation}" if posterior.interpretation else ""
            lines.append(f"- {posterior.node_id}: {probs}{interpretation}")
        if self.evidence:
            lines.append("\nEvidence Inputs:")
            for item in self.evidence:
                descriptor = f" ({item.description})" if item.description else ""
                weight = f", weight={item.weight:.2f}" if item.weight is not None else ""
                lines.append(f"- {item.node_id} = {item.state}{weight}{descriptor}")
        return "\n".join(lines)


__all__ = ["EvidenceItem", "Posterior", "CaseInsights"]
