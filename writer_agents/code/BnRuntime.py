"""Fallback Bayesian engine utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .insights import CaseInsights, EvidenceItem, Posterior


DEFAULT_MOCK_MODEL = Path(__file__).resolve().parent.parent / "data" / "mock_bn_model.json"


@dataclass(slots=True)
class MockBayesEngine:
    """Simple Bayesian engine driven by a static JSON model."""

    model_path: Path = DEFAULT_MOCK_MODEL

    def load(self) -> Dict[str, object]:
        text = self.model_path.read_text(encoding="utf-8-sig") # Handle UTF-8 BOM
        payload = json.loads(text)
        if "posteriors" not in payload:
            raise ValueError("mock BN model missing 'posteriors' field")
        return payload

    def report(self) -> str:
        payload = self.load()
        lines: List[str] = []
        lines.append("=== KEY POSTERIOR PROBABILITIES ===")
        for node_id, probs in payload.get("posteriors", {}).items():
            formatted = ", ".join(f"{state}: {value:.4f}" for state, value in probs.items())
            lines.append(f"{node_id}: {formatted}")
        mpe = payload.get("mpe", {})
        if mpe:
            lines.append("\n=== MOST PROBABLE EXPLANATION ===")
            for node_id, state in mpe.items():
                lines.append(f"{node_id}: {state}")
        summary = payload.get("summary")
        if summary:
            lines.append("\n" + str(summary))
        return "\n".join(lines)

    def insights(self, reference_id: str, summary: str) -> CaseInsights:
        payload = self.load()
        posterior_objs = [
            Posterior(node_id=node_id, probabilities=posterior)
            for node_id, posterior in payload.get("posteriors", {}).items()
        ]
        evidence_objs = [
            EvidenceItem(node_id=item["node_id"], state=item["state"], description=item.get("description"))
            for item in payload.get("evidence", [])
        ]
        return CaseInsights(
            reference_id=reference_id,
            summary=summary,
            posteriors=posterior_objs,
            evidence=evidence_objs,
            jurisdiction=str(payload.get("jurisdiction", "")) or None,
            case_style=str(payload.get("case_style", "")) or None,
        )


def build_insights_from_mock(reference_id: str, summary: str, model_path: Path | None = None) -> CaseInsights:
    engine = MockBayesEngine(model_path=model_path or DEFAULT_MOCK_MODEL)
    return engine.insights(reference_id=reference_id, summary=summary)


def format_mock_report(model_path: Path | None = None) -> str:
    engine = MockBayesEngine(model_path=model_path or DEFAULT_MOCK_MODEL)
    return engine.report()


__all__ = ["MockBayesEngine", "build_insights_from_mock", "format_mock_report"]
