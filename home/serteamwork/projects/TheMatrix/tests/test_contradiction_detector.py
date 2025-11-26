import json
from pathlib import Path
from typing import List

import pytest

from writer_agents.code.validation.contradiction_detector import (
    Contradiction,
    ContradictionDetector,
)


def _write_registry(tmp_path: Path, data: dict) -> Path:
    registry_path = tmp_path / "fact_registry.json"
    registry_path.write_text(json.dumps(data), encoding="utf-8")
    return registry_path


def test_detects_direct_citizenship_contradiction(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path, {"citizenship": "US citizen"})
    detector = ContradictionDetector(lawsuit_facts_db=registry)

    motion = "The petitioner is a PRC citizen facing retaliation."
    contradictions = detector.detect_contradictions(motion)

    assert contradictions, "Expected contradiction to be detected"
    entry = contradictions[0]
    assert entry["contradiction_type"] == "DIRECT_CONTRADICTION"
    assert entry["severity"] == "critical"
    assert "PRC citizen" in entry["claim"]


def test_in_memory_fact_registry_supports_contradiction_detection() -> None:
    detector = ContradictionDetector(fact_registry={"citizenship": "United States citizen"})

    motion = "The applicant is a PRC citizen opposing corruption."
    contradictions = detector.detect_contradictions(motion)

    assert contradictions
    entry = contradictions[0]
    assert entry["contradiction_type"] == "DIRECT_CONTRADICTION"
    assert entry["severity"] == "critical"
    assert entry["fact_type"] == "citizenship"


def test_no_contradiction_when_claim_matches_fact(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path, {"citizenship": "United States citizen"})
    detector = ContradictionDetector(lawsuit_facts_db=registry)

    motion = "The petitioner is a United States citizen seeking relief."
    contradictions = detector.detect_contradictions(motion)

    assert contradictions == []


def test_inference_flagged_when_no_fact_registry(tmp_path: Path) -> None:
    detector = ContradictionDetector()

    motion = "Because their home country of PRC cannot protect them, relief is warranted."
    contradictions = detector.detect_contradictions(motion)

    assert contradictions
    entry = contradictions[0]
    assert entry["contradiction_type"] == "INFERENCE"
    assert entry["severity"] == "warning"


def test_source_documents_prevent_false_positive(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "citizenship.txt").write_text(
        "The plaintiff is a United States citizen with longstanding ties to Boston.",
        encoding="utf-8"
    )

    detector = ContradictionDetector(source_docs_dir=docs_dir)
    motion = "The plaintiff is a United States citizen seeking protection."

    contradictions = detector.detect_contradictions(motion)
    assert contradictions == []


def test_hallucination_flagged_without_support(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "risk.txt").write_text(
        "The plaintiff faces political risk when traveling through PRC airspace.",
        encoding="utf-8"
    )

    detector = ContradictionDetector(source_docs_dir=docs_dir)
    motion = "The plaintiff is a PRC citizen in fear of persecution."

    contradictions = detector.detect_contradictions(motion)
    assert contradictions
    entry = contradictions[0]
    assert entry["contradiction_type"] == "HALLUCINATION"
    assert entry["severity"] == "critical"


def test_custom_validator_registration() -> None:
    detector = ContradictionDetector()

    def date_validator(text: str) -> List[Contradiction]:
        contradictions = []
        if "April 5, 2025" in text:
            contradictions.append(
                Contradiction(
                    claim="April 5, 2025 statement",
                    contradiction_type="DIRECT_CONTRADICTION",
                    severity="critical",
                    location="April 5, 2025 statement",
                    source_evidence="Official timeline lists April 7, 2025.",
                    fact_type="date",
                )
            )
        return contradictions

    detector.register_validator("dates", date_validator)
    motion = "The motion asserts that on April 5, 2025 the OGC responded."

    contradictions = detector.detect_contradictions(motion)
    assert contradictions
    assert contradictions[0]["fact_type"] == "date"
