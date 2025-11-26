from __future__ import annotations

from dataclasses import dataclass

from writer_agents.code.bn_structural_constraints import (
    ConstraintRule,
    StructuralConstraintConfig,
    apply_structural_constraints,
)


@dataclass
class DummyNode:
    node_id: str
    label: str
    importance_score: float = 0.5
    states: list[str] = None
    centrality_metrics: dict[str, float] = None


def test_required_edges_are_added_when_missing() -> None:
    nodes = [
        DummyNode("n1", "PRC political risk", 0.9),
        DummyNode("n2", "Plaintiff physical risk", 0.6),
    ]
    config = StructuralConstraintConfig(
        required_edges=[
            ConstraintRule(
                parent_keywords=("prc", "political", "risk"),
                child_keywords=("plaintiff", "physical", "risk"),
                reason="PRC cascade",
            )
        ]
    )

    edges = apply_structural_constraints(nodes, [], config=config)

    assert ("n1", "n2") in edges


def test_forbidden_edges_are_removed() -> None:
    nodes = [
        DummyNode("n1", "Plaintiff career history", 0.7),
        DummyNode("n2", "PRC crackdown response", 0.6),
        DummyNode("n3", "Downstream effect", 0.3),
    ]
    config = StructuralConstraintConfig(
        forbidden_edges=[
            ConstraintRule(
                parent_keywords=("plaintiff", "career"),
                child_keywords=("prc", "crackdown"),
                reason="Career history mismatch",
            )
        ]
    )

    result = apply_structural_constraints(nodes, [("n1", "n2"), ("n2", "n3")], config=config)

    assert ("n1", "n2") not in result
    assert ("n2", "n3") in result


def test_required_edges_override_parent_limit() -> None:
    nodes = [
        DummyNode("n1", "PRC political risk", 0.9),
        DummyNode("n2", "Harvard OGC alert", 0.8),
        DummyNode("n3", "Physical harm risk", 0.4),
        DummyNode("n4", "Legacy parent", 0.2),
    ]
    config = StructuralConstraintConfig(
        required_edges=[
            ConstraintRule(
                parent_keywords=("prc", "political", "risk"),
                child_keywords=("physical", "harm", "risk"),
                reason="PRC drives harm",
            )
        ],
        optional_edges=[
            ConstraintRule(
                parent_keywords=("harvard", "ogc"),
                child_keywords=("physical", "harm"),
                reason="OGC influence",
            )
        ],
    )
    initial_edges = [("n4", "n3")]

    result = apply_structural_constraints(
        nodes,
        initial_edges,
        config=config,
        max_parents=1,
    )

    assert ("n1", "n3") in result
    assert ("n4", "n3") not in result  # pruned to honor required edge
    assert ("n2", "n3") not in result  # optional edge blocked by parent cap
