"""Apply structural constraints to Bayesian Network edge sets."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokenize(pattern: Sequence[str] | str) -> List[str]:
    if isinstance(pattern, str):
        items = pattern.split()
    else:
        items = []
        for chunk in pattern:
            items.extend(chunk.split())
    return [token.strip().lower() for token in items if token.strip()]


@dataclass(frozen=True)
class ConstraintRule:
    parent_keywords: Sequence[str] | str
    child_keywords: Sequence[str] | str
    reason: str = ""
    max_edges: int | None = None
    parent_match_any: bool = False
    child_match_any: bool = False


@dataclass
class StructuralConstraintConfig:
    required_edges: List[ConstraintRule] = field(default_factory=list)
    forbidden_edges: List[ConstraintRule] = field(default_factory=list)
    optional_edges: List[ConstraintRule] = field(default_factory=list)
    domain_rules: List[ConstraintRule] = field(default_factory=list)


DEFAULT_CONSTRAINTS = StructuralConstraintConfig(
    required_edges=[
        ConstraintRule(
            parent_keywords=("prc", "political", "risk"),
            child_keywords=("plaintiff", "physical", "risk"),
            reason="PRC political risk must influence plaintiff's physical risk.",
            max_edges=2,
        ),
        ConstraintRule(
            parent_keywords=("harvard", "ogc"),
            child_keywords=("prc", "political", "risk"),
            reason="Harvard OGC communications increase PRC political risk.",
            max_edges=3,
            child_match_any=True,
        ),
    ],
    forbidden_edges=[
        ConstraintRule(
            parent_keywords=("plaintiff", "career"),
            child_keywords=("prc", "crackdown"),
            reason="Plaintiff career history cannot drive PRC crackdowns.",
        ),
        ConstraintRule(
            parent_keywords=("harm",),
            child_keywords=("statement",),
            reason="Effects cannot feed back into the original statements.",
            parent_match_any=True,
            child_match_any=True,
        ),
    ],
    optional_edges=[
        ConstraintRule(
            parent_keywords=("statement", "clarification"),
            child_keywords=("media", "amplification"),
            reason="Statements should feed media amplification nodes.",
            parent_match_any=True,
            child_match_any=True,
            max_edges=6,
        ),
        ConstraintRule(
            parent_keywords=("media", "coverage"),
            child_keywords=("prc", "political", "risk"),
            reason="Media coverage can escalate PRC political risk.",
            parent_match_any=True,
            child_match_any=True,
            max_edges=4,
        ),
    ],
    domain_rules=[
        ConstraintRule(
            parent_keywords=("prc", "ogc", "harvard"),
            child_keywords=("risk", "harm", "danger", "retaliation", "persecution"),
            reason="Institutional actors influence risk/harm outcomes.",
            parent_match_any=True,
            child_match_any=True,
            max_edges=10,
        ),
        ConstraintRule(
            parent_keywords=("email", "communication", "statement"),
            child_keywords=("media", "exposure", "wechat", "publication"),
            reason="Communications propagate to exposure nodes.",
            parent_match_any=True,
            child_match_any=True,
            max_edges=8,
        ),
    ],
)


def _match_nodes(
    nodes: Sequence[object],
    normalized_labels: dict[str, str],
    keywords: Sequence[str] | str,
    match_any: bool,
) -> List[str]:
    tokens = _tokenize(keywords)
    if not tokens:
        return []

    matches: List[str] = []
    for node in nodes:
        node_id = getattr(node, "node_id")
        label = normalized_labels.get(node_id, "")
        if not label:
            continue
        condition = any if match_any else all
        if condition(token in label for token in tokens):
            matches.append(node_id)
    return matches


def apply_structural_constraints(
    nodes: Sequence[object],
    edges: Iterable[Tuple[str, str]],
    config: StructuralConstraintConfig | None = None,
    max_parents: int | None = None,
) -> List[Tuple[str, str]]:
    """Apply structural constraints to BN edges."""
    config = config or DEFAULT_CONSTRAINTS

    node_lookup = {getattr(node, "node_id"): node for node in nodes}
    normalized_labels = {
        node_id: _normalize_text(getattr(node, "label", node_id))
        for node_id, node in node_lookup.items()
    }
    importance = {
        node_id: float(getattr(node, "importance_score", 0.0))
        for node_id, node in node_lookup.items()
    }

    # Initialize edge bookkeeping.
    unique_edges: List[Tuple[str, str]] = []
    edge_set = set()
    child_parents: dict[str, List[str]] = {node_id: [] for node_id in node_lookup}

    def _register_edge(parent: str, child: str) -> None:
        edge_set.add((parent, child))
        unique_edges.append((parent, child))
        child_parents.setdefault(child, []).append(parent)

    def _normalize_edge(parent: str, child: str) -> Tuple[str, str] | None:
        if parent == child:
            return None
        if parent not in node_lookup or child not in node_lookup:
            return None
        return parent, child

    for parent, child in edges:
        normalized = _normalize_edge(parent, child)
        if not normalized:
            continue
        if normalized in edge_set:
            continue
        _register_edge(*normalized)

    def _remove_edge(parent: str, child: str, reason: str) -> None:
        if (parent, child) not in edge_set:
            return
        edge_set.remove((parent, child))
        child_parents.get(child, []).remove(parent)
        unique_edges[:] = [edge for edge in unique_edges if edge != (parent, child)]
        logger.debug("Removed edge %s -> %s (%s)", parent, child, reason)

    def _prune_lowest_importance(child: str) -> bool:
        parents = child_parents.get(child, [])
        if not parents:
            return False
        lowest_parent = min(parents, key=lambda pid: importance.get(pid, 0.0))
        _remove_edge(lowest_parent, child, "enforcing constraints")
        return True

    def _can_add_edge(child: str, force: bool) -> bool:
        if max_parents is None:
            return True
        current = len(child_parents.get(child, []))
        if current < max_parents:
            return True
        if force:
            return _prune_lowest_importance(child)
        return False

    def _add_edge(parent: str, child: str, reason: str, force: bool = False) -> bool:
        if parent == child:
            return False
        if parent not in node_lookup or child not in node_lookup:
            return False
        if (parent, child) in edge_set:
            return False
        if not _can_add_edge(child, force):
            logger.debug("Skipped edge %s -> %s (parent cap reached)", parent, child)
            return False
        _register_edge(parent, child)
        logger.debug("Added edge %s -> %s (%s)", parent, child, reason)
        return True

    def _apply_rule(rule: ConstraintRule, *, force: bool = False, remove: bool = False) -> None:
        parents = _match_nodes(nodes, normalized_labels, rule.parent_keywords, rule.parent_match_any)
        children = _match_nodes(nodes, normalized_labels, rule.child_keywords, rule.child_match_any)
        if not parents or not children:
            logger.debug(
                "Rule skipped (%s) — parents=%s children=%s",
                rule.reason,
                bool(parents),
                bool(children),
            )
            return

        applied = 0
        for parent in parents:
            for child in children:
                if parent == child:
                    continue
                if remove:
                    _remove_edge(parent, child, rule.reason)
                    applied += 1
                else:
                    added = _add_edge(parent, child, rule.reason, force=force)
                    applied += 1 if added else 0
                if rule.max_edges and applied >= rule.max_edges:
                    return

    # Apply forbidden edges first.
    for rule in config.forbidden_edges:
        _apply_rule(rule, remove=True)

    # Required edges must exist.
    for rule in config.required_edges:
        _apply_rule(rule, force=True)

    # Optional/domain rules expand structure without forcing.
    for rule in config.optional_edges:
        _apply_rule(rule, force=False)

    for rule in config.domain_rules:
        _apply_rule(rule, force=False)

    return unique_edges


__all__ = [
    "ConstraintRule",
    "StructuralConstraintConfig",
    "DEFAULT_CONSTRAINTS",
    "apply_structural_constraints",
]
