"""Utilities for loading and sampling legal idioms."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


class IdiomRepositoryError(RuntimeError):
    """Raised when the idiom repository cannot fulfill a request."""


@dataclass(slots=True)
class Idiom:
    """Represents a reusable phrasing snippet."""

    text: str
    tags: Sequence[str]
    jurisdiction: Optional[str] = None


class IdiomRepository:
    """Loads idiom phrases from JSON and provides sampling utilities."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._index: Dict[str, List[Idiom]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            raise IdiomRepositoryError(f"Idiom database not found: {self._path}")
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise IdiomRepositoryError("Idiom database must be a JSON object keyed by tags.")
        for tag, entries in raw.items():
            if not isinstance(entries, list):
                continue
            bucket: List[Idiom] = []
            for entry in entries:
                if isinstance(entry, str):
                    bucket.append(Idiom(text=entry, tags=(tag,)))
                elif isinstance(entry, dict) and "text" in entry:
                    tags = tuple(entry.get("tags", [tag]))
                    bucket.append(
                        Idiom(
                            text=str(entry["text"]),
                            tags=tags,
                            jurisdiction=entry.get("jurisdiction"),
                        )
                    )
            if bucket:
                self._index.setdefault(tag, []).extend(bucket)

    def tags(self) -> List[str]:
        """Return all known idiom tags."""
        return sorted(self._index)

    def pick(self, tags: Iterable[str], seed: Optional[int] = None) -> Idiom:
        """Return a random idiom matching any of the requested tags."""
        tag_list = list(dict.fromkeys(tag.strip() for tag in tags if tag))
        if not tag_list:
            raise IdiomRepositoryError("No tags supplied for idiom selection.")
        population: List[Idiom] = []
        for tag in tag_list:
            population.extend(self._index.get(tag, []))
        if not population:
            raise IdiomRepositoryError(f"No idioms available for tags: {', '.join(tag_list)}")
        rng = random.Random(seed)
        return rng.choice(population)

    def pick_many(self, tags: Iterable[str], count: int, seed: Optional[int] = None) -> List[Idiom]:
        """Return multiple idioms with replacement."""
        return [self.pick(tags, seed=None if seed is None else seed + i) for i in range(count)]


class IdiomSelector:
    """Provides helper functions for agents to request idioms."""

    def __init__(self, repository: IdiomRepository) -> None:
        self._repository = repository

    def render(self, tags: Sequence[str], fallback: str = "") -> str:
        """Return idiom text if available, otherwise fallback text."""
        try:
            idiom = self._repository.pick(tags)
            return idiom.text
        except IdiomRepositoryError:
            return fallback

    def render_many(self, tags: Sequence[str], count: int = 2) -> List[str]:
        """Return several idioms for variety."""
        try:
            return [idiom.text for idiom in self._repository.pick_many(tags, count=count)]
        except IdiomRepositoryError:
            return []


__all__ = ["IdiomRepository", "IdiomSelector", "IdiomRepositoryError", "Idiom"]
