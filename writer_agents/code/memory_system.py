"""Core memory system for agent episodic memory.

Provides vector-based storage and retrieval of agent memories for enhanced
task-specific intelligence and pattern recognition.
"""

from __future__ import annotations

import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from .embeddings import EmbeddingService
except ImportError:
    from embeddings import EmbeddingService


@dataclass
class AgentMemory:
    """Single memory entry for an agent."""

    agent_type: str
    memory_id: str # UUID
    summary: str # Human-readable lesson
    context: Dict[str, Any] # Original data (job result, error, etc.)
    embedding: Optional[np.ndarray] = None # 768-dim vector
    source: str = "unknown" # "job_db" or "artifact" or "manual"
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0 # Set during retrieval


class MemoryStore:
    """Vector store for agent memories with similarity search."""

    def __init__(
        self,
        storage_path: Path = Path("memory_snapshots"),
        use_local_embeddings: bool = True, # NEW
        embedding_model: str = "all-MiniLM-L6-v2" # NEW
    ):
        """Initialize memory store.

        Args:
            storage_path: Directory to store memory snapshots
            use_local_embeddings: Whether to use local sentence-transformers
            embedding_model: Name of local model to use
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.embeddings_service = EmbeddingService(
            use_local=use_local_embeddings, # NEW
            model_name=embedding_model # NEW
        )
        self.memories: Dict[str, List[AgentMemory]] = {} # agent_type -> memories
        self._load()

    def add(self, memory: AgentMemory) -> None:
        """Add single memory with embedding.

        Args:
            memory: Memory to add
        """
        # Generate embedding if not present
        if memory.embedding is None:
            memory.embedding = self.embeddings_service.embed(memory.summary)

        # Add to agent-specific collection
        agent_memories = self.memories.setdefault(memory.agent_type, [])
        agent_memories.append(memory)

    def add_batch(self, memories: List[AgentMemory]) -> None:
        """Batch add with progress logging.

        Args:
            memories: List of memories to add
        """
        print(f"Adding {len(memories)} memories...")

        for i, mem in enumerate(memories, 1):
            self.add(mem)
            if i % 50 == 0:
                print(f" Processed {i}/{len(memories)} memories")

        self.save()
        print(f"[ok] Added {len(memories)} memories to store")

    def retrieve(
        self,
        agent_type: str,
        query: Optional[str] = None,
        k: int = 5
    ) -> List[AgentMemory]:
        """Retrieve top-k relevant memories for agent type.

        Args:
            agent_type: Type of agent to retrieve memories for
            query: Optional query string for semantic search
            k: Number of memories to return

        Returns:
            List of relevant memories, sorted by relevance
        """
        agent_memories = self.memories.get(agent_type, [])

        if not agent_memories:
            return []

        if query is None:
            # No specific query - return most recent successful ones
            successful = [
                m for m in agent_memories
                if "success" in m.summary.lower() or "completed" in m.summary.lower()
            ]
            return sorted(successful, key=lambda m: m.timestamp, reverse=True)[:k]

        # Embed query and compute similarity
        query_embedding = self.embeddings_service.embed(query)

        for mem in agent_memories:
            if mem.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, mem.embedding)
                mem.relevance_score = similarity

        # Return top-k by relevance
        return sorted(agent_memories, key=lambda m: m.relevance_score, reverse=True)[:k]

    def get_agent_stats(self, agent_type: str) -> Dict[str, Any]:
        """Get statistics for a specific agent type.

        Args:
            agent_type: Agent type to get stats for

        Returns:
            Dictionary with memory statistics
        """
        agent_memories = self.memories.get(agent_type, [])

        if not agent_memories:
            return {
                'total_memories': 0,
                'sources': {},
                'date_range': None
            }

        # Count by source
        sources = {}
        for mem in agent_memories:
            sources[mem.source] = sources.get(mem.source, 0) + 1

        # Date range
        timestamps = [mem.timestamp for mem in agent_memories]
        date_range = {
            'earliest': min(timestamps),
            'latest': max(timestamps)
        }

        return {
            'total_memories': len(agent_memories),
            'sources': sources,
            'date_range': date_range
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all agents.

        Returns:
            Dictionary with system-wide memory statistics
        """
        total_memories = sum(len(mems) for mems in self.memories.values())

        # Count by source across all agents
        all_sources = {}
        for agent_memories in self.memories.values():
            for mem in agent_memories:
                all_sources[mem.source] = all_sources.get(mem.source, 0) + 1

        return {
            'total_agents': len(self.memories),
            'total_memories': total_memories,
            'sources': all_sources,
            'agent_types': list(self.memories.keys())
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Handle zero vectors
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(a, b) / (norm_a * norm_b)

    def save(self) -> None:
        """Persist memories to disk."""
        store_file = self.storage_path / "vector_store.pkl"

        with open(store_file, "wb") as f:
            pickle.dump(self.memories, f)

        # Also save metadata
        metadata = {
            'total_memories': sum(len(mems) for mems in self.memories.values()),
            'agent_count': len(self.memories),
            'last_updated': datetime.now().isoformat(),
            'agent_stats': {
                agent_type: self.get_agent_stats(agent_type)
                for agent_type in self.memories.keys()
            }
        }

        metadata_file = self.storage_path / "system_meta.json"
        import json
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)

    def _load(self) -> None:
        """Load memories from disk."""
        store_file = self.storage_path / "vector_store.pkl"

        if store_file.exists():
            with open(store_file, "rb") as f:
                self.memories = pickle.load(f)
        else:
            self.memories = {}

    def clear(self) -> None:
        """Clear all memories (for testing)."""
        self.memories = {}
        self.save()

    def __len__(self) -> int:
        """Return total number of memories."""
        return sum(len(mems) for mems in self.memories.values())


class MemoryRetriever:
    """High-level interface for memory retrieval."""

    def __init__(self, memory_store: MemoryStore):
        """Initialize retriever.

        Args:
            memory_store: Memory store to query
        """
        self.store = memory_store

    def get_context_for_agent(
        self,
        agent_type: str,
        task_description: Optional[str] = None,
        k: int = 5
    ) -> List[str]:
        """Get contextual memories for an agent.

        Args:
            agent_type: Type of agent
            task_description: Optional description of current task
            k: Number of memories to retrieve

        Returns:
            List of memory summaries
        """
        query = task_description or f"successful execution of {agent_type}"
        memories = self.store.retrieve(agent_type, query, k)

        return [mem.summary for mem in memories]

    def get_success_patterns(self, agent_type: str, k: int = 3) -> List[str]:
        """Get successful execution patterns for an agent.

        Args:
            agent_type: Type of agent
            k: Number of patterns to retrieve

        Returns:
            List of success pattern summaries
        """
        memories = self.store.retrieve(
            agent_type,
            query="successful efficient execution",
            k=k
        )

        return [mem.summary for mem in memories]

    def get_error_patterns(self, agent_type: str, k: int = 2) -> List[str]:
        """Get common error patterns for an agent.

        Args:
            agent_type: Type of agent
            k: Number of patterns to retrieve

        Returns:
            List of error pattern summaries
        """
        memories = self.store.retrieve(
            agent_type,
            query="error failure common issue",
            k=k
        )

        return [mem.summary for mem in memories]

