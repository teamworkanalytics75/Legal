"""Embedding service for agent memory system.

Provides both local (sentence-transformers) and OpenAI embedding options
for converting text memories into vector representations.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Embedding service with local + OpenAI fallback."""

    def __init__(
        self,
        use_local: bool = True,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize embedding service.

        Args:
            use_local: Whether to use local sentence-transformers
            model_name: Name of local model to use
        """
        self.use_local = use_local
        self.model_name = model_name

        # Gate any potential network/downloads behind MATRIX_ENABLE_NETWORK_MODELS
        import os
        allow_network = os.environ.get("MATRIX_ENABLE_NETWORK_MODELS", "").strip().lower() in {"1", "true", "yes", "on"}

        if use_local:
            if allow_network:
                try:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(model_name)
                    self.mode = "local"
                    logger.info(f"Using local embedding model: {model_name}")
                except Exception as e:
                    logger.warning(f"Local sentence-transformers unavailable ({e}); falling back to network policy")
                    self.use_local = False
            else:
                # Network disabled - use offline fallback immediately
                logger.info(f"MATRIX_ENABLE_NETWORK_MODELS disabled; using offline embedder (skipping {model_name})")
                self.model = None
                self.mode = "offline"
                self.use_local = False  # Prevent retry

        if not self.use_local:
            if allow_network:
                try:
                    import openai
                    self.client = openai.Client()
                    self.mode = "openai"
                    logger.info("Using OpenAI embedding service")
                except Exception as e:
                    logger.warning(f"OpenAI embeddings unavailable ({e}); using offline fallback")
                    self.mode = "offline"
            else:
                logger.info("MATRIX_ENABLE_NETWORK_MODELS disabled; using offline embedder")
                self.mode = "offline"

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.mode == "local":
            return self.model.encode(text, convert_to_numpy=True)
        elif self.mode == "openai":
            # OpenAI text-embedding-3-small
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        else:
            # Offline deterministic vector (hash-based) as a last-resort fallback
            h = abs(hash(text))
            rng = np.random.default_rng(h % (2**32))
            vec = rng.standard_normal(384).astype(np.float32)
            vec /= (np.linalg.norm(vec) + 1e-8)
            return vec

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Batch embed for efficiency.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        if self.mode == "local":
            embeddings = []
            for batch in self._chunks(texts, batch_size):
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
            return embeddings
        elif self.mode == "openai":
            # OpenAI allows up to 2048 inputs per request
            embeddings = []
            for batch in self._chunks(texts, min(batch_size, 2048)):
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                embeddings.extend(batch_embeddings)
            return embeddings
        else:
            # Offline fallback: hash-based deterministic vectors
            return [self.embed(t) for t in texts]

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Dimension of embedding vectors
        """
        if self.mode == "local":
            # all-MiniLM-L6-v2 has 384 dimensions
            return 384
        elif self.mode == "openai":
            # text-embedding-3-small has 1536 dimensions
            return 1536
        else:
            # Offline fallback dimension matches local
            return 384

    def estimate_cost(self, num_texts: int, avg_tokens_per_text: int = 50) -> float:
        """Estimate cost for embedding texts.

        Args:
            num_texts: Number of texts to embed
            avg_tokens_per_text: Average tokens per text

        Returns:
            Estimated cost in dollars
        """
        if self.mode == "local":
            return 0.0

        total_tokens = num_texts * avg_tokens_per_text
        # OpenAI text-embedding-3-small: $0.02 per 1M tokens
        return total_tokens * 0.00002 / 1000

    def _chunks(self, lst: List[str], chunk_size: int) -> List[List[str]]:
        """Split list into chunks.

        Args:
            lst: List to chunk
            chunk_size: Size of each chunk

        Returns:
            List of chunks
        """
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]
