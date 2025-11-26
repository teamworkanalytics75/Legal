"""Chroma integration for Semantic Kernel Memory."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from semantic_kernel import Kernel

try:  # SK < 1.37
    from semantic_kernel.memory import MemoryStoreBase
except ImportError:  # SK >= 1.37
    from semantic_kernel.memory.memory_store_base import MemoryStoreBase

try:
    from ..sk_compat import get_memory_record
except ImportError:
    from sk_compat import get_memory_record  # type: ignore

MemoryRecord = get_memory_record()

try:
    import torch

    _EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _EMBEDDING_DEVICE = "cpu"

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_EMBEDDINGS: Dict[str, str] = {
    "case_law_legal": "nlpaueb/legal-bert-base-uncased",
    "case_law_general": "sentence-transformers/all-mpnet-base-v2",
    "harvard_legal": "nlpaueb/legal-bert-base-uncased",
    "harvard_general": "sentence-transformers/all-mpnet-base-v2",
    "ma_motions_legal": "nlpaueb/legal-bert-base-uncased",
    "ma_motions_general": "sentence-transformers/all-mpnet-base-v2",
}

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


class _HashingEmbeddingFunction:
    """Deterministic embedding fallback for fully-offline operation."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            for token in re.findall(r"\w+", text.lower()):
                idx = hash(token) % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec.tolist())
        return embeddings


class ChromaSKMemoryStore(MemoryStoreBase):
    """Chroma-backed implementation of Semantic Kernel MemoryStoreBase."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "sk_memory",
        *,
        collection_embedding_map: Optional[Dict[str, str]] = None,
        default_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        super().__init__()
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.collection_embedding_map = {
            **DEFAULT_COLLECTION_EMBEDDINGS,
            **(collection_embedding_map or {}),
        }
        self.default_embedding_model = default_embedding_model
        self.embedding_device = _EMBEDDING_DEVICE
        self._embedding_functions: Dict[str, SentenceTransformerEmbeddingFunction] = {}
        self._initialize_chroma()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _initialize_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )

            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Semantic Kernel memory store"},
                embedding_function=self._get_embedding_function(self.collection_name),
            )

            logger.info("Chroma SK Memory Store initialized: %s", self.collection_name)
        except ImportError:
            logger.error("ChromaDB not installed. Run `pip install chromadb`.")
            raise
        except Exception as exc:
            logger.error("Failed to initialize Chroma client: %s", exc)
            raise

    def _resolve_model_name(self, collection_name: str) -> str:
        if collection_name in self.collection_embedding_map:
            return self.collection_embedding_map[collection_name]
        if collection_name.endswith("_legal"):
            return "nlpaueb/legal-bert-base-uncased"
        if collection_name.endswith("_general"):
            return "sentence-transformers/all-mpnet-base-v2"
        return self.default_embedding_model

    def _get_embedding_function(self, collection_name: str) -> SentenceTransformerEmbeddingFunction:
        model_name = self._resolve_model_name(collection_name)
        if model_name not in self._embedding_functions:
            try:
                self._embedding_functions[model_name] = SentenceTransformerEmbeddingFunction(
                    model_name=model_name,
                    device=self.embedding_device,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load SentenceTransformer '%s' (%s). Using hashing fallback.",
                    model_name,
                    exc,
                )
                self._embedding_functions[model_name] = _HashingEmbeddingFunction()
        return self._embedding_functions[model_name]

    def _memory_record_to_chroma(self, record: MemoryRecord) -> Dict[str, Any]:
        metadata = {
            "id": record.id,
            "description": getattr(record, "description", ""),
            "additional_metadata": json.dumps(getattr(record, "additional_metadata", {}) or {}),
            "timestamp": record.timestamp.isoformat() if getattr(record, "timestamp", None) else None,
        }
        return {"document": record.text, "metadata": metadata, "embedding": record.embedding}

    def _chroma_to_memory_record(
        self,
        payload: Dict[str, Any],
        key: str,
        index: int = 0,
    ) -> Optional[MemoryRecord]:
        try:
            if not payload["ids"] or index >= len(payload["ids"]):
                return None

            document = payload["documents"][index] if payload["documents"] else ""
            metadata = payload["metadatas"][index] if payload["metadatas"] else {}
            embedding = payload["embeddings"][index] if payload.get("embeddings") else None

            additional_metadata: Dict[str, Any] = {}
            if metadata.get("additional_metadata"):
                try:
                    additional_metadata = json.loads(metadata["additional_metadata"])
                except json.JSONDecodeError:
                    pass

            from datetime import datetime

            timestamp = None
            if metadata.get("timestamp"):
                try:
                    timestamp = datetime.fromisoformat(metadata["timestamp"])
                except ValueError:
                    timestamp = None

            return MemoryRecord(
                id=payload["ids"][index],
                text=document,
                description=metadata.get("description", ""),
                additional_metadata=additional_metadata,
                embedding=embedding,
                timestamp=timestamp,
            )
        except Exception as exc:
            logger.error("Failed to convert Chroma result: %s", exc)
            return None

    # --------------------------------------------------------------------- #
    # MemoryStoreBase implementations (async)
    # --------------------------------------------------------------------- #
    async def create_collection(self, collection_name: str) -> None:
        embedding_fn = self._get_embedding_function(collection_name)
        self.chroma_client.create_collection(
            name=collection_name,
            metadata={"description": f"SK collection: {collection_name}"},
            embedding_function=embedding_fn,
        )
        logger.info("Created collection: %s", collection_name)

    async def get_collection(self, collection_name: str) -> Optional[Any]:
        try:
            embedding_fn = self._get_embedding_function(collection_name)
            return self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_fn,
            )
        except Exception as exc:
            logger.error("Failed to get collection %s: %s", collection_name, exc)
            return None

    async def delete_collection(self, collection_name: str) -> None:
        try:
            self.chroma_client.delete_collection(name=collection_name)
            logger.info("Deleted collection: %s", collection_name)
        except Exception as exc:
            logger.error("Failed to delete collection %s: %s", collection_name, exc)

    async def does_collection_exist(self, collection_name: str) -> bool:
        try:
            collections = self.chroma_client.list_collections()
            return any(col.name == collection_name for col in collections)
        except Exception as exc:
            logger.error("Failed to list collections: %s", exc)
            return False

    async def get_collections(self) -> List[str]:
        """Return all collection names (SK 1.37 abstract method)."""
        try:
            return [col.name for col in self.chroma_client.list_collections()]
        except Exception as exc:
            logger.error("Failed to read collections: %s", exc)
            return []

    async def upsert(self, collection_name: str, record: MemoryRecord) -> str:
        collection = await self.get_collection(collection_name)
        if collection is None:
            await self.create_collection(collection_name)
            collection = await self.get_collection(collection_name)

        payload = self._memory_record_to_chroma(record)
        collection.upsert(
            ids=[record.id],
            documents=[payload["document"]],
            metadatas=[payload["metadata"]],
            embeddings=[payload["embedding"]] if payload["embedding"] else None,
        )
        logger.info("Upserted record %s -> %s", record.id, collection_name)
        return record.id

    async def upsert_batch(self, collection_name: str, records: List[MemoryRecord]) -> List[str]:
        collection = await self.get_collection(collection_name)
        if collection is None:
            await self.create_collection(collection_name)
            collection = await self.get_collection(collection_name)

        payloads = [self._memory_record_to_chroma(record) for record in records]
        collection.upsert(
            ids=[record.id for record in records],
            documents=[payload["document"] for payload in payloads],
            metadatas=[payload["metadata"] for payload in payloads],
            embeddings=[payload["embedding"] for payload in payloads]
            if any(payload["embedding"] for payload in payloads)
            else None,
        )
        logger.info("Upserted %d records -> %s", len(records), collection_name)
        return [record.id for record in records]

    async def get(self, collection_name: str, key: str, with_embedding: bool = False) -> Optional[MemoryRecord]:
        collection = await self.get_collection(collection_name)
        if not collection:
            return None

        include = ["documents", "metadatas"]
        if with_embedding:
            include.append("embeddings")

        result = collection.get(ids=[key], include=include)
        if not result["ids"]:
            return None
        return self._chroma_to_memory_record(result, key)

    async def get_batch(
        self,
        collection_name: str,
        keys: List[str],
        with_embedding: bool = False,
    ) -> List[MemoryRecord]:
        collection = await self.get_collection(collection_name)
        if not collection:
            return []

        include = ["documents", "metadatas"]
        if with_embedding:
            include.append("embeddings")

        result = collection.get(ids=keys, include=include)
        records: List[MemoryRecord] = []
        for idx, key in enumerate(result["ids"]):
            record = self._chroma_to_memory_record(result, key, idx)
            if record:
                records.append(record)
        return records

    async def remove(self, collection_name: str, key: str) -> None:
        collection = await self.get_collection(collection_name)
        if not collection:
            return
        collection.delete(ids=[key])
        logger.info("Removed record %s from %s", key, collection_name)

    async def remove_batch(self, collection_name: str, keys: List[str]) -> None:
        collection = await self.get_collection(collection_name)
        if not collection:
            return
        collection.delete(ids=keys)
        logger.info("Removed %d records from %s", len(keys), collection_name)

    async def get_nearest_matches(
        self,
        collection_name: str,
        embedding: List[float],
        limit: int = 1,
        min_relevance_score: float = 0.0,
        with_embeddings: bool = False,
    ) -> List[Tuple[MemoryRecord, float]]:
        collection = await self.get_collection(collection_name)
        if not collection:
            return []

        result = collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["documents", "metadatas", "embeddings"] if with_embeddings else ["documents", "metadatas"],
        )

        matches: List[Tuple[MemoryRecord, float]] = []
        for idx, doc_id in enumerate(result["ids"][0]):
            distance = result["distances"][0][idx]
            if distance >= min_relevance_score:
                record = self._chroma_to_memory_record(result, doc_id, idx)
                if record:
                    matches.append((record, 1.0 - distance))
        return matches

    async def get_nearest_match(
        self,
        collection_name: str,
        embedding: List[float],
        min_relevance_score: float = 0.0,
        with_embedding: bool = False,
    ) -> Optional[Tuple[MemoryRecord, float]]:
        matches = await self.get_nearest_matches(
            collection_name=collection_name,
            embedding=embedding,
            limit=1,
            min_relevance_score=min_relevance_score,
            with_embeddings=with_embedding,
        )
        return matches[0] if matches else None

    async def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 10,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        collection = await self.get_collection(collection_name)
        if not collection or not query_text.strip():
            return {}

        include = ["documents", "distances"]
        if include_metadata:
            include.append("metadatas")

        def _query():
            return collection.query(query_texts=[query_text], n_results=n_results, include=include)

        try:
            return await asyncio.to_thread(_query)
        except RuntimeError:
            return collection.query(query_texts=[query_text], n_results=n_results, include=include)


class ChromaSKIntegration:
    """Convenience wrapper that wires Chroma memory into a SK kernel."""

    def __init__(
        self,
        kernel: Kernel,
        chroma_persist_directory: str,
        *,
        collection_embedding_map: Optional[Dict[str, str]] = None,
        default_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.kernel = kernel
        self.memory_store = ChromaSKMemoryStore(
            chroma_persist_directory,
            collection_name="sk_memory",
            collection_embedding_map=collection_embedding_map,
            default_embedding_model=default_embedding_model,
        )
        self.kernel.register_memory_store(self.memory_store)
        logger.info("Chroma memory store registered with SK kernel")

    async def connect_existing_collections(self, mappings: Dict[str, str]) -> None:
        for chroma_name, sk_name in mappings.items():
            if await self.memory_store.does_collection_exist(chroma_name):
                logger.info("Connected Chroma collection %s -> %s", chroma_name, sk_name)
            else:
                logger.warning("Chroma collection %s not found", chroma_name)

    async def query_memory_for_context(
        self,
        query: str,
        collection_name: str = "sk_memory",
        limit: int = 5,
    ) -> List[str]:
        result = await self.memory_store.query(collection_name, query, n_results=limit)
        if not result:
            return []
        documents = result.get("documents", [[]])
        return documents[0] if documents else []


__all__ = ["ChromaSKMemoryStore", "ChromaSKIntegration"]
