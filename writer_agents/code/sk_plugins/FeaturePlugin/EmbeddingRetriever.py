"""
Embedding Retriever - Handles semantic search over personal corpus embeddings.

Loads FAISS index and SQLite database to enable fast semantic retrieval
of relevant corpus chunks for plugins.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import os
import re
import sqlite3
import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, will use SQLite-only search")

# Try to import transformers for query encoding (gated by env)
try:
    _NETWORK_MODELS_ENABLED = os.environ.get("MATRIX_ENABLE_NETWORK_MODELS", "").strip().lower() in {"1","true","yes","on"}
    if _NETWORK_MODELS_ENABLED:
        from transformers import AutoTokenizer, AutoModel
        import torch
        TRANSFORMERS_AVAILABLE = True
    else:
        TRANSFORMERS_AVAILABLE = False
        logger.info("MATRIX_ENABLE_NETWORK_MODELS disabled; using simple text matching")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, will use simple text matching")


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk from the corpus."""
    text: str
    score: float
    source_file: str
    chunk_index: int
    metadata: Dict[str, Any]


class EmbeddingRetriever:
    """Retrieves relevant chunks from personal corpus using semantic search."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        faiss_path: Optional[Path] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize EmbeddingRetriever.

        Args:
            db_path: Path to SQLite database with embeddings
            faiss_path: Path to FAISS index file
            model_name: BERT model name or local path for query encoding. 
                       If None, checks env var EMBEDDING_MODEL_PATH, then config, then defaults to nlpaueb/legal-bert-base-uncased
        """
        if db_path is None:
            db_path = Path("case_law_data/results/personal_corpus_embeddings.db")
        if faiss_path is None:
            faiss_path = Path("case_law_data/results/personal_corpus_embeddings.faiss")

        self.db_path = Path(db_path)
        self.faiss_path = Path(faiss_path)
        
        # Determine model name/path: env var > parameter > config > default
        if model_name is None:
            model_name = os.environ.get("EMBEDDING_MODEL_PATH") or os.environ.get("LEGAL_BERT_MODEL_PATH")
        
        if model_name is None:
            # Try to load from config file
            try:
                config_path = Path("case_law_data/config/citation_pipeline_config.json")
                if config_path.exists():
                    import json
                    with open(config_path) as f:
                        config = json.load(f)
                        if config.get("embedding", {}).get("model_name"):
                            model_name = config["embedding"]["model_name"]
            except Exception:
                pass
        
        if model_name is None:
            model_name = "nlpaueb/legal-bert-base-uncased"
        
        self.model_name = model_name
        self.model_path = Path(model_name) if Path(model_name).exists() else None

        self._faiss_index = None
        self._embeddings_cache: Dict[int, np.ndarray] = {}
        self._chunks_cache: Dict[int, Dict[str, Any]] = {}
        self._model = None
        self._tokenizer = None
        self._device = None
        self._embedding_dim = 768  # Legal-BERT base dimension
        self._fallback_reason: Optional[str] = None
        self._warned_about_keyword_fallback = False
        self._semantic_enabled = False

        self._load_resources()

    def _load_resources(self) -> None:
        """Load resources with a timeout to prevent hangs."""
        timeout_seconds = float(os.environ.get("EMBEDDING_RETRIEVER_LOAD_TIMEOUT", "5"))
        timeout_seconds = max(timeout_seconds, 1.0)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._load_resources_inner)
            try:
                future.result(timeout=timeout_seconds)
            except FuturesTimeoutError:
                future.cancel()
                self._faiss_index = None
                self._model = None
                self._tokenizer = None
                self._semantic_enabled = False
                self._fallback_reason = (
                    f"Embedding resource initialization timed out after {timeout_seconds:.1f}s; "
                    "falling back to keyword search."
                )
                logger.error(self._fallback_reason)
            except Exception as exc:  # pragma: no cover - defensive guard
                self._faiss_index = None
                self._model = None
                self._tokenizer = None
                self._semantic_enabled = False
                self._fallback_reason = str(exc)
                logger.error("Embedding resource initialization failed: %s", exc)

    def _load_resources_inner(self) -> None:
        """Load FAISS index, model, and prepare for queries."""
        # Load FAISS index if available
        if FAISS_AVAILABLE and self.faiss_path.exists():
            try:
                self._faiss_index = faiss.read_index(str(self.faiss_path))
                logger.info(f"Loaded FAISS index from {self.faiss_path}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                self._faiss_index = None
        else:
            logger.info("FAISS index not available, will use SQLite-only search")

        # Load BERT model for query encoding
        if TRANSFORMERS_AVAILABLE:
            try:
                hf_offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
                local_files_only = hf_offline or not _NETWORK_MODELS_ENABLED
                
                # Check if model_name is a local path
                model_path = Path(self.model_name)
                if model_path.exists() and model_path.is_dir():
                    # Local directory path - load directly
                    logger.info(f"Loading BERT model from local path: {model_path}")
                    self._tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
                    self._model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
                else:
                    # HuggingFace model name - check cache first
                    if local_files_only:
                        logger.info(
                            "Loading BERT model in offline mode (local_files_only=True): %s",
                            self.model_name,
                        )
                    
                    # Try to use HuggingFace cache directory explicitly
                    # Check for local models_cache first, then env vars, then default
                    repo_root = Path(__file__).resolve().parents[4]
                    local_models_cache = repo_root / "models_cache"
                    cache_dir = (
                        str(local_models_cache) if local_models_cache.exists() else
                        os.environ.get("HF_HOME") or 
                        os.environ.get("TRANSFORMERS_CACHE") or 
                        os.path.expanduser("~/.cache/huggingface")
                    )
                    
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        local_files_only=local_files_only,
                        cache_dir=cache_dir,
                    )
                    self._model = AutoModel.from_pretrained(
                        self.model_name,
                        local_files_only=local_files_only,
                        cache_dir=cache_dir,
                    )
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._model.to(self._device)
                self._model.eval()
                logger.info(f"Loaded BERT model: {self.model_name} on {self._device}")
                self._semantic_enabled = True
            except Exception as e:
                logger.warning(f"Failed to load BERT model: {e}")
                self._model = None
                self._tokenizer = None
                self._fallback_reason = str(e)
        else:
            if not _NETWORK_MODELS_ENABLED:
                self._fallback_reason = (
                    "MATRIX_ENABLE_NETWORK_MODELS is disabled; semantic embeddings require this flag."
                )
            else:
                self._fallback_reason = "Transformers not available in this environment."

    def _encode_query(self, query: str) -> Optional[np.ndarray]:
        """
        Encode query text into embedding vector.

        Args:
            query: Query text

        Returns:
            Embedding vector or None if encoding fails
        """
        if not TRANSFORMERS_AVAILABLE or self._model is None:
            return None

        try:
            with torch.no_grad():
                inputs = self._tokenizer(
                    query,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self._device)

                outputs = self._model(**inputs)
                # Use [CLS] token embedding (mean pooling for better results)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Query encoding failed: {e}")
            return None

    def _load_chunk_from_db(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Load chunk metadata and text from SQLite database."""
        if chunk_id in self._chunks_cache:
            return self._chunks_cache[chunk_id]

        if not self.db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT doc_filename, chunk_index, chunk_text, start_word_idx, end_word_idx
                FROM document_chunks
                WHERE chunk_id = ?
            """, (chunk_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                chunk_data = {
                    "doc_filename": row[0],
                    "chunk_index": row[1],
                    "chunk_text": row[2],
                    "start_word_idx": row[3],
                    "end_word_idx": row[4]
                }
                self._chunks_cache[chunk_id] = chunk_data
                return chunk_data
        except Exception as e:
            logger.error(f"Failed to load chunk {chunk_id} from DB: {e}")

        return None

    def _load_embedding_from_db(self, chunk_id: int) -> Optional[np.ndarray]:
        """Load embedding vector from SQLite database."""
        if chunk_id in self._embeddings_cache:
            return self._embeddings_cache[chunk_id]

        if not self.db_path.exists():
            return None

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT embedding_blob
                FROM document_chunks
                WHERE chunk_id = ?
            """, (chunk_id,))

            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                embedding = np.frombuffer(row[0], dtype=np.float32)
                self._embeddings_cache[chunk_id] = embedding
                return embedding
        except Exception as e:
            logger.error(f"Failed to load embedding for chunk {chunk_id}: {e}")

        return None

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.3
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks using semantic search.

        Args:
            query: Search query
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of RetrievedChunk objects
        """
        if not self.db_path.exists():
            logger.warning(f"Embedding database not found: {self.db_path}")
            return []

        # Encode query (semantic) or fall back to keyword retrieval
        query_embedding = self._encode_query(query)
        if query_embedding is None:
            self._warn_keyword_fallback()
            return self._keyword_search(query, top_k)

        results = []

        # Use FAISS if available (fast)
        if self._faiss_index is not None:
            try:
                # Normalize query embedding
                query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                query_norm = query_norm.reshape(1, -1).astype(np.float32)

                # Search FAISS index
                scores, indices = self._faiss_index.search(query_norm, top_k)

                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:  # FAISS returns -1 for invalid indices
                        continue

                    chunk_id = int(idx) + 1  # FAISS is 0-indexed, DB is 1-indexed
                    chunk_data = self._load_chunk_from_db(chunk_id)

                    if chunk_data and score >= min_score:
                        results.append(RetrievedChunk(
                            text=chunk_data["chunk_text"],
                            score=float(score),
                            source_file=chunk_data["doc_filename"],
                            chunk_index=chunk_data["chunk_index"],
                            metadata={
                                "start_word_idx": chunk_data["start_word_idx"],
                                "end_word_idx": chunk_data["end_word_idx"]
                            }
                        ))
            except Exception as e:
                logger.error(f"FAISS search failed: {e}, falling back to SQLite")

        # Fallback to SQLite search if FAISS failed or unavailable
        if not results:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()

                # Get all chunks with embeddings
                cursor.execute("""
                    SELECT chunk_id, embedding_blob
                    FROM document_chunks
                    WHERE embedding_blob IS NOT NULL
                """)

                chunks_with_embeddings = cursor.fetchall()
                conn.close()

                if not chunks_with_embeddings:
                    return []

                # Compute similarities
                similarities = []
                query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

                for chunk_id, embedding_blob in chunks_with_embeddings:
                    chunk_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    chunk_norm = chunk_embedding / (np.linalg.norm(chunk_embedding) + 1e-8)
                    similarity = np.dot(query_norm, chunk_norm)
                    similarities.append((chunk_id, float(similarity)))

                # Sort by similarity and get top-k
                similarities.sort(key=lambda x: x[1], reverse=True)

                for chunk_id, score in similarities[:top_k]:
                    if score < min_score:
                        continue

                    chunk_data = self._load_chunk_from_db(chunk_id)
                    if chunk_data:
                        results.append(RetrievedChunk(
                            text=chunk_data["chunk_text"],
                            score=score,
                            source_file=chunk_data["doc_filename"],
                            chunk_index=chunk_data["chunk_index"],
                            metadata={
                                "start_word_idx": chunk_data["start_word_idx"],
                                "end_word_idx": chunk_data["end_word_idx"]
                            }
                        ))

            except Exception as e:
                logger.error(f"SQLite search failed: {e}")

        return results

    def is_available(self) -> bool:
        """Check if retriever is available (has DB and model)."""
        if not self.db_path.exists():
            return False
        return self._semantic_enabled or True  # keyword fallback is always possible

    def _keyword_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """
        Lightweight keyword-based retrieval used when embeddings are unavailable.
        """
        if not query.strip():
            return []

        tokens = re.findall(r"[a-z0-9]+", query.lower())
        tokens = [tok for tok in tokens if len(tok) > 2]
        if not tokens:
            tokens = query.lower().split()
        tokens = list(dict.fromkeys(tokens))[:5]

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            if tokens:
                like_clauses = " OR ".join("chunk_text LIKE ?" for _ in tokens)
                params = [f"%{tok}%" for tok in tokens]
                cursor.execute(
                    f"""
                    SELECT chunk_id, chunk_text, doc_filename, chunk_index
                    FROM document_chunks
                    WHERE {like_clauses}
                    LIMIT 500
                    """,
                    params,
                )
            else:
                cursor.execute(
                    """
                    SELECT chunk_id, chunk_text, doc_filename, chunk_index
                    FROM document_chunks
                    LIMIT 200
                    """
                )
            rows = cursor.fetchall()
            conn.close()
        except Exception as exc:
            logger.error(f"Keyword fallback failed: {exc}")
            return []

        scored: List[Tuple[int, float, Dict[str, Any]]] = []
        query_tokens = tokens or query.lower().split()

        for chunk_id, chunk_text, filename, chunk_index in rows:
            text_lower = chunk_text.lower()
            score = sum(text_lower.count(tok) for tok in query_tokens)
            if score <= 0:
                continue
            scored.append(
                (
                    chunk_id,
                    float(score),
                    {
                        "text": chunk_text,
                        "source_file": filename,
                        "chunk_index": chunk_index,
                    },
                )
            )

        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[RetrievedChunk] = []
        for chunk_id, score, data in scored[:top_k]:
            results.append(
                RetrievedChunk(
                    text=data["text"],
                    score=score,
                    source_file=data["source_file"],
                    chunk_index=data["chunk_index"],
                    metadata={},
                )
            )
        return results

    def _warn_keyword_fallback(self) -> None:
        """Log once when semantic retrieval degrades to keyword matching."""
        if self._warned_about_keyword_fallback:
            return
        reason = self._fallback_reason or "Unable to initialize encoder."
        logger.warning(
            "Semantic retrieval degraded to keyword matching: %s", reason
        )
        if not _NETWORK_MODELS_ENABLED:
            logger.warning(
                "Set MATRIX_ENABLE_NETWORK_MODELS=true (and restart) to enable Legal-BERT query encoding."
            )
        self._warned_about_keyword_fallback = True
