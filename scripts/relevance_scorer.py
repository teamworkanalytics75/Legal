"""
Relevance scoring utilities for STORM-inspired research.
Uses legal-domain BERT models to rank candidate web snippets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger(__name__)


@dataclass
class LegalBertRelevanceScorer:
    """
    Lightweight semantic similarity scorer built on top of LegalBERT.
    Produces cosine-normalised [CLS] embeddings for quick dot-product scoring.
    """

    model_name: str = "zlucia/legalbert-base-uncased"
    device: Optional[str] = None
    max_length: int = 512

    def __post_init__(self) -> None:
        self._cache: Dict[str, torch.Tensor] = {}
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        LOGGER.info("Loading LegalBERT relevance scorer (%s) on %s", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, text: str) -> torch.Tensor:
        """
        Encode text as an L2-normalised embedding (cached for reuse).
        """
        if text in self._cache:
            return self._cache[text]

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embedding = F.normalize(cls_embedding, p=2, dim=1)

        embedding = embedding[0].detach().cpu()
        self._cache[text] = embedding
        return embedding

    def score(self, query: str, document: str) -> float:
        """
        Compute cosine similarity between query and document embeddings.
        """
        query_emb = self._encode(query)
        doc_emb = self._encode(document)
        return float(torch.dot(query_emb, doc_emb).item())
