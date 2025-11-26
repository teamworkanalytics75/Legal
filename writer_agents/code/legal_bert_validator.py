"""Legal-BERT based quality scoring for legal documents."""

import re
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import os
import torch

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality score for a legal document section."""
    overall_score: float  # 0-1
    legal_terminology_score: float  # 0-1
    structure_score: float  # 0-1
    coherence_score: float  # 0-1
    citation_score: float  # 0-1
    details: Dict[str, Any]


class LegalBERTValidator:
    """Uses Legal-BERT (CaseHOLD variant) for quality scoring legal documents."""

    def __init__(
        self,
        model_name: str = "zlucia/casehold",  # CaseHOLD-LegalBERT - best for case law
        cache_dir: str = "./models_cache",
        device: Optional[str] = None
    ):
        """
        Initialize Legal-BERT validator.

        Args:
            model_name: HuggingFace model name for Legal-BERT
            cache_dir: Directory for model cache
            device: Device to run model on (None = auto-detect)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.tokenizer = None
        self._load_model()

        # Reference embeddings for quality comparison (lazy loaded)
        self._reference_embeddings = None

    def _load_model(self) -> None:
        """Load Legal-BERT model and tokenizer."""
        try:
            # Gate network model loads behind env var
            allow_network = os.environ.get("MATRIX_ENABLE_NETWORK_MODELS", "").strip().lower() in {"1", "true", "yes", "on"}
            if not allow_network:
                logger.warning("MATRIX_ENABLE_NETWORK_MODELS disabled; skipping Legal-BERT download and using rule-based fallback")
                self.model = None
                self.tokenizer = None
                return
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading Legal-BERT model: {self.model_name}")
            logger.info(f"Device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info("Legal-BERT model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Legal-BERT model: {e}")
            logger.warning("Quality scoring will use rule-based fallback")
            self.model = None
            self.tokenizer = None

    def score_document(self, document: str, reference_texts: Optional[List[str]] = None) -> QualityScore:
        """
        Score a legal document for quality using Legal-BERT embeddings.

        Args:
            document: Document text to score
            reference_texts: Optional list of high-quality reference documents for comparison

        Returns:
            QualityScore with detailed scoring
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("Legal-BERT not available, using rule-based scoring")
            return self._rule_based_scoring(document)

        try:
            # Generate embedding for document
            doc_embedding = self._encode_text(document)

            # Score different aspects
            legal_terminology_score = self._score_legal_terminology(document)
            structure_score = self._score_structure(document)
            coherence_score = self._score_coherence(document, doc_embedding)
            citation_score = self._score_citations(document)

            # Compare with reference texts if provided
            reference_score = 0.0
            if reference_texts:
                reference_score = self._compare_with_references(doc_embedding, reference_texts)

            # Calculate overall score (weighted average)
            overall_score = (
                legal_terminology_score * 0.30 +
                structure_score * 0.25 +
                coherence_score * 0.25 +
                citation_score * 0.20
            )

            # Boost if similar to high-quality references
            if reference_texts and reference_score > 0.8:
                overall_score = min(1.0, overall_score * 1.1)

            return QualityScore(
                overall_score=float(overall_score),
                legal_terminology_score=float(legal_terminology_score),
                structure_score=float(structure_score),
                coherence_score=float(coherence_score),
                citation_score=float(citation_score),
                details={
                    "reference_similarity": float(reference_score) if reference_texts else None,
                    "embedding_dim": len(doc_embedding),
                    "word_count": len(document.split()),
                    "char_count": len(document)
                }
            )

        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            return self._rule_based_scoring(document)

    def _encode_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Encode text into embedding using Legal-BERT.

        Args:
            text: Text to encode
            max_length: Maximum token length

        Returns:
            Normalized embedding vector
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _score_legal_terminology(self, text: str) -> float:
        """Score use of legal terminology (0-1)."""
        # Legal terminology that should appear in good legal documents
        legal_terms = [
            "court", "plaintiff", "defendant", "motion", "statute", "precedent",
            "case law", "jurisdiction", "discovery", "seal", "pseudonym",
            "privacy", "harm", "balancing", "test", "interest", "public interest",
            "private", "constitutional", "right", "due process", "first amendment",
            "section 1782", "federal rules", "civil procedure", "evidence",
            "confidential", "sensitive", "disclosure", "redaction"
        ]

        text_lower = text.lower()
        found_terms = sum(1 for term in legal_terms if term in text_lower)

        # Score based on percentage of terms found
        max_score = len(legal_terms)
        score = min(1.0, found_terms / (max_score * 0.3))  # 30% of terms = perfect score

        return float(score)

    def _score_structure(self, text: str) -> float:
        """Score document structure and formatting (0-1)."""
        score = 0.0

        # Check for proper headings/sections
        has_headings = bool(re.search(r'^[IVX]+\.|^###|^##', text, re.MULTILINE))
        if has_headings:
            score += 0.3

        # Check for proper paragraphs (line breaks)
        paragraphs = text.split('\n\n')
        if 3 <= len(paragraphs) <= 15:
            score += 0.3
        elif len(paragraphs) > 1:
            score += 0.15

        # Check for proper sentence structure
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if 10 <= avg_sentence_length <= 25:
            score += 0.2
        else:
            score += 0.1

        # Check for proper formatting (has punctuation, capitalization)
        has_punctuation = any(p in text for p in ['.', ';', ',', ':'])
        has_capitalization = any(c.isupper() for c in text[:100])
        if has_punctuation and has_capitalization:
            score += 0.2
        else:
            score += 0.1

        return min(1.0, score)

    def _score_coherence(self, text: str, embedding: Optional[np.ndarray] = None) -> float:
        """
        Score text coherence (0-1).

        Uses embedding similarity between adjacent paragraphs to measure coherence.
        """
        if embedding is None:
            # Fallback: simple coherence check
            return self._simple_coherence_check(text)

        try:
            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

            if len(paragraphs) < 2:
                return 0.5  # Can't measure coherence with single paragraph

            # Encode each paragraph
            para_embeddings = []
            for para in paragraphs[:10]:  # Limit to first 10 paragraphs
                try:
                    emb = self._encode_text(para)
                    para_embeddings.append(emb)
                except Exception:
                    continue

            if len(para_embeddings) < 2:
                return self._simple_coherence_check(text)

            # Calculate pairwise similarities between adjacent paragraphs
            similarities = []
            for i in range(len(para_embeddings) - 1):
                sim = np.dot(para_embeddings[i], para_embeddings[i + 1])
                similarities.append(sim)

            # Higher average similarity = better coherence
            avg_similarity = np.mean(similarities) if similarities else 0.0
            # Normalize: similarity ranges from -1 to 1, we want 0-1
            coherence_score = max(0.0, min(1.0, (avg_similarity + 1.0) / 2.0))

            return float(coherence_score)

        except Exception as e:
            logger.warning(f"Coherence scoring failed: {e}")
            return self._simple_coherence_check(text)

    def _simple_coherence_check(self, text: str) -> float:
        """Simple coherence check using word overlap between paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(paragraphs) < 2:
            return 0.5

        # Check transition words and topic consistency
        transition_words = ['furthermore', 'moreover', 'however', 'therefore', 'consequently',
                          'additionally', 'specifically', 'in particular', 'for example']
        has_transitions = any(word in text.lower() for word in transition_words)

        # Check for repeated key terms (indicates topic consistency)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # High frequency of certain terms indicates coherence
        max_freq = max(word_freq.values()) if word_freq else 0
        has_repeated_terms = max_freq >= 3

        score = 0.5
        if has_transitions:
            score += 0.25
        if has_repeated_terms:
            score += 0.25

        return min(1.0, score)

    def _score_citations(self, text: str) -> float:
        """Score citation quality and presence (0-1)."""
        import re

        # Citation patterns
        citation_patterns = [
            r'\d+\s+U\.S\.C\.\s*§?\s*\d+',  # USC citations
            r'\d+\s+F\.\d+[dD]\s+\d+',  # Federal Reporter
            r'\d+\s+S\.Ct\.\s+\d+',  # Supreme Court
            r'\d+\s+F\.\s*Supp\.\s*\d+[dD]?\s+\d+',  # Federal Supplement
            r'\d+\s+F\.\s*R\.\s*Civ\.\s*P\.',  # Federal Rules
            r'Fed\.\s*R\.\s*Civ\.\s*P\.',  # Federal Rules
        ]

        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)

        # Score based on citation count and variety
        unique_citations = set(citations)

        if len(unique_citations) >= 3:
            return 1.0
        elif len(unique_citations) == 2:
            return 0.75
        elif len(unique_citations) == 1:
            return 0.5
        elif len(citations) > 0:
            return 0.3
        else:
            return 0.1

    def _compare_with_references(
        self,
        embedding: np.ndarray,
        reference_texts: List[str]
    ) -> float:
        """
        Compare document embedding with reference high-quality documents.

        Args:
            embedding: Document embedding
            reference_texts: List of reference document texts

        Returns:
            Average similarity score (0-1)
        """
        if not reference_texts:
            return 0.0

        try:
            # Encode reference texts
            ref_embeddings = []
            for ref_text in reference_texts[:5]:  # Limit to 5 references
                try:
                    ref_emb = self._encode_text(ref_text)
                    ref_embeddings.append(ref_emb)
                except Exception:
                    continue

            if not ref_embeddings:
                return 0.0

            # Calculate similarity with each reference
            similarities = []
            for ref_emb in ref_embeddings:
                sim = np.dot(embedding, ref_emb)
                similarities.append(sim)

            # Average similarity
            avg_similarity = np.mean(similarities) if similarities else 0.0
            # Normalize to 0-1
            normalized = max(0.0, min(1.0, (avg_similarity + 1.0) / 2.0))

            return float(normalized)

        except Exception as e:
            logger.warning(f"Reference comparison failed: {e}")
            return 0.0

    def _rule_based_scoring(self, document: str) -> QualityScore:
        """Fallback rule-based scoring when Legal-BERT is not available."""
        legal_terminology_score = self._score_legal_terminology(document)
        structure_score = self._score_structure(document)
        coherence_score = self._simple_coherence_check(document)
        citation_score = self._score_citations(document)

        overall_score = (
            legal_terminology_score * 0.30 +
            structure_score * 0.25 +
            coherence_score * 0.25 +
            citation_score * 0.20
        )

        return QualityScore(
            overall_score=float(overall_score),
            legal_terminology_score=float(legal_terminology_score),
            structure_score=float(structure_score),
            coherence_score=float(coherence_score),
            citation_score=float(citation_score),
            details={
                "method": "rule_based",
                "word_count": len(document.split()),
                "char_count": len(document)
            }
        )


__all__ = ["LegalBERTValidator", "QualityScore"]
