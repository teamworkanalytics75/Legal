"""Sophisticated section-by-section merging of multi-model drafts."""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a section of a legal document."""
    title: str
    content: str
    start_pos: int
    end_pos: int
    section_type: str  # e.g., "introduction", "factual_background", "legal_argument", "conclusion"


@dataclass
class SectionComparison:
    """Comparison result between two sections."""
    primary_section: Section
    secondary_section: Section
    quality_score: float  # 0-1
    semantic_similarity: float  # 0-1
    combined_score: float  # Weighted combination
    selected_section: Section  # Which one was selected
    selection_reason: str


class SectionMerger:
    """Merges drafts from multiple models by comparing and selecting best sections."""

    def __init__(
        self,
        quality_weight: float = 0.70,
        semantic_weight: float = 0.30,
        quality_threshold: float = 0.70,
        semantic_similarity_threshold: float = 0.80
    ):
        """
        Initialize section merger.

        Args:
            quality_weight: Weight for quality score in selection (0-1)
            semantic_weight: Weight for semantic similarity in selection (0-1)
            quality_threshold: Minimum quality score to accept section
            semantic_similarity_threshold: Minimum similarity for comparison
        """
        self.quality_weight = quality_weight
        self.semantic_weight = semantic_weight
        self.quality_threshold = quality_threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold

        # Legal-BERT embedding model (lazy loaded)
        self._embedding_model = None
        self._embedding_tokenizer = None

    def _load_embedding_model(self):
        """Lazy load Legal-BERT model for embeddings."""
        if self._embedding_model is None:
            try:
                import os
                allow_net = os.environ.get("MATRIX_ENABLE_NETWORK_MODELS", "").strip().lower() in {"1","true","yes","on"}
                if not allow_net:
                    logger.warning("MATRIX_ENABLE_NETWORK_MODELS disabled; skipping transformers embedding model load")
                    self._embedding_model = None
                    self._embedding_tokenizer = None
                    return
                from transformers import AutoModel, AutoTokenizer
                import torch

                model_name = "zlucia/casehold"  # CaseHOLD-LegalBERT - best for case law
                logger.info(f"Loading Legal-BERT model: {model_name}")

                self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir="./models_cache"
                )
                self._embedding_model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir="./models_cache"
                )

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._embedding_model.to(device)
                self._embedding_model.eval()

                self._device = device
                logger.info(f"Legal-BERT model loaded on {device}")

            except Exception as e:
                logger.warning(f"Failed to load Legal-BERT for embeddings: {e}")
                logger.warning("Will use rule-based quality scoring instead")
                self._embedding_model = None
                self._embedding_tokenizer = None

    def merge_drafts(
        self,
        primary_draft: str,
        secondary_draft: str,
        primary_model_name: str = "phi3:mini",
        secondary_model_name: str = "qwen2.5:14b"
    ) -> Dict[str, Any]:
        """
        Merge two drafts by comparing sections and selecting best parts.

        Args:
            primary_draft: Draft from primary model
            secondary_draft: Draft from secondary model
            primary_model_name: Name of primary model
            secondary_model_name: Name of secondary model

        Returns:
            Dictionary with merged draft and comparison details
        """
        logger.info("Starting section-by-section merge...")

        # Segment both drafts into sections
        primary_sections = self._segment_document(primary_draft)
        secondary_sections = self._segment_document(secondary_draft)

        logger.info(f"Primary draft: {len(primary_sections)} sections")
        logger.info(f"Secondary draft: {len(secondary_sections)} sections")

        # Compare and select best sections
        selected_sections = []
        comparisons = []

        # Match sections by type (introduction, factual_background, etc.)
        section_types = set([s.section_type for s in primary_sections + secondary_sections])

        for section_type in sorted(section_types):
            primary_section = self._find_section_by_type(primary_sections, section_type)
            secondary_section = self._find_section_by_type(secondary_sections, section_type)

            if primary_section and secondary_section:
                # Compare both sections
                comparison = self._compare_sections(
                    primary_section,
                    secondary_section,
                    primary_model_name,
                    secondary_model_name
                )
                comparisons.append(comparison)
                selected_sections.append(comparison.selected_section)
                logger.info(f"{section_type}: Selected {comparison.selected_section.title} "
                          f"(quality: {comparison.quality_score:.2f}, "
                          f"similarity: {comparison.semantic_similarity:.2f})")
            elif primary_section:
                # Only primary has this section
                selected_sections.append(primary_section)
                logger.info(f"{section_type}: Using primary (secondary missing)")
            elif secondary_section:
                # Only secondary has this section
                selected_sections.append(secondary_section)
                logger.info(f"{section_type}: Using secondary (primary missing)")

        # Merge selected sections into final draft
        merged_draft = self._assemble_draft(selected_sections)

        # Calculate merge statistics
        primary_count = sum(1 for c in comparisons if c.selected_section == c.primary_section)
        secondary_count = sum(1 for c in comparisons if c.selected_section == c.secondary_section)

        return {
            "merged_draft": merged_draft,
            "comparisons": [
                {
                    "section_type": c.primary_section.section_type,
                    "selected_model": primary_model_name if c.selected_section == c.primary_section else secondary_model_name,
                    "quality_score": c.quality_score,
                    "semantic_similarity": c.semantic_similarity,
                    "combined_score": c.combined_score,
                    "selection_reason": c.selection_reason
                }
                for c in comparisons
            ],
            "statistics": {
                "total_sections": len(selected_sections),
                "primary_selected": primary_count,
                "secondary_selected": secondary_count,
                "average_quality": np.mean([c.quality_score for c in comparisons]) if comparisons else 0.0,
                "average_similarity": np.mean([c.semantic_similarity for c in comparisons]) if comparisons else 0.0
            }
        }

    def _segment_document(self, document: str) -> List[Section]:
        """
        Segment document into sections based on structural markers.

        Args:
            document: Full document text

        Returns:
            List of Section objects
        """
        sections = []

        # Common legal document section patterns
        section_patterns = [
            (r'(?i)^(?:I\.|1\.)\s*(?:INTRODUCTION|INTRO)', 'introduction'),
            (r'(?i)^(?:II\.|2\.)\s*(?:FACTUAL\s*BACKGROUND|BACKGROUND)', 'factual_background'),
            (r'(?i)^(?:III\.|3\.)\s*(?:LEGAL\s*ARGUMENT|ARGUMENT)', 'legal_argument'),
            (r'(?i)^(?:IV\.|4\.)\s*(?:PRIVACY\s*HARM|HARM)', 'privacy_harm'),
            (r'(?i)^(?:V\.|5\.)\s*(?:BALANCING\s*TEST|BALANCING)', 'balancing_test'),
            (r'(?i)^(?:VI\.|6\.)\s*(?:REQUESTED\s*RELIEF|RELIEF|CONCLUSION)', 'conclusion'),
            (r'(?i)^(?:###|##)\s*(?:INTRODUCTION|INTRO)', 'introduction'),
            (r'(?i)^(?:###|##)\s*(?:FACTUAL\s*BACKGROUND|BACKGROUND)', 'factual_background'),
            (r'(?i)^(?:###|##)\s*(?:LEGAL\s*ARGUMENT|ARGUMENT)', 'legal_argument'),
            (r'(?i)^(?:###|##)\s*(?:PRIVACY\s*HARM|HARM)', 'privacy_harm'),
            (r'(?i)^(?:###|##)\s*(?:BALANCING\s*TEST|BALANCING)', 'balancing_test'),
            (r'(?i)^(?:###|##)\s*(?:REQUESTED\s*RELIEF|RELIEF|CONCLUSION)', 'conclusion'),
        ]

        lines = document.split('\n')
        current_section: Optional[Section] = None
        current_content = []
        current_type = "unknown"
        current_start = 0

        for i, line in enumerate(lines):
            # Check if this line starts a new section
            matched_type = None
            for pattern, section_type in section_patterns:
                if re.match(pattern, line.strip()):
                    matched_type = section_type
                    break

            if matched_type:
                # Save previous section
                if current_section is not None:
                    sections.append(Section(
                        title=current_section.title,
                        content='\n'.join(current_content),
                        start_pos=current_start,
                        end_pos=i,
                        section_type=current_type
                    ))

                # Start new section
                current_type = matched_type
                current_start = i
                current_content = [line]
                current_section = Section(
                    title=line.strip(),
                    content=line,
                    start_pos=i,
                    end_pos=i,
                    section_type=matched_type
                )
            else:
                # Continue current section
                if current_content or line.strip():
                    current_content.append(line)

        # Save last section
        if current_section is not None and current_content:
            sections.append(Section(
                title=current_section.title,
                content='\n'.join(current_content),
                start_pos=current_start,
                end_pos=len(lines),
                section_type=current_type
            ))

        # If no sections found, treat entire document as one section
        if not sections:
            sections.append(Section(
                title="Document",
                content=document,
                start_pos=0,
                end_pos=len(lines),
                section_type="full_document"
            ))

        return sections

    def _find_section_by_type(self, sections: List[Section], section_type: str) -> Optional[Section]:
        """Find first section of given type."""
        for section in sections:
            if section.section_type == section_type:
                return section
        return None

    def _compare_sections(
        self,
        primary: Section,
        secondary: Section,
        primary_model_name: str,
        secondary_model_name: str
    ) -> SectionComparison:
        """
        Compare two sections and select the best one.

        Args:
            primary: Primary model's section
            secondary: Secondary model's section
            primary_model_name: Name of primary model
            secondary_model_name: Name of secondary model

        Returns:
            SectionComparison with selection
        """
        # Calculate quality scores
        primary_quality = self._calculate_quality_score(primary.content)
        secondary_quality = self._calculate_quality_score(secondary.content)

        # Calculate semantic similarity
        semantic_similarity = self._calculate_semantic_similarity(primary.content, secondary.content)

        # Calculate combined scores (for selection)
        primary_combined = (primary_quality * self.quality_weight) + (semantic_similarity * self.semantic_weight)
        secondary_combined = (secondary_quality * self.quality_weight) + (semantic_similarity * self.semantic_weight)

        # Select best section
        if primary_combined >= secondary_combined:
            selected = primary
            selected_model = primary_model_name
            reason = f"Higher combined score ({primary_combined:.2f} vs {secondary_combined:.2f})"
        else:
            selected = secondary
            selected_model = secondary_model_name
            reason = f"Higher combined score ({secondary_combined:.2f} vs {primary_combined:.2f})"

        return SectionComparison(
            primary_section=primary,
            secondary_section=secondary,
            quality_score=max(primary_quality, secondary_quality),
            semantic_similarity=semantic_similarity,
            combined_score=max(primary_combined, secondary_combined),
            selected_section=selected,
            selection_reason=reason
        )

    def _calculate_quality_score(self, text: str) -> float:
        """
        Calculate quality score for a section (0-1).

        Uses rule-based scoring:
        - Word count (appropriate length)
        - Legal terminology usage
        - Citation presence
        - Structure completeness
        """
        score = 0.0

        # Word count check (300-800 words ideal for sections)
        word_count = len(text.split())
        if 300 <= word_count <= 800:
            score += 0.25
        elif 200 <= word_count < 300 or 800 < word_count <= 1200:
            score += 0.15
        else:
            score += 0.05

        # Legal terminology check
        legal_terms = [
            "court", "plaintiff", "defendant", "motion", "statute", "precedent",
            "jurisdiction", "discovery", "seal", "pseudonym", "privacy", "harm",
            "balancing", "test", "interest", "public", "private", "constitutional"
        ]
        term_count = sum(1 for term in legal_terms if term.lower() in text.lower())
        if term_count >= 5:
            score += 0.25
        elif term_count >= 3:
            score += 0.15
        else:
            score += 0.05

        # Citation check
        citation_patterns = [r'\d+\s+U\.S\.C\.', r'\d+\s+F\.\d+[dD]', r'\d+\s+S\.Ct\.', r'\d+\s+F\.\s*Supp\.']
        has_citation = any(re.search(pattern, text) for pattern in citation_patterns)
        if has_citation:
            score += 0.25
        else:
            score += 0.10

        # Structure completeness (has paragraphs, proper formatting)
        has_paragraphs = '\n\n' in text or text.count('\n') >= 3
        proper_formatting = any(marker in text for marker in [':', ';', '.'])
        if has_paragraphs and proper_formatting:
            score += 0.25
        else:
            score += 0.10

        return min(1.0, score)

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using Legal-BERT embeddings.

        Returns:
            Similarity score (0-1), where 1.0 means identical meaning
        """
        # Load embedding model if needed
        self._load_embedding_model()

        if self._embedding_model is None:
            # Fallback: Use simple text similarity
            logger.warning("Using fallback text similarity (Legal-BERT not available)")
            return self._simple_text_similarity(text1, text2)

        try:
            import torch

            # Encode both texts
            emb1 = self._encode_text(text1)
            emb2 = self._encode_text(text2)

            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            # Normalize to 0-1 range
            similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))

            return float(similarity)

        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}, using fallback")
            return self._simple_text_similarity(text1, text2)

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using Legal-BERT model."""
        if self._embedding_model is None or self._embedding_tokenizer is None:
            raise RuntimeError("Embedding model not loaded")

        import torch

        # Truncate if too long
        max_length = 512
        inputs = self._embedding_tokenizer(
            text[:max_length * 4],  # Approximate character limit
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._embedding_model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Fallback text similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _assemble_draft(self, sections: List[Section]) -> str:
        """Assemble selected sections into final draft."""
        return '\n\n'.join([section.content for section in sections])


__all__ = ["SectionMerger", "Section", "SectionComparison"]
