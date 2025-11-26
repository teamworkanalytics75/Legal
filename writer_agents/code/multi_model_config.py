"""Configuration for multi-model ensemble system."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class ComparisonStrategy(Enum):
    """Strategies for comparing and merging multi-model outputs."""
    VOTE = "vote"  # Simple voting (not implemented)
    CONSENSUS = "consensus"  # Merge when both agree (not implemented)
    QUALITY_SCORE = "quality_score"  # Pick highest quality score (not implemented)
    SECTION_BY_SECTION = "section_by_section"  # Cherry-pick best sections (sophisticated)


@dataclass
class MultiModelConfig:
    """Configuration for multi-model ensemble system."""

    # Enable multi-model system
    enabled: bool = True

    # Drafting models (primary + secondary)
    primary_drafting_model: str = "phi3:mini"  # Phi-3 Mini 128K (context advantage)
    secondary_drafting_model: str = "qwen2.5:14b"  # Qwen2.5 14B (reasoning advantage)

    # Validation models
    quality_scorer_model: str = "legal-bert-casehold"  # Legal-BERT for quality scoring
    logical_reviewer_model: str = "qwen2.5:14b"  # Qwen2.5 for logical review

    # Review models
    review_model: str = "qwen2.5:14b"  # Better reasoning for review

    # Comparison strategy
    comparison_strategy: ComparisonStrategy = ComparisonStrategy.SECTION_BY_SECTION

    # Quality thresholds
    quality_threshold: float = 0.70  # Minimum quality score to accept section
    semantic_similarity_threshold: float = 0.80  # Minimum similarity for section comparison

    # Weighting for section selection (quality_score + semantic_relevance)
    quality_weight: float = 0.70  # Weight for quality score
    semantic_weight: float = 0.30  # Weight for semantic relevance

    # Validation weights (Legal-BERT + Qwen2.5 + CatBoost)
    legal_bert_weight: float = 0.40  # Legal-BERT quality scoring
    qwen_review_weight: float = 0.40  # Qwen2.5 logical review
    catboost_weight: float = 0.20  # CatBoost feature validation

    # Phase-specific model usage
    use_multi_model_drafting: bool = True  # Use both models in DRAFT phase
    use_multi_model_validation: bool = True  # Use Legal-BERT + Qwen2.5 in VALIDATE phase
    use_qwen_review: bool = True  # Use Qwen2.5 in REVIEW phase
    use_multi_model_refinement: bool = True  # Use both models in REFINE phase

    # Ollama configuration
    ollama_base_url: str = "http://localhost:11434"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "primary_drafting_model": self.primary_drafting_model,
            "secondary_drafting_model": self.secondary_drafting_model,
            "quality_scorer_model": self.quality_scorer_model,
            "logical_reviewer_model": self.logical_reviewer_model,
            "review_model": self.review_model,
            "comparison_strategy": self.comparison_strategy.value,
            "quality_threshold": self.quality_threshold,
            "semantic_similarity_threshold": self.semantic_similarity_threshold,
            "quality_weight": self.quality_weight,
            "semantic_weight": self.semantic_weight,
            "legal_bert_weight": self.legal_bert_weight,
            "qwen_review_weight": self.qwen_review_weight,
            "catboost_weight": self.catboost_weight,
            "use_multi_model_drafting": self.use_multi_model_drafting,
            "use_multi_model_validation": self.use_multi_model_validation,
            "use_qwen_review": self.use_qwen_review,
            "use_multi_model_refinement": self.use_multi_model_refinement,
            "ollama_base_url": self.ollama_base_url,
        }


def get_default_multi_model_config() -> MultiModelConfig:
    """Get default multi-model configuration optimized for legal writing."""
    return MultiModelConfig()


__all__ = ["MultiModelConfig", "ComparisonStrategy", "get_default_multi_model_config"]

