#!/usr/bin/env python3
"""
Memory Manager - Handles memory storage and retrieval for analysis results.

Extracted from RefinementLoop to follow single responsibility principle.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """Handles memory storage and retrieval for analysis results."""

    def __init__(self, memory_store=None):
        """
        Initialize MemoryManager.

        Args:
            memory_store: EpisodicMemoryBank instance (optional)
        """
        self.memory_store = memory_store
        self._recent_analyses: List[Dict[str, Any]] = []
        self._cache_limit = 5

    def retrieve_similar_analyses(self, draft_text: str, k: int = 3) -> List[Dict]:
        """
        Retrieve similar past analyses from memory.

        Args:
            draft_text: The draft text to find similar analyses for
            k: Number of similar analyses to retrieve

        Returns:
            List of similar analysis dictionaries, or empty list if none found
        """
        query_text = draft_text[:500]

        if self.memory_store:
            try:
                past_memories = self.memory_store.retrieve(
                    agent_type="RefinementLoop",
                    query=query_text,
                    k=k,
                    memory_types=["execution", "query", "analysis"]
                )
                if past_memories:
                    logger.info(
                        "Found %d similar past CatBoost analyses - using insights to inform analysis",
                        len(past_memories)
                    )
                    return past_memories
            except Exception as e:
                logger.debug(f"Could not query past CatBoost analyses: {e}")

        # Fallback to local in-process cache so repeated calls avoid recompute
        if self._recent_analyses:
            logger.debug(
                "Using %d locally cached CatBoost analyses as fallback",
                min(k, len(self._recent_analyses))
            )
            return list(self._recent_analyses[-k:])

        return []

    def store_analysis(
        self,
        analysis_data: Dict[str, Any],
        draft_text: str,
        weak_features: Dict[str, Any],
        features: Dict[str, float],
        success_prob: Optional[float] = None,
        prediction: Optional[int] = None,
        confidence: Optional[float] = None,
        shap_insights: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store analysis results in memory.

        Args:
            analysis_data: Summary of analysis (weak_features_count, etc.)
            draft_text: The draft text that was analyzed
            weak_features: Dictionary of weak features
            features: All feature values
            success_prob: Success probability (optional)
            prediction: Model prediction (optional)
            confidence: Prediction confidence (optional)
            shap_insights: SHAP insights (optional)

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            analysis_summary = analysis_data.copy() if analysis_data else {
                "weak_features_count": len(weak_features),
                "weak_features": list(weak_features.keys()),
                "features_analyzed": len(features),
                "success_probability": success_prob,
                "prediction": int(prediction) if prediction is not None else None,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "shap_available": shap_insights is not None and shap_insights.get("shap_available", False)
            }

            # Store comprehensive context including all feature scores
            memory_context = {
                "agent_type": "RefinementLoop",
                "operation": "catboost_analysis",
                "analysis_summary": analysis_summary,
                "weak_features": weak_features,  # Store full weak_features dict
                "feature_scores": features,  # Store all feature scores
                "draft_length": len(draft_text),
                "total_features_analyzed": len(features),
                "success_probability": success_prob,
                "prediction": int(prediction) if prediction is not None else None,
                "confidence": confidence,
                "shap_insights": shap_insights if shap_insights else None
            }

            query_text = (
                f"CatBoost analysis: {len(weak_features)} weak features, {success_prob:.2%} success probability"
                if success_prob
                else f"CatBoost analysis: {len(weak_features)} weak features identified in draft"
            )

            # Cache locally regardless of downstream storage availability
            cache_entry = {
                "summary": query_text,
                "context": memory_context,
                "timestamp": datetime.now().isoformat()
            }
            self._recent_analyses.append(cache_entry)
            if len(self._recent_analyses) > self._cache_limit:
                self._recent_analyses = self._recent_analyses[-self._cache_limit:]

            if not self.memory_store:
                return True

            # Use EpisodicMemoryBank.add() method
            try:
                from EpisodicMemoryBank import EpisodicMemoryEntry
                memory_entry = EpisodicMemoryEntry(
                    agent_type="RefinementLoop",
                    memory_id=f"catboost_{datetime.now().isoformat()}",
                    summary=query_text,
                    context=memory_context,
                    source="refinement_loop",
                    timestamp=datetime.now(),
                    memory_type="analysis"
                )
                self.memory_store.add(memory_entry)
            except ImportError:
                # Fallback: try store_memory if it exists
                if hasattr(self.memory_store, 'store_memory'):
                    self.memory_store.store_memory(
                        agent_type="RefinementLoop",
                        query=query_text,
                        result=json.dumps(analysis_summary, indent=2),
                        context=memory_context,
                        memory_type="analysis"
                    )
                else:
                    logger.warning("Memory store doesn't support add() or store_memory() methods")
                    return False

            # Avoid invalid format spec when success_prob is None
            success_prob_str = f"{success_prob:.3f}" if success_prob is not None else "N/A"
            logger.debug(
                f"Stored CatBoost analysis in memory: {len(weak_features)} weak features, "
                f"success_prob={success_prob_str}"
            )
            return True

        except Exception as e:
            logger.debug(f"Failed to store CatBoost analysis in memory: {e}")
            return False
