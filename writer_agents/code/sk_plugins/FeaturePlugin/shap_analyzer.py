#!/usr/bin/env python3
"""
SHAP Analyzer - Handles SHAP insights computation.

Extracted from RefinementLoop to follow single responsibility principle.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Handles SHAP insights computation for explainable recommendations."""

    def __init__(self, catboost_predictor):
        """
        Initialize SHAPAnalyzer.

        Args:
            catboost_predictor: CatBoostPredictor instance with loaded model
        """
        self.catboost_predictor = catboost_predictor
        self._shap_plugin = None

    def compute_insights(self, features: Dict[str, float], top_n: int = 10) -> Dict[str, Any]:
        """
        Compute SHAP insights for a feature vector.

        Args:
            features: Dictionary of feature name -> value
            top_n: Number of top features to highlight

        Returns:
            Dictionary with SHAP insights and recommendations, or None if unavailable
        """
        if not self.catboost_predictor or not self.catboost_predictor.model:
            return None

        try:
            # Lazy load SHAPInsightPlugin
            if self._shap_plugin is None:
                try:
                    from ...SHAPInsightPlugin import SHAPInsightPlugin
                    self._shap_plugin = SHAPInsightPlugin(model=self.catboost_predictor.model)
                except ImportError as e:
                    logger.debug(f"SHAPInsightPlugin not available: {e}")
                    return None

            shap_insights = self._shap_plugin.compute_shap_insights(features, top_n=top_n)

            if shap_insights and shap_insights.get("shap_available"):
                logger.info(f"✅ SHAP insights computed: {len(shap_insights.get('recommendations', []))} actionable recommendations")

                # Log top insights
                top_helping = list(shap_insights.get('top_helping_features', {}).keys())[:3]
                top_hurting = list(shap_insights.get('top_hurting_features', {}).keys())[:3]
                if top_helping:
                    logger.info(f"   Top helping features: {top_helping}")
                if top_hurting:
                    logger.info(f"   Top hurting features: {top_hurting}")

                return shap_insights
            else:
                return None

        except Exception as e:
            logger.warning(f"SHAP insights computation failed (will use basic recommendations): {e}")
            return None

