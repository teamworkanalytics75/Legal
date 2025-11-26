"""
SHAP Insight Plugin for Semantic Kernel

Provides explainable AI insights based on SHAP values.
Converts CatBoost predictions into actionable recommendations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Sequence, Callable
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from case_law_data.model_registry import CatBoostModelRegistry
except ImportError:
    CatBoostModelRegistry = None

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class SHAPInsightPlugin:
    """
    Plugin that provides SHAP-based insights for CatBoost models.

    Converts model predictions into actionable recommendations by:
    - Computing SHAP values for specific cases
    - Identifying which features matter most
    - Generating feature-specific improvement suggestions
    - Providing context-aware recommendations
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model: Optional[Any] = None,
        preferred_model_names: Optional[Sequence[str]] = None,
        auto_train: bool = False,
        registry_strategy: str = "priority",
        trainer: Optional[Callable[..., Dict[str, Any]]] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize SHAP Insight Plugin.

        Args:
            model_path: Path to CatBoost model file
            model: Pre-loaded CatBoost model (optional)
            preferred_model_names: explicit filenames (in priority order) to
                resolve via the CatBoostModelRegistry when no path/model is provided
            auto_train: when True, run the registry's real-data training pipeline
                if no saved model exists
            registry_strategy: strategy to choose among multiple saved models
            trainer: optional custom training callable compatible with the registry API
            trainer_kwargs: kwargs passed to the training callable
        """
        self.model = model
        self.model_path = model_path
        self.feature_names = None
        self.preferred_model_names = list(preferred_model_names or [])
        self.auto_train = auto_train
        self.registry_strategy = registry_strategy
        self.registry_trainer = trainer
        self.registry_trainer_kwargs = trainer_kwargs or {}

        if model is not None:
            self._initialize_from_model(model)
        elif model_path and model_path.exists():
            self._load_model(model_path)
        else:
            self._resolve_model_from_registry()

    def _load_model(self, model_path: Path):
        """Load CatBoost model from file."""
        if not CATBOOST_AVAILABLE:
            logger.error("CatBoost not available")
            return

        try:
            self.model = CatBoostClassifier()
            self.model.load_model(str(model_path))
            self._initialize_from_model(self.model)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def _resolve_model_from_registry(self):
        """Attempt to locate or train a CatBoost model via the registry."""
        if CatBoostModelRegistry is None:
            logger.warning("CatBoostModelRegistry unavailable - provide model_path or model")
            return

        try:
            registry = CatBoostModelRegistry()
            record, trained_model = registry.resolve_or_train(
                preferred_models=self.preferred_model_names,
                strategy=self.registry_strategy,
                train_if_missing=self.auto_train,
                trainer=self.registry_trainer,
                trainer_kwargs=self.registry_trainer_kwargs,
            )
        except Exception as exc:
            logger.warning(f"Failed to resolve CatBoost model via registry: {exc}")
            return

        if record is None and trained_model is None:
            logger.warning("No CatBoost model available - SHAP insights unavailable")
            return

        if trained_model is not None:
            logger.info(f"Using freshly trained CatBoost model: {record.path.name if record else 'unknown'}")
            self.model = trained_model
            self._initialize_from_model(trained_model)
            return

        if record:
            self.model_path = record.path
            self._load_model(record.path)

    def _initialize_from_model(self, model: Any):
        """Initialize feature names from model."""
        if hasattr(model, 'feature_names_') and model.feature_names_ is not None:
            self.feature_names = model.feature_names_
        else:
            logger.warning("Model does not have feature_names_ - will use feature_df.columns as fallback")
            self.feature_names = None

    def compute_shap_insights(
        self,
        features: Dict[str, float],
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Compute SHAP insights for a feature vector.

        Args:
            features: Dictionary of feature name -> value
            top_n: Number of top features to highlight

        Returns:
            Dictionary with SHAP insights and recommendations
        """
        if self.model is None:
            return {
                "error": "No model available",
                "shap_available": False
            }

        try:
            # Convert features to DataFrame
            feature_df = pd.DataFrame([features])

            # Ensure all model features are present and determine feature names for this run.
            if self.feature_names:
                for col in self.feature_names:
                    if col not in feature_df.columns:
                        feature_df[col] = 0.0
                feature_df = feature_df[self.feature_names]
                resolved_feature_names = list(self.feature_names)
            else:
                resolved_feature_names = list(feature_df.columns)
                logger.debug(
                    "Model did not provide feature names; using feature_df columns as fallback: %s",
                    resolved_feature_names,
                )

            # Get prediction
            prediction = self.model.predict(feature_df)[0]
            proba = self.model.predict_proba(feature_df)[0]
            confidence = float(np.max(proba))

            # Compute SHAP values
            pool = Pool(feature_df, feature_names=resolved_feature_names)
            shap_values = self.model.get_feature_importance(pool, type="ShapValues")

            # Handle multi-class output
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]  # Positive class
            elif (
                shap_values.ndim == 2
                and shap_values.shape[1] == len(resolved_feature_names) + 1
            ):
                shap_values = shap_values[:, :-1]  # Remove bias

            shap_index = resolved_feature_names
            if len(shap_index) != shap_values.shape[-1]:
                shap_index = [f"feature_{i}" for i in range(shap_values.shape[-1])]

            # Convert to series
            shap_series = pd.Series(shap_values[0], index=shap_index)

            # Get feature values
            feature_values = pd.Series(features)

            # Analyze SHAP values
            top_positive = shap_series[shap_series > 0].nlargest(top_n)
            top_negative = shap_series[shap_series < 0].nsmallest(top_n)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                top_positive, top_negative, feature_values, shap_series
            )

            feature_contributions = {
                feat: {
                    "shap_value": float(shap_series[feat]),
                    "current_value": float(feature_values.get(feat, 0)),
                    "impact": "positive" if shap_series[feat] > 0 else "negative"
                }
                for feat in shap_series.index[: min(len(shap_series), top_n * 2)]
            }

            return {
                "shap_available": True,
                "prediction": int(prediction),
                "confidence": confidence,
                "probabilities": {
                    f"class_{i}": float(p) for i, p in enumerate(proba)
                },
                "shap_values": shap_series.to_dict(),
                "top_helping_features": top_positive.to_dict(),
                "top_hurting_features": top_negative.to_dict(),
                "recommendations": recommendations,
                "feature_contributions": feature_contributions
            }

        except Exception as e:
            logger.error(f"SHAP computation failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "shap_available": False
            }

    def _generate_recommendations(
        self,
        top_positive: pd.Series,
        top_negative: pd.Series,
        feature_values: pd.Series,
        shap_series: pd.Series
    ) -> List[str]:
        """Generate actionable recommendations from SHAP values."""
        recommendations = []

        # Positive features (helping)
        for feature, impact in top_positive.items():
            current_val = feature_values.get(feature, 0)

            if impact > 0.1:  # High impact
                recommendations.append(
                    f"✅ **Strengthen {feature}**: Currently helping significantly "
                    f"(impact: +{impact:.3f}). This is a key strength - maintain or enhance."
                )
            elif impact > 0.05:  # Medium impact
                recommendations.append(
                    f"✅ **Maintain {feature}**: Contributing positively "
                    f"(impact: +{impact:.3f}). Keep current level."
                )

        # Negative features (hurting)
        for feature, impact in top_negative.items():
            current_val = feature_values.get(feature, 0)

            if impact < -0.1:  # High negative impact
                recommendations.append(
                    f"⚠️ **Reduce {feature}**: Significantly hurting chances "
                    f"(impact: {impact:.3f}). Consider minimizing or removing."
                )
            elif impact < -0.05:  # Medium negative impact
                recommendations.append(
                    f"⚠️ **Review {feature}**: Slightly negative impact "
                    f"(impact: {impact:.3f}). Could be improved."
                )

        # Feature interactions (if available)
        # TODO: Add interaction analysis if 3-way interactions are computed

        return recommendations

    def get_feature_explanation(
        self,
        features: Dict[str, float],
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for a specific feature.

        Args:
            features: Feature vector
            feature_name: Name of feature to explain

        Returns:
            Explanation of how this feature contributes
        """
        insights = self.compute_shap_insights(features)

        if not insights.get("shap_available"):
            return {"error": "SHAP insights unavailable"}

        shap_value = insights["shap_values"].get(feature_name, 0)
        current_value = features.get(feature_name, 0)

        return {
            "feature_name": feature_name,
            "current_value": current_value,
            "shap_contribution": float(shap_value),
            "impact": "positive" if shap_value > 0 else "negative",
            "magnitude": abs(shap_value),
            "interpretation": self._interpret_feature_contribution(
                feature_name, shap_value, current_value
            )
        }

    def _interpret_feature_contribution(
        self,
        feature_name: str,
        shap_value: float,
        current_value: float
    ) -> str:
        """Generate human-readable interpretation of feature contribution."""
        if abs(shap_value) < 0.01:
            return f"{feature_name} has minimal impact on this prediction."

        direction = "increasing" if shap_value > 0 else "decreasing"
        magnitude = "significantly" if abs(shap_value) > 0.1 else "moderately"

        return (
            f"{feature_name} is {magnitude} {direction} the predicted success probability. "
            f"Current value: {current_value:.2f}, Contribution: {shap_value:+.3f}"
        )
