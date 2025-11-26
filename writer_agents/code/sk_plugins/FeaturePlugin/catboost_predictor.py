#!/usr/bin/env python3
"""
CatBoost Predictor - Handles CatBoost model loading and predictions.

Extracted from RefinementLoop to follow single responsibility principle.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CatBoostPredictor:
    """Handles CatBoost model loading and predictions."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        shap_importance: Optional[Dict[str, float]] = None
    ):
        """
        Initialize CatBoostPredictor.

        Args:
            model_path: Path to CatBoost model file (optional, will auto-detect if None)
            shap_importance: Pre-loaded SHAP importance values (optional)
        """
        self.model = None
        self.model_path = model_path
        self.shap_importance = shap_importance or {}

        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        else:
            # Try to find model automatically
            found_path = self._find_model()
            if found_path:
                self.load_model(found_path)

    def load_model(self, model_path: Path) -> bool:
        """
        Load CatBoost model from file path.

        Args:
            model_path: Path to the .cbm model file

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from catboost import CatBoostClassifier

            if not model_path.exists():
                logger.warning(f"CatBoost model not found at {model_path}, continuing without model")
                return False

            # Try to use GPU if available (faster inference)
            try:
                # Test if GPU is available for CatBoost
                test_model = CatBoostClassifier(task_type='GPU', devices='0', iterations=1, verbose=False)
                import pandas as pd
                import numpy as np
                test_X = pd.DataFrame(np.random.rand(5, 3))
                test_y = np.random.randint(0, 2, 5)
                test_model.fit(test_X, test_y, verbose=False)
                # GPU works, use it
                self.model = CatBoostClassifier(task_type='GPU', devices='0')
                logger.info("🚀 CatBoost using GPU acceleration")
            except Exception:
                # GPU not available or failed, use CPU
                self.model = CatBoostClassifier()
                logger.info("CatBoost using CPU (GPU not available)")
            
            self.model.load_model(str(model_path))
            self.model_path = model_path
            logger.info(f"Loaded CatBoost model from {model_path}")

            # Extract feature names if available
            if hasattr(self.model, 'feature_names_'):
                logger.info(f"Model has {len(self.model.feature_names_)} features")

            # Load SHAP importance from summary file if available, otherwise extract from model
            # Add error handling to prevent hanging
            if not self.shap_importance:
                try:
                    logger.debug("Attempting to load SHAP importance from summary file...")
                    self._load_shap_from_summary()
                    if self.shap_importance:
                        logger.info(f"Successfully loaded SHAP importance ({len(self.shap_importance)} features)")
                except Exception as e:
                    logger.warning(f"SHAP summary loading failed: {e}, will extract from model")
                
                if not self.shap_importance:
                    try:
                        logger.debug("Extracting SHAP importance from model...")
                        self._extract_shap_importance()
                        if self.shap_importance:
                            logger.info(f"Successfully extracted SHAP importance from model ({len(self.shap_importance)} features)")
                    except Exception as e:
                        logger.warning(f"SHAP extraction from model failed: {e}, continuing without SHAP")
                        # Set empty dict to avoid retrying
                        self.shap_importance = {}

            return True

        except ImportError:
            logger.warning("CatBoost not available, model loading skipped")
            return False
        except Exception as e:
            logger.error(f"Failed to load CatBoost model: {e}")
            return False

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Run prediction on features.

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Dictionary with:
                - prediction: int (0 or 1)
                - confidence: float (max probability)
                - probabilities: array of probabilities
                - success_probability: float (same as confidence)
        """
        if not self.model:
            return {
                "prediction": None,
                "confidence": None,
                "probabilities": None,
                "success_probability": None
            }

        try:
            import pandas as pd
            import numpy as np

            # Convert features to DataFrame format expected by model
            feature_df = pd.DataFrame([features])

            # Ensure all required columns exist
            if hasattr(self.model, 'feature_names_'):
                for col in self.model.feature_names_:
                    if col not in feature_df.columns:
                        feature_df[col] = 0
                feature_df = feature_df[self.model.feature_names_]

            # Get prediction probabilities
            proba = self.model.predict_proba(feature_df)[0]
            prediction = self.model.predict(feature_df)[0]
            confidence = float(np.max(proba))
            success_prob = confidence  # Use max probability as success probability

            logger.debug(f"CatBoost prediction: confidence={confidence:.3f}, prediction={prediction}")

            return {
                "prediction": int(prediction),
                "confidence": confidence,
                "probabilities": proba.tolist(),
                "success_probability": success_prob
            }

        except Exception as e:
            logger.warning(f"Failed to run CatBoost prediction: {e}")
            return {
                "prediction": None,
                "confidence": None,
                "probabilities": None,
                "success_probability": None
            }

    def _find_model(self) -> Optional[Path]:
        """
        Find CatBoost model by priority order.

        Returns:
            Path to model file if found, None otherwise
        """
        search_dir = Path("case_law_data/models")
        order = [
            "catboost_outline_unified.cbm",
            "catboost_outline_both_interactions.cbm",
            "catboost_motion_seal_pseudonym.cbm",
            "catboost_motion.cbm",
            "section_1782_discovery_model.cbm",
        ]
        for name in order:
            p = search_dir / name
            if p.exists():
                return p
        # Fallback to any .cbm
        for p in search_dir.glob("*.cbm"):
            return p
        return None

    def _load_shap_from_summary(self) -> None:
        """Load SHAP importance values from section_1782_discovery_SHAP_SUMMARY.json.

        Filters out opinion artifact features that correlate with outcomes rather than
        petition characteristics (e.g., 'keyword_affirmed', 'keyword_reversed').
        """
        try:
            # Try to find SHAP summary file
            base_dir = Path(__file__).parents[5]  # Navigate to project root
            shap_summary_paths = [
                base_dir / "case_law_data" / "analysis" / "section_1782_discovery_SHAP_SUMMARY.json",
                Path(__file__).parents[4] / "case_law_data" / "analysis" / "section_1782_discovery_SHAP_SUMMARY.json",
            ]

            shap_path = None
            for path in shap_summary_paths:
                if path.exists():
                    shap_path = path
                    break

            if not shap_path:
                logger.debug("SHAP summary file not found, will extract from model instead")
                return

            # Load opinion artifact filters
            opinion_artifacts = self._load_opinion_artifact_filters()

            with open(shap_path, 'r') as f:
                shap_data = json.load(f)

            # Extract mean_abs_shap values from top_features, filtering opinion artifacts
            if 'top_features' in shap_data:
                filtered_importance = {}
                filtered_count = 0

                for item in shap_data['top_features']:
                    feature_name = item['feature']
                    shap_value = item['mean_abs_shap']

                    # Skip opinion artifact features
                    if feature_name in opinion_artifacts:
                        filtered_count += 1
                        logger.debug(f"Filtered opinion artifact feature: {feature_name} (SHAP: {shap_value:.4f})")
                        continue

                    filtered_importance[feature_name] = shap_value

                self.shap_importance = filtered_importance
                logger.info(f"Loaded SHAP importance for {len(self.shap_importance)} features from {shap_path.name}")
                if filtered_count > 0:
                    logger.info(f"Filtered {filtered_count} opinion artifact features (e.g., keyword_affirmed, keyword_reversed)")
            else:
                logger.warning(f"SHAP summary file missing 'top_features' key")

        except Exception as e:
            logger.warning(f"Failed to load SHAP summary: {e}")
            # Continue without SHAP - will try to extract from model instead

    def _load_opinion_artifact_filters(self) -> set:
        """Load opinion artifact features that should be filtered out.

        These features (e.g., 'keyword_affirmed', 'keyword_reversed') appear primarily
        in court opinions/orders describing decisions, not in original petitions/motions.
        Using them would cause data leakage since they describe outcomes rather than
        petition characteristics.
        """
        try:
            # Try to find filter config file
            base_dir = Path(__file__).parents[5]
            filter_paths = [
                base_dir / "case_law_data" / "config" / "petition_feature_filters.json",
                Path(__file__).parents[4] / "case_law_data" / "config" / "petition_feature_filters.json",
            ]

            filter_path = None
            for path in filter_paths:
                if path.exists():
                    filter_path = path
                    break

            if not filter_path:
                logger.debug("Opinion artifact filter config not found, using default filters")
                # Default opinion artifact features
                return {
                    'keyword_affirmed', 'keyword_affirmed_rate',
                    'keyword_reversed', 'keyword_reversed_rate',
                    'keyword_judgment', 'keyword_judgment_rate',
                    'keyword_granted_rate',  # May appear in opinions
                    'analysis_ratio_excluding_standard'
                }

            with open(filter_path, 'r') as f:
                filter_config = json.load(f)

            artifact_features = set(filter_config.get('opinion_artifact_features', []))
            logger.debug(f"Loaded {len(artifact_features)} opinion artifact filters from config")
            return artifact_features

        except Exception as e:
            logger.warning(f"Failed to load opinion artifact filters: {e}")
            # Return default set
            return {
                'keyword_affirmed', 'keyword_affirmed_rate',
                'keyword_reversed', 'keyword_reversed_rate',
                'keyword_judgment', 'keyword_judgment_rate',
                'keyword_granted_rate',
                'analysis_ratio_excluding_standard'
            }

    def _extract_shap_importance(self) -> None:
        """Extract SHAP importance values from loaded CatBoost model."""
        try:
            if not self.model:
                return

            # Get feature importance (CatBoost's built-in importance)
            if hasattr(self.model, 'get_feature_importance'):
                importance = self.model.get_feature_importance()
                feature_names = self.model.feature_names_ if hasattr(self.model, 'feature_names_') else []

                if feature_names and len(importance) == len(feature_names):
                    self.shap_importance = dict(zip(feature_names, importance))
                    logger.info(f"Extracted SHAP importance for {len(self.shap_importance)} features")
                else:
                    logger.warning("Feature names don't match importance array length")
            else:
                logger.warning("Model doesn't support get_feature_importance()")

        except Exception as e:
            logger.error(f"Failed to extract SHAP importance: {e}")

