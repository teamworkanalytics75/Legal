"""Inference pipeline for ML models.

This module provides:
1. Model loading and inference
2. Batch prediction capabilities
3. Model registry and versioning
4. Performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime
import joblib

# Import our ML components
from ..models.supervised import CaseOutcomePredictor, DocumentClassifier, AgentPerformancePredictor
from ..models.deep_learning import LegalLSTM, LegalBERT, LegalCNN, AttentionModel
from ..data import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing trained models."""

    def __init__(self, registry_path: str = 'ml_system/models/registry.json'):
        """Initialize model registry.

        Args:
            registry_path: Path to registry file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.metadata = {}

        self._load_registry()

    def _load_registry(self):
        """Load existing registry."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    self.models = data.get('models', {})
                    self.metadata = data.get('metadata', {})
                logger.info(f"Loaded registry with {len(self.models)} models")
            except Exception as e:
                logger.warning(f"Error loading registry: {e}")
                self.models = {}
                self.metadata = {}

    def _save_registry(self):
        """Save registry to file."""
        try:
            data = {
                'models': self.models,
                'metadata': self.metadata,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def register_model(self, model_name: str, model_path: str,
                      metadata: Dict[str, Any]) -> bool:
        """Register a new model.

        Args:
            model_name: Name of the model
            model_path: Path to the saved model
            metadata: Model metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            version = metadata.get('version', '1.0.0')
            model_key = f"{model_name}_v{version}"

            self.models[model_key] = {
                'name': model_name,
                'version': version,
                'path': model_path,
                'metadata': metadata,
                'registered_at': datetime.now().isoformat()
            }

            # Update metadata
            if model_name not in self.metadata:
                self.metadata[model_name] = {
                    'latest_version': version,
                    'total_versions': 1,
                    'first_registered': datetime.now().isoformat()
                }
            else:
                self.metadata[model_name]['total_versions'] += 1
                if version > self.metadata[model_name]['latest_version']:
                    self.metadata[model_name]['latest_version'] = version

            self._save_registry()
            logger.info(f"Registered model: {model_key}")
            return True

        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
            return False

    def load_model(self, model_name: str, version: str = 'latest') -> Optional[Any]:
        """Load a model from registry.

        Args:
            model_name: Name of the model
            version: Version to load ('latest' or specific version)

        Returns:
            Loaded model or None if not found
        """
        try:
            if version == 'latest':
                if model_name not in self.metadata:
                    logger.error(f"Model {model_name} not found in registry")
                    return None
                version = self.metadata[model_name]['latest_version']

            model_key = f"{model_name}_v{version}"
            if model_key not in self.models:
                logger.error(f"Model {model_key} not found in registry")
                return None

            model_info = self.models[model_key]
            model_path = Path(model_info['path'])

            # Load model based on type
            model_type = model_info['metadata'].get('type', 'unknown')

            if model_type == 'outcome_predictor':
                model = CaseOutcomePredictor()
                model.load_models(str(model_path))
            elif model_type == 'document_classifier':
                model = DocumentClassifier()
                model.load_model(str(model_path))
            elif model_type == 'agent_predictor':
                model = AgentPerformancePredictor()
                model.load_models(str(model_path))
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None

            logger.info(f"Loaded model: {model_key}")
            return model

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models.

        Returns:
            List of model information
        """
        models_list = []
        for model_key, model_info in self.models.items():
            models_list.append({
                'key': model_key,
                'name': model_info['name'],
                'version': model_info['version'],
                'path': model_info['path'],
                'registered_at': model_info['registered_at'],
                'metadata': model_info['metadata']
            })
        return models_list

    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Compare multiple models.

        Args:
            model_names: List of model names to compare

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for model_name in model_names:
            if model_name in self.metadata:
                metadata = self.metadata[model_name]
                comparison_data.append({
                    'model_name': model_name,
                    'latest_version': metadata['latest_version'],
                    'total_versions': metadata['total_versions'],
                    'first_registered': metadata['first_registered']
                })

        return pd.DataFrame(comparison_data)


class InferencePipeline:
    """Pipeline for running inference on ML models."""

    def __init__(self, registry_path: str = 'ml_system/models/registry.json'):
        """Initialize inference pipeline.

        Args:
            registry_path: Path to model registry
        """
        self.model_registry = ModelRegistry(registry_path)
        self.feature_engineer = FeatureEngineer()
        self.loaded_models = {}
        self.inference_history = []

        logger.info("InferencePipeline initialized")

    def load_model(self, model_name: str, version: str = 'latest') -> bool:
        """Load a model for inference.

        Args:
            model_name: Name of the model
            version: Version to load

        Returns:
            True if successful, False otherwise
        """
        try:
            model = self.model_registry.load_model(model_name, version)
            if model is not None:
                self.loaded_models[model_name] = model
                logger.info(f"Model {model_name} loaded for inference")
                return True
            else:
                logger.error(f"Failed to load model {model_name}")
                return False
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    def predict(self, input_data: Union[np.ndarray, Dict[str, Any], str],
                model_name: str = 'best') -> Dict[str, Any]:
        """Make a prediction using the specified model.

        Args:
            input_data: Input data for prediction
            model_name: Name of the model to use

        Returns:
            Prediction results
        """
        start_time = time.time()

        try:
            # Determine which model to use
            if model_name == 'best':
                model_name = self._get_best_model()

            if model_name not in self.loaded_models:
                # Try to load the model
                if not self.load_model(model_name):
                    return {'error': f'Model {model_name} not available'}

            model = self.loaded_models[model_name]

            # Prepare input data
            if isinstance(input_data, dict) or isinstance(input_data, str):
                # Extract features from input data (dict or raw text)
                features = self._extract_features_from_dict(input_data)
            else:
                features = input_data

            # Make prediction
            if hasattr(model, 'predict_outcome'):
                # Case outcome predictor
                prediction = model.predict_outcome(features)
            elif hasattr(model, 'classify_document'):
                # Document classifier
                prediction = model.classify_document(features)
            elif hasattr(model, 'predict_job_success'):
                # Agent performance predictor
                prediction = model.predict_job_success(features)
            else:
                return {'error': f'Unknown model type for {model_name}'}

            # Record inference
            inference_time = time.time() - start_time
            self._record_inference(model_name, inference_time, prediction)

            return {
                'model_name': model_name,
                'prediction': prediction,
                'inference_time': inference_time,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {e}")
            return {'error': str(e)}

    def batch_predict(self, input_batch: List[Union[np.ndarray, Dict[str, Any]]],
                     model_name: str = 'best') -> List[Dict[str, Any]]:
        """Make batch predictions.

        Args:
            input_batch: List of input data
            model_name: Name of the model to use

        Returns:
            List of prediction results
        """
        results = []

        for i, input_data in enumerate(input_batch):
            try:
                result = self.predict(input_data, model_name)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch prediction {i}: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e)
                })

        return results

    def _get_best_model(self) -> str:
        """Get the best available model."""
        # This is a simplified implementation
        # In practice, you would use performance metrics to determine the best model

        available_models = list(self.loaded_models.keys())
        if not available_models:
            # Try to load default models
            default_models = ['outcome_predictor', 'document_classifier', 'agent_predictor']
            for model_name in default_models:
                if self.load_model(model_name):
                    return model_name

        return available_models[0] if available_models else 'outcome_predictor'

    def _extract_features_from_dict(self, input_data: Union[Dict[str, Any], str]) -> np.ndarray:
        """Extract features from a dict or raw text using FeatureEngineer."""
        try:
            if isinstance(input_data, str):
                row = {'opinion_text': input_data}
            else:
                # Accept either 'case_text'/'document_text' or pre-extracted
                text = input_data.get('opinion_text') or input_data.get('case_text') or input_data.get('document_text') or ''
                meta = input_data.get('case_metadata') or input_data.get('document_metadata') or {}
                # Merge flat keys too
                flat_meta = {k: v for k, v in input_data.items() if k not in ('opinion_text', 'case_text', 'document_text', 'case_metadata', 'document_metadata')}
                merged = {**meta, **flat_meta}
                row = {'opinion_text': text, **merged}

            df = pd.DataFrame([row])
            feats = self.feature_engineer.extract_all_features(df, 'legal')
            X = self.feature_engineer.create_feature_matrix(feats)
            if X is None or X.size == 0:
                # Fallback minimal vector if feature extraction yielded nothing
                return np.array([[0.0] * 16])
            return X
        except Exception as e:
            logger.warning(f"Feature extraction failed, using fallback vector: {e}")
            return np.array([[0.0] * 16])

    def _record_inference(self, model_name: str, inference_time: float, prediction: Any):
        """Record inference for monitoring."""
        self.inference_history.append({
            'model_name': model_name,
            'inference_time': inference_time,
            'timestamp': datetime.now().isoformat(),
            'prediction_keys': list(prediction.keys()) if isinstance(prediction, dict) else []
        })

        # Keep only recent history (last 1000 inferences)
        if len(self.inference_history) > 1000:
            self.inference_history = self.inference_history[-1000:]

    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics.

        Returns:
            Inference statistics
        """
        if not self.inference_history:
            return {'total_inferences': 0}

        # Calculate statistics
        total_inferences = len(self.inference_history)
        avg_inference_time = np.mean([inf['inference_time'] for inf in self.inference_history])

        # Model usage statistics
        model_usage = {}
        for inf in self.inference_history:
            model_name = inf['model_name']
            model_usage[model_name] = model_usage.get(model_name, 0) + 1

        return {
            'total_inferences': total_inferences,
            'avg_inference_time': avg_inference_time,
            'model_usage': model_usage,
            'recent_inferences': self.inference_history[-10:]  # Last 10 inferences
        }

    def monitor_performance(self) -> Dict[str, Any]:
        """Monitor model performance.

        Returns:
            Performance monitoring results
        """
        stats = self.get_inference_stats()

        # Check for performance issues
        issues = []

        if stats['avg_inference_time'] > 1.0:  # More than 1 second
            issues.append('Slow inference time')

        if stats['total_inferences'] > 0:
            # Check for recent errors
            recent_errors = [inf for inf in self.inference_history[-100:] if 'error' in inf]
            if len(recent_errors) > 10:  # More than 10% error rate
                issues.append('High error rate')

        return {
            'status': 'healthy' if not issues else 'issues_detected',
            'issues': issues,
            'stats': stats
        }


# Convenience functions
def load_model_for_inference(model_name: str, version: str = 'latest') -> Optional[Any]:
    """Load a model for inference."""
    pipeline = InferencePipeline()
    return pipeline.load_model(model_name, version)

def predict_with_model(input_data: Union[np.ndarray, Dict[str, Any]],
                      model_name: str = 'best') -> Dict[str, Any]:
    """Make a prediction using a model."""
    pipeline = InferencePipeline()
    return pipeline.predict(input_data, model_name)

def batch_predict_with_model(input_batch: List[Union[np.ndarray, Dict[str, Any]]],
                           model_name: str = 'best') -> List[Dict[str, Any]]:
    """Make batch predictions using a model."""
    pipeline = InferencePipeline()
    return pipeline.batch_predict(input_batch, model_name)
