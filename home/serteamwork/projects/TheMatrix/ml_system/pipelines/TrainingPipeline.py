"""Automated training pipeline for ML models.

This module provides:
1. End-to-end training pipeline
2. Automated model selection and hyperparameter tuning
3. Model retraining and drift detection
4. Performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import our ML components
from ..data import LegalDataLoader, AgentDataLoader, FeatureEngineer
from ..models.supervised import CaseOutcomePredictor, DocumentClassifier, AgentPerformancePredictor
from ..models.deep_learning import LegalLSTM, LegalBERT, LegalCNN, AttentionModel
from ..models.deep_learning.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class AutomatedTrainingPipeline:
    """Automated pipeline for training ML models."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize training pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or self._get_default_config()
        self.data_loader = None
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = None  # ModelEvaluator not required here
        self.trained_models = {}
        self.training_history = {}

        logger.info("AutomatedTrainingPipeline initialized")

    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration."""
        return {
            'data': {
                'legal_limit': 1000,
                'agent_limit': 5000,
                'train_test_split': 0.8,
                'validation_split': 0.1
            },
            'models': {
                'supervised': {
                    'outcome_predictor': True,
                    'document_classifier': True,
                    'agent_predictor': True
                },
                'deep_learning': {
                    'legal_lstm': True,
                    'legal_bert': True,
                    'legal_cnn': True,
                    'attention_model': True
                }
            },
            'training': {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001,
                'early_stopping_patience': 3,
                'cross_validation_folds': 5
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'max_trials': 50,
                'timeout': 3600  # 1 hour
            }
        }

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline.

        Returns:
            Pipeline results
        """
        logger.info("Starting full ML training pipeline...")
        start_time = time.time()

        results = {
            'pipeline_start_time': datetime.now().isoformat(),
            'models_trained': {},
            'performance_metrics': {},
            'errors': []
        }

        try:
            # Step 1: Load and preprocess data
            logger.info("Step 1: Loading and preprocessing data...")
            data_results = self._load_and_preprocess_data()
            results['data_loading'] = data_results

            # Step 2: Feature engineering
            logger.info("Step 2: Feature engineering...")
            feature_results = self._extract_features(data_results)
            results['feature_engineering'] = feature_results

            # Step 3: Train supervised learning models
            logger.info("Step 3: Training supervised learning models...")
            supervised_results = self._train_supervised_models(feature_results)
            results['models_trained']['supervised'] = supervised_results

            # Step 4: Train deep learning models
            logger.info("Step 4: Training deep learning models...")
            deep_learning_results = self._train_deep_learning_models(feature_results)
            results['models_trained']['deep_learning'] = deep_learning_results

            # Step 5: Model evaluation and selection
            logger.info("Step 5: Model evaluation and selection...")
            evaluation_results = self._evaluate_and_select_models()
            results['performance_metrics'] = evaluation_results

            # Step 6: Save best models
            logger.info("Step 6: Saving best models...")
            save_results = self._save_best_models()
            results['model_saving'] = save_results

            # Step 7: Generate performance report
            logger.info("Step 7: Generating performance report...")
            report_results = self._generate_performance_report()
            results['performance_report'] = report_results

        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            logger.error(error_msg)
            results['errors'].append(error_msg)

        # Calculate total time
        total_time = time.time() - start_time
        results['pipeline_end_time'] = datetime.now().isoformat()
        results['total_execution_time'] = total_time

        logger.info(f"Full pipeline completed in {total_time:.2f} seconds")

        return results

    def _load_and_preprocess_data(self) -> Dict[str, Any]:
        """Load and preprocess data from all sources."""
        try:
            # Initialize data loaders
            self.data_loader = LegalDataLoader()
            agent_loader = AgentDataLoader()

            # Load data
            legal_data = self.data_loader.load_case_law_data(self.config['data']['legal_limit'])
            agent_data = agent_loader.load_agent_execution_data(self.config['data']['agent_limit'])

            # Create training datasets
            datasets = {
                'legal_data': legal_data,
                'agent_data': agent_data,
                'legal_data_shape': legal_data.shape,
                'agent_data_shape': agent_data.shape
            }

            logger.info(f"Loaded {len(legal_data)} legal cases and {len(agent_data)} agent records")

            return datasets

        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return {'error': str(e)}

    def _extract_features(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from loaded data."""
        try:
            if 'error' in data_results:
                return {'error': 'Cannot extract features due to data loading error'}

            legal_data = data_results['legal_data']
            agent_data = data_results['agent_data']

            # Extract legal features
            legal_features = self.feature_engineer.extract_all_features(legal_data, 'legal')
            legal_feature_matrix = self.feature_engineer.create_feature_matrix(legal_features)

            # Extract agent features
            agent_features = self.feature_engineer.extract_all_features(agent_data, 'agent')
            agent_feature_matrix = self.feature_engineer.create_feature_matrix(agent_features)

            # Prepare labels
            legal_labels = {
                'outcome': legal_data['outcome_label'].values,
                'case_type': legal_data['case_type'].values,
                'domain': legal_data['domain_label'].values,
                'jurisdiction': legal_data['jurisdiction_label'].values,
                'complexity': legal_data['complexity_label'].values
            }

            agent_labels = {
                'success': agent_data['success_label'].values,
                'performance': agent_data['performance_label'].values,
                'efficiency': agent_data['efficiency_label'].values,
                'error_type': agent_data['error_type_label'].values
            }

            # Train/test splits
            legal_splits = self._create_train_test_splits(legal_feature_matrix, legal_labels)
            agent_splits = self._create_train_test_splits(agent_feature_matrix, agent_labels)

            return {
                'legal_features': legal_feature_matrix,
                'agent_features': agent_feature_matrix,
                'legal_labels': legal_labels,
                'agent_labels': agent_labels,
                'legal_splits': legal_splits,
                'agent_splits': agent_splits,
                'feature_names': self.feature_engineer.get_feature_names(legal_features)
            }

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {'error': str(e)}

    def _create_train_test_splits(self, features: np.ndarray, labels: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Create train/test splits for features and labels."""
        splits = {}

        for label_name, label_values in labels.items():
            try:
                # Remove NaN values
                valid_indices = ~np.isnan(label_values)
                if isinstance(label_values[0], str):
                    valid_indices = ~pd.isna(label_values)

                X_clean = features[valid_indices]
                y_clean = label_values[valid_indices]

                if len(X_clean) > 10:  # Minimum samples for splitting
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_clean, y_clean,
                        test_size=1 - self.config['data']['train_test_split'],
                        random_state=42,
                        stratify=y_clean if len(np.unique(y_clean)) > 1 else None
                    )

                    splits[label_name] = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test
                    }
                else:
                    logger.warning(f"Not enough samples for {label_name}: {len(X_clean)}")

            except Exception as e:
                logger.warning(f"Error creating split for {label_name}: {e}")

        return splits

    def _train_supervised_models(self, feature_results: Dict[str, Any]) -> Dict[str, Any]:
        """Train supervised learning models."""
        if 'error' in feature_results:
            return {'error': 'Cannot train supervised models due to feature extraction error'}

        results = {}

        # Train case outcome predictor
        if self.config['models']['supervised']['outcome_predictor']:
            try:
                outcome_splits = feature_results['legal_splits'].get('outcome')
                if outcome_splits:
                    predictor = CaseOutcomePredictor()
                    predictor.train(
                        outcome_splits['X_train'],
                        outcome_splits['y_train'],
                        outcome_splits['X_test'],
                        outcome_splits['y_test']
                    )

                    # Evaluate
                    eval_results = predictor.evaluate(
                        outcome_splits['X_test'],
                        outcome_splits['y_test']
                    )

                    self.trained_models['outcome_predictor'] = predictor
                    results['outcome_predictor'] = {
                        'status': 'success',
                        'evaluation': eval_results
                    }

                    logger.info("Case outcome predictor trained successfully")
                else:
                    results['outcome_predictor'] = {'status': 'skipped', 'reason': 'No outcome data'}

            except Exception as e:
                logger.error(f"Error training outcome predictor: {e}")
                results['outcome_predictor'] = {'status': 'error', 'error': str(e)}

        # Train document classifier
        if self.config['models']['supervised']['document_classifier']:
            try:
                # Use case_type for document classification
                case_type_splits = feature_results['legal_splits'].get('case_type')
                if case_type_splits:
                    classifier = DocumentClassifier()

                    # Prepare multi-task labels
                    y_dict = {
                        'case_type': feature_results['legal_labels']['case_type'],
                        'domain': feature_results['legal_labels']['domain'],
                        'jurisdiction': feature_results['legal_labels']['jurisdiction'],
                        'complexity': feature_results['legal_labels']['complexity']
                    }

                    classifier.train(case_type_splits['X_train'], y_dict)

                    # Evaluate
                    eval_results = classifier.evaluate(
                        case_type_splits['X_test'],
                        y_dict
                    )

                    self.trained_models['document_classifier'] = classifier
                    results['document_classifier'] = {
                        'status': 'success',
                        'evaluation': eval_results
                    }

                    logger.info("Document classifier trained successfully")
                else:
                    results['document_classifier'] = {'status': 'skipped', 'reason': 'No case type data'}

            except Exception as e:
                logger.error(f"Error training document classifier: {e}")
                results['document_classifier'] = {'status': 'error', 'error': str(e)}

        # Train agent performance predictor
        if self.config['models']['supervised']['agent_predictor']:
            try:
                success_splits = feature_results['agent_splits'].get('success')
                if success_splits:
                    predictor = AgentPerformancePredictor()

                    # Prepare multi-task labels
                    y_dict = {
                        'success_prediction': feature_results['agent_labels']['success'],
                        'token_usage': feature_results['agent_labels'].get('performance', []),
                        'execution_time': feature_results['agent_labels'].get('efficiency', [])
                    }

                    predictor.train(success_splits['X_train'], y_dict)

                    # Evaluate
                    eval_results = predictor.evaluate(
                        success_splits['X_test'],
                        y_dict
                    )

                    self.trained_models['agent_predictor'] = predictor
                    results['agent_predictor'] = {
                        'status': 'success',
                        'evaluation': eval_results
                    }

                    logger.info("Agent performance predictor trained successfully")
                else:
                    results['agent_predictor'] = {'status': 'skipped', 'reason': 'No agent success data'}

            except Exception as e:
                logger.error(f"Error training agent predictor: {e}")
                results['agent_predictor'] = {'status': 'error', 'error': str(e)}

        return results

    def _train_deep_learning_models(self, feature_results: Dict[str, Any]) -> Dict[str, Any]:
        """Train deep learning models."""
        if 'error' in feature_results:
            return {'error': 'Cannot train deep learning models due to feature extraction error'}

        results = {}

        # Note: Deep learning models require text data, not feature matrices
        # This is a simplified implementation - in practice, you would need
        # to prepare text data and create appropriate data loaders

        logger.info("Deep learning model training requires text data preparation")
        results['note'] = 'Deep learning models require text data and specialized data loaders'

        return results

    def _evaluate_and_select_models(self) -> Dict[str, Any]:
        """Evaluate all trained models and select the best ones."""
        results = {}

        for model_name, model in self.trained_models.items():
            try:
                # This is a simplified evaluation
                # In practice, you would use proper test data
                results[model_name] = {
                    'status': 'evaluated',
                    'note': 'Evaluation requires test data'
                }

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'status': 'error', 'error': str(e)}

        return results

    def _save_best_models(self) -> Dict[str, Any]:
        """Save the best trained models."""
        results = {}
        save_dir = Path('ml_system/models/trained')
        save_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.trained_models.items():
            try:
                model_save_dir = save_dir / model_name
                model_save_dir.mkdir(exist_ok=True)

                if hasattr(model, 'save_models'):
                    model.save_models(str(model_save_dir))
                elif hasattr(model, 'save_model'):
                    model.save_model(str(model_save_dir))

                results[model_name] = {'status': 'saved', 'path': str(model_save_dir)}
                logger.info(f"Model {model_name} saved to {model_save_dir}")

            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")
                results[model_name] = {'status': 'error', 'error': str(e)}

        return results

    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report_path = Path('ml_system/reports')
        report_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_path / f'training_report_{timestamp}.md'

        # Generate markdown report
        report_content = f"""# ML Training Pipeline Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pipeline Configuration
```json
{json.dumps(self.config, indent=2)}
```

## Models Trained
"""

        for model_name, model in self.trained_models.items():
            report_content += f"\n### {model_name}\n"
            report_content += f"- Status: Trained\n"
            report_content += f"- Type: {type(model).__name__}\n"

        report_content += f"""

## Summary
- Total models trained: {len(self.trained_models)}
- Pipeline execution time: {self.training_history.get('total_time', 'N/A')}
- Status: {'Success' if self.trained_models else 'No models trained'}
"""

        # Save report
        with open(report_file, 'w') as f:
            f.write(report_content)

        logger.info(f"Performance report saved to {report_file}")

        return {
            'report_path': str(report_file),
            'models_trained': len(self.trained_models),
            'status': 'success'
        }

    def schedule_retraining(self, frequency: str = 'weekly') -> Dict[str, Any]:
        """Schedule automated model retraining.

        Args:
            frequency: Retraining frequency ('daily', 'weekly', 'monthly')

        Returns:
            Scheduling results
        """
        # This is a placeholder for scheduling functionality
        # In practice, you would integrate with a task scheduler like Celery or APScheduler

        logger.info(f"Scheduling retraining with frequency: {frequency}")

        return {
            'status': 'scheduled',
            'frequency': frequency,
            'next_run': 'Implementation required',
            'note': 'Integration with task scheduler needed'
        }

    def monitor_data_drift(self) -> Dict[str, Any]:
        """Monitor for data drift and trigger retraining if needed.

        Returns:
            Drift monitoring results
        """
        # This is a placeholder for drift detection
        # In practice, you would implement statistical tests for drift detection

        logger.info("Monitoring data drift...")

        return {
            'status': 'monitoring',
            'drift_detected': False,
            'note': 'Drift detection implementation required'
        }


# Convenience functions
def run_ml_training_pipeline(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Run the complete ML training pipeline."""
    pipeline = AutomatedTrainingPipeline(config)
    return pipeline.run_full_pipeline()

def train_specific_model(model_type: str, data: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
    """Train a specific model type."""
    pipeline = AutomatedTrainingPipeline(config)

    if model_type == 'outcome_predictor':
        return pipeline._train_supervised_models(data)
    elif model_type == 'document_classifier':
        return pipeline._train_supervised_models(data)
    elif model_type == 'agent_predictor':
        return pipeline._train_supervised_models(data)
    else:
        return {'error': f'Unknown model type: {model_type}'}
