"""Model evaluation and monitoring pipeline.

This module provides:
1. Comprehensive model evaluation
2. Performance monitoring
3. Drift detection
4. Evaluation report generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""

    def __init__(self):
        """Initialize evaluator."""
        self.evaluation_results = {}
        self.evaluation_history = []

    def evaluate_model(self, model: Any, test_data: Union[Tuple[np.ndarray, np.ndarray], Dict[str, Any]],
                      model_name: str = 'model') -> Dict[str, Any]:
        """Evaluate a model comprehensively.

        Args:
            model: Model to evaluate
            test_data: Test data (features, labels) or dictionary
            model_name: Name for the model

        Returns:
            Evaluation metrics
        """
        start_time = time.time()

        try:
            # Prepare test data
            if isinstance(test_data, tuple):
                X_test, y_test = test_data
            else:
                X_test = test_data.get('features')
                y_test = test_data.get('labels')

            if X_test is None or y_test is None:
                return {'error': 'Invalid test data format'}

            # Determine model type and evaluation method
            model_type = self._detect_model_type(model)

            if model_type == 'classification':
                metrics = self._evaluate_classification_model(model, X_test, y_test)
            elif model_type == 'regression':
                metrics = self._evaluate_regression_model(model, X_test, y_test)
            elif model_type == 'multi_task':
                metrics = self._evaluate_multi_task_model(model, X_test, y_test)
            else:
                return {'error': f'Unknown model type: {model_type}'}

            # Add metadata
            metrics['model_name'] = model_name
            metrics['model_type'] = model_type
            metrics['evaluation_time'] = time.time() - start_time
            metrics['timestamp'] = datetime.now().isoformat()
            metrics['test_samples'] = len(X_test)

            # Store results
            self.evaluation_results[model_name] = metrics
            self.evaluation_history.append({
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })

            logger.info(f"Model {model_name} evaluated successfully")
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {'error': str(e)}

    def _detect_model_type(self, model: Any) -> str:
        """Detect the type of model."""
        model_class = type(model).__name__

        if 'Classifier' in model_class or 'Predictor' in model_class:
            return 'classification'
        elif 'Regressor' in model_class:
            return 'regression'
        elif 'MultiTask' in model_class or 'DocumentClassifier' in model_class:
            return 'multi_task'
        else:
            # Try to determine from methods
            if hasattr(model, 'predict_proba'):
                return 'classification'
            elif hasattr(model, 'predict'):
                return 'regression'
            else:
                return 'unknown'

    def _evaluate_classification_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a classification model."""
        try:
            # Get predictions
            y_pred = model.predict(X_test)

            # Basic metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            # ROC-AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                except:
                    pass

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()

            # Per-class metrics
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                metrics['per_class_metrics'] = report
            except:
                pass

            return metrics

        except Exception as e:
            logger.error(f"Error in classification evaluation: {e}")
            return {'error': str(e)}

    def _evaluate_regression_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate a regression model."""
        try:
            # Get predictions
            y_pred = model.predict(X_test)

            # Regression metrics
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }

            # Additional metrics
            metrics['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

            return metrics

        except Exception as e:
            logger.error(f"Error in regression evaluation: {e}")
            return {'error': str(e)}

    def _evaluate_multi_task_model(self, model: Any, X_test: np.ndarray, y_test: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate a multi-task model."""
        try:
            metrics = {}

            # Evaluate each task
            for task_name, task_labels in y_test.items():
                try:
                    if hasattr(model, 'classify_document'):
                        # Document classifier
                        result = model.classify_document(X_test[0:1])  # Single sample
                        task_metrics = {f'{task_name}_confidence': result.get(task_name, {}).get('confidence', 0)}
                    else:
                        # Generic multi-task evaluation
                        task_metrics = {'accuracy': 0.0}  # Placeholder

                    metrics.update(task_metrics)

                except Exception as e:
                    logger.warning(f"Error evaluating task {task_name}: {e}")
                    metrics[f'{task_name}_error'] = str(e)

            return metrics

        except Exception as e:
            logger.error(f"Error in multi-task evaluation: {e}")
            return {'error': str(e)}

    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Compare multiple models.

        Args:
            model_names: List of model names to compare

        Returns:
            Comparison DataFrame
        """
        comparison_data = []

        for model_name in model_names:
            if model_name in self.evaluation_results:
                metrics = self.evaluation_results[model_name]

                # Extract key metrics
                row = {'model_name': model_name}

                if 'accuracy' in metrics:
                    row['accuracy'] = metrics['accuracy']
                if 'f1_score' in metrics:
                    row['f1_score'] = metrics['f1_score']
                if 'precision' in metrics:
                    row['precision'] = metrics['precision']
                if 'recall' in metrics:
                    row['recall'] = metrics['recall']
                if 'r2' in metrics:
                    row['r2'] = metrics['r2']
                if 'mae' in metrics:
                    row['mae'] = metrics['mae']

                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def generate_evaluation_report(self, output_path: str, include_plots: bool = True):
        """Generate a comprehensive evaluation report.

        Args:
            output_path: Path to save the report
            include_plots: Whether to include plots
        """
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Create markdown report
        report_content = f"""# Model Evaluation Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

Total models evaluated: {len(self.evaluation_results)}

"""

        # Model comparison table
        if self.evaluation_results:
            comparison_df = self.compare_models(list(self.evaluation_results.keys()))
            if not comparison_df.empty:
                report_content += "## Model Comparison\n\n"
                report_content += comparison_df.to_markdown(index=False) + "\n\n"

        # Detailed results for each model
        for model_name, metrics in self.evaluation_results.items():
            report_content += f"## {model_name}\n\n"

            if 'error' in metrics:
                report_content += f"**Error**: {metrics['error']}\n\n"
                continue

            # Basic metrics
            if 'accuracy' in metrics:
                report_content += f"- **Accuracy**: {metrics['accuracy']:.4f}\n"
            if 'precision' in metrics:
                report_content += f"- **Precision**: {metrics['precision']:.4f}\n"
            if 'recall' in metrics:
                report_content += f"- **Recall**: {metrics['recall']:.4f}\n"
            if 'f1_score' in metrics:
                report_content += f"- **F1 Score**: {metrics['f1_score']:.4f}\n"
            if 'r2' in metrics:
                report_content += f"- **R² Score**: {metrics['r2']:.4f}\n"
            if 'mae' in metrics:
                report_content += f"- **MAE**: {metrics['mae']:.4f}\n"
            if 'rmse' in metrics:
                report_content += f"- **RMSE**: {metrics['rmse']:.4f}\n"

            # Additional info
            if 'test_samples' in metrics:
                report_content += f"- **Test Samples**: {metrics['test_samples']}\n"
            if 'evaluation_time' in metrics:
                report_content += f"- **Evaluation Time**: {metrics['evaluation_time']:.2f}s\n"

            report_content += "\n"

        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)

        # Generate plots if requested
        if include_plots and self.evaluation_results:
            self._generate_evaluation_plots(report_path.parent)

        logger.info(f"Evaluation report saved to {output_path}")

    def _generate_evaluation_plots(self, output_dir: Path):
        """Generate evaluation plots."""
        try:
            # Model comparison plot
            if len(self.evaluation_results) > 1:
                comparison_df = self.compare_models(list(self.evaluation_results.keys()))

                # Accuracy comparison
                if 'accuracy' in comparison_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.bar(comparison_df['model_name'], comparison_df['accuracy'])
                    plt.title('Model Accuracy Comparison')
                    plt.ylabel('Accuracy')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(output_dir / 'accuracy_comparison.png')
                    plt.close()

                # F1 Score comparison
                if 'f1_score' in comparison_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.bar(comparison_df['model_name'], comparison_df['f1_score'])
                    plt.title('Model F1 Score Comparison')
                    plt.ylabel('F1 Score')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(output_dir / 'f1_score_comparison.png')
                    plt.close()

            # Confusion matrices
            for model_name, metrics in self.evaluation_results.items():
                if 'confusion_matrix' in metrics:
                    cm = np.array(metrics['confusion_matrix'])
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix - {model_name}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.tight_layout()
                    plt.savefig(output_dir / f'confusion_matrix_{model_name}.png')
                    plt.close()

            logger.info(f"Evaluation plots saved to {output_dir}")

        except Exception as e:
            logger.warning(f"Error generating plots: {e}")


class PerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(self, monitoring_window: int = 100):
        """Initialize performance monitor.

        Args:
            monitoring_window: Number of recent predictions to monitor
        """
        self.monitoring_window = monitoring_window
        self.performance_history = []
        self.drift_detector = DriftDetector()

    def record_prediction(self, model_name: str, prediction: Any,
                         actual: Optional[Any] = None, metadata: Optional[Dict] = None):
        """Record a prediction for monitoring.

        Args:
            model_name: Name of the model
            prediction: Model prediction
            actual: Actual value (if available)
            metadata: Additional metadata
        """
        record = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual,
            'metadata': metadata or {}
        }

        self.performance_history.append(record)

        # Keep only recent history
        if len(self.performance_history) > self.monitoring_window:
            self.performance_history = self.performance_history[-self.monitoring_window:]

    def monitor_production_performance(self) -> Dict[str, Any]:
        """Monitor production performance.

        Returns:
            Performance monitoring results
        """
        if not self.performance_history:
            return {'status': 'no_data', 'message': 'No predictions recorded'}

        # Calculate performance metrics
        recent_predictions = self.performance_history[-50:]  # Last 50 predictions

        # Accuracy (if actual values available)
        accuracy = None
        if any(record['actual'] is not None for record in recent_predictions):
            correct = sum(1 for record in recent_predictions
                         if record['actual'] is not None and record['prediction'] == record['actual'])
            total = sum(1 for record in recent_predictions if record['actual'] is not None)
            if total > 0:
                accuracy = correct / total

        # Response time
        response_times = [record['metadata'].get('response_time', 0) for record in recent_predictions]
        avg_response_time = np.mean(response_times) if response_times else 0

        # Error rate
        errors = sum(1 for record in recent_predictions if 'error' in record['metadata'])
        error_rate = errors / len(recent_predictions)

        # Drift detection
        drift_results = self.drift_detector.detect_drift(self.performance_history)

        return {
            'status': 'healthy' if error_rate < 0.1 and avg_response_time < 1.0 else 'issues_detected',
            'accuracy': accuracy,
            'avg_response_time': avg_response_time,
            'error_rate': error_rate,
            'total_predictions': len(self.performance_history),
            'drift_detected': drift_results['drift_detected'],
            'drift_score': drift_results['drift_score']
        }

    def generate_performance_report(self, output_path: str):
        """Generate a performance monitoring report.

        Args:
            output_path: Path to save the report
        """
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Get performance summary
        performance_summary = self.monitor_production_performance()

        # Create report
        report_content = f"""# Performance Monitoring Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

- **Status**: {performance_summary['status']}
- **Total Predictions**: {performance_summary['total_predictions']}
- **Average Response Time**: {performance_summary['avg_response_time']:.3f}s
- **Error Rate**: {performance_summary['error_rate']:.3f}
- **Drift Detected**: {performance_summary['drift_detected']}
- **Drift Score**: {performance_summary['drift_score']:.3f}

"""

        if performance_summary['accuracy'] is not None:
            report_content += f"- **Accuracy**: {performance_summary['accuracy']:.3f}\n\n"

        # Recent predictions
        report_content += "## Recent Predictions\n\n"
        recent_predictions = self.performance_history[-10:]
        for record in recent_predictions:
            report_content += f"- **{record['timestamp']}**: {record['model_name']} - {record['prediction']}\n"

        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)

        logger.info(f"Performance report saved to {output_path}")


class DriftDetector:
    """Detect data drift in model predictions."""

    def __init__(self, window_size: int = 50):
        """Initialize drift detector.

        Args:
            window_size: Size of window for drift detection
        """
        self.window_size = window_size
        self.baseline_stats = None

    def detect_drift(self, performance_history: List[Dict]) -> Dict[str, Any]:
        """Detect drift in performance history.

        Args:
            performance_history: List of performance records

        Returns:
            Drift detection results
        """
        if len(performance_history) < self.window_size * 2:
            return {'drift_detected': False, 'drift_score': 0.0, 'message': 'Insufficient data'}

        try:
            # Extract predictions
            predictions = [record['prediction'] for record in performance_history]

            # Split into baseline and current windows
            baseline = predictions[:self.window_size]
            current = predictions[-self.window_size:]

            # Calculate drift score (simplified)
            drift_score = self._calculate_drift_score(baseline, current)

            # Threshold for drift detection
            drift_threshold = 0.1
            drift_detected = drift_score > drift_threshold

            return {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'threshold': drift_threshold,
                'baseline_size': len(baseline),
                'current_size': len(current)
            }

        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return {'drift_detected': False, 'drift_score': 0.0, 'error': str(e)}

    def _calculate_drift_score(self, baseline: List, current: List) -> float:
        """Calculate drift score between baseline and current data."""
        try:
            # Convert to numpy arrays
            baseline_array = np.array(baseline)
            current_array = np.array(current)

            # Calculate statistical difference
            baseline_mean = np.mean(baseline_array)
            current_mean = np.mean(current_array)

            baseline_std = np.std(baseline_array)
            current_std = np.std(current_array)

            # Simple drift score based on mean and std differences
            mean_diff = abs(baseline_mean - current_mean) / (baseline_std + 1e-8)
            std_diff = abs(baseline_std - current_std) / (baseline_std + 1e-8)

            drift_score = (mean_diff + std_diff) / 2

            return float(drift_score)

        except Exception as e:
            logger.error(f"Error calculating drift score: {e}")
            return 0.0


# Convenience functions
def evaluate_model_comprehensive(model: Any, test_data: Union[Tuple[np.ndarray, np.ndarray], Dict[str, Any]],
                                model_name: str = 'model') -> Dict[str, Any]:
    """Evaluate a model comprehensively."""
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(model, test_data, model_name)

def compare_models_performance(model_names: List[str], evaluator: ModelEvaluator) -> pd.DataFrame:
    """Compare multiple models."""
    return evaluator.compare_models(model_names)

def generate_evaluation_report(models_results: Dict[str, Dict], output_path: str):
    """Generate evaluation report for multiple models."""
    evaluator = ModelEvaluator()
    evaluator.evaluation_results = models_results
    evaluator.generate_evaluation_report(output_path)
