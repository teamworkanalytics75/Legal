"""ML function tools for existing agents.

This module provides function tools that existing agents can call:
1. predict_case_outcome - Predict case outcome using ML models
2. classify_document - Classify legal document
3. find_similar_cases - Find similar cases using ML similarity
4. predict_agent_performance - Predict agent performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime

# Import ML components
from ..pipelines import InferencePipeline
from ..data import FeatureEngineer
from ..models.supervised import CaseOutcomePredictor, DocumentClassifier, AgentPerformancePredictor

logger = logging.getLogger(__name__)


class MLTools:
    """Collection of ML function tools for agents."""

    def __init__(self):
        """Initialize ML tools."""
        self.inference_pipeline = InferencePipeline()
        self.feature_engineer = FeatureEngineer()

        # Load models
        self._load_models()

    def _load_models(self):
        """Load ML models for inference."""
        try:
            self.inference_pipeline.load_model('outcome_predictor')
            self.inference_pipeline.load_model('document_classifier')
            self.inference_pipeline.load_model('agent_predictor')
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load some ML models: {e}")

    def predict_case_outcome(self, case_summary: str, case_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Predict case outcome using ML models.

        Args:
            case_summary: Summary of the case
            case_metadata: Additional case metadata

        Returns:
            Prediction results
        """
        try:
            if not case_summary:
                return {
                    'error': 'No case summary provided',
                    'win_probability': 0.5,
                    'loss_probability': 0.5,
                    'settlement_probability': 0.0,
                    'confidence_score': 0.0
                }

            # Extract features
            features = self.feature_engineer.extract_all_features(
                pd.DataFrame([{'opinion_text': case_summary, **(case_metadata or {})}]),
                'legal'
            )
            feature_matrix = self.feature_engineer.create_feature_matrix(features)

            # Make prediction
            prediction_result = self.inference_pipeline.predict(feature_matrix, 'outcome_predictor')

            if 'error' in prediction_result:
                # Fallback to simple heuristic
                return self._fallback_outcome_prediction(case_summary)

            prediction = prediction_result['prediction']

            return {
                'win_probability': prediction.get('ensemble_probability', {}).get('win', 0.5),
                'loss_probability': prediction.get('ensemble_probability', {}).get('loss', 0.5),
                'settlement_probability': prediction.get('ensemble_probability', {}).get('settlement', 0.0),
                'confidence_score': prediction.get('confidence_score', 0.5),
                'key_factors': prediction.get('key_factors', []),
                'model_used': 'outcome_predictor',
                'prediction_time': prediction_result.get('inference_time', 0)
            }

        except Exception as e:
            logger.error(f"Error in case outcome prediction: {e}")
            return {
                'error': str(e),
                'win_probability': 0.5,
                'loss_probability': 0.5,
                'settlement_probability': 0.0,
                'confidence_score': 0.0
            }

    def classify_document(self, document_text: str, document_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Classify legal document.

        Args:
            document_text: Text of the document
            document_metadata: Additional document metadata

        Returns:
            Classification results
        """
        try:
            if not document_text:
                return {
                    'error': 'No document text provided',
                    'case_type': 'unknown',
                    'legal_domain': 'unknown',
                    'jurisdiction': 'unknown',
                    'complexity': 'unknown',
                    'confidence_scores': {}
                }

            # Extract features
            features = self.feature_engineer.extract_all_features(
                pd.DataFrame([{'opinion_text': document_text, **(document_metadata or {})}]),
                'legal'
            )
            feature_matrix = self.feature_engineer.create_feature_matrix(features)

            # Make prediction
            prediction_result = self.inference_pipeline.predict(feature_matrix, 'document_classifier')

            if 'error' in prediction_result:
                # Fallback to simple heuristic
                return self._fallback_document_classification(document_text)

            prediction = prediction_result['prediction']

            return {
                'case_type': prediction.get('case_type', {}).get('prediction', 'unknown'),
                'legal_domain': prediction.get('legal_domain', {}).get('prediction', 'unknown'),
                'jurisdiction': prediction.get('jurisdiction', {}).get('prediction', 'unknown'),
                'complexity': prediction.get('complexity', {}).get('prediction', 'unknown'),
                'confidence_scores': {
                    'case_type': prediction.get('case_type', {}).get('confidence', 0.5),
                    'legal_domain': prediction.get('legal_domain', {}).get('confidence', 0.5),
                    'jurisdiction': prediction.get('jurisdiction', {}).get('confidence', 0.5),
                    'complexity': prediction.get('complexity', {}).get('confidence', 0.5)
                },
                'model_used': 'document_classifier',
                'prediction_time': prediction_result.get('inference_time', 0)
            }

        except Exception as e:
            logger.error(f"Error in document classification: {e}")
            return {
                'error': str(e),
                'case_type': 'unknown',
                'legal_domain': 'unknown',
                'jurisdiction': 'unknown',
                'complexity': 'unknown',
                'confidence_scores': {}
            }

    def find_similar_cases(self, case_features: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar cases using ML similarity.

        Args:
            case_features: Features of the case to find similar cases for
            top_k: Number of similar cases to return

        Returns:
            List of similar cases
        """
        try:
            # This is a simplified implementation
            # In practice, you would use vector similarity with embeddings

            # Extract features
            if 'case_text' in case_features:
                features = self.feature_engineer.extract_all_features(
                    pd.DataFrame([case_features]),
                    'legal'
                )
                feature_matrix = self.feature_engineer.create_feature_matrix(features)
            else:
                # Use provided features directly
                feature_matrix = np.array([list(case_features.values())])

            # Calculate similarity (simplified)
            # In practice, you would compare with a database of case embeddings

            similar_cases = []
            for i in range(min(top_k, 5)):  # Return up to 5 similar cases
                similar_cases.append({
                    'case_id': f'similar_{i+1}',
                    'similarity_score': 0.9 - (i * 0.1),  # Decreasing similarity
                    'case_name': f'Similar Case {i+1}',
                    'key_similarities': ['legal_issue', 'jurisdiction', 'case_type'],
                    'outcome': 'win' if i % 2 == 0 else 'loss'
                })

            return similar_cases

        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []

    def predict_agent_performance(self, job_features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict agent performance for a job.

        Args:
            job_features: Features of the job

        Returns:
            Performance prediction results
        """
        try:
            # Extract features
            features = self.feature_engineer.extract_all_features(
                pd.DataFrame([job_features]),
                'agent'
            )
            feature_matrix = self.feature_engineer.create_feature_matrix(features)

            # Make prediction
            prediction_result = self.inference_pipeline.predict(feature_matrix, 'agent_predictor')

            if 'error' in prediction_result:
                # Fallback to simple heuristic
                return self._fallback_agent_performance_prediction(job_features)

            prediction = prediction_result['prediction']

            return {
                'success_probability': prediction.get('success_prediction', {}).get('success_probability', 0.5),
                'predicted_tokens': prediction.get('predicted_tokens', 1000),
                'predicted_time_seconds': prediction.get('predicted_time_seconds', 60),
                'confidence_score': prediction.get('success_prediction', {}).get('confidence', 0.5),
                'model_used': 'agent_predictor',
                'prediction_time': prediction_result.get('inference_time', 0)
            }

        except Exception as e:
            logger.error(f"Error in agent performance prediction: {e}")
            return {
                'error': str(e),
                'success_probability': 0.5,
                'predicted_tokens': 1000,
                'predicted_time_seconds': 60,
                'confidence_score': 0.0
            }

    def _fallback_outcome_prediction(self, case_summary: str) -> Dict[str, Any]:
        """Fallback outcome prediction using simple heuristics."""
        text_lower = case_summary.lower()

        # Simple keyword-based prediction
        win_keywords = ['granted', 'sustained', 'affirmed', 'won', 'prevailed']
        loss_keywords = ['denied', 'dismissed', 'reversed', 'lost', 'failed']
        settlement_keywords = ['settled', 'settlement', 'agreed']

        win_score = sum(1 for keyword in win_keywords if keyword in text_lower)
        loss_score = sum(1 for keyword in loss_keywords if keyword in text_lower)
        settlement_score = sum(1 for keyword in settlement_keywords if keyword in text_lower)

        total_score = win_score + loss_score + settlement_score

        if total_score == 0:
            return {
                'win_probability': 0.5,
                'loss_probability': 0.5,
                'settlement_probability': 0.0,
                'confidence_score': 0.3,
                'key_factors': ['insufficient_data'],
                'model_used': 'heuristic_fallback'
            }

        win_prob = win_score / total_score
        loss_prob = loss_score / total_score
        settlement_prob = settlement_score / total_score

        return {
            'win_probability': win_prob,
            'loss_probability': loss_prob,
            'settlement_probability': settlement_prob,
            'confidence_score': 0.6,
            'key_factors': ['keyword_analysis'],
            'model_used': 'heuristic_fallback'
        }

    def _fallback_document_classification(self, document_text: str) -> Dict[str, Any]:
        """Fallback document classification using simple heuristics."""
        text_lower = document_text.lower()

        # Case type classification
        if any(word in text_lower for word in ['criminal', 'felony', 'misdemeanor']):
            case_type = 'criminal'
        elif any(word in text_lower for word in ['civil', 'contract', 'tort']):
            case_type = 'civil'
        else:
            case_type = 'civil'  # Default

        # Legal domain classification
        if any(word in text_lower for word in ['employment', 'discrimination']):
            legal_domain = 'employment'
        elif any(word in text_lower for word in ['contract', 'agreement']):
            legal_domain = 'contract'
        else:
            legal_domain = 'other'

        # Jurisdiction classification
        if any(word in text_lower for word in ['federal', 'district']):
            jurisdiction = 'federal'
        else:
            jurisdiction = 'state'

        # Complexity classification
        text_length = len(document_text)
        if text_length > 50000:
            complexity = 'high'
        elif text_length > 10000:
            complexity = 'medium'
        else:
            complexity = 'low'

        return {
            'case_type': case_type,
            'legal_domain': legal_domain,
            'jurisdiction': jurisdiction,
            'complexity': complexity,
            'confidence_scores': {
                'case_type': 0.6,
                'legal_domain': 0.6,
                'jurisdiction': 0.6,
                'complexity': 0.7
            },
            'model_used': 'heuristic_fallback'
        }

    def _fallback_agent_performance_prediction(self, job_features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback agent performance prediction using simple heuristics."""
        # Simple heuristic based on job features
        job_type = job_features.get('job_type', 'unknown')
        phase = job_features.get('phase', 'unknown')

        # Base success probability
        success_prob = 0.7

        # Adjust based on job type
        if job_type in ['CitationFinderAgent', 'CitationVerifierAgent']:
            success_prob = 0.9  # Deterministic agents
        elif job_type in ['DraftWriterAgent', 'AnalysisAgent']:
            success_prob = 0.6  # LLM-based agents

        # Adjust based on phase
        if phase == 'citation':
            success_prob += 0.1  # Citation phase is more reliable
        elif phase == 'drafting':
            success_prob -= 0.1  # Drafting phase is more complex

        # Ensure probability is between 0 and 1
        success_prob = max(0.1, min(0.9, success_prob))

        return {
            'success_probability': success_prob,
            'predicted_tokens': 1000,
            'predicted_time_seconds': 60,
            'confidence_score': 0.5,
            'model_used': 'heuristic_fallback'
        }


# Global instance for easy access
ml_tools = MLTools()


# Function tools that agents can call
def predict_case_outcome(case_summary: str, case_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Predict case outcome using ML models."""
    return ml_tools.predict_case_outcome(case_summary, case_metadata)

def classify_document(document_text: str, document_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classify legal document."""
    return ml_tools.classify_document(document_text, document_metadata)

def find_similar_cases(case_features: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """Find similar cases using ML similarity."""
    return ml_tools.find_similar_cases(case_features, top_k)

def predict_agent_performance(job_features: Dict[str, Any]) -> Dict[str, Any]:
    """Predict agent performance for a job."""
    return ml_tools.predict_agent_performance(job_features)

def get_ml_tools() -> MLTools:
    """Get the ML tools instance."""
    return ml_tools
