"""ML-focused atomic agents for The Matrix legal AI system.

This module implements:
1. OutcomePredictorAgent - Predicts legal case outcomes
2. DocumentClassifierAgent - Classifies legal documents
3. PatternRecognizerAgent - Recognizes patterns in legal data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import asyncio

# Import atomic agent base class
try:
    from writer_agents.code.atomic_agent import AtomicAgent
    ATOMIC_AGENT_AVAILABLE = True
except ImportError:
    # Fallback stub so class definitions don't fail
    class AtomicAgent(object):
        def __init__(self, *args, **kwargs):
            pass

    ATOMIC_AGENT_AVAILABLE = False
    logging.warning("AtomicAgent not available. ML agents will be disabled.")

# Import ML components
from ..models.supervised import CaseOutcomePredictor, DocumentClassifier, AgentPerformancePredictor
from ..pipelines import InferencePipeline
from ..data import FeatureEngineer

logger = logging.getLogger(__name__)


class OutcomePredictorAgent(AtomicAgent):
    """Predicts legal case outcome using ML models."""

    def __init__(self):
        """Initialize outcome predictor agent."""
        if not ATOMIC_AGENT_AVAILABLE:
            raise ImportError("AtomicAgent is required for OutcomePredictorAgent")

        super().__init__()
        self.duty = "Predict legal case outcome probability"
        self.is_deterministic = False  # Uses ML models
        self.cost_tier = "low"  # Fast inference
        self.inference_pipeline = None
        self.feature_engineer = FeatureEngineer()

        # Initialize inference pipeline
        try:
            self.inference_pipeline = InferencePipeline()
            self.inference_pipeline.load_model('outcome_predictor')
        except Exception as e:
            logger.warning(f"Could not load outcome predictor model: {e}")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute outcome prediction.

        Args:
            input_data: Should contain case information

        Returns:
            Prediction results with confidence
        """
        try:
            # Extract case information
            case_text = input_data.get('case_text', '')
            case_metadata = input_data.get('case_metadata', {})

            if not case_text:
                return {
                    'error': 'No case text provided',
                    'win_probability': 0.5,
                    'loss_probability': 0.5,
                    'settlement_probability': 0.0,
                    'confidence_score': 0.0
                }

            # Extract features
            features = self.feature_engineer.extract_all_features(
                pd.DataFrame([{'opinion_text': case_text, **case_metadata}]),
                'legal'
            )
            feature_matrix = self.feature_engineer.create_feature_matrix(features)

            # Make prediction
            if self.inference_pipeline:
                prediction_result = self.inference_pipeline.predict(feature_matrix, 'outcome_predictor')

                if 'error' in prediction_result:
                    # Fallback to simple heuristic
                    return self._fallback_prediction(case_text, case_metadata)

                prediction = prediction_result['prediction']

                # Extract probabilities
                ensemble_prob = prediction.get('ensemble_probability', {})

                return {
                    'win_probability': ensemble_prob.get('win', 0.5),
                    'loss_probability': ensemble_prob.get('loss', 0.5),
                    'settlement_probability': ensemble_prob.get('settlement', 0.0),
                    'confidence_score': prediction.get('confidence_score', 0.5),
                    'key_factors': prediction.get('key_factors', []),
                    'model_used': 'outcome_predictor',
                    'prediction_time': prediction_result.get('inference_time', 0)
                }
            else:
                return self._fallback_prediction(case_text, case_metadata)

        except Exception as e:
            logger.error(f"Error in outcome prediction: {e}")
            return {
                'error': str(e),
                'win_probability': 0.5,
                'loss_probability': 0.5,
                'settlement_probability': 0.0,
                'confidence_score': 0.0
            }

    def _fallback_prediction(self, case_text: str, case_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction using simple heuristics."""
        text_lower = case_text.lower()

        # Simple keyword-based prediction
        win_keywords = ['granted', 'sustained', 'affirmed', 'won', 'prevailed', 'successful']
        loss_keywords = ['denied', 'dismissed', 'reversed', 'lost', 'failed', 'unsuccessful']
        settlement_keywords = ['settled', 'settlement', 'agreed', 'compromise']

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


class DocumentClassifierAgent(AtomicAgent):
    """Classifies legal documents by type and domain."""

    def __init__(self):
        """Initialize document classifier agent."""
        if not ATOMIC_AGENT_AVAILABLE:
            raise ImportError("AtomicAgent is required for DocumentClassifierAgent")

        super().__init__()
        self.duty = "Classify legal document type and domain"
        self.is_deterministic = False
        self.cost_tier = "low"
        self.inference_pipeline = None
        self.feature_engineer = FeatureEngineer()

        # Initialize inference pipeline
        try:
            self.inference_pipeline = InferencePipeline()
            self.inference_pipeline.load_model('document_classifier')
        except Exception as e:
            logger.warning(f"Could not load document classifier model: {e}")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document classification.

        Args:
            input_data: Should contain document text

        Returns:
            Classification results
        """
        try:
            # Extract document information
            document_text = input_data.get('document_text', '')
            document_metadata = input_data.get('document_metadata', {})

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
                pd.DataFrame([{'opinion_text': document_text, **document_metadata}]),
                'legal'
            )
            feature_matrix = self.feature_engineer.create_feature_matrix(features)

            # Make prediction
            if self.inference_pipeline:
                prediction_result = self.inference_pipeline.predict(feature_matrix, 'document_classifier')

                if 'error' in prediction_result:
                    # Fallback to simple heuristic
                    return self._fallback_classification(document_text, document_metadata)

                prediction = prediction_result['prediction']

                # Extract classifications
                case_type = prediction.get('case_type', {}).get('prediction', 'unknown')
                legal_domain = prediction.get('legal_domain', {}).get('prediction', 'unknown')
                jurisdiction = prediction.get('jurisdiction', {}).get('prediction', 'unknown')
                complexity = prediction.get('complexity', {}).get('prediction', 'unknown')

                confidence_scores = {
                    'case_type': prediction.get('case_type', {}).get('confidence', 0.5),
                    'legal_domain': prediction.get('legal_domain', {}).get('confidence', 0.5),
                    'jurisdiction': prediction.get('jurisdiction', {}).get('confidence', 0.5),
                    'complexity': prediction.get('complexity', {}).get('confidence', 0.5)
                }

                return {
                    'case_type': case_type,
                    'legal_domain': legal_domain,
                    'jurisdiction': jurisdiction,
                    'complexity': complexity,
                    'confidence_scores': confidence_scores,
                    'model_used': 'document_classifier',
                    'prediction_time': prediction_result.get('inference_time', 0)
                }
            else:
                return self._fallback_classification(document_text, document_metadata)

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

    def _fallback_classification(self, document_text: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback classification using simple heuristics."""
        text_lower = document_text.lower()

        # Case type classification
        if any(word in text_lower for word in ['criminal', 'felony', 'misdemeanor']):
            case_type = 'criminal'
        elif any(word in text_lower for word in ['civil', 'contract', 'tort']):
            case_type = 'civil'
        elif any(word in text_lower for word in ['administrative', 'regulatory']):
            case_type = 'administrative'
        else:
            case_type = 'civil'  # Default

        # Legal domain classification
        if any(word in text_lower for word in ['employment', 'discrimination', 'harassment']):
            legal_domain = 'employment'
        elif any(word in text_lower for word in ['contract', 'agreement', 'breach']):
            legal_domain = 'contract'
        elif any(word in text_lower for word in ['tort', 'negligence', 'liability']):
            legal_domain = 'tort'
        else:
            legal_domain = 'other'

        # Jurisdiction classification
        if any(word in text_lower for word in ['federal', 'district', 'circuit']):
            jurisdiction = 'federal'
        elif any(word in text_lower for word in ['supreme', 'court']):
            jurisdiction = 'supreme'
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


class PatternRecognizerAgent(AtomicAgent):
    """Recognizes patterns in legal cases and agent execution."""

    def __init__(self):
        """Initialize pattern recognizer agent."""
        if not ATOMIC_AGENT_AVAILABLE:
            raise ImportError("AtomicAgent is required for PatternRecognizerAgent")

        super().__init__()
        self.duty = "Identify patterns in legal data and agent behavior"
        self.is_deterministic = False
        self.cost_tier = "medium"
        self.inference_pipeline = None
        self.feature_engineer = FeatureEngineer()

        # Initialize inference pipeline
        try:
            self.inference_pipeline = InferencePipeline()
            self.inference_pipeline.load_model('agent_predictor')
        except Exception as e:
            logger.warning(f"Could not load agent predictor model: {e}")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pattern recognition.

        Args:
            input_data: Should contain data to analyze

        Returns:
            Pattern recognition results
        """
        try:
            # Extract data to analyze
            data_type = input_data.get('data_type', 'legal')  # 'legal' or 'agent'
            data_text = input_data.get('data_text', '')
            data_metadata = input_data.get('data_metadata', {})

            if not data_text:
                return {
                    'error': 'No data provided for pattern recognition',
                    'similar_cases': [],
                    'success_patterns': [],
                    'risk_factors': [],
                    'recommendations': []
                }

            # Extract features
            features = self.feature_engineer.extract_all_features(
                pd.DataFrame([{'opinion_text': data_text, **data_metadata}]),
                data_type
            )
            feature_matrix = self.feature_engineer.create_feature_matrix(features)

            # Analyze patterns
            patterns = self._analyze_patterns(data_text, data_metadata, data_type)

            # Get similar cases (simplified)
            similar_cases = self._find_similar_cases(feature_matrix, data_type)

            # Get success patterns
            success_patterns = self._identify_success_patterns(data_text, data_metadata)

            # Get risk factors
            risk_factors = self._identify_risk_factors(data_text, data_metadata)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                patterns, similar_cases, success_patterns, risk_factors
            )

            return {
                'similar_cases': similar_cases,
                'success_patterns': success_patterns,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'patterns_detected': patterns,
                'data_type': data_type,
                'analysis_time': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in pattern recognition: {e}")
            return {
                'error': str(e),
                'similar_cases': [],
                'success_patterns': [],
                'risk_factors': [],
                'recommendations': []
            }

    def _analyze_patterns(self, text: str, metadata: Dict[str, Any], data_type: str) -> List[str]:
        """Analyze patterns in the data."""
        patterns = []
        text_lower = text.lower()

        # Legal patterns
        if data_type == 'legal':
            if 'precedent' in text_lower:
                patterns.append('precedent_reference')
            if 'statute' in text_lower:
                patterns.append('statutory_interpretation')
            if 'constitutional' in text_lower:
                patterns.append('constitutional_issue')
            if 'appeal' in text_lower:
                patterns.append('appellate_procedure')

        # Agent patterns
        elif data_type == 'agent':
            if 'error' in text_lower:
                patterns.append('error_pattern')
            if 'timeout' in text_lower:
                patterns.append('timeout_pattern')
            if 'success' in text_lower:
                patterns.append('success_pattern')

        return patterns

    def _find_similar_cases(self, features: np.ndarray, data_type: str) -> List[Dict[str, Any]]:
        """Find similar cases (simplified implementation)."""
        # This is a placeholder - in practice, you would use vector similarity
        return [
            {
                'case_id': 'similar_1',
                'similarity_score': 0.85,
                'case_name': 'Similar Case 1',
                'key_similarities': ['legal_issue', 'jurisdiction']
            },
            {
                'case_id': 'similar_2',
                'similarity_score': 0.78,
                'case_name': 'Similar Case 2',
                'key_similarities': ['case_type', 'outcome']
            }
        ]

    def _identify_success_patterns(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Identify success patterns."""
        patterns = []
        text_lower = text.lower()

        if 'strong_precedent' in text_lower:
            patterns.append('strong_precedent')
        if 'favorable_jurisdiction' in text_lower:
            patterns.append('favorable_jurisdiction')
        if 'clear_evidence' in text_lower:
            patterns.append('clear_evidence')
        if 'experienced_counsel' in text_lower:
            patterns.append('experienced_counsel')

        return patterns

    def _identify_risk_factors(self, text: str, metadata: Dict[str, Any]) -> List[str]:
        """Identify risk factors."""
        risks = []
        text_lower = text.lower()

        if 'weak_precedent' in text_lower:
            risks.append('weak_precedent')
        if 'adverse_jurisdiction' in text_lower:
            risks.append('adverse_jurisdiction')
        if 'insufficient_evidence' in text_lower:
            risks.append('insufficient_evidence')
        if 'novel_legal_issue' in text_lower:
            risks.append('novel_legal_issue')

        return risks

    def _generate_recommendations(self, patterns: List[str], similar_cases: List[Dict],
                                success_patterns: List[str], risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Recommendations based on success patterns
        if 'strong_precedent' in success_patterns:
            recommendations.append('Leverage strong precedent in argumentation')
        if 'favorable_jurisdiction' in success_patterns:
            recommendations.append('Emphasize jurisdictional advantages')

        # Recommendations based on risk factors
        if 'weak_precedent' in risk_factors:
            recommendations.append('Address precedent weaknesses proactively')
        if 'novel_legal_issue' in risk_factors:
            recommendations.append('Develop novel legal arguments carefully')

        # General recommendations
        if similar_cases:
            recommendations.append('Study similar cases for strategic insights')

        if not recommendations:
            recommendations.append('Conduct thorough legal research')

        return recommendations


# Convenience functions
def create_outcome_predictor_agent() -> OutcomePredictorAgent:
    """Create an outcome predictor agent."""
    return OutcomePredictorAgent()

def create_document_classifier_agent() -> DocumentClassifierAgent:
    """Create a document classifier agent."""
    return DocumentClassifierAgent()

def create_pattern_recognizer_agent() -> PatternRecognizerAgent:
    """Create a pattern recognizer agent."""
    return PatternRecognizerAgent()

def create_all_ml_agents() -> Dict[str, AtomicAgent]:
    """Create all ML agents."""
    return {
        'outcome_predictor': create_outcome_predictor_agent(),
        'document_classifier': create_document_classifier_agent(),
        'pattern_recognizer': create_pattern_recognizer_agent()
    }
