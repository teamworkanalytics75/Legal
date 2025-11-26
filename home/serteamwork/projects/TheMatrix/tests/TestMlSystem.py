"""Unit tests for ML system.

This module provides comprehensive unit tests for:
1. Data loaders and feature engineering
2. Supervised learning models
3. Deep learning models
4. ML pipelines and agents
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import tempfile
import os
from pathlib import Path

# Import ML components
from ml_system.data import LegalDataLoader, AgentDataLoader, FeatureEngineer
from ml_system.models.supervised import CaseOutcomePredictor, DocumentClassifier, AgentPerformancePredictor
from ml_system.models.deep_learning import LegalLSTM, LegalBERT, LegalCNN, AttentionModel
from ml_system.pipelines import AutomatedTrainingPipeline, InferencePipeline, ModelEvaluator
from ml_system.agents import OutcomePredictorAgent, DocumentClassifierAgent, PatternRecognizerAgent
from ml_system.tools import MLTools


class TestDataLoaders(unittest.TestCase):
    """Test data loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.legal_loader = LegalDataLoader()
        self.agent_loader = AgentDataLoader()

    def test_legal_data_loader_initialization(self):
        """Test legal data loader initialization."""
        self.assertIsInstance(self.legal_loader, LegalDataLoader)
        self.assertIsNotNone(self.legal_loader.mysql_config)

    def test_agent_data_loader_initialization(self):
        """Test agent data loader initialization."""
        self.assertIsInstance(self.agent_loader, AgentDataLoader)
        self.assertEqual(self.agent_loader.db_path, "jobs.db")

    @patch('ml_system.data.data_loader.mysql.connector.connect')
    def test_load_case_law_data_mock(self, mock_connect):
        """Test loading case law data with mocked database."""
        # Mock database connection and cursor
        mock_cursor = Mock()
        mock_cursor.description = [('id',), ('case_name',), ('opinion_text',)]
        mock_cursor.fetchall.return_value = [
            (1, 'Test Case', 'This is a test case opinion text'),
            (2, 'Another Case', 'Another test case opinion text')
        ]

        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        # Test loading data
        df = self.legal_loader.load_case_law_data(limit=2)

        # Verify results
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn('outcome_label', df.columns)
        self.assertIn('complexity_label', df.columns)

    @patch('ml_system.data.data_loader.sqlite3.connect')
    def test_load_agent_execution_data_mock(self, mock_connect):
        """Test loading agent execution data with mocked database."""
        # Mock database connection
        mock_connection = Mock()
        mock_connect.return_value = mock_connection

        # Mock pandas read_sql_query
        with patch('ml_system.data.data_loader.pd.read_sql_query') as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame({
                'job_id': [1, 2],
                'job_type': ['CitationFinderAgent', 'DraftWriterAgent'],
                'status': ['succeeded', 'failed'],
                'tokens_out': [100, 200]
            })

            # Test loading data
            df = self.agent_loader.load_agent_execution_data(limit=2)

            # Verify results
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
            self.assertIn('success_label', df.columns)
            self.assertIn('performance_label', df.columns)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.feature_engineer = FeatureEngineer()
        self.sample_texts = [
            "This is a test legal document with some legal terminology.",
            "Another test document with different content and structure.",
            "A third document for testing feature extraction."
        ]

    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization."""
        self.assertIsInstance(self.feature_engineer, FeatureEngineer)
        self.assertIsNotNone(self.feature_engineer.text_extractor)
        self.assertIsNotNone(self.feature_engineer.metadata_extractor)
        self.assertIsNotNone(self.feature_engineer.agent_extractor)

    def test_extract_text_features(self):
        """Test text feature extraction."""
        features = self.feature_engineer.text_extractor.extract_text_features(self.sample_texts)

        # Verify basic features are extracted
        self.assertIn('char_length', features)
        self.assertIn('word_length', features)
        self.assertIn('sentence_length', features)

        # Verify feature arrays have correct shape
        self.assertEqual(len(features['char_length']), len(self.sample_texts))
        self.assertEqual(len(features['word_length']), len(self.sample_texts))

    def test_extract_legal_features(self):
        """Test legal-specific feature extraction."""
        features = self.feature_engineer.text_extractor.extract_legal_features(self.sample_texts)

        # Verify legal features are extracted
        self.assertIn('citation_count', features)
        self.assertIn('legal_term_count', features)
        self.assertIn('procedural_term_count', features)

        # Verify feature arrays have correct shape
        self.assertEqual(len(features['citation_count']), len(self.sample_texts))

    def test_create_feature_matrix(self):
        """Test feature matrix creation."""
        features = self.feature_engineer.text_extractor.extract_text_features(self.sample_texts)
        feature_matrix = self.feature_engineer.create_feature_matrix(features)

        # Verify feature matrix has correct shape
        self.assertEqual(feature_matrix.shape[0], len(self.sample_texts))
        self.assertGreater(feature_matrix.shape[1], 0)


class TestSupervisedModels(unittest.TestCase):
    """Test supervised learning models."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.choice(['win', 'loss', 'settlement'], 100)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.choice(['win', 'loss', 'settlement'], 20)

    def test_case_outcome_predictor_initialization(self):
        """Test case outcome predictor initialization."""
        predictor = CaseOutcomePredictor()
        self.assertIsInstance(predictor, CaseOutcomePredictor)
        self.assertIn('logistic', predictor.models)
        self.assertIn('random_forest', predictor.models)
        self.assertIn('gradient_boost', predictor.models)
        self.assertIn('svm', predictor.models)

    def test_case_outcome_predictor_training(self):
        """Test case outcome predictor training."""
        predictor = CaseOutcomePredictor()

        # Mock the training to avoid actual model training
        with patch.object(predictor.models['logistic'], 'fit') as mock_fit:
            with patch.object(predictor.models['logistic'], 'score') as mock_score:
                mock_score.return_value = 0.8

                # Test training
                scores = predictor.train(self.X_train, self.y_train)

                # Verify training was called
                mock_fit.assert_called_once()
                self.assertIn('logistic_train', scores)

    def test_document_classifier_initialization(self):
        """Test document classifier initialization."""
        classifier = DocumentClassifier()
        self.assertIsInstance(classifier, DocumentClassifier)
        self.assertIsNotNone(classifier.multi_task_classifier)

    def test_agent_performance_predictor_initialization(self):
        """Test agent performance predictor initialization."""
        predictor = AgentPerformancePredictor()
        self.assertIsInstance(predictor, AgentPerformancePredictor)
        self.assertIn('success_prediction', predictor.models)
        self.assertIn('token_usage', predictor.models)
        self.assertIn('execution_time', predictor.models)


class TestDeepLearningModels(unittest.TestCase):
    """Test deep learning models."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.num_classes = 3
        self.max_length = 512

    def test_legal_lstm_initialization(self):
        """Test Legal LSTM initialization."""
        try:
            lstm = LegalLSTM(self.vocab_size, num_classes=self.num_classes)
            self.assertIsInstance(lstm, LegalLSTM)
            self.assertEqual(lstm.vocab_size, self.vocab_size)
            self.assertEqual(lstm.num_classes, self.num_classes)
        except ImportError:
            self.skipTest("PyTorch not available")

    def test_legal_cnn_initialization(self):
        """Test Legal CNN initialization."""
        try:
            cnn = LegalCNN(self.vocab_size, num_classes=self.num_classes)
            self.assertIsInstance(cnn, LegalCNN)
            self.assertEqual(cnn.vocab_size, self.vocab_size)
            self.assertEqual(cnn.num_classes, self.num_classes)
        except ImportError:
            self.skipTest("TensorFlow not available")

    def test_attention_model_initialization(self):
        """Test Attention Model initialization."""
        try:
            attention = AttentionModel(self.vocab_size, num_classes=self.num_classes)
            self.assertIsInstance(attention, AttentionModel)
            self.assertEqual(attention.vocab_size, self.vocab_size)
            self.assertEqual(attention.num_classes, self.num_classes)
        except ImportError:
            self.skipTest("TensorFlow not available")


class TestMLPipelines(unittest.TestCase):
    """Test ML pipelines."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = AutomatedTrainingPipeline()
        self.inference_pipeline = InferencePipeline()
        self.evaluator = ModelEvaluator()

    def test_automated_training_pipeline_initialization(self):
        """Test automated training pipeline initialization."""
        self.assertIsInstance(self.pipeline, AutomatedTrainingPipeline)
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNotNone(self.pipeline.feature_engineer)

    def test_inference_pipeline_initialization(self):
        """Test inference pipeline initialization."""
        self.assertIsInstance(self.inference_pipeline, InferencePipeline)
        self.assertIsNotNone(self.inference_pipeline.model_registry)
        self.assertIsNotNone(self.inference_pipeline.feature_engineer)

    def test_model_evaluator_initialization(self):
        """Test model evaluator initialization."""
        self.assertIsInstance(self.evaluator, ModelEvaluator)
        self.assertEqual(len(self.evaluator.evaluation_results), 0)
        self.assertEqual(len(self.evaluator.evaluation_history), 0)


class TestMLAgents(unittest.TestCase):
    """Test ML atomic agents."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_input = {
            'case_text': 'This is a test case with some legal content.',
            'case_metadata': {'court': 'federal', 'case_type': 'civil'}
        }

    def test_outcome_predictor_agent_initialization(self):
        """Test outcome predictor agent initialization."""
        try:
            agent = OutcomePredictorAgent()
            self.assertIsInstance(agent, OutcomePredictorAgent)
            self.assertEqual(agent.duty, "Predict legal case outcome probability")
            self.assertFalse(agent.is_deterministic)
            self.assertEqual(agent.cost_tier, "low")
        except ImportError:
            self.skipTest("AtomicAgent not available")

    def test_document_classifier_agent_initialization(self):
        """Test document classifier agent initialization."""
        try:
            agent = DocumentClassifierAgent()
            self.assertIsInstance(agent, DocumentClassifierAgent)
            self.assertEqual(agent.duty, "Classify legal document type and domain")
            self.assertFalse(agent.is_deterministic)
            self.assertEqual(agent.cost_tier, "low")
        except ImportError:
            self.skipTest("AtomicAgent not available")

    def test_pattern_recognizer_agent_initialization(self):
        """Test pattern recognizer agent initialization."""
        try:
            agent = PatternRecognizerAgent()
            self.assertIsInstance(agent, PatternRecognizerAgent)
            self.assertEqual(agent.duty, "Identify patterns in legal data and agent behavior")
            self.assertFalse(agent.is_deterministic)
            self.assertEqual(agent.cost_tier, "medium")
        except ImportError:
            self.skipTest("AtomicAgent not available")


class TestMLTools(unittest.TestCase):
    """Test ML tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.ml_tools = MLTools()

    def test_ml_tools_initialization(self):
        """Test ML tools initialization."""
        self.assertIsInstance(self.ml_tools, MLTools)
        self.assertIsNotNone(self.ml_tools.inference_pipeline)
        self.assertIsNotNone(self.ml_tools.feature_engineer)

    def test_predict_case_outcome(self):
        """Test case outcome prediction."""
        result = self.ml_tools.predict_case_outcome("Test case summary")

        # Verify result structure
        self.assertIn('win_probability', result)
        self.assertIn('loss_probability', result)
        self.assertIn('settlement_probability', result)
        self.assertIn('confidence_score', result)

        # Verify probabilities sum to 1
        total_prob = (result['win_probability'] +
                     result['loss_probability'] +
                     result['settlement_probability'])
        self.assertAlmostEqual(total_prob, 1.0, places=1)

    def test_classify_document(self):
        """Test document classification."""
        result = self.ml_tools.classify_document("Test document text")

        # Verify result structure
        self.assertIn('case_type', result)
        self.assertIn('legal_domain', result)
        self.assertIn('jurisdiction', result)
        self.assertIn('complexity', result)
        self.assertIn('confidence_scores', result)

    def test_find_similar_cases(self):
        """Test finding similar cases."""
        case_features = {'case_text': 'Test case', 'court': 'federal'}
        result = self.ml_tools.find_similar_cases(case_features, top_k=3)

        # Verify result structure
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 3)

        if result:
            self.assertIn('case_id', result[0])
            self.assertIn('similarity_score', result[0])
            self.assertIn('case_name', result[0])

    def test_predict_agent_performance(self):
        """Test agent performance prediction."""
        job_features = {'job_type': 'CitationFinderAgent', 'phase': 'citation'}
        result = self.ml_tools.predict_agent_performance(job_features)

        # Verify result structure
        self.assertIn('success_probability', result)
        self.assertIn('predicted_tokens', result)
        self.assertIn('predicted_time_seconds', result)
        self.assertIn('confidence_score', result)


class TestIntegration(unittest.TestCase):
    """Test integration between ML components."""

    def test_end_to_end_prediction(self):
        """Test end-to-end prediction workflow."""
        # Create sample data
        case_text = "This is a test legal case with some content."
        case_metadata = {'court': 'federal', 'case_type': 'civil'}

        # Test outcome prediction
        ml_tools = MLTools()
        outcome_result = ml_tools.predict_case_outcome(case_text, case_metadata)

        # Verify outcome prediction
        self.assertIn('win_probability', outcome_result)
        self.assertIn('loss_probability', outcome_result)

        # Test document classification
        doc_result = ml_tools.classify_document(case_text, case_metadata)

        # Verify document classification
        self.assertIn('case_type', doc_result)
        self.assertIn('legal_domain', doc_result)

        # Test similar cases
        similar_cases = ml_tools.find_similar_cases({'case_text': case_text})

        # Verify similar cases
        self.assertIsInstance(similar_cases, list)

    def test_model_registry_integration(self):
        """Test model registry integration."""
        from ml_system.models.model_registry import ModelRegistry

        # Create temporary registry
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, 'test_registry.json')
            registry = ModelRegistry(registry_path)

            # Test model registration
            test_metadata = {
                'type': 'test_model',
                'accuracy': 0.85,
                'created_at': '2024-01-01'
            }

            success = registry.register_model('test_model', temp_dir, test_metadata)
            self.assertTrue(success)

            # Test model listing
            models = registry.list_models()
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]['name'], 'test_model')


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestDataLoaders,
        TestFeatureEngineering,
        TestSupervisedModels,
        TestDeepLearningModels,
        TestMLPipelines,
        TestMLAgents,
        TestMLTools,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
