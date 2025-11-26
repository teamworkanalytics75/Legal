"""Integration tests for ML system.

This module provides integration tests for:
1. End-to-end ML pipeline testing
2. ML agent integration with atomic agent system
3. Model training and inference workflows
4. Performance and reliability testing
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import tempfile
import os
import asyncio
from pathlib import Path

# Import ML components
from ml_system.pipelines import AutomatedTrainingPipeline, InferencePipeline
from ml_system.agents import OutcomePredictorAgent, DocumentClassifierAgent, PatternRecognizerAgent
from ml_system.tools import MLTools
from ml_system.models.model_registry import ModelRegistry


class TestMLPipelineIntegration(unittest.TestCase):
    """Test ML pipeline integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = AutomatedTrainingPipeline()
        self.inference_pipeline = InferencePipeline()

    @patch('ml_system.pipelines.training_pipeline.LegalDataLoader')
    @patch('ml_system.pipelines.training_pipeline.AgentDataLoader')
    def test_full_pipeline_execution(self, mock_agent_loader, mock_legal_loader):
        """Test full ML pipeline execution."""
        # Mock data loaders
        mock_legal_data = pd.DataFrame({
            'case_name': ['Case 1', 'Case 2', 'Case 3'],
            'opinion_text': ['Text 1', 'Text 2', 'Text 3'],
            'court': ['federal', 'state', 'federal'],
            'case_type': ['civil', 'criminal', 'civil'],
            'outcome_label': ['win', 'loss', 'settlement'],
            'complexity_label': ['high', 'medium', 'low'],
            'domain_label': ['employment', 'contract', 'tort'],
            'jurisdiction_label': ['federal', 'state', 'federal']
        })

        mock_agent_data = pd.DataFrame({
            'job_type': ['Agent1', 'Agent2', 'Agent3'],
            'phase': ['research', 'drafting', 'citation'],
            'status': ['succeeded', 'failed', 'succeeded'],
            'tokens_out': [100, 200, 150],
            'success_label': [1, 0, 1],
            'performance_label': ['high', 'low', 'medium'],
            'efficiency_label': ['high', 'low', 'medium'],
            'error_type_label': ['none', 'timeout', 'none']
        })

        mock_legal_loader.return_value.load_case_law_data.return_value = mock_legal_data
        mock_agent_loader.return_value.load_agent_execution_data.return_value = mock_agent_data

        # Run pipeline
        results = self.pipeline.run_full_pipeline()

        # Verify results
        self.assertIn('pipeline_start_time', results)
        self.assertIn('pipeline_end_time', results)
        self.assertIn('total_execution_time', results)
        self.assertIn('models_trained', results)

    def test_inference_pipeline_integration(self):
        """Test inference pipeline integration."""
        # Test with sample data
        sample_features = np.random.rand(1, 10)

        # Test prediction (will use fallback if models not available)
        result = self.inference_pipeline.predict(sample_features, 'outcome_predictor')

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)

    def test_model_registry_integration(self):
        """Test model registry integration."""
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

            # Test model loading
            model = registry.load_model('test_model')
            # Model loading may fail if actual model files don't exist, which is expected

            # Test model listing
            models = registry.list_models()
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]['name'], 'test_model')


class TestMLAgentIntegration(unittest.TestCase):
    """Test ML agent integration with atomic agent system."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_input = {
            'case_text': 'This is a test legal case with some content.',
            'case_metadata': {'court': 'federal', 'case_type': 'civil'}
        }

    def test_outcome_predictor_agent_integration(self):
        """Test outcome predictor agent integration."""
        try:
            agent = OutcomePredictorAgent()

            # Test agent execution
            result = asyncio.run(agent.execute(self.sample_input))

            # Verify result structure
            self.assertIn('win_probability', result)
            self.assertIn('loss_probability', result)
            self.assertIn('settlement_probability', result)
            self.assertIn('confidence_score', result)

            # Verify probabilities are valid
            self.assertGreaterEqual(result['win_probability'], 0.0)
            self.assertLessEqual(result['win_probability'], 1.0)
            self.assertGreaterEqual(result['loss_probability'], 0.0)
            self.assertLessEqual(result['loss_probability'], 1.0)

        except ImportError:
            self.skipTest("AtomicAgent not available")

    def test_document_classifier_agent_integration(self):
        """Test document classifier agent integration."""
        try:
            agent = DocumentClassifierAgent()

            # Test agent execution
            result = asyncio.run(agent.execute(self.sample_input))

            # Verify result structure
            self.assertIn('case_type', result)
            self.assertIn('legal_domain', result)
            self.assertIn('jurisdiction', result)
            self.assertIn('complexity', result)
            self.assertIn('confidence_scores', result)

            # Verify confidence scores are valid
            for score in result['confidence_scores'].values():
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

        except ImportError:
            self.skipTest("AtomicAgent not available")

    def test_pattern_recognizer_agent_integration(self):
        """Test pattern recognizer agent integration."""
        try:
            agent = PatternRecognizerAgent()

            # Test agent execution
            result = asyncio.run(agent.execute(self.sample_input))

            # Verify result structure
            self.assertIn('similar_cases', result)
            self.assertIn('success_patterns', result)
            self.assertIn('risk_factors', result)
            self.assertIn('recommendations', result)
            self.assertIn('patterns_detected', result)

            # Verify result types
            self.assertIsInstance(result['similar_cases'], list)
            self.assertIsInstance(result['success_patterns'], list)
            self.assertIsInstance(result['risk_factors'], list)
            self.assertIsInstance(result['recommendations'], list)

        except ImportError:
            self.skipTest("AtomicAgent not available")


class TestMLToolsIntegration(unittest.TestCase):
    """Test ML tools integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.ml_tools = MLTools()

    def test_ml_tools_case_outcome_prediction(self):
        """Test ML tools case outcome prediction."""
        case_summary = "This is a test case summary with legal content."
        case_metadata = {'court': 'federal', 'case_type': 'civil'}

        result = self.ml_tools.predict_case_outcome(case_summary, case_metadata)

        # Verify result structure
        self.assertIn('win_probability', result)
        self.assertIn('loss_probability', result)
        self.assertIn('settlement_probability', result)
        self.assertIn('confidence_score', result)

        # Verify probabilities sum to approximately 1
        total_prob = (result['win_probability'] +
                     result['loss_probability'] +
                     result['settlement_probability'])
        self.assertAlmostEqual(total_prob, 1.0, places=1)

    def test_ml_tools_document_classification(self):
        """Test ML tools document classification."""
        document_text = "This is a test legal document with various content."
        document_metadata = {'court': 'state', 'case_type': 'criminal'}

        result = self.ml_tools.classify_document(document_text, document_metadata)

        # Verify result structure
        self.assertIn('case_type', result)
        self.assertIn('legal_domain', result)
        self.assertIn('jurisdiction', result)
        self.assertIn('complexity', result)
        self.assertIn('confidence_scores', result)

        # Verify confidence scores are valid
        for score in result['confidence_scores'].values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_ml_tools_similar_cases(self):
        """Test ML tools similar cases finding."""
        case_features = {
            'case_text': 'Test case with legal content',
            'court': 'federal',
            'case_type': 'civil'
        }

        result = self.ml_tools.find_similar_cases(case_features, top_k=3)

        # Verify result structure
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 3)

        if result:
            for case in result:
                self.assertIn('case_id', case)
                self.assertIn('similarity_score', case)
                self.assertIn('case_name', case)
                self.assertGreaterEqual(case['similarity_score'], 0.0)
                self.assertLessEqual(case['similarity_score'], 1.0)

    def test_ml_tools_agent_performance_prediction(self):
        """Test ML tools agent performance prediction."""
        job_features = {
            'job_type': 'CitationFinderAgent',
            'phase': 'citation',
            'priority': 1
        }

        result = self.ml_tools.predict_agent_performance(job_features)

        # Verify result structure
        self.assertIn('success_probability', result)
        self.assertIn('predicted_tokens', result)
        self.assertIn('predicted_time_seconds', result)
        self.assertIn('confidence_score', result)

        # Verify values are valid
        self.assertGreaterEqual(result['success_probability'], 0.0)
        self.assertLessEqual(result['success_probability'], 1.0)
        self.assertGreaterEqual(result['predicted_tokens'], 0)
        self.assertGreaterEqual(result['predicted_time_seconds'], 0)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end ML workflow."""

    def test_complete_ml_workflow(self):
        """Test complete ML workflow from data to prediction."""
        # Step 1: Create sample data
        sample_data = {
            'case_text': 'This is a comprehensive test case with legal content.',
            'case_metadata': {
                'court': 'federal',
                'case_type': 'civil',
                'jurisdiction': 'federal'
            }
        }

        # Step 2: Test ML tools
        ml_tools = MLTools()

        # Test outcome prediction
        outcome_result = ml_tools.predict_case_outcome(
            sample_data['case_text'],
            sample_data['case_metadata']
        )

        # Test document classification
        doc_result = ml_tools.classify_document(
            sample_data['case_text'],
            sample_data['case_metadata']
        )

        # Test similar cases
        similar_cases = ml_tools.find_similar_cases(sample_data)

        # Step 3: Test ML agents
        try:
            # Test outcome predictor agent
            outcome_agent = OutcomePredictorAgent()
            outcome_agent_result = asyncio.run(outcome_agent.execute(sample_data))

            # Test document classifier agent
            doc_agent = DocumentClassifierAgent()
            doc_agent_result = asyncio.run(doc_agent.execute(sample_data))

            # Test pattern recognizer agent
            pattern_agent = PatternRecognizerAgent()
            pattern_agent_result = asyncio.run(pattern_agent.execute(sample_data))

            # Verify all results are valid
            self.assertIsInstance(outcome_result, dict)
            self.assertIsInstance(doc_result, dict)
            self.assertIsInstance(similar_cases, list)
            self.assertIsInstance(outcome_agent_result, dict)
            self.assertIsInstance(doc_agent_result, dict)
            self.assertIsInstance(pattern_agent_result, dict)

        except ImportError:
            # Skip agent tests if AtomicAgent not available
            pass

    def test_model_registry_workflow(self):
        """Test model registry workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, 'workflow_registry.json')
            registry = ModelRegistry(registry_path)

            # Test model registration
            test_models = [
                ('outcome_predictor', {'type': 'outcome_predictor', 'accuracy': 0.85}),
                ('document_classifier', {'type': 'document_classifier', 'accuracy': 0.80}),
                ('agent_predictor', {'type': 'agent_predictor', 'accuracy': 0.75})
            ]

            for model_name, metadata in test_models:
                success = registry.register_model(model_name, temp_dir, metadata)
                self.assertTrue(success)

            # Test model listing
            models = registry.list_models()
            self.assertEqual(len(models), 3)

            # Test model comparison
            comparison = registry.compare_models(['outcome_predictor', 'document_classifier'])
            self.assertIn('outcome_predictor', comparison)
            self.assertIn('document_classifier', comparison)

            # Test model archiving
            success = registry.archive_model('outcome_predictor', '1.0.0')
            self.assertTrue(success)

            # Test model deletion
            success = registry.delete_model('agent_predictor', '1.0.0')
            self.assertTrue(success)

            # Verify final state
            models = registry.list_models()
            self.assertEqual(len(models), 2)  # One archived, one deleted


class TestPerformanceAndReliability(unittest.TestCase):
    """Test performance and reliability of ML system."""

    def test_prediction_performance(self):
        """Test prediction performance."""
        ml_tools = MLTools()

        # Test with multiple predictions
        test_cases = [
            "Short case summary",
            "Medium length case summary with more detailed content",
            "Very long case summary with extensive legal content and multiple issues"
        ]

        results = []
        for case in test_cases:
            result = ml_tools.predict_case_outcome(case)
            results.append(result)

        # Verify all predictions completed
        self.assertEqual(len(results), len(test_cases))

        # Verify all results have required fields
        for result in results:
            self.assertIn('win_probability', result)
            self.assertIn('loss_probability', result)
            self.assertIn('settlement_probability', result)

    def test_error_handling(self):
        """Test error handling in ML system."""
        ml_tools = MLTools()

        # Test with invalid input
        invalid_inputs = [
            "",  # Empty string
            None,  # None input
            {},  # Empty dict
            "x" * 10000  # Very long string
        ]

        for invalid_input in invalid_inputs:
            try:
                if isinstance(invalid_input, str):
                    result = ml_tools.predict_case_outcome(invalid_input)
                else:
                    result = ml_tools.classify_document(str(invalid_input))

                # Should handle gracefully
                self.assertIsInstance(result, dict)
                self.assertIn('error', result)

            except Exception as e:
                # Should not crash
                self.fail(f"ML system crashed on invalid input: {e}")

    def test_concurrent_predictions(self):
        """Test concurrent predictions."""
        import threading
        import time

        ml_tools = MLTools()
        results = []
        errors = []

        def make_prediction(case_text):
            try:
                result = ml_tools.predict_case_outcome(case_text)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=make_prediction,
                args=(f"Test case {i}",)
            )
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestMLPipelineIntegration,
        TestMLAgentIntegration,
        TestMLToolsIntegration,
        TestEndToEndWorkflow,
        TestPerformanceAndReliability
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\nIntegration Tests Summary:")
    print(f"Tests run: {result.testsRun}")
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
