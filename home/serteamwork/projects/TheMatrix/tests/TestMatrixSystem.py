#!/usr/bin/env python
"""Comprehensive pytest-based test suite for The Matrix system."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from writer_agents.code.insights import CaseInsights, Posterior
from writer_agents.settlement_optimizer import SettlementOptimizer, SettlementConfig
from writer_agents.game_theory import BATNAAnalyzer, NashEquilibriumCalculator, StrategicRecommender
from writer_agents.scenario_war_gaming import ScenarioDefinition, ScenarioComparator
from writer_agents.reputation_risk import ReputationRiskScorer


class TestSettlementOptimizer:
    """Test the settlement optimizer with realistic inputs."""

    @pytest.fixture
    def realistic_insights(self):
        """Create realistic case insights for testing."""
        posteriors = [
            Posterior(
                node_id="LegalSuccess_US",
                probabilities={
                    "High": 0.65,
                    "Moderate": 0.25,
                    "Low": 0.10
                },
                interpretation="Strong evidence of discrimination"
            ),
            Posterior(
                node_id="FinancialDamage",
                probabilities={
                    "Material": 0.70,
                    "Moderate": 0.25,
                    "Minor": 0.05
                },
                interpretation="Harvard has substantial resources"
            )
        ]

        return CaseInsights(
            reference_id="test_harvard_case",
            summary="Harvard discrimination case test",
            posteriors=posteriors
        )

    @pytest.fixture
    def settlement_config(self):
        """Create reasonable settlement configuration."""
        return SettlementConfig(
            monte_carlo_iterations=1000,  # Fast for testing
            expected_legal_costs=750_000,
            risk_aversion_coefficient=0.3
        )

    def test_settlement_optimizer_basic(self, realistic_insights, settlement_config):
        """Test basic settlement optimization functionality."""
        optimizer = SettlementOptimizer()
        settlement_rec = optimizer.optimize_settlement(realistic_insights, settlement_config)

        # Test that we get reasonable values
        assert settlement_rec.optimal_settlement > 0, "Optimal settlement should be positive"
        assert settlement_rec.optimal_settlement < 50_000_000, "Optimal settlement should be reasonable (<$50M)"
        assert settlement_rec.settlement_range[0] >= 0, "Settlement range lower bound should be non-negative"
        assert settlement_rec.settlement_range[1] > settlement_rec.settlement_range[0], "Settlement range should be valid"

        # Test EV analysis
        assert settlement_rec.ev_analysis.ev_mean > 0, "Expected value should be positive"
        assert settlement_rec.ev_analysis.certainty_equivalent > 0, "Certainty equivalent should be positive"
        assert settlement_rec.ev_analysis.downside_probability >= 0, "Downside probability should be non-negative"
        assert settlement_rec.ev_analysis.downside_probability <= 1, "Downside probability should be <= 1"

    def test_settlement_optimizer_performance(self, realistic_insights, settlement_config):
        """Test settlement optimizer performance."""
        import time

        optimizer = SettlementOptimizer()
        start_time = time.time()
        settlement_rec = optimizer.optimize_settlement(realistic_insights, settlement_config)
        execution_time = time.time() - start_time

        # Should complete in reasonable time
        assert execution_time < 5.0, f"Settlement optimization took {execution_time:.2f}s, should be <5s"

        # Should have generated outcomes
        assert len(settlement_rec.monte_carlo_outcomes) == settlement_config.monte_carlo_iterations

    def test_settlement_optimizer_reproducibility(self, realistic_insights, settlement_config):
        """Test that settlement optimization is reproducible."""
        optimizer = SettlementOptimizer()

        # Run twice with same inputs
        result1 = optimizer.optimize_settlement(realistic_insights, settlement_config)
        result2 = optimizer.optimize_settlement(realistic_insights, settlement_config)

        # Results should be very close (allowing for small Monte Carlo variance)
        assert abs(result1.optimal_settlement - result2.optimal_settlement) < 100_000, "Results should be reproducible"


class TestGameTheory:
    """Test game theory modules."""

    @pytest.fixture
    def mock_settlement_rec(self):
        """Create mock settlement recommendation."""
        mock_rec = Mock()
        mock_rec.optimal_settlement = 5_000_000
        mock_rec.settlement_range = (2_000_000, 8_000_000)
        mock_rec.break_even_point = 750_000  # Add missing attribute
        mock_rec.ev_analysis.ev_mean = 5_000_000
        mock_rec.ev_analysis.certainty_equivalent = 4_000_000
        return mock_rec

    @pytest.fixture
    def mock_insights(self):
        """Create mock case insights."""
        posteriors = [
            Posterior(
                node_id="LegalSuccess_US",
                probabilities={"High": 0.7, "Moderate": 0.2, "Low": 0.1},
                interpretation="Test case"
            )
        ]
        return CaseInsights(
            reference_id="test_case",
            summary="Test case for game theory",
            posteriors=posteriors
        )

    def test_batna_analyzer(self, mock_insights, mock_settlement_rec):
        """Test BATNA analyzer functionality."""
        batna_analyzer = BATNAAnalyzer()
        batna_result = batna_analyzer.analyze_batna(mock_insights, mock_settlement_rec)

        assert batna_result.your_batna is not None, "Your BATNA should be calculated"
        assert batna_result.their_batna is not None, "Their BATNA should be calculated"
        assert isinstance(batna_result.zopa_exists, bool), "ZOPA existence should be boolean"

    def test_nash_equilibrium_calculator(self, mock_insights, mock_settlement_rec):
        """Test Nash equilibrium calculator."""
        batna_analyzer = BATNAAnalyzer()
        batna_result = batna_analyzer.analyze_batna(mock_insights, mock_settlement_rec)

        nash_calc = NashEquilibriumCalculator()
        nash_result = nash_calc.calculate_nash_settlement(batna_result)

        if nash_result is not None:
            assert isinstance(nash_result, (int, float)), "Nash equilibrium should be numeric"

    def test_strategic_recommender(self, mock_insights, mock_settlement_rec):
        """Test strategic recommender."""
        batna_analyzer = BATNAAnalyzer()
        batna_result = batna_analyzer.analyze_batna(mock_insights, mock_settlement_rec)

        nash_calc = NashEquilibriumCalculator()
        nash_result = nash_calc.calculate_nash_settlement(batna_result)

        strategic_rec = StrategicRecommender()
        recommendations = strategic_rec.recommend_strategy(batna_result, nash_result, mock_settlement_rec, mock_insights)

        assert recommendations.first_offer is not None, "First offer should be calculated"
        assert len(recommendations.target_range) == 2, "Target range should have 2 values"
        assert recommendations.walkaway_point is not None, "Walk-away point should be calculated"


class TestReputationRisk:
    """Test reputation risk scorer."""

    @pytest.fixture
    def mock_insights(self):
        """Create mock case insights for reputation risk testing."""
        posteriors = [
            Posterior(
                node_id="LegalSuccess_US",
                probabilities={"High": 0.7, "Moderate": 0.2, "Low": 0.1},
                interpretation="Test case"
            ),
            Posterior(
                node_id="FinancialDamage",
                probabilities={"Material": 0.6, "Moderate": 0.3, "Minor": 0.1},
                interpretation="Test case"
            )
        ]
        return CaseInsights(
            reference_id="test_reputation_case",
            summary="Test case for reputation risk",
            posteriors=posteriors
        )

    def test_reputation_risk_scorer(self, mock_insights):
        """Test reputation risk scorer functionality."""
        risk_scorer = ReputationRiskScorer()
        risk_assessments = risk_scorer.score_reputation_risk(mock_insights)

        assert len(risk_assessments) > 0, "Should have risk assessments for multiple outcomes"

        for outcome, assessment in risk_assessments.items():
            assert hasattr(assessment, 'overall_score'), "Assessment should have overall_score"
            assert isinstance(assessment.overall_score, (int, float)), "Overall score should be numeric"
            assert -20 <= assessment.overall_score <= 0, "Risk score should be in expected range (-20 to 0)"


class TestScenarioWarGaming:
    """Test scenario war gaming functionality."""

    def test_scenario_definition(self):
        """Test scenario definition creation."""
        scenario = ScenarioDefinition(
            scenario_id="test_scenario",
            name="Test Scenario",
            description="Test scenario description",
            evidence={
                "LegalSuccess_US": "High",
                "FinancialDamage": "Material"
            },
            assumptions=["Test assumption"]
        )

        assert scenario.scenario_id == "test_scenario"
        assert scenario.name == "Test Scenario"
        assert len(scenario.evidence) == 2
        assert len(scenario.assumptions) == 1

    def test_scenario_comparator(self):
        """Test scenario comparator functionality."""
        # Create mock scenario results instead of definitions
        from unittest.mock import Mock

        mock_result1 = Mock()
        mock_result1.scenario.name = "Strong Case"
        mock_result1.settlement.optimal_settlement = 8_000_000
        mock_result1.nash_equilibrium = 7_500_000
        mock_result1.settlement.ev_analysis.ev_mean = 8_000_000
        mock_result1.settlement.ev_analysis.downside_probability = 0.1
        mock_result1.insights = CaseInsights(
            reference_id="test1",
            summary="Test case 1",
            posteriors=[Posterior(node_id="LegalSuccess_US", probabilities={"High": 0.8}, interpretation="Strong")]
        )

        mock_result2 = Mock()
        mock_result2.scenario.name = "Weak Case"
        mock_result2.settlement.optimal_settlement = 2_000_000
        mock_result2.nash_equilibrium = 1_500_000
        mock_result2.settlement.ev_analysis.ev_mean = 2_000_000
        mock_result2.settlement.ev_analysis.downside_probability = 0.3
        mock_result2.insights = CaseInsights(
            reference_id="test2",
            summary="Test case 2",
            posteriors=[Posterior(node_id="LegalSuccess_US", probabilities={"Low": 0.8}, interpretation="Weak")]
        )

        scenarios = [mock_result1, mock_result2]

        comparator = ScenarioComparator()
        comparison = comparator.compare(scenarios)

        assert hasattr(comparison, 'comparison_dict'), "Comparison should have comparison_dict"
        assert len(comparison.comparison_dict) > 0, "Should have comparison results"


class TestIntegration:
    """Test integration between modules."""

    @pytest.fixture
    def realistic_insights(self):
        """Create realistic case insights for integration testing."""
        posteriors = [
            Posterior(
                node_id="LegalSuccess_US",
                probabilities={"High": 0.65, "Moderate": 0.25, "Low": 0.10},
                interpretation="Strong evidence"
            ),
            Posterior(
                node_id="FinancialDamage",
                probabilities={"Material": 0.70, "Moderate": 0.25, "Minor": 0.05},
                interpretation="Substantial resources"
            )
        ]
        return CaseInsights(
            reference_id="integration_test",
            summary="Integration test case",
            posteriors=posteriors
        )

    def test_settlement_to_game_theory_integration(self, realistic_insights):
        """Test integration from settlement optimization to game theory."""
        # Run settlement optimization
        optimizer = SettlementOptimizer()
        config = SettlementConfig(monte_carlo_iterations=500)  # Fast for testing
        settlement_rec = optimizer.optimize_settlement(realistic_insights, config)

        # Run game theory analysis
        batna_analyzer = BATNAAnalyzer()
        batna_result = batna_analyzer.analyze_batna(realistic_insights, settlement_rec)

        # Test that integration works
        assert batna_result.your_batna is not None
        assert batna_result.their_batna is not None

        # Run Nash equilibrium calculation
        nash_calc = NashEquilibriumCalculator()
        nash_result = nash_calc.calculate_nash_settlement(batna_result)

        # Run strategic recommendations
        strategic_rec = StrategicRecommender()
        recommendations = strategic_rec.recommend_strategy(batna_result, nash_result, settlement_rec, realistic_insights)

        assert recommendations.first_offer is not None
        assert len(recommendations.target_range) == 2

    def test_game_theory_to_reputation_risk_integration(self, realistic_insights):
        """Test integration from game theory to reputation risk."""
        # Run reputation risk analysis
        risk_scorer = ReputationRiskScorer()
        risk_assessments = risk_scorer.score_reputation_risk(realistic_insights)

        # Test that we get reasonable risk assessments
        assert len(risk_assessments) > 0

        for outcome, assessment in risk_assessments.items():
            assert -20 <= assessment.overall_score <= 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
