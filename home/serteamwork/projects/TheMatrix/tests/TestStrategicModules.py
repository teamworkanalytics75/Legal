"""Tests for strategic analysis modules.

This test suite covers:
- Settlement optimization
- Game theory calculations
- Scenario war gaming
- Reputation risk analysis
- Integration layer
"""

import pytest
from pathlib import Path
from typing import Dict

# Import modules to test
from writer_agents.code.insights import CaseInsights, Posterior, EvidenceItem
from writer_agents.settlement_optimizer import (
    SettlementConfig,
    SettlementOptimizer,
    MonteCarloSimulator,
    ExpectedValueCalculator,
)
from writer_agents.game_theory import (
    BATNAAnalyzer,
    NashEquilibriumCalculator,
    StrategicRecommender,
)
from writer_agents.reputation_risk import (
    ReputationFactorAnalyzer,
    MediaImpactModeler,
    ReputationRiskScorer,
)
from writer_agents.scenario_war_gaming import (
    ScenarioDefinition,
    ScenarioComparator,
)


# Test fixtures

@pytest.fixture
def sample_posteriors():
    """Create sample posterior distributions for testing."""
    return [
        Posterior(
            node_id="LegalSuccess_US",
            probabilities={"High": 0.3, "Moderate": 0.5, "Low": 0.2}
        ),
        Posterior(
            node_id="Financial_Damages",
            probabilities={"Material": 0.4, "Moderate": 0.4, "Minor": 0.2}
        ),
    ]


@pytest.fixture
def sample_insights(sample_posteriors):
    """Create sample CaseInsights for testing."""
    return CaseInsights(
        reference_id="TEST-001",
        summary="Test case summary",
        posteriors=sample_posteriors,
        evidence=[
            EvidenceItem(node_id="OGC_Email_Apr18_2025", state="Sent"),
            EvidenceItem(node_id="PRC_Awareness", state="Direct"),
        ],
        jurisdiction="US",
    )


# Settlement Optimizer Tests

def test_settlement_config_defaults():
    """Test that SettlementConfig has sensible defaults."""
    config = SettlementConfig()

    assert config.expected_legal_costs == 500_000.0
    assert config.monte_carlo_iterations == 10_000
    assert config.risk_aversion_coefficient == 0.3
    assert "High" in config.success_outcomes
    assert "Material" in config.damages_outcomes


def test_monte_carlo_simulator(sample_insights):
    """Test Monte Carlo simulation."""
    config = SettlementConfig(monte_carlo_iterations=100) # Small for testing
    simulator = MonteCarloSimulator(config)

    outcomes = simulator.run_simulation(sample_insights)

    assert len(outcomes) == 100
    assert all(hasattr(o, 'net_outcome') for o in outcomes)
    assert all(hasattr(o, 'gross_recovery') for o in outcomes)
    assert all(hasattr(o, 'legal_costs') for o in outcomes)


def test_expected_value_calculator(sample_insights):
    """Test expected value calculations."""
    config = SettlementConfig(monte_carlo_iterations=100)
    simulator = MonteCarloSimulator(config)
    outcomes = simulator.run_simulation(sample_insights)

    calculator = ExpectedValueCalculator(config)
    ev_analysis = calculator.calculate_ev_trial(outcomes)

    assert hasattr(ev_analysis, 'ev_mean')
    assert hasattr(ev_analysis, 'certainty_equivalent')
    assert hasattr(ev_analysis, 'downside_probability')
    assert 0 <= ev_analysis.downside_probability <= 1


def test_settlement_optimizer(sample_insights):
    """Test settlement optimization."""
    optimizer = SettlementOptimizer()
    recommendation = optimizer.optimize_settlement(sample_insights)

    assert hasattr(recommendation, 'optimal_settlement')
    assert hasattr(recommendation, 'settlement_range')
    assert hasattr(recommendation, 'strategy_recommendation')
    assert recommendation.settlement_range[0] <= recommendation.settlement_range[1]


def test_settlement_recommendation_report(sample_insights):
    """Test settlement recommendation report generation."""
    optimizer = SettlementOptimizer()
    recommendation = optimizer.optimize_settlement(sample_insights)

    report = recommendation.to_report()

    assert "Settlement Analysis" in report
    assert "Optimal Settlement:" in report
    assert "$" in report


# Game Theory Tests

def test_batna_analyzer(sample_insights):
    """Test BATNA analysis."""
    # First get settlement recommendation
    optimizer = SettlementOptimizer()
    settlement_rec = optimizer.optimize_settlement(sample_insights)

    # Then analyze BATNA
    analyzer = BATNAAnalyzer()
    batna = analyzer.analyze_batna(sample_insights, settlement_rec)

    assert hasattr(batna, 'your_batna')
    assert hasattr(batna, 'their_batna')
    assert hasattr(batna, 'zopa_exists')
    assert isinstance(batna.zopa_exists, bool)


def test_nash_equilibrium_calculator(sample_insights):
    """Test Nash equilibrium calculation."""
    # Setup
    optimizer = SettlementOptimizer()
    settlement_rec = optimizer.optimize_settlement(sample_insights)

    analyzer = BATNAAnalyzer()
    batna = analyzer.analyze_batna(sample_insights, settlement_rec)

    # Calculate Nash equilibrium
    calculator = NashEquilibriumCalculator()
    nash = calculator.calculate_nash_settlement(batna)

    if batna.zopa_exists:
        assert nash is not None
        assert nash > 0
    else:
        assert nash is None


def test_strategic_recommender(sample_insights):
    """Test strategic recommendations."""
    # Setup
    optimizer = SettlementOptimizer()
    settlement_rec = optimizer.optimize_settlement(sample_insights)

    analyzer = BATNAAnalyzer()
    batna = analyzer.analyze_batna(sample_insights, settlement_rec)

    calculator = NashEquilibriumCalculator()
    nash = calculator.calculate_nash_settlement(batna)

    # Get recommendations
    recommender = StrategicRecommender()
    strategy = recommender.recommend_strategy(batna, nash, settlement_rec, sample_insights)

    assert hasattr(strategy, 'first_offer')
    assert hasattr(strategy, 'walkaway_point')
    assert hasattr(strategy, 'strategy_narrative')
    assert strategy.first_offer > 0


# Reputation Risk Tests

def test_reputation_factor_analyzer(sample_insights):
    """Test reputation factor analysis."""
    analyzer = ReputationFactorAnalyzer()

    impact = analyzer.analyze_reputation_impact(sample_insights, "trial_lose")

    assert hasattr(impact, 'overall_score')
    assert hasattr(impact, 'factor_impacts')
    assert impact.overall_score < 0 # Should be negative (damage)
    assert "academic_prestige" in impact.factor_impacts


def test_media_impact_modeler():
    """Test media impact modeling."""
    modeler = MediaImpactModeler()

    score = modeler.model_media_coverage(
        case_severity="high",
        involves_china=True,
        involves_student_safety=True
    )

    assert hasattr(score, 'interest_score')
    assert hasattr(score, 'expected_article_count')
    assert 0 <= score.interest_score <= 100
    assert score.expected_article_count > 0


def test_reputation_risk_scorer(sample_insights):
    """Test overall reputation risk scoring."""
    scorer = ReputationRiskScorer()

    impacts = scorer.score_reputation_risk(sample_insights)

    assert len(impacts) == 4 # Four default outcomes
    assert "trial_lose" in impacts
    assert "settle_high" in impacts
    assert all(impact.overall_score < 0 for impact in impacts.values())


# Scenario War Gaming Tests

def test_scenario_definition():
    """Test scenario definition creation."""
    scenario = ScenarioDefinition(
        scenario_id="S1",
        name="Test Scenario",
        description="Test description",
        evidence={"TestNode": "TestState"},
        assumptions=["Assumption 1", "Assumption 2"]
    )

    assert scenario.scenario_id == "S1"
    assert scenario.name == "Test Scenario"
    assert len(scenario.assumptions) == 2


def test_scenario_comparator():
    """Test scenario comparison."""
    from writer_agents.scenario_war_gaming import ScenarioResult

    # Create mock results
    results = []
    for i in range(3):
        optimizer = SettlementOptimizer()
        recommendation = optimizer.optimize_settlement(
            CaseInsights(
                reference_id=f"TEST-{i}",
                summary="Test",
                posteriors=[
                    Posterior(
                        node_id="LegalSuccess_US",
                        probabilities={"High": 0.3, "Moderate": 0.5, "Low": 0.2}
                    ),
                    Posterior(
                        node_id="Financial_Damages",
                        probabilities={"Material": 0.4, "Moderate": 0.4, "Minor": 0.2}
                    ),
                ],
            )
        )

        scenario = ScenarioDefinition(
            scenario_id=f"S{i}",
            name=f"Scenario {i}",
            description="Test",
            evidence={},
            assumptions=[]
        )

        results.append(ScenarioResult(
            scenario=scenario,
            insights=sample_insights,
            settlement=recommendation,
            batna=None,
            nash_equilibrium=None,
        ))

    comparator = ScenarioComparator()
    comparison = comparator.compare(results)

    assert hasattr(comparison, 'comparison_dict')
    assert hasattr(comparison, 'best_scenario')
    assert hasattr(comparison, 'worst_scenario')
    assert len(comparison.comparison_dict) == 3


# Integration Tests

def test_settlement_to_game_theory_pipeline(sample_insights):
    """Test that settlement flows into game theory correctly."""
    # Settlement optimization
    optimizer = SettlementOptimizer()
    settlement = optimizer.optimize_settlement(sample_insights)

    # Game theory using settlement results
    analyzer = BATNAAnalyzer()
    batna = analyzer.analyze_batna(sample_insights, settlement)

    calculator = NashEquilibriumCalculator()
    nash = calculator.calculate_nash_settlement(batna)

    recommender = StrategicRecommender()
    strategy = recommender.recommend_strategy(batna, nash, settlement, sample_insights)

    # Verify pipeline works
    assert settlement.optimal_settlement > 0
    assert batna.your_batna == settlement.ev_analysis.certainty_equivalent
    if batna.zopa_exists:
        assert nash is not None
        assert strategy.first_offer > nash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

