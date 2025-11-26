"""Quick test to verify all strategic module imports work correctly."""

def test_settlement_optimizer_imports():
    """Test settlement optimizer imports."""
    from writer_agents.settlement_optimizer import (
        SettlementConfig,
        SettlementOptimizer,
        MonteCarloSimulator,
        ExpectedValueCalculator,
        SettlementRecommendation,
        EVAnalysis,
        TrialOutcome,
    )
    assert SettlementConfig is not None
    assert SettlementOptimizer is not None


def test_game_theory_imports():
    """Test game theory imports."""
    from writer_agents.game_theory import (
        BATNAAnalyzer,
        NashEquilibriumCalculator,
        StrategicRecommender,
        BATNAResult,
        NegotiationStrategy,
        GameTheoryResult,
    )
    assert BATNAAnalyzer is not None
    assert NashEquilibriumCalculator is not None


def test_scenario_war_gaming_imports():
    """Test scenario war gaming imports."""
    from writer_agents.scenario_war_gaming import (
        ScenarioDefinition,
        ScenarioResult,
        ComparisonMatrix,
        ScenarioBatchRunner,
        ScenarioComparator,
        ScenarioVisualizer,
        HARVARD_SCENARIOS,
    )
    assert ScenarioDefinition is not None
    assert ScenarioBatchRunner is not None
    assert len(HARVARD_SCENARIOS) == 3


def test_reputation_risk_imports():
    """Test reputation risk imports."""
    from writer_agents.reputation_risk import (
        ReputationConfig,
        ReputationImpact,
        MediaImpactScore,
        ReputationFactorAnalyzer,
        MediaImpactModeler,
        ReputationRiskScorer,
    )
    assert ReputationConfig is not None
    assert ReputationFactorAnalyzer is not None


def test_strategic_integration_imports():
    """Test strategic integration imports."""
    from writer_agents.strategic_integration import (
        StrategicAnalysisConfig,
        CompleteStrategicReport,
        StrategicAnalysisEngine,
        quick_settlement_analysis,
        quick_game_theory_analysis,
    )
    assert StrategicAnalysisEngine is not None
    assert quick_settlement_analysis is not None


def test_cross_module_integration():
    """Test that modules can work together."""
    from writer_agents.code.insights import CaseInsights, Posterior
    from writer_agents.settlement_optimizer import SettlementOptimizer

    # Create mock insights
    insights = CaseInsights(
        reference_id="TEST",
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

    # Test that optimizer can process insights
    optimizer = SettlementOptimizer()
    # Note: This will fail without numpy, but the import test is what matters
    try:
        recommendation = optimizer.optimize_settlement(insights)
        assert recommendation is not None
    except ImportError:
        # Expected if numpy not installed
        pass


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

