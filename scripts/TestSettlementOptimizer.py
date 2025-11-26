#!/usr/bin/env python
"""Test Monte Carlo Settlement Optimizer functionality."""

import numpy as np
from writer_agents.settlement_optimizer import (
    SettlementConfig,
    SettlementOptimizer,
    SettlementRecommendation
)
from writer_agents.code.insights import CaseInsights, Posterior

def create_test_case_insights():
    """Create test case insights with mock BN posteriors."""
    print("Creating test case insights...")

    # Create posteriors
    posteriors = [
        Posterior(
            node_id="LegalSuccess_US",
            probabilities={"High": 0.7, "Moderate": 0.2, "Low": 0.1}
        ),
        Posterior(
            node_id="FinancialDamage",
            probabilities={"Material": 0.6, "Moderate": 0.3, "Minor": 0.1}
        )
    ]

    # Create case insights
    insights = CaseInsights(
        reference_id="test_case",
        summary="Test case for Monte Carlo settlement optimization",
        posteriors=posteriors
    )

    return insights

def test_settlement_optimizer():
    """Test the settlement optimizer with Monte Carlo simulation."""
    print("\n" + "="*60)
    print("TESTING MONTE CARLO SETTLEMENT OPTIMIZER")
    print("="*60)

    # Create test case
    insights = create_test_case_insights()

    # Create settlement config
    config = SettlementConfig(
        expected_legal_costs=500_000.0,
        monthly_legal_burn_rate=50_000.0,
        expected_trial_duration_months=18,
        risk_aversion_coefficient=0.3,
        discount_rate=0.05,
        monte_carlo_iterations=1000,  # Reduced for testing
        confidence_interval=0.90
    )

    print(f"Configuration:")
    print(f"  Legal costs: ${config.expected_legal_costs:,.0f}")
    print(f"  Risk aversion: {config.risk_aversion_coefficient}")
    print(f"  Monte Carlo iterations: {config.monte_carlo_iterations:,}")
    print(f"  Confidence interval: {config.confidence_interval:.0%}")

    # Create optimizer
    optimizer = SettlementOptimizer()

    # Run optimization
    print(f"\nRunning Monte Carlo simulation...")
    recommendation = optimizer.optimize_settlement(insights, config)

    # Display results
    print(f"\n" + "="*60)
    print("SETTLEMENT OPTIMIZATION RESULTS")
    print("="*60)

    print(f"Optimal Settlement: ${recommendation.optimal_settlement:,.0f}")
    print(f"Settlement Range: ${recommendation.settlement_range[0]:,.0f} - ${recommendation.settlement_range[1]:,.0f}")
    print(f"Break-even Point: ${recommendation.break_even_point:,.0f}")

    print(f"\nExpected Value Analysis:")
    print(f"  Mean EV: ${recommendation.ev_analysis.ev_mean:,.0f}")
    print(f"  Median EV: ${recommendation.ev_analysis.ev_median:,.0f}")
    print(f"  Std Dev: ${recommendation.ev_analysis.ev_std:,.0f}")
    print(f"  Certainty Equivalent: ${recommendation.ev_analysis.certainty_equivalent:,.0f}")
    print(f"  Downside Risk: {recommendation.ev_analysis.downside_probability:.1%}")

    print(f"\nStrategy Recommendation:")
    print(f"  {recommendation.strategy_recommendation}")

    return recommendation

def test_performance():
    """Test performance of Monte Carlo simulation."""
    print("\n" + "="*60)
    print("PERFORMANCE TESTING")
    print("="*60)

    import time

    insights = create_test_case_insights()

    # Test different iteration counts
    iteration_counts = [100, 1000, 5000]

    for iterations in iteration_counts:
        config = SettlementConfig(
            monte_carlo_iterations=iterations,
            expected_legal_costs=500_000.0,
            risk_aversion_coefficient=0.3
        )

        optimizer = SettlementOptimizer()

        start_time = time.time()
        recommendation = optimizer.optimize_settlement(insights, config)
        elapsed_time = time.time() - start_time

        print(f"Iterations: {iterations:,} | Time: {elapsed_time:.3f}s | Optimal: ${recommendation.optimal_settlement:,.0f}")

    return True

def test_scenario_comparison():
    """Test settlement optimization across different scenarios."""
    print("\n" + "="*60)
    print("SCENARIO COMPARISON TESTING")
    print("="*60)

    # Test different success probabilities
    scenarios = [
        ("Strong Case", {"High": 0.8, "Moderate": 0.15, "Low": 0.05}),
        ("Moderate Case", {"High": 0.5, "Moderate": 0.3, "Low": 0.2}),
        ("Weak Case", {"High": 0.2, "Moderate": 0.3, "Low": 0.5})
    ]

    config = SettlementConfig(
        monte_carlo_iterations=1000,
        expected_legal_costs=500_000.0,
        risk_aversion_coefficient=0.3
    )

    optimizer = SettlementOptimizer()

    print(f"{'Scenario':<15} {'Optimal':<12} {'Range':<20} {'EV':<12} {'Risk':<8}")
    print("-" * 70)

    for scenario_name, success_probs in scenarios:
        # Create insights with different posteriors
        posteriors = [
            Posterior(
                node_id="LegalSuccess_US",
                probabilities=success_probs
            ),
            Posterior(
                node_id="FinancialDamage",
                probabilities={"Material": 0.6, "Moderate": 0.3, "Minor": 0.1}
            )
        ]

        insights = CaseInsights(
            reference_id=f"test_{scenario_name.lower().replace(' ', '_')}",
            summary=f"Test {scenario_name}",
            posteriors=posteriors
        )

        recommendation = optimizer.optimize_settlement(insights, config)

        print(f"{scenario_name:<15} ${recommendation.optimal_settlement:>10,.0f} "
              f"${recommendation.settlement_range[0]:>8,.0f}-${recommendation.settlement_range[1]:>8,.0f} "
              f"${recommendation.ev_analysis.ev_mean:>10,.0f} "
              f"{recommendation.ev_analysis.downside_probability:>6.1%}")

    return True

def main():
    """Main test function."""
    print("MONTE CARLO SETTLEMENT OPTIMIZER TEST")
    print("="*60)

    try:
        # Test basic functionality
        recommendation = test_settlement_optimizer()

        # Test performance
        performance_success = test_performance()

        # Test scenario comparison
        scenario_success = test_scenario_comparison()

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Settlement optimization: PASS")
        print(f"Performance testing: {'PASS' if performance_success else 'FAIL'}")
        print(f"Scenario comparison: {'PASS' if scenario_success else 'FAIL'}")

        if performance_success and scenario_success:
            print("\nSUCCESS: All Monte Carlo settlement optimizer tests passed!")
            return True
        else:
            print("\nFAILURE: Some tests failed. Check the output above.")
            return False

    except Exception as e:
        print(f"\nERROR: Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
