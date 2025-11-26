#!/usr/bin/env python3
"""
Simple proof-of-concept test for The Matrix strategic modules.
Tests basic import and functionality without complex emojis.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_settlement_optimizer():
    """Test settlement optimizer basic functionality."""
    print("=" * 60)
    print("TESTING SETTLEMENT OPTIMIZER")
    print("=" * 60)

    try:
        # Change to writer_agents directory for imports
        import os
        os.chdir(PROJECT_ROOT / "writer_agents")

        from settlement_optimizer import SettlementOptimizer, SettlementConfig

        # Create sample posteriors
        posteriors = {
            "LegalSuccess_US": {"High": 0.65, "Moderate": 0.25, "Low": 0.10},
            "Financial_Damages": {"Material": 0.45, "Moderate": 0.35, "Minor": 0.20}
        }

        # Configure for a $5M lawsuit
        config = SettlementConfig(
            expected_legal_costs=500_000.0,
            monthly_legal_burn_rate=50_000.0,
            expected_trial_duration_months=18,
            risk_aversion_coefficient=0.3,
            discount_rate=0.05,
            monte_carlo_iterations=1000,  # Reduced for faster testing
            confidence_interval=0.90
        )

        optimizer = SettlementOptimizer(config)

        # Run optimization
        recommendation = optimizer.optimize_settlement(posteriors)

        print("OK Settlement Optimizer Test PASSED")
        print(f"Optimal Settlement: ${recommendation.optimal_settlement:,.0f}")
        print(f"Settlement Range: ${recommendation.settlement_range[0]:,.0f} - ${recommendation.settlement_range[1]:,.0f}")
        print(f"Risk-Adjusted EV: ${recommendation.ev_analysis.certainty_equivalent:,.0f}")
        print(f"Downside Risk: {recommendation.ev_analysis.downside_probability:.1%}")

        return True

    except Exception as e:
        print(f"X Settlement Optimizer Test FAILED: {e}")
        return False

def test_game_theory():
    """Test game theory module basic functionality."""
    print("\n" + "=" * 60)
    print("TESTING GAME THEORY MODULE")
    print("=" * 60)

    try:
        from game_theory import GameTheoryAnalyzer, GameTheoryConfig

        # Create sample posteriors
        posteriors = {
            "LegalSuccess_US": {"High": 0.65, "Moderate": 0.25, "Low": 0.10},
            "Financial_Damages": {"Material": 0.45, "Moderate": 0.35, "Minor": 0.20}
        }

        # Configure for negotiation analysis
        config = GameTheoryConfig(
            your_trial_costs=500_000.0,
            their_trial_costs=750_000.0,
            your_success_probability=0.65,
            their_success_probability=0.35,
            expected_damages=5_000_000.0,
            negotiation_timeframe_months=6
        )

        analyzer = GameTheoryAnalyzer(config)

        # Run BATNA analysis
        batna_result = analyzer.analyze_batna(posteriors)

        print("OK Game Theory Test PASSED")
        print(f"Your BATNA: ${batna_result.your_batna:,.0f}")
        print(f"Their BATNA: ${abs(batna_result.their_batna):,.0f}")
        print(f"ZOPA Exists: {batna_result.zopa_exists}")
        if batna_result.zopa_range:
            print(f"ZOPA Range: ${batna_result.zopa_range[0]:,.0f} - ${abs(batna_result.zopa_range[1]):,.0f}")

        return True

    except Exception as e:
        print(f"X Game Theory Test FAILED: {e}")
        return False

def test_scenario_war_gaming():
    """Test scenario war gaming basic functionality."""
    print("\n" + "=" * 60)
    print("TESTING SCENARIO WAR GAMING")
    print("=" * 60)

    try:
        from scenario_war_gaming import ScenarioWarGamer, ScenarioConfig

        # Define multiple scenarios
        scenarios = {
            "Base Case": {
                "LegalSuccess_US": {"High": 0.65, "Moderate": 0.25, "Low": 0.10},
                "Financial_Damages": {"Material": 0.45, "Moderate": 0.35, "Minor": 0.20}
            },
            "Strong Evidence": {
                "LegalSuccess_US": {"High": 0.85, "Moderate": 0.12, "Low": 0.03},
                "Financial_Damages": {"Material": 0.45, "Moderate": 0.35, "Minor": 0.20}
            }
        }

        config = ScenarioConfig(
            expected_legal_costs=500_000.0,
            expected_damages=5_000_000.0,
            risk_aversion_coefficient=0.3
        )

        gamer = ScenarioWarGamer(config)

        # Run scenario analysis
        results = gamer.analyze_scenarios(scenarios)

        print("OK Scenario War Gaming Test PASSED")
        print(f"Scenarios Analyzed: {len(results.scenario_results)}")

        # Show best and worst scenarios
        best_scenario = results.best_scenario
        worst_scenario = results.worst_scenario

        print(f"Best Scenario: {best_scenario.scenario_name}")
        print(f"  Expected Value: ${best_scenario.expected_value:,.0f}")

        print(f"Worst Scenario: {worst_scenario.scenario_name}")
        print(f"  Expected Value: ${worst_scenario.expected_value:,.0f}")

        return True

    except Exception as e:
        print(f"X Scenario War Gaming Test FAILED: {e}")
        return False

def test_reputation_risk():
    """Test reputation risk module basic functionality."""
    print("\n" + "=" * 60)
    print("TESTING REPUTATION RISK MODULE")
    print("=" * 60)

    try:
        from reputation_risk import ReputationRiskAnalyzer, ReputationConfig

        # Configure for Harvard (or similar institution)
        config = ReputationConfig(
            institution_type="university",
            institution_name="Harvard University",
            academic_prestige_weight=0.25,
            federal_funding_weight=0.20,
            donor_relations_weight=0.20,
            student_enrollment_weight=0.15,
            media_perception_weight=0.10,
            alumni_trust_weight=0.10
        )

        analyzer = ReputationRiskAnalyzer(config)

        # Create sample posteriors
        posteriors = {
            "LegalSuccess_US": {"High": 0.65, "Moderate": 0.25, "Low": 0.10},
            "Financial_Damages": {"Material": 0.45, "Moderate": 0.35, "Minor": 0.20}
        }

        # Run reputation risk analysis
        risk_analysis = analyzer.analyze_reputation_risk(posteriors)

        print("OK Reputation Risk Test PASSED")
        print(f"Institution: {risk_analysis.institution_name}")

        # Show impact scores for different outcomes
        print(f"Impact Scores:")
        print(f"  Trial Loss: {risk_analysis.trial_loss_impact:.1f}")
        print(f"  High Settlement: {risk_analysis.high_settlement_impact:.1f}")
        print(f"  Low Settlement: {risk_analysis.low_settlement_impact:.1f}")
        print(f"  Trial Win: {risk_analysis.trial_win_impact:.1f}")

        return True

    except Exception as e:
        print(f"X Reputation Risk Test FAILED: {e}")
        return False

def main():
    """Run all strategic module tests."""
    print("The Matrix Strategic Modules Proof-of-Concept Test")
    print("Testing with sample lawsuit posteriors...")

    results = []

    # Test each module
    results.append(test_settlement_optimizer())
    results.append(test_game_theory())
    results.append(test_scenario_war_gaming())
    results.append(test_reputation_risk())

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total:.1%}")

    if passed == total:
        print("ALL STRATEGIC MODULES WORKING!")
        print("Ready to proceed with full integration.")
    else:
        print("WARNING: Some modules need attention before proceeding.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
