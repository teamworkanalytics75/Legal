#!/usr/bin/env python3
"""
Proof-of-concept test for The Matrix strategic modules.

Tests settlement optimizer, game theory, scenario war gaming, and reputation risk
with sample lawsuit posteriors to verify they produce intelligent results.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def create_sample_posteriors():
    """Create sample BN posteriors for testing strategic modules."""
    return {
        # Legal success probabilities
        "LegalSuccess_US": {
            "High": 0.65,      # 65% chance of high success
            "Moderate": 0.25,   # 25% chance of moderate success
            "Low": 0.10        # 10% chance of low success
        },

        # Financial damages probabilities
        "Financial_Damages": {
            "Material": 0.45,   # 45% chance of material damages ($10M+)
            "Moderate": 0.35,   # 35% chance of moderate damages ($3M)
            "Minor": 0.20       # 20% chance of minor damages ($500K)
        },

        # Evidence strength
        "Evidence_Strength": {
            "Strong": 0.70,    # 70% chance of strong evidence
            "Moderate": 0.25,   # 25% chance of moderate evidence
            "Weak": 0.05        # 5% chance of weak evidence
        },

        # Opponent awareness
        "Opponent_Awareness": {
            "Direct": 0.60,     # 60% chance of direct awareness
            "Indirect": 0.30,   # 30% chance of indirect awareness
            "None": 0.10        # 10% chance of no awareness
        }
    }

def test_settlement_optimizer():
    """Test settlement optimizer with sample posteriors."""
    print("=" * 60)
    print("TESTING SETTLEMENT OPTIMIZER")
    print("=" * 60)

    try:
        from settlement_optimizer import SettlementOptimizer, SettlementConfig

        # Create sample posteriors
        posteriors = create_sample_posteriors()

        # Configure for a $5M lawsuit
        config = SettlementConfig(
            expected_legal_costs=500_000.0,
            monthly_legal_burn_rate=50_000.0,
            expected_trial_duration_months=18,
            risk_aversion_coefficient=0.3,
            discount_rate=0.05,
            monte_carlo_iterations=10_000,
            confidence_interval=0.90
        )

        optimizer = SettlementOptimizer(config)

        # Run optimization
        recommendation = optimizer.optimize_settlement(posteriors)

        print(f"OK Settlement Optimizer Test PASSED")
        print(f"Optimal Settlement: ${recommendation.optimal_settlement:,.0f}")
        print(f"Settlement Range: ${recommendation.settlement_range[0]:,.0f} - ${recommendation.settlement_range[1]:,.0f}")
        print(f"Risk-Adjusted EV: ${recommendation.ev_analysis.certainty_equivalent:,.0f}")
        print(f"Downside Risk: {recommendation.ev_analysis.downside_probability:.1%}")
        print(f"Strategic Recommendation: {recommendation.strategic_recommendation}")

        return True

    except Exception as e:
        print(f"X Settlement Optimizer Test FAILED: {e}")
        return False

def test_game_theory():
    """Test game theory module with sample posteriors."""
    print("\n" + "=" * 60)
    print("TESTING GAME THEORY MODULE")
    print("=" * 60)

    try:
        from writer_agents.game_theory import GameTheoryAnalyzer, GameTheoryConfig

        # Create sample posteriors
        posteriors = create_sample_posteriors()

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

        # Run Nash equilibrium analysis
        nash_result = analyzer.calculate_nash_equilibrium(posteriors)

        # Generate negotiation strategy
        strategy = analyzer.generate_negotiation_strategy(posteriors)

        print(f"✅ Game Theory Test PASSED")
        print(f"Your BATNA: ${batna_result.your_batna:,.0f}")
        print(f"Their BATNA: ${abs(batna_result.their_batna):,.0f}")
        print(f"ZOPA Exists: {batna_result.zopa_exists}")
        if batna_result.zopa_range:
            print(f"ZOPA Range: ${batna_result.zopa_range[0]:,.0f} - ${abs(batna_result.zopa_range[1]):,.0f}")
        print(f"Nash Equilibrium: ${nash_result.nash_amount:,.0f}")
        print(f"First Offer: ${strategy.first_offer:,.0f}")
        print(f"Walk-Away Point: ${strategy.walk_away_point:,.0f}")

        return True

    except Exception as e:
        print(f"❌ Game Theory Test FAILED: {e}")
        return False

def test_scenario_war_gaming():
    """Test scenario war gaming with multiple evidence scenarios."""
    print("\n" + "=" * 60)
    print("TESTING SCENARIO WAR GAMING")
    print("=" * 60)

    try:
        from writer_agents.scenario_war_gaming import ScenarioWarGamer, ScenarioConfig

        # Define multiple scenarios
        scenarios = {
            "Base Case": create_sample_posteriors(),
            "Strong Evidence": {
                **create_sample_posteriors(),
                "Evidence_Strength": {"Strong": 0.90, "Moderate": 0.08, "Weak": 0.02}
            },
            "Weak Evidence": {
                **create_sample_posteriors(),
                "Evidence_Strength": {"Strong": 0.30, "Moderate": 0.50, "Weak": 0.20}
            },
            "Direct Awareness": {
                **create_sample_posteriors(),
                "Opponent_Awareness": {"Direct": 0.90, "Indirect": 0.08, "None": 0.02}
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

        print(f"✅ Scenario War Gaming Test PASSED")
        print(f"Scenarios Analyzed: {len(results.scenario_results)}")

        # Show best and worst scenarios
        best_scenario = results.best_scenario
        worst_scenario = results.worst_scenario

        print(f"Best Scenario: {best_scenario.scenario_name}")
        print(f"  Expected Value: ${best_scenario.expected_value:,.0f}")
        print(f"  Settlement Range: ${best_scenario.settlement_range[0]:,.0f} - ${best_scenario.settlement_range[1]:,.0f}")

        print(f"Worst Scenario: {worst_scenario.scenario_name}")
        print(f"  Expected Value: ${worst_scenario.expected_value:,.0f}")
        print(f"  Settlement Range: ${worst_scenario.settlement_range[0]:,.0f} - ${worst_scenario.settlement_range[1]:,.0f}")

        # Show evidence impact
        print(f"\nEvidence Impact Analysis:")
        for impact in results.evidence_impacts[:3]:  # Top 3
            print(f"  {impact.evidence_factor}: {impact.impact_score:.2f} (${impact.value_difference:,.0f})")

        return True

    except Exception as e:
        print(f"❌ Scenario War Gaming Test FAILED: {e}")
        return False

def test_reputation_risk():
    """Test reputation risk module with institutional data."""
    print("\n" + "=" * 60)
    print("TESTING REPUTATION RISK MODULE")
    print("=" * 60)

    try:
        from writer_agents.reputation_risk import ReputationRiskAnalyzer, ReputationConfig

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
        posteriors = create_sample_posteriors()

        # Run reputation risk analysis
        risk_analysis = analyzer.analyze_reputation_risk(posteriors)

        print(f"✅ Reputation Risk Test PASSED")
        print(f"Institution: {risk_analysis.institution_name}")

        # Show impact scores for different outcomes
        print(f"\nImpact Scores:")
        print(f"  Trial Loss: {risk_analysis.trial_loss_impact:.1f}")
        print(f"  High Settlement: {risk_analysis.high_settlement_impact:.1f}")
        print(f"  Low Settlement: {risk_analysis.low_settlement_impact:.1f}")
        print(f"  Trial Win: {risk_analysis.trial_win_impact:.1f}")

        # Show media coverage predictions
        print(f"\nMedia Coverage Predictions:")
        print(f"  National Coverage: {risk_analysis.media_coverage.national_coverage:.1%}")
        print(f"  Industry Coverage: {risk_analysis.media_coverage.industry_coverage:.1%}")
        print(f"  Local Coverage: {risk_analysis.media_coverage.local_coverage:.1%}")

        # Show financial risk quantification
        print(f"\nFinancial Risk Quantification:")
        print(f"  Federal Funding at Risk: ${risk_analysis.financial_risks.federal_funding_at_risk:,.0f}")
        print(f"  Donor Impact: ${risk_analysis.financial_risks.donor_impact:,.0f}")
        print(f"  Enrollment Impact: ${risk_analysis.financial_risks.enrollment_impact:,.0f}")

        return True

    except Exception as e:
        print(f"❌ Reputation Risk Test FAILED: {e}")
        return False

def test_strategic_integration():
    """Test strategic integration module end-to-end."""
    print("\n" + "=" * 60)
    print("TESTING STRATEGIC INTEGRATION")
    print("=" * 60)

    try:
        from writer_agents.strategic_integration import StrategicAnalysisEngine

        # Create sample posteriors
        posteriors = create_sample_posteriors()

        # Initialize engine
        engine = StrategicAnalysisEngine()

        # Run complete analysis
        report = engine.run_complete_analysis(
            posteriors=posteriors,
            case_summary="Sample lawsuit involving institutional liability",
            evidence_scenarios={
                "Base Case": posteriors,
                "Strong Evidence": {
                    **posteriors,
                    "Evidence_Strength": {"Strong": 0.90, "Moderate": 0.08, "Weak": 0.02}
                }
            }
        )

        print(f"✅ Strategic Integration Test PASSED")
        print(f"Report Generated: {len(report.sections)} sections")
        print(f"Analysis Complete: {report.analysis_complete}")
        print(f"Total Strategic Value: ${report.total_strategic_value:,.0f}")

        # Show key recommendations
        print(f"\nKey Recommendations:")
        for i, rec in enumerate(report.key_recommendations[:3], 1):
            print(f"  {i}. {rec}")

        return True

    except Exception as e:
        print(f"❌ Strategic Integration Test FAILED: {e}")
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
    results.append(test_strategic_integration())

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
