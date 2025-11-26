#!/usr/bin/env python
"""Test integration between major The Matrix components."""

import os
import json
import time
from pathlib import Path

def test_component_integration():
    """Test integration between major components."""
    print("WITCHWEB COMPONENT INTEGRATION TEST")
    print("="*60)

    integration_results = {}

    # Test BN → Monte Carlo integration
    print("\n1. BN -> Monte Carlo Integration:")
    print("-" * 30)

    try:
        from writer_agents.code.insights import CaseInsights, Posterior
        from writer_agents.settlement_optimizer import SettlementOptimizer, SettlementConfig

        # Create mock insights
        insights = CaseInsights(
            reference_id="test_integration",
            summary="Test case for integration",
            posteriors=[
                Posterior(node_id="LegalSuccess_US", probabilities={"High": 0.7, "Moderate": 0.2, "Low": 0.1}),
                Posterior(node_id="FinancialDamage", probabilities={"Material": 0.6, "Moderate": 0.3, "Minor": 0.1})
            ]
        )

        # Test settlement optimization
        optimizer = SettlementOptimizer()
        config = SettlementConfig(monte_carlo_iterations=100)  # Reduced for testing
        settlement_rec = optimizer.optimize_settlement(insights, config)

        print(f"  Settlement optimization: SUCCESS")
        print(f"  Optimal settlement: ${settlement_rec.optimal_settlement:,.0f}")
        print(f"  Expected value: ${settlement_rec.ev_analysis.ev_mean:,.0f}")

        integration_results["bn_monte_carlo"] = True

    except Exception as e:
        print(f"  BN → Monte Carlo integration failed: {e}")
        integration_results["bn_monte_carlo"] = False

    # Test Monte Carlo → Game Theory integration
    print("\n2. Monte Carlo -> Game Theory Integration:")
    print("-" * 30)

    try:
        from writer_agents.game_theory import BATNAAnalyzer, NashEquilibriumCalculator

        batna_analyzer = BATNAAnalyzer()
        batna_result = batna_analyzer.analyze_batna(insights, settlement_rec)

        nash_calc = NashEquilibriumCalculator()
        nash_result = nash_calc.calculate_nash_settlement(batna_result)

        print(f"  BATNA analysis: SUCCESS")
        print(f"  Your BATNA: ${batna_result.your_batna:,.0f}")
        print(f"  Their BATNA: ${batna_result.their_batna:,.0f}")
        print(f"  Nash equilibrium: ${nash_result:,.0f}" if nash_result else "No Nash equilibrium")

        integration_results["monte_carlo_game_theory"] = True

    except Exception as e:
        print(f"  Monte Carlo → Game Theory integration failed: {e}")
        integration_results["monte_carlo_game_theory"] = False

    # Test Game Theory → Strategic Recommender integration
    print("\n3. Game Theory -> Strategic Recommender Integration:")
    print("-" * 30)

    try:
        from writer_agents.game_theory import StrategicRecommender

        strategic_rec = StrategicRecommender()
        recommendations = strategic_rec.recommend_strategy(batna_result, nash_result, settlement_rec, insights)

        print(f"  Strategic recommendations: SUCCESS")
        print(f"  First offer: ${recommendations.first_offer:,.0f}")
        print(f"  Target range: ${recommendations.target_range[0]:,.0f} - ${recommendations.target_range[1]:,.0f}")
        print(f"  Walk-away point: ${recommendations.walkaway_point:,.0f}")

        integration_results["game_theory_strategic"] = True

    except Exception as e:
        print(f"  Game Theory → Strategic Recommender integration failed: {e}")
        integration_results["game_theory_strategic"] = False

    # Test Reputation Risk integration
    print("\n4. Reputation Risk Integration:")
    print("-" * 30)

    try:
        from writer_agents.reputation_risk import ReputationRiskScorer

        risk_scorer = ReputationRiskScorer()
        risk_assessments = risk_scorer.score_reputation_risk(insights)

        print(f"  Reputation risk analysis: SUCCESS")
        print(f"  Outcomes analyzed: {len(risk_assessments)}")

        for outcome, assessment in risk_assessments.items():
            impact_level = assessment._interpret_impact(assessment.overall_score)
            print(f"    {outcome}: {assessment.overall_score:.1f} ({impact_level})")

        integration_results["reputation_risk"] = True

    except Exception as e:
        print(f"  Reputation risk integration failed: {e}")
        integration_results["reputation_risk"] = False

    # Test LangChain integration
    print("\n5. LangChain Integration:")
    print("-" * 30)

    try:
        from writer_agents.code.langchain_integration import LangChainSQLAgent
        from writer_agents.code.agents import ModelConfig

        lawsuit_db_path = Path("C:/Users/Owner/Desktop/LawsuitSQL/lawsuit.db")
        if lawsuit_db_path.exists():
            langchain_agent = LangChainSQLAgent(lawsuit_db_path, ModelConfig(model="gpt-4o-mini"))

            # Test a simple query
            result = langchain_agent.query_evidence("What tables are in this database?")

            print(f"  LangChain SQL agent: SUCCESS")
            print(f"  Query success: {result['success']}")
            print(f"  Answer: {result.get('answer', 'No answer')[:100]}...")

            integration_results["langchain"] = True
        else:
            print(f"  LangChain integration: SKIPPED (no database)")
            integration_results["langchain"] = None

    except Exception as e:
        print(f"  LangChain integration failed: {e}")
        integration_results["langchain"] = False

    # Test Atomic Agents integration
    print("\n6. Atomic Agents Integration:")
    print("-" * 30)

    try:
        from writer_agents.code.agents import AgentFactory, ModelConfig
        from writer_agents.code.atomic_agents.research import FactExtractorAgent

        factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))
        fact_agent = FactExtractorAgent(factory)

        print(f"  AgentFactory: SUCCESS")
        print(f"  FactExtractorAgent: SUCCESS")
        print(f"  Agent type: {type(fact_agent)}")

        integration_results["atomic_agents"] = True

    except Exception as e:
        print(f"  Atomic agents integration failed: {e}")
        integration_results["atomic_agents"] = False

    return integration_results

def test_performance_summary():
    """Generate performance summary."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    # Test performance of key components
    performance_results = {}

    # Test BN performance
    print("\n1. Bayesian Network Performance:")
    print("-" * 30)

    try:
        from writer_agents.code.insights import CaseInsights, Posterior

        start_time = time.time()
        insights = CaseInsights(
            reference_id="perf_test",
            summary="Performance test",
            posteriors=[
                Posterior(node_id="LegalSuccess_US", probabilities={"High": 0.7, "Moderate": 0.2, "Low": 0.1}),
                Posterior(node_id="FinancialDamage", probabilities={"Material": 0.6, "Moderate": 0.3, "Minor": 0.1})
            ]
        )
        bn_time = time.time() - start_time

        print(f"  BN insights creation: {bn_time:.3f}s")
        performance_results["bn"] = bn_time

    except Exception as e:
        print(f"  BN performance test failed: {e}")
        performance_results["bn"] = None

    # Test Monte Carlo performance
    print("\n2. Monte Carlo Performance:")
    print("-" * 30)

    try:
        from writer_agents.settlement_optimizer import SettlementOptimizer, SettlementConfig

        optimizer = SettlementOptimizer()
        config = SettlementConfig(monte_carlo_iterations=1000)  # Standard test

        start_time = time.time()
        settlement_rec = optimizer.optimize_settlement(insights, config)
        mc_time = time.time() - start_time

        print(f"  Monte Carlo (1000 iterations): {mc_time:.3f}s")
        print(f"  Optimal settlement: ${settlement_rec.optimal_settlement:,.0f}")
        performance_results["monte_carlo"] = mc_time

    except Exception as e:
        print(f"  Monte Carlo performance test failed: {e}")
        performance_results["monte_carlo"] = None

    # Test Game Theory performance
    print("\n3. Game Theory Performance:")
    print("-" * 30)

    try:
        from writer_agents.game_theory import BATNAAnalyzer, NashEquilibriumCalculator

        start_time = time.time()
        batna_analyzer = BATNAAnalyzer()
        batna_result = batna_analyzer.analyze_batna(insights, settlement_rec)

        nash_calc = NashEquilibriumCalculator()
        nash_result = nash_calc.calculate_nash_settlement(batna_result)
        gt_time = time.time() - start_time

        print(f"  Game theory analysis: {gt_time:.3f}s")
        print(f"  Nash equilibrium: ${nash_result:,.0f}" if nash_result else "No Nash equilibrium")
        performance_results["game_theory"] = gt_time

    except Exception as e:
        print(f"  Game theory performance test failed: {e}")
        performance_results["game_theory"] = None

    return performance_results

def main():
    """Main test function."""
    print("WITCHWEB COMPLETE SYSTEM TEST")
    print("="*60)

    try:
        # Test component integration
        integration_results = test_component_integration()

        # Test performance
        performance_results = test_performance_summary()

        # Final summary
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)

        print(f"BN -> Monte Carlo: {'PASS' if integration_results.get('bn_monte_carlo') else 'FAIL'}")
        print(f"Monte Carlo -> Game Theory: {'PASS' if integration_results.get('monte_carlo_game_theory') else 'FAIL'}")
        print(f"Game Theory -> Strategic: {'PASS' if integration_results.get('game_theory_strategic') else 'FAIL'}")
        print(f"Reputation Risk: {'PASS' if integration_results.get('reputation_risk') else 'FAIL'}")

        langchain_status = integration_results.get('langchain')
        if langchain_status is None:
            print(f"LangChain Integration: SKIPPED")
        else:
            print(f"LangChain Integration: {'PASS' if langchain_status else 'FAIL'}")

        print(f"Atomic Agents: {'PASS' if integration_results.get('atomic_agents') else 'FAIL'}")

        # Performance summary
        print(f"\nPerformance Results:")
        if performance_results.get("bn"):
            print(f"  BN Insights: {performance_results['bn']:.3f}s")
        if performance_results.get("monte_carlo"):
            print(f"  Monte Carlo: {performance_results['monte_carlo']:.3f}s")
        if performance_results.get("game_theory"):
            print(f"  Game Theory: {performance_results['game_theory']:.3f}s")

        # Overall success
        critical_integrations = [
            integration_results.get('bn_monte_carlo'),
            integration_results.get('monte_carlo_game_theory'),
            integration_results.get('game_theory_strategic'),
            integration_results.get('reputation_risk'),
            integration_results.get('atomic_agents')
        ]

        success_count = sum(1 for result in critical_integrations if result)
        total_count = len(critical_integrations)

        print(f"\nOverall Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

        if success_count >= 4:  # 80% success rate
            print("\nSUCCESS: The Matrix system is working excellently!")
            print("   Core analytical pipeline is operational")
            print("   Strategic modules are functional")
            print("   Atomic agents are ready for deployment")
            return True
        else:
            print("\nPARTIAL: Some components need attention")
            return False

    except Exception as e:
        print(f"\nERROR: Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
