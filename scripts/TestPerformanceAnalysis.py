#!/usr/bin/env python
"""Benchmark performance, analyze costs, and validate accuracy."""

import time
import json
from pathlib import Path
from writer_agents.code.insights import CaseInsights, Posterior
from writer_agents.settlement_optimizer import SettlementOptimizer, SettlementConfig
from writer_agents.game_theory import BATNAAnalyzer, NashEquilibriumCalculator, StrategicRecommender
from writer_agents.reputation_risk import ReputationRiskScorer
from writer_agents.code.langchain_integration import LangChainSQLAgent
from writer_agents.code.agents import AgentFactory, ModelConfig


def benchmark_settlement_optimizer():
    """Benchmark settlement optimizer performance."""
    print("\n" + "="*60)
    print("BENCHMARKING SETTLEMENT OPTIMIZER")
    print("="*60)
    
    # Create test case
    posteriors = [
        Posterior(
            node_id="LegalSuccess_US",
            probabilities={"High": 0.7, "Moderate": 0.2, "Low": 0.1},
            interpretation="Test case"
        ),
        Posterior(
            node_id="FinancialDamage",
            probabilities={"Material": 0.7, "Moderate": 0.2, "Minor": 0.1},
            interpretation="Test case"
        )
    ]
    
    insights = CaseInsights(
        reference_id="performance_test",
        summary="Performance test case",
        posteriors=posteriors
    )
    
    # Test different iteration counts
    iteration_counts = [100, 500, 1000, 5000, 10000]
    results = {}
    
    for iterations in iteration_counts:
        print(f"\nTesting {iterations:,} iterations:")
        
        config = SettlementConfig(monte_carlo_iterations=iterations)
        optimizer = SettlementOptimizer()
        
        start_time = time.time()
        settlement_rec = optimizer.optimize_settlement(insights, config)
        execution_time = time.time() - start_time
        
        results[iterations] = {
            "execution_time": execution_time,
            "optimal_settlement": settlement_rec.optimal_settlement,
            "iterations_per_second": iterations / execution_time
        }
        
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Iterations/second: {iterations/execution_time:,.0f}")
        print(f"  Optimal settlement: ${settlement_rec.optimal_settlement:,.0f}")
    
    return results


def benchmark_game_theory():
    """Benchmark game theory modules."""
    print("\n" + "="*60)
    print("BENCHMARKING GAME THEORY MODULES")
    print("="*60)
    
    # Create test case
    posteriors = [
        Posterior(
            node_id="LegalSuccess_US",
            probabilities={"High": 0.7, "Moderate": 0.2, "Low": 0.1},
            interpretation="Test case"
        ),
        Posterior(
            node_id="FinancialDamage",
            probabilities={"Material": 0.7, "Moderate": 0.2, "Minor": 0.1},
            interpretation="Test case"
        )
    ]
    
    insights = CaseInsights(
        reference_id="game_theory_performance_test",
        summary="Game theory performance test",
        posteriors=posteriors
    )
    
    # Settlement optimization for BATNA
    optimizer = SettlementOptimizer()
    config = SettlementConfig(monte_carlo_iterations=1000)
    settlement_rec = optimizer.optimize_settlement(insights, config)
    
    results = {}
    
    # BATNA Analysis
    print("\nBATNA Analysis:")
    start_time = time.time()
    batna_analyzer = BATNAAnalyzer()
    batna_result = batna_analyzer.analyze_batna(insights, settlement_rec)
    batna_time = time.time() - start_time
    
    results["batna"] = {
        "execution_time": batna_time,
        "your_batna": batna_result.your_batna,
        "their_batna": batna_result.their_batna,
        "zopa_exists": batna_result.zopa_exists
    }
    
    print(f"  Execution time: {batna_time:.3f}s")
    print(f"  Your BATNA: ${batna_result.your_batna:,.0f}")
    print(f"  Their BATNA: ${batna_result.their_batna:,.0f}")
    
    # Nash Equilibrium
    print("\nNash Equilibrium:")
    start_time = time.time()
    nash_calc = NashEquilibriumCalculator()
    nash_result = nash_calc.calculate_nash_settlement(batna_result)
    nash_time = time.time() - start_time
    
    results["nash"] = {
        "execution_time": nash_time,
        "nash_equilibrium": nash_result
    }
    
    print(f"  Execution time: {nash_time:.3f}s")
    print(f"  Nash equilibrium: ${nash_result:,.0f}" if nash_result else "No equilibrium")
    
    # Strategic Recommendations
    print("\nStrategic Recommendations:")
    start_time = time.time()
    strategic_rec = StrategicRecommender()
    recommendations = strategic_rec.recommend_strategy(batna_result, nash_result, settlement_rec, insights)
    strategic_time = time.time() - start_time
    
    results["strategic"] = {
        "execution_time": strategic_time,
        "first_offer": recommendations.first_offer,
        "target_range": recommendations.target_range,
        "walkaway_point": recommendations.walkaway_point
    }
    
    print(f"  Execution time: {strategic_time:.3f}s")
    print(f"  First offer: ${recommendations.first_offer:,.0f}")
    
    return results


def benchmark_reputation_risk():
    """Benchmark reputation risk scorer."""
    print("\n" + "="*60)
    print("BENCHMARKING REPUTATION RISK SCORER")
    print("="*60)
    
    # Create test case
    posteriors = [
        Posterior(
            node_id="LegalSuccess_US",
            probabilities={"High": 0.7, "Moderate": 0.2, "Low": 0.1},
            interpretation="Test case"
        ),
        Posterior(
            node_id="FinancialDamage",
            probabilities={"Material": 0.7, "Moderate": 0.2, "Minor": 0.1},
            interpretation="Test case"
        )
    ]
    
    insights = CaseInsights(
        reference_id="reputation_performance_test",
        summary="Reputation risk performance test",
        posteriors=posteriors
    )
    
    start_time = time.time()
    risk_scorer = ReputationRiskScorer()
    risk_assessments = risk_scorer.score_reputation_risk(insights)
    execution_time = time.time() - start_time
    
    results = {
        "execution_time": execution_time,
        "outcomes_assessed": len(risk_assessments),
        "assessments": {}
    }
    
    print(f"Execution time: {execution_time:.3f}s")
    print(f"Outcomes assessed: {len(risk_assessments)}")
    
    for outcome, assessment in risk_assessments.items():
        results["assessments"][outcome] = assessment.overall_score
        print(f"  {outcome}: {assessment.overall_score:.1f}")
    
    return results


def benchmark_langchain_queries():
    """Benchmark LangChain query performance."""
    print("\n" + "="*60)
    print("BENCHMARKING LANGCHAIN QUERIES")
    print("="*60)
    
    lawsuit_db_path = Path("C:/Users/Owner/Desktop/LawsuitSQL/lawsuit.db")
    if not lawsuit_db_path.exists():
        print("  Database not found - SKIPPED")
        return None
    
    try:
        model_config = ModelConfig(model="gpt-4o-mini")
        langchain_agent = LangChainSQLAgent(lawsuit_db_path, model_config)
        
        # Test queries of different complexity
        test_queries = [
            "What tables are in this database?",
            "Find documents mentioning Harvard University",
            "What are the most common legal citations?",
            "Show documents from 2019 related to discrimination",
            "Find evidence of institutional knowledge of discriminatory practices"
        ]
        
        results = {}
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query[:50]}...")
            
            start_time = time.time()
            result = langchain_agent.query_evidence(query)
            execution_time = time.time() - start_time
            
            results[f"query_{i}"] = {
                "query": query,
                "execution_time": execution_time,
                "success": result['success'],
                "answer_length": len(result.get('answer', '')) if result['success'] else 0
            }
            
            print(f"  Execution time: {execution_time:.2f}s")
            print(f"  Success: {result['success']}")
            if result['success']:
                print(f"  Answer length: {len(result['answer'])} characters")
        
        return results
        
    except Exception as e:
        print(f"  LangChain benchmarking failed: {e}")
        return None


def benchmark_atomic_agents():
    """Benchmark atomic agent creation and execution."""
    print("\n" + "="*60)
    print("BENCHMARKING ATOMIC AGENTS")
    print("="*60)
    
    try:
        factory = AgentFactory(ModelConfig(model="gpt-4o-mini"))
        
        # Test agent creation
        print("\nAgent Creation:")
        start_time = time.time()
        
        agent_types = [
            "FactExtractorAgent",
            "OutlineBuilderAgent", 
            "CitationFinderAgent",
            "MarkdownExporterAgent"
        ]
        
        agents = []
        for agent_type in agent_types:
            try:
                from writer_agents.code.atomic_agents.research import FactExtractorAgent
                from writer_agents.code.atomic_agents.drafting import OutlineBuilderAgent
                from writer_agents.code.atomic_agents.citations import CitationFinderAgent
                from writer_agents.code.atomic_agents.output import MarkdownExporterAgent
                
                if agent_type == "FactExtractorAgent":
                    agent = FactExtractorAgent(factory)
                elif agent_type == "OutlineBuilderAgent":
                    agent = OutlineBuilderAgent(factory)
                elif agent_type == "CitationFinderAgent":
                    agent = CitationFinderAgent(factory)
                elif agent_type == "MarkdownExporterAgent":
                    agent = MarkdownExporterAgent()
                
                agents.append(agent)
                
            except Exception as e:
                print(f"  Failed to create {agent_type}: {e}")
        
        creation_time = time.time() - start_time
        
        results = {
            "agent_creation": {
                "execution_time": creation_time,
                "agents_created": len(agents),
                "total_types": len(agent_types),
                "success_rate": len(agents) / len(agent_types)
            }
        }
        
        print(f"  Creation time: {creation_time:.3f}s")
        print(f"  Agents created: {len(agents)}/{len(agent_types)}")
        print(f"  Success rate: {len(agents)/len(agent_types)*100:.1f}%")
        
        # Test MasterSupervisor creation
        print("\nMasterSupervisor Creation:")
        start_time = time.time()
        
        from writer_agents.code.master_supervisor import MasterSupervisor
        master_supervisor = MasterSupervisor(factory)
        
        supervisor_time = time.time() - start_time
        
        results["master_supervisor"] = {
            "execution_time": supervisor_time,
            "created": master_supervisor is not None
        }
        
        print(f"  Creation time: {supervisor_time:.3f}s")
        print(f"  Created: {master_supervisor is not None}")
        
        return results
        
    except Exception as e:
        print(f"  Atomic agent benchmarking failed: {e}")
        return None


def analyze_costs():
    """Analyze estimated costs for different operations."""
    print("\n" + "="*60)
    print("COST ANALYSIS")
    print("="*60)
    
    # OpenAI pricing (as of 2024)
    gpt4o_mini_pricing = {
        "input": 0.00015,   # $0.15 per 1K tokens
        "output": 0.0006    # $0.60 per 1K tokens
    }
    
    # Estimated token usage
    operations = {
        "settlement_optimization": {
            "tokens": 0,  # Deterministic
            "cost": 0.0
        },
        "game_theory_analysis": {
            "tokens": 0,  # Deterministic
            "cost": 0.0
        },
        "reputation_risk_scoring": {
            "tokens": 0,  # Deterministic
            "cost": 0.0
        },
        "langchain_query": {
            "tokens": 2000,  # Estimated
            "cost": 2000 * (gpt4o_mini_pricing["input"] + gpt4o_mini_pricing["output"]) / 1000
        },
        "atomic_agent_execution": {
            "tokens": 1000,  # Estimated per agent
            "cost": 1000 * (gpt4o_mini_pricing["input"] + gpt4o_mini_pricing["output"]) / 1000
        }
    }
    
    print("Estimated costs per operation:")
    total_cost = 0
    
    for operation, data in operations.items():
        cost = data["cost"]
        total_cost += cost
        print(f"  {operation}: ${cost:.4f}")
    
    print(f"\nTotal estimated cost per analysis: ${total_cost:.4f}")
    
    # Cost optimization analysis
    print(f"\nCost Optimization Analysis:")
    deterministic_operations = ["settlement_optimization", "game_theory_analysis", "reputation_risk_scoring"]
    deterministic_count = len(deterministic_operations)
    total_operations = len(operations)
    
    print(f"  Deterministic operations: {deterministic_count}/{total_operations} ({deterministic_count/total_operations*100:.1f}%)")
    print(f"  Cost savings from deterministic operations: 100%")
    
    return operations


def validate_accuracy():
    """Validate accuracy of results."""
    print("\n" + "="*60)
    print("ACCURACY VALIDATION")
    print("="*60)
    
    # Test settlement optimization accuracy
    print("\nSettlement Optimization Accuracy:")
    
    # Test with known probabilities
    posteriors = [
        Posterior(
            node_id="LegalSuccess_US",
            probabilities={"High": 1.0, "Moderate": 0.0, "Low": 0.0},  # 100% success
            interpretation="Perfect case"
        ),
        Posterior(
            node_id="FinancialDamage",
            probabilities={"Material": 1.0, "Moderate": 0.0, "Minor": 0.0},  # 100% material
            interpretation="Maximum damages"
        )
    ]
    
    insights = CaseInsights(
        reference_id="accuracy_test",
        summary="Accuracy test case",
        posteriors=posteriors
    )
    
    optimizer = SettlementOptimizer()
    config = SettlementConfig(monte_carlo_iterations=1000)
    settlement_rec = optimizer.optimize_settlement(insights, config)
    
    print(f"  Perfect case settlement: ${settlement_rec.optimal_settlement:,.0f}")
    print(f"  Expected: High value (should be > $5M)")
    print(f"  Accuracy: {'PASS' if settlement_rec.optimal_settlement > 5_000_000 else 'FAIL'}")
    
    # Test reproducibility
    print(f"\nReproducibility Test:")
    settlements = []
    for i in range(3):
        settlement_rec = optimizer.optimize_settlement(insights, config)
        settlements.append(settlement_rec.optimal_settlement)
    
    max_diff = max(settlements) - min(settlements)
    print(f"  Settlement range: ${min(settlements):,.0f} - ${max(settlements):,.0f}")
    print(f"  Maximum difference: ${max_diff:,.0f}")
    print(f"  Reproducibility: {'PASS' if max_diff < 100_000 else 'FAIL'}")
    
    return {
        "settlement_accuracy": settlement_rec.optimal_settlement > 5_000_000,
        "reproducibility": max_diff < 100_000,
        "settlements": settlements
    }


def main():
    """Run comprehensive performance analysis."""
    print("WITCHWEB PERFORMANCE ANALYSIS")
    print("="*60)
    print("Benchmarking performance, analyzing costs, and validating accuracy")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Run all benchmarks
        settlement_results = benchmark_settlement_optimizer()
        game_theory_results = benchmark_game_theory()
        reputation_results = benchmark_reputation_risk()
        langchain_results = benchmark_langchain_queries()
        atomic_results = benchmark_atomic_agents()
        
        # Analyze costs
        cost_analysis = analyze_costs()
        
        # Validate accuracy
        accuracy_results = validate_accuracy()
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        
        report = {
            "performance_analysis": {
                "total_execution_time": total_time,
                "settlement_optimizer": settlement_results,
                "game_theory": game_theory_results,
                "reputation_risk": reputation_results,
                "langchain_queries": langchain_results,
                "atomic_agents": atomic_results
            },
            "cost_analysis": cost_analysis,
            "accuracy_validation": accuracy_results
        }
        
        # Save report
        report_file = Path("performance_analysis_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total analysis time: {total_time:.2f} seconds")
        print(f"Report saved to: {report_file}")
        
        # Performance targets
        print(f"\nPerformance Targets:")
        print(f"  Settlement optimization (5K iterations): {'PASS' if settlement_results[5000]['execution_time'] < 1.0 else 'FAIL'}")
        print(f"  Game theory analysis: {'PASS' if sum(game_theory_results[k]['execution_time'] for k in game_theory_results) < 0.1 else 'FAIL'}")
        print(f"  Reputation risk scoring: {'PASS' if reputation_results['execution_time'] < 0.1 else 'FAIL'}")
        print(f"  LangChain queries: {'PASS' if langchain_results and all(r['execution_time'] < 20 for r in langchain_results.values()) else 'FAIL'}")
        
        # Accuracy targets
        print(f"\nAccuracy Targets:")
        print(f"  Settlement accuracy: {'PASS' if accuracy_results['settlement_accuracy'] else 'FAIL'}")
        print(f"  Reproducibility: {'PASS' if accuracy_results['reproducibility'] else 'FAIL'}")
        
        # Cost targets
        print(f"\nCost Targets:")
        total_cost = sum(cost_analysis[op]['cost'] for op in cost_analysis)
        print(f"  Total cost per analysis: ${total_cost:.4f} {'PASS' if total_cost < 0.50 else 'FAIL'}")
        
        print(f"\nPerformance analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nERROR: Performance analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
