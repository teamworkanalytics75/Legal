"""Strategic integration layer for complete legal analysis.

This module orchestrates all strategic analysis modules (settlement optimization,
game theory, scenario analysis, reputation risk) into a unified workflow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from writer_agents.code.bn_adapter import run_bn_inference
from writer_agents.code.insights import CaseInsights
from .game_theory import (
    BATNAAnalyzer,
    GameTheoryResult,
    NashEquilibriumCalculator,
    StrategicRecommender,
)
from .reputation_risk import ReputationImpact, ReputationRiskScorer
from .scenario_war_gaming import (
    ComparisonMatrix,
    ScenarioBatchRunner,
    ScenarioComparator,
    ScenarioDefinition,
)
from .settlement_optimizer import SettlementOptimizer, SettlementRecommendation

logger = logging.getLogger(__name__)


@dataclass
class StrategicAnalysisConfig:
    """Configuration for strategic analysis engine."""

    model_path: Path
    enable_settlement_optimization: bool = True
    enable_game_theory: bool = True
    enable_scenario_gaming: bool = True
    enable_reputation_analysis: bool = True
    institution: str = "Harvard"

    # Settlement config overrides
    legal_costs: Optional[float] = None
    risk_aversion: Optional[float] = None


@dataclass
class CompleteStrategicReport:
    """Complete strategic analysis report."""

    insights: CaseInsights
    settlement: SettlementRecommendation
    game_theory: GameTheoryResult
    scenarios: Optional[ComparisonMatrix]
    reputation: Dict[str, ReputationImpact]
    timestamp: datetime

    def to_markdown_report(self) -> str:
        """Generate comprehensive markdown report."""
        report = []

        # Header
        report.append("# Complete Strategic Analysis Report")
        report.append(f"\n**Case:** {self.insights.reference_id}")
        report.append(f"**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Jurisdiction:** {self.insights.jurisdiction or 'Not specified'}")
        report.append("\n---\n")

        # Executive Summary
        report.append("## Executive Summary\n")
        report.append(f"**Optimal Settlement:** ${self.settlement.optimal_settlement:,.0f}")

        if self.game_theory.nash_equilibrium:
            report.append(f"**Nash Equilibrium:** ${self.game_theory.nash_equilibrium:,.0f}")

        report.append(f"**Expected Trial Value:** ${self.settlement.ev_analysis.ev_mean:,.0f}")
        report.append(f"**Downside Risk:** {self.settlement.ev_analysis.downside_probability:.1%}")
        report.append("\n---\n")

        # Settlement Analysis
        report.append(self.settlement.to_report())
        report.append("\n---\n")

        # Game Theory Analysis
        report.append("## Game Theory Analysis\n")
        report.append(self.game_theory.batna.to_report())
        report.append("\n### Negotiation Strategy\n")
        report.append(self.game_theory.strategy.strategy_narrative)
        report.append("\n---\n")

        # Scenario Comparison
        if self.scenarios:
            report.append("## Scenario Analysis\n")
            report.append(self.scenarios.to_report())
            report.append("\n---\n")

        # Reputation Risk
        report.append("## Reputation Risk Assessment\n")
        for outcome, impact in sorted(self.reputation.items()):
            report.append(impact.to_report())
            report.append("\n")

        report.append("\n---\n")

        # Bayesian Network Insights
        report.append("## Bayesian Network Analysis\n")
        report.append(self.insights.to_prompt_block())

        return "\n".join(report)


class StrategicAnalysisEngine:
    """Unified engine for complete strategic analysis."""

    def __init__(self, config: Optional[StrategicAnalysisConfig] = None):
        """
        Initialize strategic analysis engine.

        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config

    async def run_complete_analysis(
        self,
        model_path: Path,
        evidence: Dict[str, str],
        summary: str,
        *,
        scenarios: Optional[List[ScenarioDefinition]] = None,
        institution: str = "Harvard",
        reference_id: str = "case"
    ) -> CompleteStrategicReport:
        """
        Run complete strategic analysis.

        Workflow:
        1. BN inference -> CaseInsights
        2. Settlement optimization
        3. Game theory analysis
        4. Scenario comparison (if provided)
        5. Reputation analysis
        6. Generate unified report

        Args:
            model_path: Path to BN model
            evidence: Evidence dictionary for BN
            summary: Case summary
            scenarios: Optional list of scenarios for comparison
            institution: Institution name for reputation analysis
            reference_id: Case reference identifier

        Returns:
            CompleteStrategicReport with all analyses
        """
        logger.info("="*80)
        logger.info("STARTING COMPLETE STRATEGIC ANALYSIS")
        logger.info("="*80)

        # Step 1: BN Inference
        logger.info("\n[Step 1/5] Running Bayesian Network inference...")
        insights, _ = run_bn_inference(
            model_path,
            evidence,
            summary,
            reference_id=reference_id
        )
        logger.info(f" -> Generated {len(insights.posteriors)} posterior distributions")

        # Step 2: Settlement Optimization
        logger.info("\n[Step 2/5] Running settlement optimization...")
        settlement_opt = SettlementOptimizer()
        settlement_rec = settlement_opt.optimize_settlement(insights)
        logger.info(f" -> Optimal settlement: ${settlement_rec.optimal_settlement:,.0f}")

        # Step 3: Game Theory Analysis
        logger.info("\n[Step 3/5] Running game theory analysis...")
        batna_analyzer = BATNAAnalyzer()
        batna = batna_analyzer.analyze_batna(insights, settlement_rec)

        nash_calc = NashEquilibriumCalculator()
        nash = nash_calc.calculate_nash_settlement(batna)

        recommender = StrategicRecommender()
        strategy = recommender.recommend_strategy(batna, nash, settlement_rec, insights)

        game_theory_result = GameTheoryResult(
            batna=batna,
            nash_equilibrium=nash,
            strategy=strategy,
        )
        logger.info(f" -> Nash equilibrium: ${nash:,.0f if nash else 0}")

        # Step 4: Scenario Analysis (if provided)
        scenario_comparison = None
        if scenarios:
            logger.info(f"\n[Step 4/5] Running scenario war gaming ({len(scenarios)} scenarios)...")
            batch_runner = ScenarioBatchRunner(model_path)
            scenario_results = await batch_runner.run_scenarios(scenarios, summary)

            if scenario_results:
                comparator = ScenarioComparator()
                scenario_comparison = comparator.compare(scenario_results)
                logger.info(f" -> Completed {len(scenario_results)} scenarios")
        else:
            logger.info("\n[Step 4/5] Skipping scenario analysis (no scenarios provided)")

        # Step 5: Reputation Analysis
        logger.info("\n[Step 5/5] Running reputation risk analysis...")
        rep_scorer = ReputationRiskScorer()
        rep_impacts = rep_scorer.score_reputation_risk(insights)
        logger.info(f" -> Analyzed {len(rep_impacts)} outcome scenarios")

        # Generate Report
        logger.info("\n[Complete] Generating strategic report...")
        report = CompleteStrategicReport(
            insights=insights,
            settlement=settlement_rec,
            game_theory=game_theory_result,
            scenarios=scenario_comparison,
            reputation=rep_impacts,
            timestamp=datetime.now(),
        )

        logger.info("="*80)
        logger.info("STRATEGIC ANALYSIS COMPLETE")
        logger.info("="*80)

        return report

    def run_complete_analysis_sync(
        self,
        model_path: Path,
        evidence: Dict[str, str],
        summary: str,
        **kwargs
    ) -> CompleteStrategicReport:
        """
        Synchronous version of complete analysis (no scenario gaming).

        Args:
            model_path: Path to BN model
            evidence: Evidence dictionary
            summary: Case summary
            **kwargs: Additional arguments

        Returns:
            CompleteStrategicReport
        """
        import asyncio

        # Run without scenarios in sync mode
        return asyncio.run(self.run_complete_analysis(
            model_path,
            evidence,
            summary,
            scenarios=None, # Skip scenarios in sync mode
            **kwargs
        ))


def quick_settlement_analysis(
    model_path: Path,
    evidence: Dict[str, str],
    summary: str
) -> SettlementRecommendation:
    """
    Quick helper function for just settlement analysis.

    Args:
        model_path: Path to BN model
        evidence: Evidence dictionary
        summary: Case summary

    Returns:
        SettlementRecommendation
    """
    logger.info("Running quick settlement analysis...")

    # Run BN inference
    insights, _ = run_bn_inference(model_path, evidence, summary)

    # Run settlement optimization
    optimizer = SettlementOptimizer()
    recommendation = optimizer.optimize_settlement(insights)

    logger.info(f"Quick analysis complete: ${recommendation.optimal_settlement:,.0f}")
    return recommendation


def quick_game_theory_analysis(
    model_path: Path,
    evidence: Dict[str, str],
    summary: str
) -> GameTheoryResult:
    """
    Quick helper function for game theory analysis.

    Args:
        model_path: Path to BN model
        evidence: Evidence dictionary
        summary: Case summary

    Returns:
        GameTheoryResult
    """
    logger.info("Running quick game theory analysis...")

    # Run BN inference
    insights, _ = run_bn_inference(model_path, evidence, summary)

    # Settlement optimization
    optimizer = SettlementOptimizer()
    settlement = optimizer.optimize_settlement(insights)

    # Game theory
    batna_analyzer = BATNAAnalyzer()
    batna = batna_analyzer.analyze_batna(insights, settlement)

    nash_calc = NashEquilibriumCalculator()
    nash = nash_calc.calculate_nash_settlement(batna)

    recommender = StrategicRecommender()
    strategy = recommender.recommend_strategy(batna, nash, settlement, insights)

    result = GameTheoryResult(
        batna=batna,
        nash_equilibrium=nash,
        strategy=strategy,
    )

    logger.info(f"Quick analysis complete: Nash=${nash:,.0f if nash else 0}")
    return result


__all__ = [
    "StrategicAnalysisConfig",
    "CompleteStrategicReport",
    "StrategicAnalysisEngine",
    "quick_settlement_analysis",
    "quick_game_theory_analysis",
]

