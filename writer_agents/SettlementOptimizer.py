"""Settlement optimization engine using Monte Carlo simulation and risk analysis.

This module provides tools to calculate optimal settlement ranges based on
Bayesian network posteriors, legal costs, and risk preferences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from writer_agents.code.insights import CaseInsights

logger = logging.getLogger(__name__)


@dataclass
class SettlementConfig:
    """Configuration for settlement optimization."""

    # Financial parameters
    expected_legal_costs: float = 500_000.0 # Total legal costs if go to trial
    monthly_legal_burn_rate: float = 50_000.0
    expected_trial_duration_months: int = 18

    # Risk parameters
    risk_aversion_coefficient: float = 0.3 # 0=risk-neutral, 1=very risk-averse
    discount_rate: float = 0.05 # Annual discount rate for time value

    # Simulation parameters
    monte_carlo_iterations: int = 10_000
    confidence_interval: float = 0.90 # 90% confidence interval

    # BN node mappings (customize for your model)
    success_node_id: str = "LegalSuccess_US"
    damages_node_id: str = "FinancialDamage"  # Updated to match actual BN model

    # Outcome value mappings
    success_outcomes: Dict[str, float] = field(default_factory=lambda: {
        "High": 1.0,
        "Moderate": 0.5,
        "Low": 0.1,
    })

    damages_outcomes: Dict[str, float] = field(default_factory=lambda: {
        "Material": 10_000_000.0,
        "Moderate": 3_000_000.0,
        "Minor": 500_000.0,
    })


@dataclass
class TrialOutcome:
    """Single trial outcome from Monte Carlo simulation."""
    gross_recovery: float
    legal_costs: float
    net_outcome: float
    success_state: str
    damages_state: str


@dataclass
class EVAnalysis:
    """Expected value analysis results."""
    ev_mean: float
    ev_median: float
    ev_std: float
    certainty_equivalent: float
    confidence_interval: Tuple[float, float]
    downside_probability: float


@dataclass
class SettlementRecommendation:
    """Complete settlement recommendation with analysis."""
    optimal_settlement: float
    settlement_range: Tuple[float, float]
    ev_analysis: EVAnalysis
    strategy_recommendation: str
    break_even_point: float
    monte_carlo_outcomes: List[TrialOutcome]

    def to_report(self) -> str:
        """Generate markdown report."""
        return f"""
## Settlement Analysis

**Optimal Settlement:** ${self.optimal_settlement:,.0f}

**Acceptable Range:** ${self.settlement_range[0]:,.0f} - ${self.settlement_range[1]:,.0f}

**Expected Value of Trial:** ${self.ev_analysis.ev_mean:,.0f} ${self.ev_analysis.ev_std:,.0f}

**Risk-Adjusted Value:** ${self.ev_analysis.certainty_equivalent:,.0f}

**Downside Risk:** {self.ev_analysis.downside_probability:.1%} chance of negative outcome

**Break-Even Point:** ${self.break_even_point:,.0f} (legal costs)

**Strategy:** {self.strategy_recommendation}
"""


class MonteCarloSimulator:
    """Runs Monte Carlo simulations based on BN posteriors."""

    def __init__(self, config: SettlementConfig):
        self.config = config

    def run_simulation(self, insights: CaseInsights) -> List[TrialOutcome]:
        """
        Run Monte Carlo simulation using posterior probabilities.

        Algorithm:
        1. Extract posteriors for success and damages nodes
        2. For each iteration:
           - Sample success outcome from posterior distribution
           - Sample damages amount from posterior distribution
           - Calculate net outcome = (success * damages) - costs
        3. Return list of all outcomes

        Args:
            insights: CaseInsights from BN inference

        Returns:
            List of TrialOutcome objects
        """
        posteriors = insights.posterior_lookup()

        # Get posteriors for key nodes
        success_posterior = posteriors.get(self.config.success_node_id)
        damages_posterior = posteriors.get(self.config.damages_node_id)

        if not success_posterior:
            raise ValueError(
                f"Success node '{self.config.success_node_id}' not found in posteriors. "
                f"Available nodes: {list(posteriors.keys())}"
            )
        if not damages_posterior:
            raise ValueError(
                f"Damages node '{self.config.damages_node_id}' not found in posteriors. "
                f"Available nodes: {list(posteriors.keys())}"
            )

        outcomes = []
        rng = np.random.default_rng(seed=42)

        logger.info(f"Running {self.config.monte_carlo_iterations} Monte Carlo iterations...")

        for _ in range(self.config.monte_carlo_iterations):
            # Sample success (convert to probability multiplier)
            success_state = self._sample_from_distribution(
                success_posterior.probabilities,
                rng
            )
            success_multiplier = self.config.success_outcomes.get(success_state, 0.0)

            # Sample damages amount
            damages_state = self._sample_from_distribution(
                damages_posterior.probabilities,
                rng
            )
            damages_amount = self.config.damages_outcomes.get(damages_state, 0.0)

            # Calculate outcome
            gross_recovery = success_multiplier * damages_amount
            legal_costs = self.config.expected_legal_costs
            net_outcome = gross_recovery - legal_costs

            outcomes.append(TrialOutcome(
                gross_recovery=gross_recovery,
                legal_costs=legal_costs,
                net_outcome=net_outcome,
                success_state=success_state,
                damages_state=damages_state,
            ))

        logger.info(f"Monte Carlo simulation complete. Generated {len(outcomes)} outcomes.")
        return outcomes

    def _sample_from_distribution(
        self,
        probs: Dict[str, float],
        rng: np.random.Generator
    ) -> str:
        """Sample a state from probability distribution."""
        states = list(probs.keys())
        probabilities = [probs[s] for s in states]

        # Normalize probabilities to sum to 1.0
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # Equal probability if all zeros
            probabilities = [1.0 / len(states)] * len(states)

        return str(rng.choice(states, p=probabilities))


class ExpectedValueCalculator:
    """Calculate risk-adjusted expected values."""

    def __init__(self, config: SettlementConfig):
        self.config = config

    def calculate_ev_trial(self, outcomes: List[TrialOutcome]) -> EVAnalysis:
        """
        Calculate expected value of going to trial.

        Includes:
        - Raw expected value (mean)
        - Risk-adjusted value (incorporates risk aversion)
        - Confidence intervals
        - Downside risk metrics

        Args:
            outcomes: List of trial outcomes from Monte Carlo

        Returns:
            EVAnalysis with comprehensive statistics
        """
        net_outcomes = np.array([o.net_outcome for o in outcomes])

        # Basic statistics
        ev_mean = float(np.mean(net_outcomes))
        ev_median = float(np.median(net_outcomes))
        ev_std = float(np.std(net_outcomes))

        # Confidence intervals
        ci_percent_lower = (1 - self.config.confidence_interval) / 2 * 100
        ci_percent_upper = (1 + self.config.confidence_interval) / 2 * 100
        ci_lower = float(np.percentile(net_outcomes, ci_percent_lower))
        ci_upper = float(np.percentile(net_outcomes, ci_percent_upper))

        # Risk adjustment using certainty equivalent
        # CE = EV - risk_aversion_coefficient * std_deviation
        # This is a more standard and reasonable risk adjustment
        # Use standard deviation instead of variance to avoid extreme values
        risk_adjustment = self.config.risk_aversion_coefficient * ev_std
        certainty_equivalent = ev_mean - risk_adjustment

        # Downside risk (probability of negative outcome)
        downside_prob = float(np.sum(net_outcomes < 0) / len(net_outcomes))

        logger.info(f"EV Analysis: Mean=${ev_mean:,.0f}, CE=${certainty_equivalent:,.0f}, "
                   f"Downside={downside_prob:.1%}")

        return EVAnalysis(
            ev_mean=ev_mean,
            ev_median=ev_median,
            ev_std=ev_std,
            certainty_equivalent=certainty_equivalent,
            confidence_interval=(ci_lower, ci_upper),
            downside_probability=downside_prob,
        )


class SettlementOptimizer:
    """Main class that recommends settlement strategy."""

    def optimize_settlement(
        self,
        insights: CaseInsights,
        config: Optional[SettlementConfig] = None
    ) -> SettlementRecommendation:
        """
        Generate settlement recommendation based on BN analysis.

        This uses Monte Carlo simulation to model trial outcomes and
        calculates optimal settlement ranges using risk-adjusted expected values.

        Args:
            insights: CaseInsights from BN inference
            config: Optional configuration (uses defaults if not provided)

        Returns:
            SettlementRecommendation with optimal range and strategy
        """
        config = config or SettlementConfig()

        logger.info(f"Optimizing settlement for case: {insights.reference_id}")

        # Step 1: Run Monte Carlo simulation
        simulator = MonteCarloSimulator(config)
        outcomes = simulator.run_simulation(insights)

        # Step 2: Calculate expected value of trial
        ev_calc = ExpectedValueCalculator(config)
        ev_analysis = ev_calc.calculate_ev_trial(outcomes)

        # Step 3: Determine optimal settlement point
        # Optimal settlement = expected value (not certainty equivalent for settlement)
        # Certainty equivalent is too conservative for settlement negotiations
        optimal_settlement = ev_analysis.ev_mean

        # Step 4: Calculate acceptable settlement range
        # Range is based on confidence interval and strategic considerations
        # Lower bound: 80% of confidence interval lower bound
        # Upper bound: 120% of confidence interval upper bound
        min_acceptable = max(0, ev_analysis.confidence_interval[0] * 0.8)
        max_offer = ev_analysis.confidence_interval[1] * 1.2

        # Step 5: Generate strategic recommendation
        if optimal_settlement <= 0:
            strategy = (
                "Strong position to reject settlement. Expected value is negative, "
                "suggesting the case favors the defendant. Consider demanding minimal "
                "payment or proceeding to trial if you have strong evidence."
            )
        elif optimal_settlement < config.expected_legal_costs * 0.5:
            strategy = (
                f"Favor settlement to avoid legal costs. Expected value "
                f"(${optimal_settlement:,.0f}) is significantly below break-even point "
                f"(${config.expected_legal_costs:,.0f}). Settlement saves costs even at "
                f"lower amounts."
            )
        elif optimal_settlement < config.expected_legal_costs:
            strategy = (
                f"Settlement makes economic sense. Expected value (${optimal_settlement:,.0f}) "
                f"is below legal costs (${config.expected_legal_costs:,.0f}). Settling "
                f"within the recommended range avoids the risk and cost of trial."
            )
        else:
            strategy = (
                f"Strong case justifies trial consideration. Expected value "
                f"(${optimal_settlement:,.0f}) exceeds legal costs. Settlement should "
                f"only be considered if offer is within or above the recommended range."
            )

        logger.info(f"Settlement optimization complete. Optimal: ${optimal_settlement:,.0f}")

        return SettlementRecommendation(
            optimal_settlement=optimal_settlement,
            settlement_range=(min_acceptable, max_offer),
            ev_analysis=ev_analysis,
            strategy_recommendation=strategy,
            break_even_point=config.expected_legal_costs,
            monte_carlo_outcomes=outcomes,
        )


__all__ = [
    "SettlementConfig",
    "TrialOutcome",
    "EVAnalysis",
    "SettlementRecommendation",
    "MonteCarloSimulator",
    "ExpectedValueCalculator",
    "SettlementOptimizer",
]

