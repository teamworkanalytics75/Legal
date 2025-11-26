"""Settlement optimizer agent - runs Monte Carlo simulations."""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..core.agent import BackgroundAgent, AgentConfig


class SettlementOptimizerAgent(BackgroundAgent):
    """Runs Monte Carlo simulations for settlement optimization (deterministic)."""

    def __init__(self, config: AgentConfig):
        # Override model to None since this is deterministic
        config.model = None
        super().__init__(config)
        self.output_dir = Path("background_agents/outputs/settlements")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def process(self, task: Any) -> Any:
        """
        Run settlement optimization.

        Args:
            task: Dict with case parameters

        Returns:
            Dict with optimization results
        """
        # Validate input
        if not isinstance(task, dict):
            self.logger.error(f"Invalid task data: {task}")
            return {'error': 'Task must be a dict'}

        # Extract parameters with defaults
        success_prob = task.get('success_probability', 0.5)
        damages_mean = task.get('damages_mean', 1000000)
        damages_std = task.get('damages_std', 200000)
        legal_costs = task.get('legal_costs', 50000)
        iterations = task.get('iterations', 10000)
        risk_aversion = task.get('risk_aversion', 0.5)

        # Run Monte Carlo simulation
        results = self._run_monte_carlo(
            success_prob,
            damages_mean,
            damages_std,
            legal_costs,
            iterations
        )

        # Calculate optimal settlement
        optimal = self._calculate_optimal_settlement(results, risk_aversion)

        result = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'success_probability': success_prob,
                'damages_mean': damages_mean,
                'damages_std': damages_std,
                'legal_costs': legal_costs,
                'iterations': iterations,
                'risk_aversion': risk_aversion
            },
            'simulation_results': results,
            'optimal_settlement': optimal
        }

        # Save result
        case_name = task.get('case_name', 'unknown')
        output_file = self.output_dir / f"settlement_{case_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def _run_monte_carlo(
        self,
        success_prob: float,
        damages_mean: float,
        damages_std: float,
        legal_costs: float,
        iterations: int
    ) -> Dict:
        """Run Monte Carlo simulation."""
        # Generate random outcomes
        outcomes = np.random.binomial(1, success_prob, iterations)

        # Generate damages (only for successful outcomes)
        damages = np.random.normal(damages_mean, damages_std, iterations)
        damages = np.maximum(damages, 0)  # Can't be negative

        # Calculate net recovery for each trial
        net_recovery = np.where(
            outcomes == 1,
            damages - legal_costs,  # Win: damages minus costs
            -legal_costs  # Lose: just costs
        )

        # Calculate statistics
        return {
            'expected_value': float(np.mean(net_recovery)),
            'median': float(np.median(net_recovery)),
            'std_dev': float(np.std(net_recovery)),
            'percentile_5': float(np.percentile(net_recovery, 5)),
            'percentile_25': float(np.percentile(net_recovery, 25)),
            'percentile_75': float(np.percentile(net_recovery, 75)),
            'percentile_95': float(np.percentile(net_recovery, 95)),
            'min': float(np.min(net_recovery)),
            'max': float(np.max(net_recovery)),
            'win_rate': float(np.mean(outcomes))
        }

    def _calculate_optimal_settlement(
        self,
        simulation: Dict,
        risk_aversion: float
    ) -> Dict:
        """Calculate optimal settlement range."""
        ev = simulation['expected_value']
        std = simulation['std_dev']

        # Risk-adjusted certainty equivalent
        certainty_equivalent = ev - (risk_aversion * std)

        # Settlement range (80% to 100% of certainty equivalent)
        min_acceptable = certainty_equivalent * 0.8
        optimal = certainty_equivalent
        aggressive = ev  # For aggressive negotiation

        return {
            'certainty_equivalent': float(certainty_equivalent),
            'minimum_acceptable': float(min_acceptable),
            'optimal_settlement': float(optimal),
            'aggressive_target': float(aggressive),
            'expected_value': float(ev),
            'recommendation': self._generate_recommendation(
                min_acceptable,
                optimal,
                aggressive,
                simulation
            )
        }

    def _generate_recommendation(
        self,
        min_acceptable: float,
        optimal: float,
        aggressive: float,
        simulation: Dict
    ) -> str:
        """Generate settlement recommendation text."""
        return f"""Settlement Recommendation:

Optimal Settlement Range: ${min_acceptable:,.0f} - ${optimal:,.0f}

- Minimum Acceptable: ${min_acceptable:,.0f}
  (Walk away if offered less)

- Optimal Settlement: ${optimal:,.0f}
  (Risk-adjusted fair value)

- Aggressive Target: ${aggressive:,.0f}
  (Expected value without risk adjustment)

Trial Outcome Analysis:
- Expected Value: ${simulation['expected_value']:,.0f}
- Downside Risk (5th percentile): ${simulation['percentile_5']:,.0f}
- Upside Potential (95th percentile): ${simulation['percentile_95']:,.0f}

Recommendation: Accept any offer above ${optimal:,.0f}.
Negotiate aggressively between ${min_acceptable:,.0f} and ${optimal:,.0f}.
Reject offers below ${min_acceptable:,.0f} unless other factors apply."""

