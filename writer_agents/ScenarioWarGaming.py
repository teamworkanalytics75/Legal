"""Scenario war gaming for batch analysis of multiple evidence configurations.

This module enables comparative analysis across different scenarios to understand
how changes in evidence affect case outcomes and settlement recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False
    pd = None

from writer_agents.code.bn_adapter import run_bn_inference
from writer_agents.code.insights import CaseInsights
from .game_theory import BATNAAnalyzer, NashEquilibriumCalculator
from .settlement_optimizer import SettlementOptimizer, SettlementRecommendation

logger = logging.getLogger(__name__)


@dataclass
class ScenarioDefinition:
    """Defines a single scenario to analyze."""
    scenario_id: str
    name: str
    description: str
    evidence: Dict[str, str] # Evidence dictionary for BN
    assumptions: List[str] # Human-readable assumptions


@dataclass
class ScenarioResult:
    """Results from analyzing a single scenario."""
    scenario: ScenarioDefinition
    insights: CaseInsights
    settlement: SettlementRecommendation
    batna: any # BATNAResult
    nash_equilibrium: Optional[float]


@dataclass
class ComparisonMatrix:
    """Comparison of results across scenarios."""
    data_frame: any # pandas DataFrame if available
    comparison_dict: Dict[str, Dict[str, float]] # Fallback dictionary
    best_scenario: Dict[str, any]
    worst_scenario: Dict[str, any]

    def to_report(self) -> str:
        """Generate markdown comparison report."""
        report = ["## Scenario Comparison\n"]

        if HAVE_PANDAS and self.data_frame is not None:
            report.append("```")
            report.append(str(self.data_frame.to_string()))
            report.append("```\n")
        else:
            # Fallback to dictionary format
            report.append("| Scenario | Optimal Settlement | Nash Equilibrium | EV Trial | Downside Risk |")
            report.append("|----------|-------------------|------------------|----------|---------------|")
            for scenario, data in self.comparison_dict.items():
                report.append(
                    f"| {scenario} | ${data.get('Optimal Settlement', 0):,.0f} | "
                    f"${data.get('Nash Equilibrium', 0):,.0f} | "
                    f"${data.get('EV Trial', 0):,.0f} | "
                    f"{data.get('Downside Risk', 0):.1%} |"
                )
            report.append("")

        report.append(f"\n**Best Scenario:** {self.best_scenario.get('Scenario', 'N/A')}")
        report.append(f"- Optimal Settlement: ${self.best_scenario.get('Optimal Settlement', 0):,.0f}")

        report.append(f"\n**Worst Scenario:** {self.worst_scenario.get('Scenario', 'N/A')}")
        report.append(f"- Optimal Settlement: ${self.worst_scenario.get('Optimal Settlement', 0):,.0f}")

        return "\n".join(report)


class ScenarioBatchRunner:
    """Run BN inference across multiple scenarios."""

    def __init__(self, model_path: Path):
        """
        Initialize batch runner.

        Args:
            model_path: Path to BN model file (.xdsl or .pkl)
        """
        self.model_path = model_path

    async def run_scenarios(
        self,
        scenarios: List[ScenarioDefinition],
        summary: str,
        use_fallback: bool = True
    ) -> List[ScenarioResult]:
        """
        Run all scenarios in batch.

        For each scenario:
        1. Run BN inference
        2. Run settlement optimization
        3. Run game theory analysis
        4. Package results

        Args:
            scenarios: List of scenario definitions
            summary: Case summary text
            use_fallback: Whether to use fallback if BN inference fails

        Returns:
            List of ScenarioResult objects
        """
        results = []

        logger.info(f"Running {len(scenarios)} scenarios...")

        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"[{i}/{len(scenarios)}] Running scenario: {scenario.name}")

            try:
                # Run BN inference
                insights, _ = run_bn_inference(
                    self.model_path,
                    scenario.evidence,
                    summary,
                    reference_id=scenario.scenario_id
                )

                # Run settlement optimization
                settlement_opt = SettlementOptimizer()
                settlement_rec = settlement_opt.optimize_settlement(insights)

                # Run game theory
                batna_analyzer = BATNAAnalyzer()
                batna = batna_analyzer.analyze_batna(insights, settlement_rec)

                nash_calc = NashEquilibriumCalculator()
                nash = nash_calc.calculate_nash_settlement(batna)

                # Package result
                results.append(ScenarioResult(
                    scenario=scenario,
                    insights=insights,
                    settlement=settlement_rec,
                    batna=batna,
                    nash_equilibrium=nash,
                ))

                logger.info(
                    f" -> Optimal Settlement: ${settlement_rec.optimal_settlement:,.0f}, "
                    f"Nash: ${nash:,.0f if nash else 0}"
                )

            except Exception as e:
                logger.error(f"Failed to run scenario {scenario.name}: {e}")
                if not use_fallback:
                    raise
                # Continue with other scenarios
                continue

        logger.info(f"Completed {len(results)}/{len(scenarios)} scenarios successfully")
        return results


class ScenarioComparator:
    """Compare results across scenarios."""

    def compare(self, results: List[ScenarioResult]) -> ComparisonMatrix:
        """
        Generate comparison matrix.

        Compares:
        - Settlement ranges
        - Nash equilibria
        - Key posterior probabilities
        - Risk metrics

        Args:
            results: List of scenario results

        Returns:
            ComparisonMatrix with analysis
        """
        logger.info(f"Comparing {len(results)} scenarios...")

        comparison_data = []

        for result in results:
            row = {
                "Scenario": result.scenario.name,
                "Optimal Settlement": result.settlement.optimal_settlement,
                "Nash Equilibrium": result.nash_equilibrium if result.nash_equilibrium else 0.0,
                "EV Trial": result.settlement.ev_analysis.ev_mean,
                "Downside Risk": result.settlement.ev_analysis.downside_probability,
                "Success Prob": self._get_success_prob(result.insights),
            }
            comparison_data.append(row)

        # Create comparison structures
        if HAVE_PANDAS:
            df = pd.DataFrame(comparison_data)

            # Identify best/worst scenarios
            best_idx = df["Optimal Settlement"].idxmax()
            worst_idx = df["Optimal Settlement"].idxmin()

            best_scenario = df.loc[best_idx].to_dict()
            worst_scenario = df.loc[worst_idx].to_dict()

            comparison_dict = {row["Scenario"]: row for row in comparison_data}
        else:
            df = None
            comparison_dict = {row["Scenario"]: row for row in comparison_data}

            # Find best/worst without pandas
            best_scenario = max(comparison_data, key=lambda x: x["Optimal Settlement"])
            worst_scenario = min(comparison_data, key=lambda x: x["Optimal Settlement"])

        return ComparisonMatrix(
            data_frame=df,
            comparison_dict=comparison_dict,
            best_scenario=best_scenario,
            worst_scenario=worst_scenario,
        )

    def _get_success_prob(self, insights: CaseInsights) -> float:
        """Extract success probability from insights."""
        posteriors = insights.posterior_lookup()
        success_post = posteriors.get("LegalSuccess_US")

        if not success_post:
            return 0.0

        # Sum probabilities of positive outcomes
        success_prob = 0.0
        for state, prob in success_post.probabilities.items():
            if state in ["High", "Moderate"]:
                success_prob += prob

        return success_prob


class ScenarioVisualizer:
    """Visualize scenario comparisons (requires matplotlib)."""

    def __init__(self):
        """Initialize visualizer."""
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.have_matplotlib = True
        except ImportError:
            self.plt = None
            self.have_matplotlib = False
            logger.warning("matplotlib not available - visualization disabled")

    def plot_settlement_comparison(
        self,
        comparison: ComparisonMatrix,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Create bar chart comparing settlements across scenarios.

        Args:
            comparison: Comparison matrix
            output_path: Optional path to save plot

        Returns:
            Path to saved plot, or None if matplotlib not available
        """
        if not self.have_matplotlib:
            logger.warning("Cannot plot - matplotlib not installed")
            return None

        if not HAVE_PANDAS or comparison.data_frame is None:
            logger.warning("Cannot plot - pandas DataFrame not available")
            return None

        df = comparison.data_frame

        # Create bar chart
        fig, ax = self.plt.subplots(figsize=(10, 6))

        scenarios = df["Scenario"]
        settlements = df["Optimal Settlement"]

        ax.bar(scenarios, settlements, color='steelblue', alpha=0.7)
        ax.set_xlabel("Scenario", fontsize=12)
        ax.set_ylabel("Optimal Settlement ($)", fontsize=12)
        ax.set_title("Settlement Comparison Across Scenarios", fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for i, v in enumerate(settlements):
            ax.text(i, v + 50000, f"${v:,.0f}", ha='center', va='bottom', fontsize=9)

        self.plt.tight_layout()

        if output_path:
            self.plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {output_path}")
            return output_path

        return None


# Pre-defined Harvard scenario templates
HARVARD_SCENARIOS = [
    ScenarioDefinition(
        scenario_id="S1_strong",
        name="Strong Case",
        description="All evidence admitted, strong case",
        evidence={
            "OGC_Email_Apr18_2025": "Sent",
            "PRC_Awareness": "Direct",
        },
        assumptions=["OGC email proven", "Direct PRC awareness established"]
    ),
    ScenarioDefinition(
        scenario_id="S2_moderate",
        name="Moderate Case",
        description="Email excluded, but other evidence holds",
        evidence={
            "OGC_Email_Apr18_2025": "Not_Sent",
            "PRC_Awareness": "Direct",
        },
        assumptions=["Email evidence excluded", "PRC awareness via other means"]
    ),
    ScenarioDefinition(
        scenario_id="S3_weak",
        name="Weak Case",
        description="Only circumstantial evidence",
        evidence={
            "OGC_Email_Apr18_2025": "Not_Sent",
            "PRC_Awareness": "Indirect",
        },
        assumptions=["Email excluded", "Only indirect PRC awareness"]
    ),
]


__all__ = [
    "ScenarioDefinition",
    "ScenarioResult",
    "ComparisonMatrix",
    "ScenarioBatchRunner",
    "ScenarioComparator",
    "ScenarioVisualizer",
    "HARVARD_SCENARIOS",
]

