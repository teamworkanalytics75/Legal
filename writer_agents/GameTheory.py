"""Game theory models for legal settlement negotiations.

This module provides tools for calculating BATNA (Best Alternative To Negotiated Agreement),
Nash equilibrium, and strategic negotiation recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from writer_agents.code.insights import CaseInsights
from .settlement_optimizer import SettlementRecommendation, SettlementConfig

logger = logging.getLogger(__name__)


@dataclass
class BATNAResult:
    """Best Alternative To Negotiated Agreement analysis."""
    your_batna: float # Your expected value from going to trial
    their_batna: float # Opponent's expected cost (negative)
    zopa_exists: bool # Zone of Possible Agreement exists
    zopa_range: Optional[Tuple[float, float]] # (min, max) if ZOPA exists

    def to_report(self) -> str:
        """Generate markdown report."""
        zopa_status = "[ok] YES" if self.zopa_exists else " NO"
        zopa_range_str = (
            f"${self.zopa_range[0]:,.0f} - ${abs(self.zopa_range[1]):,.0f}"
            if self.zopa_range
            else "N/A"
        )

        return f"""
### BATNA Analysis

**Your BATNA (Expected Trial Value):** ${self.your_batna:,.0f}

**Opponent's BATNA (Expected Cost):** ${abs(self.their_batna):,.0f}

**Zone of Possible Agreement (ZOPA):** {zopa_status}

**ZOPA Range:** {zopa_range_str}
"""


@dataclass
class NegotiationStrategy:
    """Strategic negotiation recommendations."""
    first_offer: float
    target_range: Tuple[float, float]
    walkaway_point: float
    nash_equilibrium: Optional[float]
    strategy_narrative: str


@dataclass
class GameTheoryResult:
    """Complete game theory analysis results."""
    batna: BATNAResult
    nash_equilibrium: Optional[float]
    strategy: NegotiationStrategy


class BATNAAnalyzer:
    """Analyze Best Alternative To Negotiated Agreement."""

    def __init__(self, opponent_cost_multiplier: float = 1.5):
        """
        Initialize BATNA analyzer.

        Args:
            opponent_cost_multiplier: Multiplier for opponent's legal costs
                                     (typically higher for institutions like Harvard)
        """
        self.opponent_cost_multiplier = opponent_cost_multiplier

    def analyze_batna(
        self,
        insights: CaseInsights,
        settlement_rec: SettlementRecommendation,
        opponent_legal_costs: float = 750_000.0
    ) -> BATNAResult:
        """
        Calculate BATNA for both parties.

        Your BATNA = Expected value of going to trial (from settlement optimizer)
        Their BATNA = Their expected cost of trial (estimated from your posteriors)

        Args:
            insights: CaseInsights from BN inference
            settlement_rec: Settlement recommendation from optimizer
            opponent_legal_costs: Estimated legal costs for opponent

        Returns:
            BATNAResult with both parties' BATNAs and ZOPA analysis
        """
        logger.info("Analyzing BATNA for both parties...")

        # Your BATNA is the certainty equivalent from settlement analysis
        your_batna = settlement_rec.ev_analysis.certainty_equivalent

        # Estimate opponent's BATNA
        # They face: probability of losing * damages + their legal costs
        posteriors = insights.posterior_lookup()

        # Calculate their expected loss
        expected_damages = self._calculate_expected_damages(
            posteriors,
            SettlementConfig() # Use default config for standard mappings
        )

        # Their BATNA is negative (it's a cost to them)
        their_batna = -(expected_damages + opponent_legal_costs)

        # ZOPA (Zone of Possible Agreement) exists if:
        # your minimum acceptable < their maximum willing to pay
        # In other words: your_batna < abs(their_batna)
        zopa_exists = your_batna < abs(their_batna)

        if zopa_exists:
            # ZOPA range: from your minimum to their maximum
            zopa_range = (your_batna, abs(their_batna))
        else:
            zopa_range = None

        logger.info(
            f"BATNA Analysis: Your=${your_batna:,.0f}, "
            f"Their=${their_batna:,.0f}, ZOPA={zopa_exists}"
        )

        return BATNAResult(
            your_batna=your_batna,
            their_batna=their_batna,
            zopa_exists=zopa_exists,
            zopa_range=zopa_range,
        )

    def _calculate_expected_damages(
        self,
        posteriors: Dict[str, any],
        config: SettlementConfig
    ) -> float:
        """Calculate opponent's expected damage payment."""
        success_posterior = posteriors.get(config.success_node_id)
        damages_posterior = posteriors.get(config.damages_node_id)

        if not success_posterior or not damages_posterior:
            # Fallback to conservative estimate
            return 5_000_000.0

        # Calculate expected damages
        expected_damages = 0.0
        for success_state, success_prob in success_posterior.probabilities.items():
            success_mult = config.success_outcomes.get(success_state, 0.0)

            for damage_state, damage_prob in damages_posterior.probabilities.items():
                damage_amount = config.damages_outcomes.get(damage_state, 0.0)

                # Expected value contribution
                expected_damages += success_prob * damage_prob * success_mult * damage_amount

        return expected_damages


class NashEquilibriumCalculator:
    """Calculate Nash equilibrium for settlement negotiations."""

    def calculate_nash_settlement(
        self,
        batna_result: BATNAResult,
        bargaining_power: float = 0.5
    ) -> Optional[float]:
        """
        Calculate Nash bargaining solution.

        The Nash bargaining solution maximizes the product of gains above each party's BATNA:
        Nash = arg max [(your_payoff - your_batna)^ * (their_payoff - their_batna)^(1-)]

        For equal bargaining power (=0.5), this simplifies to splitting the surplus equally.

        Args:
            batna_result: BATNA analysis results
            bargaining_power: Your relative power (0.5 = equal, >0.5 = you have more power)

        Returns:
            Nash equilibrium settlement amount, or None if no ZOPA exists
        """
        if not batna_result.zopa_exists:
            logger.warning("No ZOPA exists - Nash equilibrium not calculable")
            return None

        # Total surplus to be split between parties
        # Surplus = what they're willing to pay - what you need
        total_surplus = abs(batna_result.their_batna) - batna_result.your_batna

        # Nash solution with power asymmetry
        # If you have more power (>0.5), you get more of the surplus
        your_share = bargaining_power * total_surplus
        nash_settlement = batna_result.your_batna + your_share

        logger.info(
            f"Nash Equilibrium: ${nash_settlement:,.0f} "
            f"(Power={bargaining_power:.2f}, Surplus=${total_surplus:,.0f})"
        )

        return nash_settlement


class StrategicRecommender:
    """Generate negotiation strategy recommendations."""

    def recommend_strategy(
        self,
        batna_result: BATNAResult,
        nash_settlement: Optional[float],
        settlement_rec: SettlementRecommendation,
        insights: CaseInsights
    ) -> NegotiationStrategy:
        """
        Provide strategic recommendations for negotiation.

        Args:
            batna_result: BATNA analysis
            nash_settlement: Nash equilibrium settlement
            settlement_rec: Settlement recommendation
            insights: Case insights for evidence reference

        Returns:
            NegotiationStrategy with specific recommendations
        """
        logger.info("Generating negotiation strategy...")

        if not batna_result.zopa_exists or nash_settlement is None:
            # No ZOPA - prepare for trial
            strategy_text = """
**Negotiation Position:** WEAK - No Zone of Possible Agreement

**Analysis:** The expected value analysis shows no overlap between what you need
and what the opponent is likely willing to pay. This suggests:
- Your BATNA is too high (strong case for you)
- Their expected costs are lower than your expectations
- Settlement is unlikely without significant new evidence or strategy changes

**Recommendation:** Prepare for trial. If settlement discussions occur:
1. Start with a very high anchor to test their reaction
2. Focus on strengthening your case evidence
3. Consider what new information might change the calculation
4. Be prepared to walk away
"""
            return NegotiationStrategy(
                first_offer=batna_result.your_batna * 1.5, # High anchor
                target_range=(batna_result.your_batna, batna_result.your_batna),
                walkaway_point=batna_result.your_batna,
                nash_equilibrium=nash_settlement,
                strategy_narrative=strategy_text,
            )

        # ZOPA exists - settlement is possible
        # First offer (anchoring): start high but credible
        first_offer = nash_settlement * 1.25
        if first_offer > abs(batna_result.their_batna):
            first_offer = abs(batna_result.their_batna) * 0.95 # Just below their max

        # Walk-away point: your BATNA minus small buffer
        walkaway_point = batna_result.your_batna * 0.9

        # Target range around Nash equilibrium
        target_min = nash_settlement * 0.9
        target_max = nash_settlement * 1.1

        # Get strongest evidence for justification
        strongest_evidence = self._get_strongest_evidence(insights)

        # Strategy narrative
        strategy_text = f"""
**Negotiation Position:** STRONG - ZOPA Exists

**Zone of Possible Agreement:** ${batna_result.zopa_range[0]:,.0f} - ${abs(batna_result.zopa_range[1]):,.0f}

**Opening Move:**
- **First Offer:** ${first_offer:,.0f}
- **Justification:** Based on expected trial outcome and strength of evidence
- **Anchoring Strategy:** Start high to shape opponent's expectations

**Target Settlement Zone:** ${target_min:,.0f} - ${target_max:,.0f}
- This range is centered on the Nash equilibrium: ${nash_settlement:,.0f}
- Represents fair split of surplus given relative bargaining positions

**Walk-Away Point:** ${walkaway_point:,.0f}
- Below this, you're better off going to trial
- This is {(walkaway_point / batna_result.your_batna - 1) * 100:.0f}% below your BATNA for buffer

**Recommended Tactics:**

1. **Anchor High:** Open with ${first_offer:,.0f} to set high reference point
2. **Justify with Evidence:** Emphasize strongest elements: {strongest_evidence}
3. **Show Credible Threat:** Demonstrate willingness and preparation to go to trial
4. **Negotiate Toward Nash:** Use principled negotiation to move toward ${nash_settlement:,.0f}
5. **Know Your Floor:** Don't accept below ${walkaway_point:,.0f}

**Information Strategy:**
- **Reveal:** Strength of evidence, case preparation, trial readiness
- **Emphasize:** Risks they face (legal costs, reputational damage, uncertainty)
- **Conceal:** Time pressures, cost sensitivities, internal constraints
- **Signal:** Flexibility within target zone, but firmness on walkaway point

**Concession Strategy:**
- Start with small concessions (2-5%) to show good faith
- Slow down concessions as you approach target range
- Make final offers around Nash equilibrium
- Anchor final positions with principled justifications

**Timeline Leverage:**
- Their costs accumulate over time (${settlement_rec.break_even_point:,.0f} in legal fees)
- Use trial preparation deadlines as leverage points
- Consider splitting discussions: liability first, damages second
"""

        return NegotiationStrategy(
            first_offer=first_offer,
            target_range=(target_min, target_max),
            walkaway_point=walkaway_point,
            nash_equilibrium=nash_settlement,
            strategy_narrative=strategy_text,
        )

    def _get_strongest_evidence(self, insights: CaseInsights) -> str:
        """Identify strongest evidence items for narrative."""
        if not insights.evidence:
            return "Posterior probability analysis"

        # For now, return first few evidence items
        evidence_list = [f"{e.node_id}={e.state}" for e in insights.evidence[:3]]
        return ", ".join(evidence_list)


__all__ = [
    "BATNAResult",
    "NegotiationStrategy",
    "GameTheoryResult",
    "BATNAAnalyzer",
    "NashEquilibriumCalculator",
    "StrategicRecommender",
]

