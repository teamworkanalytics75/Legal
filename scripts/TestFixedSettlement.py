#!/usr/bin/env python
"""Test the fixed settlement optimizer."""

import sys
from pathlib import Path
from writer_agents.code.insights import CaseInsights, Posterior
from writer_agents.settlement_optimizer import SettlementOptimizer, SettlementConfig

def test_fixed_settlement_optimizer():
    """Test the settlement optimizer with realistic inputs."""
    print("Testing Fixed Settlement Optimizer")
    print("="*50)
    
    # Create realistic posteriors
    posteriors = [
        Posterior(
            node_id="LegalSuccess_US",
            probabilities={
                "High": 0.65,
                "Moderate": 0.25,
                "Low": 0.10
            },
            interpretation="Strong evidence of discrimination"
        ),
        Posterior(
            node_id="FinancialDamage",
            probabilities={
                "Material": 0.70,
                "Moderate": 0.25,
                "Minor": 0.05
            },
            interpretation="Harvard has substantial resources"
        )
    ]
    
    insights = CaseInsights(
        reference_id="test_fixed_settlement",
        summary="Test case for fixed settlement optimizer",
        posteriors=posteriors
    )
    
    # Test with reasonable configuration
    config = SettlementConfig(
        monte_carlo_iterations=1000,  # Faster for testing
        expected_legal_costs=750_000,
        risk_aversion_coefficient=0.3
    )
    
    optimizer = SettlementOptimizer()
    settlement_rec = optimizer.optimize_settlement(insights, config)
    
    print(f"Optimal settlement: ${settlement_rec.optimal_settlement:,.0f}")
    print(f"Settlement range: ${settlement_rec.settlement_range[0]:,.0f} - ${settlement_rec.settlement_range[1]:,.0f}")
    print(f"Expected trial value: ${settlement_rec.ev_analysis.ev_mean:,.0f}")
    print(f"Certainty equivalent: ${settlement_rec.ev_analysis.certainty_equivalent:,.0f}")
    print(f"Downside risk: {settlement_rec.ev_analysis.downside_probability:.1%}")
    
    # Check if values are reasonable
    if abs(settlement_rec.optimal_settlement) > 100_000_000:  # More than $100M
        print("FAIL: Settlement values are still unrealistic")
        return False
    elif settlement_rec.optimal_settlement < 0:
        print("FAIL: Optimal settlement is negative")
        return False
    else:
        print("PASS: Settlement values are reasonable")
        return True

if __name__ == "__main__":
    success = test_fixed_settlement_optimizer()
    sys.exit(0 if success else 1)
