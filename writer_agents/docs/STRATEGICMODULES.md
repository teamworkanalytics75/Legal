# Strategic Analysis Modules Documentation

## Overview

The strategic analysis modules provide decision support tools for legal settlement negotiations. These modules integrate with your existing Bayesian network system to provide:

1. **Settlement Optimization** - Calculate optimal settlement ranges using Monte Carlo simulation
2. **Game Theory Analysis** - BATNA, Nash equilibrium, and negotiation strategy
3. **Scenario War Gaming** - Compare outcomes across different evidence scenarios
4. **Reputation Risk Assessment** - Quantify reputational damage for different outcomes

---

## Architecture

```

                   Existing BN System 
  bn_adapter.py -> CaseInsights (posteriors + evidence) 

                     
         
                                
                                
    
  Settlement Game Theory 
  Optimizer Module 
    
         
         

    Strategic Integration Layer 
  - Combines all module outputs 
  - Generates unified reports 

         
          Scenario War Gaming (batch mode)
          Reputation Risk Analysis
```

---

## Module 1: Settlement Optimization

### Overview

Uses Monte Carlo simulation to model trial outcomes and calculate optimal settlement ranges based on Bayesian network posteriors.

### Key Classes

#### `SettlementConfig`

Configuration for settlement optimization:

```python
from writer_agents.settlement_optimizer import SettlementConfig

config = SettlementConfig(
    expected_legal_costs=500_000.0, # Total legal costs if go to trial
    risk_aversion_coefficient=0.3, # 0=risk-neutral, 1=very risk-averse
    monte_carlo_iterations=10_000, # Number of simulations
    confidence_interval=0.90, # 90% confidence interval

    # BN node mappings
    success_node_id="LegalSuccess_US",
    damages_node_id="Financial_Damages",
)
```

#### `SettlementOptimizer`

Main class for optimization:

```python
from writer_agents.settlement_optimizer import SettlementOptimizer

optimizer = SettlementOptimizer()
recommendation = optimizer.optimize_settlement(insights, config)

print(f"Optimal: ${recommendation.optimal_settlement:,.0f}")
print(f"Range: ${recommendation.settlement_range[0]:,.0f} - ${recommendation.settlement_range[1]:,.0f}")
```

### Algorithm

The settlement optimizer uses a sophisticated approach:

1. **Monte Carlo Simulation**: Sample from posterior distributions 10,000 times
2. **Expected Value Calculation**: Compute mean, median, standard deviation
3. **Risk Adjustment**: Apply certainty equivalent formula: `CE = - (/2)2`
4. **Range Determination**: Use confidence intervals with strategic buffers

### Mathematical Formulas

**Certainty Equivalent:**
```
CE = - (/2) 2
```
Where:
- = expected value (mean)
- = risk aversion coefficient
- 2 = variance

**Settlement Range:**
```
Lower Bound = CI_lower 0.8
Upper Bound = CI_upper 1.2
```

### Output

`SettlementRecommendation` contains:
- `optimal_settlement`: Risk-adjusted optimal value
- `settlement_range`: (min, max) acceptable range
- `ev_analysis`: Expected value statistics
- `strategy_recommendation`: Text recommendation
- `monte_carlo_outcomes`: Full simulation results

---

## Module 2: Game Theory

### Overview

Applies game theory concepts to settlement negotiations, including BATNA (Best Alternative To Negotiated Agreement) and Nash equilibrium.

### Key Classes

#### `BATNAAnalyzer`

Calculates BATNA for both parties:

```python
from writer_agents.game_theory import BATNAAnalyzer

analyzer = BATNAAnalyzer()
batna = analyzer.analyze_batna(insights, settlement_rec)

if batna.zopa_exists:
    print(f"ZOPA: ${batna.zopa_range[0]:,.0f} - ${batna.zopa_range[1]:,.0f}")
```

#### `NashEquilibriumCalculator`

Calculates Nash bargaining solution:

```python
from writer_agents.game_theory import NashEquilibriumCalculator

calculator = NashEquilibriumCalculator()
nash = calculator.calculate_nash_settlement(batna, bargaining_power=0.5)

print(f"Nash Equilibrium: ${nash:,.0f}")
```

#### `StrategicRecommender`

Generates negotiation strategy:

```python
from writer_agents.game_theory import StrategicRecommender

recommender = StrategicRecommender()
strategy = recommender.recommend_strategy(batna, nash, settlement_rec, insights)

print(f"First Offer: ${strategy.first_offer:,.0f}")
print(f"Walk-Away: ${strategy.walkaway_point:,.0f}")
print(strategy.strategy_narrative)
```

### Mathematical Formulas

**Nash Bargaining Solution:**
```
Nash = arg max [(your_payoff - your_BATNA)^ (their_payoff - their_BATNA)^(1-)]
```

For equal power (=0.5):
```
Nash = your_BATNA + (their_BATNA - your_BATNA)
```

**ZOPA (Zone of Possible Agreement):**
```
ZOPA exists if: your_BATNA < |their_BATNA|
ZOPA range: [your_BATNA, |their_BATNA|]
```

### Concepts

**BATNA**: Your best alternative if negotiation fails (typically trial expected value)

**ZOPA**: Range where both parties prefer settlement to their BATNA

**Nash Equilibrium**: Fair split of surplus given relative bargaining power

**Anchoring**: Starting with high first offer to set reference point

---

## Module 3: Scenario War Gaming

### Overview

Batch analysis across multiple evidence scenarios to understand sensitivity and identify optimal strategies.

### Key Classes

#### `ScenarioDefinition`

Define a scenario to analyze:

```python
from writer_agents.scenario_war_gaming import ScenarioDefinition

scenario = ScenarioDefinition(
    scenario_id="S1",
    name="Strong Case",
    description="All evidence admitted",
    evidence={
        "OGC_Email_Apr18_2025": "Sent",
        "PRC_Awareness": "Direct",
    },
    assumptions=["Email proven", "Direct awareness"]
)
```

#### `ScenarioBatchRunner`

Run scenarios in batch:

```python
from writer_agents.scenario_war_gaming import ScenarioBatchRunner

runner = ScenarioBatchRunner(model_path)
results = await runner.run_scenarios(scenarios, summary)
```

#### `ScenarioComparator`

Compare results:

```python
from writer_agents.scenario_war_gaming import ScenarioComparator

comparator = ScenarioComparator()
comparison = comparator.compare(results)

print(comparison.data_frame) # Pandas DataFrame
print(f"Best: {comparison.best_scenario['Scenario']}")
```

### Usage Pattern

```python
# Define scenarios
scenarios = [
    ScenarioDefinition(...), # Strong case
    ScenarioDefinition(...), # Moderate case
    ScenarioDefinition(...), # Weak case
]

# Run batch analysis
runner = ScenarioBatchRunner(model_path)
results = await runner.run_scenarios(scenarios, summary)

# Compare
comparator = ScenarioComparator()
comparison = comparator.compare(results)

# Visualize
visualizer = ScenarioVisualizer()
visualizer.plot_settlement_comparison(comparison, output_path="chart.png")
```

---

## Module 4: Reputation Risk

### Overview

Quantifies reputational damage to institutions (like Harvard) across different case outcomes.

### Key Classes

#### `ReputationFactorAnalyzer`

Analyzes reputation factors:

```python
from writer_agents.reputation_risk import ReputationFactorAnalyzer

analyzer = ReputationFactorAnalyzer()
impact = analyzer.analyze_reputation_impact(insights, "trial_lose")

print(f"Overall Impact: {impact.overall_score:.1f}")
for factor, score in impact.factor_impacts.items():
    print(f" {factor}: {score:.1f}")
```

#### `MediaImpactModeler`

Models media coverage:

```python
from writer_agents.reputation_risk import MediaImpactModeler

modeler = MediaImpactModeler()
media = modeler.model_media_coverage(
    case_severity="high",
    involves_china=True,
    involves_student_safety=True
)

print(f"Interest: {media.interest_score}/100")
print(f"Expected Articles: {media.expected_article_count}")
```

#### `ReputationRiskScorer`

Complete reputation analysis:

```python
from writer_agents.reputation_risk import ReputationRiskScorer

scorer = ReputationRiskScorer()
impacts = scorer.score_reputation_risk(insights)

for outcome, impact in impacts.items():
    print(f"{outcome}: {impact.overall_score:.1f}")
```

### Reputation Factors

**Harvard-Specific Weights:**
- Academic Prestige: 25%
- Federal Funding: 20%
- Donor Relations: 20%
- Student Enrollment: 15%
- Media Perception: 10%
- Alumni Trust: 10%

### Impact Scores

**Interpretation:**
- 0 to -2: Minimal impact
- -2 to -5: Minor impact
- -5 to -10: Moderate impact
- -10 to -15: Significant impact
- < -15: Severe impact

---

## Module 5: Strategic Integration

### Overview

Unified interface that orchestrates all modules into a complete analysis.

### Key Classes

#### `StrategicAnalysisEngine`

Main engine for complete analysis:

```python
from writer_agents.strategic_integration import StrategicAnalysisEngine

engine = StrategicAnalysisEngine()
report = await engine.run_complete_analysis(
    model_path=Path("model.xdsl"),
    evidence={"Node1": "State1"},
    summary="Case summary",
    scenarios=[...], # Optional
    institution="Harvard"
)

# Print markdown report
print(report.to_markdown_report())
```

### Quick Helper Functions

For simpler analyses:

```python
from writer_agents.strategic_integration import (
    quick_settlement_analysis,
    quick_game_theory_analysis
)

# Just settlement
settlement = quick_settlement_analysis(model_path, evidence, summary)

# Settlement + game theory
game_theory = quick_game_theory_analysis(model_path, evidence, summary)
```

### Complete Workflow

```python
import asyncio
from pathlib import Path
from writer_agents.strategic_integration import StrategicAnalysisEngine
from writer_agents.scenario_war_gaming import ScenarioDefinition

async def analyze_case():
    # Define scenarios
    scenarios = [...]

    # Run complete analysis
    engine = StrategicAnalysisEngine()
    report = await engine.run_complete_analysis(
        model_path=Path("model.xdsl"),
        evidence={"OGC_Email": "Sent"},
        summary="Case summary",
        scenarios=scenarios,
    )

    # Access results
    print(f"Settlement: ${report.settlement.optimal_settlement:,.0f}")
    print(f"Nash: ${report.game_theory.nash_equilibrium:,.0f}")

    if report.scenarios:
        print(report.scenarios.to_report())

    for outcome, impact in report.reputation.items():
        print(f"{outcome}: {impact.overall_score:.1f}")

    # Save report
    with open("report.md", "w") as f:
        f.write(report.to_markdown_report())

asyncio.run(analyze_case())
```

---

## Integration with Existing System

### BN Model Requirements

Your BN model must have:
1. **Success Node**: Outcome likelihood (e.g., "LegalSuccess_US")
   - States: "High", "Moderate", "Low"
2. **Damages Node**: Financial impact (e.g., "Financial_Damages")
   - States: "Material", "Moderate", "Minor"

Configure node IDs in `SettlementConfig`:

```python
config = SettlementConfig(
    success_node_id="YourSuccessNode",
    damages_node_id="YourDamagesNode",
    success_outcomes={"Win": 1.0, "Partial": 0.5, "Lose": 0.0},
    damages_outcomes={"High": 5_000_000, "Med": 2_000_000, "Low": 500_000},
)
```

### Using with Existing `CaseInsights`

All modules accept `CaseInsights` from your existing BN system:

```python
from writer_agents.bn_adapter import run_bn_inference
from writer_agents.settlement_optimizer import SettlementOptimizer

# Your existing BN inference
insights, _ = run_bn_inference(model_path, evidence, summary)

# Strategic analysis
optimizer = SettlementOptimizer()
recommendation = optimizer.optimize_settlement(insights)
```

---

## Dependencies

Required packages:

```bash
pip install numpy scipy
```

Optional (for enhanced features):

```bash
pip install pandas matplotlib nashpy
```

---

## Examples

See `examples/harvard_strategic_analysis.py` for complete working example.

### Quick Settlement Analysis

```python
from writer_agents.strategic_integration import quick_settlement_analysis
from pathlib import Path

settlement = quick_settlement_analysis(
    model_path=Path("model.xdsl"),
    evidence={"Node1": "State1"},
    summary="Brief case summary"
)

print(settlement.to_report())
```

### Scenario Comparison

```python
from writer_agents.scenario_war_gaming import ScenarioBatchRunner, ScenarioComparator

scenarios = [...] # Define your scenarios

runner = ScenarioBatchRunner(model_path)
results = await runner.run_scenarios(scenarios, summary)

comparator = ScenarioComparator()
comparison = comparator.compare(results)

print(comparison.to_report())
```

---

## Performance Notes

- **Monte Carlo**: 10,000 iterations takes ~1-2 seconds
- **Scenario Gaming**: N scenarios 2-3 seconds per scenario
- **Complete Analysis**: 5-15 seconds depending on scenario count

---

## Customization

### Custom Risk Aversion

```python
config = SettlementConfig(risk_aversion_coefficient=0.5) # More risk-averse
```

### Custom Bargaining Power

```python
nash = calculator.calculate_nash_settlement(batna, bargaining_power=0.6) # You have more power
```

### Custom Reputation Factors

```python
from writer_agents.reputation_risk import ReputationConfig

config = ReputationConfig(
    institution_name="MyInstitution",
    factors={
        "factor1": {"weight": 0.3, "baseline_score": 85},
        "factor2": {"weight": 0.7, "baseline_score": 90},
    }
)
```

---

## Troubleshooting

### Missing BN Nodes

**Error**: `Node 'LegalSuccess_US' not found in posteriors`

**Solution**: Configure correct node IDs:
```python
config = SettlementConfig(
    success_node_id="YourActualNodeName",
    damages_node_id="YourDamagesNodeName"
)
```

### Pandas Not Available

Scenario comparison works without pandas but with reduced features. Install for full functionality:
```bash
pip install pandas
```

### Negative Optimal Settlement

This can occur if case is very weak. Interpretation:
- Negative value = case favors defendant
- Consider rejecting settlement or demanding minimal payment

---

## API Reference

### Settlement Optimizer

**Classes:**
- `SettlementConfig`: Configuration
- `SettlementOptimizer`: Main optimizer
- `MonteCarloSimulator`: Simulation engine
- `ExpectedValueCalculator`: Statistics calculator

**Key Methods:**
- `optimize_settlement(insights, config=None)`: Main entry point
- `run_simulation(insights)`: Monte Carlo simulation
- `calculate_ev_trial(outcomes)`: EV calculation

### Game Theory

**Classes:**
- `BATNAAnalyzer`: BATNA calculation
- `NashEquilibriumCalculator`: Nash equilibrium
- `StrategicRecommender`: Strategy generation

**Key Methods:**
- `analyze_batna(insights, settlement_rec)`: Calculate BATNAs
- `calculate_nash_settlement(batna, power=0.5)`: Nash equilibrium
- `recommend_strategy(...)`: Generate strategy

### Reputation Risk

**Classes:**
- `ReputationFactorAnalyzer`: Factor analysis
- `MediaImpactModeler`: Media modeling
- `ReputationRiskScorer`: Overall scoring

**Key Methods:**
- `analyze_reputation_impact(insights, outcome)`: Single outcome
- `score_reputation_risk(insights)`: All outcomes
- `model_media_coverage(severity, ...)`: Media impact

---

## Version History

**v1.0.0** (October 2025)
- Initial release
- Settlement optimization with Monte Carlo
- Game theory (BATNA, Nash equilibrium)
- Scenario war gaming
- Reputation risk analysis
- Complete integration layer

---

## Support

For issues or questions:
1. Check this documentation
2. Review examples in `examples/harvard_strategic_analysis.py`
3. Run tests: `pytest tests/test_strategic_modules.py`
4. Check integration with your BN model

---

## License

Same license as parent project.

