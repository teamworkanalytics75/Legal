# Enhanced Bayesian Network System

## Quick Start

### Installation

```bash
# Install required packages
uv run pip install pgmpy networkx pandas

# Verify installation
uv run python -c "import pgmpy; import networkx; print('Ready!')"
```

### Build Your First Enhanced BN

```bash
# 1. Build structure from knowledge graph
uv run python -m writer_agents.build_bn_structure_from_kg \
  --entities complete_analysis_fast/entities_all.json \
  --graph analysis_results/cooccurrence_graph_pruned.gpickle \
  --output analysis_results/bn_structure \
  --max-nodes 150

# 2. Construct initial model
uv run python -m writer_agents.bn_constructor \
  --structure analysis_results/bn_structure/bn_structure.pkl \
  --output analysis_results/bn_model_initial.pkl

# 3. Learn parameters from data (if available)
uv run python -m writer_agents.parameter_learning \
  --model analysis_results/bn_model_initial.pkl \
  --data data/training_cases.csv \
  --output analysis_results/bn_model_learned.pkl \
  --method bayesian

# 4. Validate the model
uv run python -m writer_agents.bn_validation \
  --model analysis_results/bn_model_learned.pkl \
  --output analysis_results/validation_report.json
```

### Or Use the Integrated Pipeline

```bash
uv run python -m writer_agents.bn_integration build \
  --entities complete_analysis_fast/entities_all.json \
  --graph analysis_results/cooccurrence_graph_pruned.gpickle \
  --data data/training_cases.csv \
  --output analysis_results/enhanced_bn \
  --max-nodes 150
```

## Architecture

```
Knowledge Graph (entities + co-occurrence)
    v
Node Selection (centrality-based, <=150 nodes)
    v
Structure Building (DAG with <=3 parents)
    v
Initial Model (expert priors)
    v
Parameter Learning (MLE/Bayesian/EM)
    v
Validation (structure, performance, quality)
    v
Integration (with existing bn_adapter.py)
    v
Inference (Variable Elimination)
```

## Modules

### 1. `build_bn_structure_from_kg.py`

**Purpose**: Select and structure nodes from knowledge graph.

**Key Features**:
- Centrality-based node ranking (degree, betweenness, PageRank, eigenvector)
- Automatic state generation based on entity type
- Redundant node merging
- DAG enforcement
- Parent limit enforcement (<=3)

**Input**:
- `entities_all.json`: Entity list with labels and counts
- `cooccurrence_graph_pruned.gpickle`: NetworkX graph

**Output**:
- `selected_nodes.json`: Node list with states and metrics
- `edges.json`: Parent-child relationships
- `bn_structure.pkl`: Complete structure for next step
- `metadata.json`: Summary statistics

**Configuration**:
```python
config = NodeSelectionConfig(
    max_nodes=150,
    max_parents_per_node=3,
    max_states_per_variable=4,
    centrality_percentile=0.25,
    outcome_variables=['LegalSuccess_US', 'FinancialDamage'],
)
```

### 2. `bn_constructor.py`

**Purpose**: Build pgmpy BayesianNetwork from structure.

**Key Features**:
- Uniform or expert prior initialization
- TabularCPD creation
- Variable Elimination setup
- Model validation
- Save/load functionality

**Expert Priors**:
```python
priors = {
    'LegalSuccess_US': {'True': 0.3, 'False': 0.7},
    'FinancialDamage': {'Low': 0.4, 'Medium': 0.3, 'High': 0.2, 'VeryHigh': 0.1},
}
```

### 3. `parameter_learning.py`

**Purpose**: Learn CPTs from data automatically.

**Methods**:

1. **MLE** (Maximum Likelihood):
   ```python
   config = LearningConfig(method="mle")
   ```

2. **Bayesian** (Recommended):
   ```python
   config = LearningConfig(
       method="bayesian",
       prior_type="BDeu",
       equivalent_sample_size=5,
   )
   ```

3. **EM** (Expectation-Maximization):
   ```python
   config = LearningConfig(
       method="em",
       max_em_iterations=100,
       em_tolerance=1e-4,
   )
   ```

**Data Format**:
- CSV with columns matching node IDs
- JSON array of case objects
- SQLite database with evidence tables

### 4. `bn_validation.py`

**Purpose**: Validate model quality and performance.

**Checks**:
- [ok] Node count <= max_nodes
- [ok] Parents <= max_parents
- [ok] DAG structure
- [ok] Inference time < max_time
- [ok] CPD variance > min_variance
- [ok] No extreme probabilities (without justification)

**Output**: JSON validation report with pass/fail and recommendations.

### 5. `bn_integration.py`

**Purpose**: Integrate with existing system.

**Features**:
- Automatic learned model detection
- Fallback chain (learned -> expert -> PySMILE -> pgmpy)
- End-to-end pipeline
- CLI for common operations

**CLI Commands**:
```bash
# Build
bn_integration build --entities ... --graph ... --output ...

# Infer
bn_integration infer --model ... --evidence "X=Y,A=B" --summary "..."

# Validate
bn_integration validate --model ... --output ...
```

## Node Selection Criteria

A node is selected if it meets **any** of:

1. **High centrality**: Top 25% in degree, betweenness, PageRank, or eigenvector
2. **Outcome variable**: Explicitly listed (e.g., `LegalSuccess_US`)
3. **Strong connection**: Edge weight >= threshold to outcome variable
4. **Observable**: Has evidence in database (count > 0)

## Parameter Learning Pipeline

### Step 1: Extract Evidence

```python
from writer_agents.parameter_learning import extract_evidence_from_corpus

data = extract_evidence_from_corpus('corpus.json', model)
# Returns: DataFrame with columns for each BN variable
```

### Step 2: Learn

```python
from writer_agents.parameter_learning import learn_parameters

learned_model = learn_parameters(model, data, config)
```

### Step 3: Evaluate

```python
from writer_agents.parameter_learning import evaluate_learned_parameters

evaluation = evaluate_learned_parameters(original_model, learned_model)
# Returns: Dict with avg_change, max_change, large_changes list
```

### Step 4: Save

```python
from writer_agents.parameter_learning import save_learned_model

save_learned_model(learned_model, output_path, evaluation)
```

## Validation

### Run Validation

```python
from writer_agents.bn_validation import run_full_validation

report = run_full_validation(model, config)

print(f"Passed: {report.passed}")
print(f"Inference time: {report.inference_time}s")
print(f"Recommendations: {report.recommendations}")
```

### Interpret Results

- **PASSED**: Model meets all constraints
- **FAILED**: Review `performance_issues` and `recommendations`

Common issues:
- Too many nodes -> Increase `centrality_percentile` or reduce `max_nodes`
- Slow inference -> Prune low-variance nodes
- Extreme CPDs -> Review data quality or increase `equivalent_sample_size`

## Integration with WizardWeb

### Option 1: Replace Model File

```bash
# Build learned model
uv run python -m writer_agents.bn_integration build \
  --entities complete_analysis_fast/entities_all.json \
  --graph analysis_results/cooccurrence_graph_pruned.gpickle \
  --data data/cases.csv \
  --output experiments/

# Copy to WizardWeb location
cp experiments/bn_model_learned.xdsl experiments/WizardWeb1.1.3.xdsl
```

### Option 2: Modify `bn_adapter.py`

Add learned model support to `run_bn_inference_with_fallback`:

```python
# In bn_adapter.py
def run_bn_inference_with_fallback(...):
    # Try learned model first
    learned_path = model_path.with_stem(model_path.stem + "_learned")
    if learned_path.exists():
        return run_pgmpy_inference(learned_path, evidence, summary, reference_id)
    
    # Fall back to original logic
    ...
```

### Option 3: Use Integrated Function

```python
from writer_agents.bn_integration import run_inference_with_learned_model

insights, posteriors = run_inference_with_learned_model(
    model_path=MODEL_PATH,
    evidence=evidence,
    summary=summary,
    reference_id="case_001",
    data_path=TRAINING_DATA_PATH, # Optional: for on-demand learning
    force_relearn=False,
)
```

## Performance Tuning

### Slow Inference

1. **Reduce nodes**: Increase `centrality_percentile` from 0.25 to 0.15
2. **Prune nodes**: Run validation and remove low-variance nodes
3. **Limit parents**: Reduce `max_parents_per_node` from 3 to 2
4. **Use approximate inference**: Consider Junction Tree with approximation

### Poor Learning Results

1. **More data**: Aim for N >= 100 records
2. **Stronger prior**: Increase `equivalent_sample_size` from 5 to 10
3. **Better data**: Reduce missing values (< 20%)
4. **Expert review**: Manually refine states and structure

### Out of Memory

1. **Reduce nodes**: Set `max_nodes=100` instead of 150
2. **Simplify states**: Reduce `max_states_per_variable` to 3
3. **Split model**: Create sub-models by domain (e.g., legal, financial)

## Testing

### Unit Tests

```bash
uv run pytest tests/test_bn_structure.py
uv run pytest tests/test_parameter_learning.py
uv run pytest tests/test_bn_validation.py
```

### Integration Test

```bash
# Run full pipeline on small dataset
uv run python -m writer_agents.bn_integration build \
  --entities tests/fixtures/entities_small.json \
  --graph tests/fixtures/graph_small.gpickle \
  --output tests/outputs/test_bn \
  --max-nodes 50
```

### Regression Test

Keep a frozen test dataset to ensure consistent results:

```bash
# Save baseline posteriors
uv run python save_baseline_posteriors.py

# Run regression test
uv run pytest tests/test_regression.py
```

## Troubleshooting

### Error: "pgmpy not available"

```bash
uv run pip install pgmpy networkx
```

### Error: "Model validation failed"

Check for:
- Cycles in edge list (use `ensure_acyclic`)
- Mismatched parent counts
- Invalid CPT dimensions

### Error: "Insufficient data"

Reduce `min_data_size` in `LearningConfig` or collect more cases.

### Warning: "EM did not converge"

Increase `max_em_iterations` or check for degenerate cases.

## API Reference

See individual module docstrings:

```python
help(build_bn_structure_from_kg)
help(parameter_learning)
help(bn_validation)
```

## Contributing

When adding features:

1. Follow existing patterns (dataclasses, type hints, logging)
2. Add docstrings to all public functions
3. Create unit tests
4. Update this README
5. Run `make format` and `make lint`

## License

Same as parent project (see LICENSE file).

---

**Questions?** Review `analysis_results/parameter_learning_report.md` for detailed documentation.

