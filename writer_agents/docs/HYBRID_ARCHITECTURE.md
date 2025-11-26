# Hybrid SK-AutoGen Writing System

## 🎯 Overview

The Hybrid SK-AutoGen Writing System combines **Microsoft Semantic Kernel (SK)** and **AutoGen** to create a powerful legal document drafting pipeline. This system leverages the strengths of both frameworks:

- **AutoGen**: Creative exploration, brainstorming, and iterative refinement
- **Semantic Kernel**: Deterministic drafting, structured validation, and quality gates

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Hybrid Orchestration Controller                │
│  (writer_agents/code/HybridOrchestrator.py)             │
└───────────────┬─────────────────────┬────────────────────┘
                │                     │
    ┌───────────▼──────────┐  ┌──────▼────────────────────┐
    │   AutoGen Agents     │  │  Semantic Kernel Engine   │
    │  (Exploration)       │  │  (Production Draft)       │
    │                      │  │                           │
    │ • Brainstorming      │  │ • Native Functions        │
    │ • Argument Discovery │  │ • Semantic Functions      │
    │ • Alternative Ideas  │  │ • Validators (Plugins)    │
    │ • KB Querying        │  │ • SK Planner Pipeline     │
    └──────────────────────┘  └───────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install Semantic Kernel
pip install semantic-kernel

# Ensure you have OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
# OR create .openai_api_key.txt file
```

### 2. Basic Usage

```python
import asyncio
from writer_agents.code.EnhancedOrchestrator import EnhancedWriterOrchestrator, EnhancedOrchestratorConfig
from writer_agents.code.insights import CaseInsights

async def main():
    # Create case insights
    insights = CaseInsights(
        summary="Privacy harm case involving data exposure",
        evidence={"OGC_Email_Apr18_2025": "Sent", "PRC_Awareness": "Direct"},
        posteriors={"PrivacyHarm": 0.85, "Causation": 0.78},
        jurisdiction="US",
        case_style="Motion for Sealing"  # Triggers hybrid_sk workflow
    )

    # Create orchestrator with hybrid SK enabled
    config = EnhancedOrchestratorConfig()
    config.enable_sk_hybrid = True

    orchestrator = EnhancedWriterOrchestrator(config)

    try:
        # Run hybrid workflow
        result = await orchestrator.run_intelligent_workflow(insights)

        print(f"Generated document: {result.edited_document}")
        print(f"Workflow type: {result.metadata['workflow_type']}")

    finally:
        await orchestrator.close()

asyncio.run(main())
```

### 3. Run Demo

```bash
cd writer_agents
python demo_hybrid_system.py
```

## 🔧 Components

### Core Components

1. **HybridOrchestrator** (`writer_agents/code/HybridOrchestrator.py`)
   - Main orchestration controller
   - Manages workflow phases between AutoGen and SK
   - Handles iteration and quality gates

2. **SK Configuration** (`writer_agents/code/sk_config.py`)
   - Semantic Kernel setup and initialization
   - Model configuration and memory integration
   - Convenience functions for kernel creation

3. **Plugin System** (`writer_agents/code/sk_plugins/`)
   - Base classes for SK plugins
   - Privacy Harm drafting plugin (prototype)
   - Extensible architecture for new plugins

4. **Quality Gates** (`writer_agents/code/quality_gates.py`)
   - Automated validation pipeline
   - Citation, structure, and tone validation
   - Configurable thresholds and requirements

5. **AutoGen-SK Bridge** (`writer_agents/code/autogen_sk_bridge.py`)
   - Integration between AutoGen and SK
   - SK functions as AutoGen tools
   - Context translation and data flow

### Workflow Phases

1. **EXPLORE** (AutoGen): Brainstorm arguments, query knowledge base
2. **PLAN** (SK Planner): Generate function execution plan
3. **DRAFT** (SK Functions): Execute native/semantic functions
4. **VALIDATE** (SK Validation): Run quality gates
5. **REVIEW** (AutoGen): Critique if validation fails
6. **REFINE** (SK): Re-run functions with revision context
7. **COMMIT**: Save to outputs, log to system memory

## 📋 Available Plugins

### Privacy Harm Plugin

**Location**: `writer_agents/code/sk_plugins/DraftingPlugin/privacy_harm_function.py`

**Functions**:
- `PrivacyHarmNative`: Deterministic template-based drafting
- `PrivacyHarmSemantic`: LLM-based drafting with structured prompts

**Usage**:
```python
from writer_agents.code.sk_plugins.DraftingPlugin.privacy_harm_function import PrivacyHarmPlugin

plugin = PrivacyHarmPlugin(kernel)
await plugin.initialize()

# Use native function
result = await plugin.native_function.execute(
    evidence={"OGC_Email": "Sent"},
    posteriors={"PrivacyHarm": 0.85},
    case_summary="Privacy violation case",
    jurisdiction="US"
)
```

## 🛡️ Quality Gates

The system includes comprehensive quality gates:

### Required Gates
- **Citation Validity**: Ensures proper `[Node:State]` format
- **Structure Complete**: Verifies all required sections present
- **Evidence Grounding**: All claims have supporting evidence
- **Argument Coherence**: Logical flow and transitions

### Optional Gates
- **Tone Consistency**: Professional legal language
- **Grammar/Spelling**: Basic language quality
- **Legal Accuracy**: Jurisdiction-specific validation

### Configuration
```python
from writer_agents.code.quality_gates import QUALITY_GATES, QualityGateRunner

# Run quality gates
runner = QualityGateRunner(sk_kernel)
results = await runner.run_all_gates(document, context)

print(f"Overall score: {results.overall_score}")
print(f"Can proceed: {results.can_proceed}")
```

## 🔄 Workflow Selection

The Enhanced Orchestrator automatically selects the best workflow:

### Hybrid SK Workflow
**Triggers**: Cases with `case_style` containing "motion", "seal", or "pseudonym"
**Features**:
- AutoGen exploration + SK structured drafting
- Quality gate validation
- Iterative refinement
- Production-ready output

### Traditional Workflow
**Triggers**: Simple cases, low complexity
**Features**:
- Standard AutoGen pipeline
- Basic validation
- Faster execution

### Advanced Workflow
**Triggers**: High complexity cases
**Features**:
- Multi-order planning
- Nested review processes
- Research agents
- Quality assurance gates

## 📊 Performance Monitoring

The system includes built-in performance monitoring:

```python
config = EnhancedOrchestratorConfig()
config.enable_performance_monitoring = True

# Results include performance metrics
result = await orchestrator.run_intelligent_workflow(insights)
metrics = result.metadata['performance_metrics']

print(f"Execution time: {metrics['execution_time']}s")
print(f"Agent interactions: {metrics['agent_interactions']}")
print(f"Quality score: {metrics['quality_score']}")
```

## 🔌 Integration Points

### With Existing Systems

1. **Bayesian Network Integration**
   ```python
   # BN posteriors automatically passed to SK context
   insights = CaseInsights(
       posteriors={"PrivacyHarm": 0.85, "Causation": 0.78}
   )
   ```

2. **Chroma Collections**
   ```python
   # SK Memory can connect to existing Chroma collections
   kernel = create_sk_kernel_with_chroma("path/to/chroma/collections")
   ```

3. **Research Modules**
   - ML-enriched evidence flows to both AutoGen and SK
   - Vector database queries available to SK functions
   - Historical drafts accessible for consistency

### Memory Integration

- AutoGen sessions logged to `system_memory/`
- SK functions can read historical drafts
- Workflow state persisted for debugging

## 🧪 Testing

### Run Tests
```bash
cd writer_agents
python test_hybrid_system.py
```

### Test Coverage
- SK Kernel creation and initialization
- Plugin registration and execution
- Hybrid workflow end-to-end
- Enhanced Orchestrator integration
- Quality gate validation

## 📈 Benchmarks

The system supports performance comparison:

```python
# Compare workflows on same case
traditional_result = await run_traditional_workflow(insights)
hybrid_result = await run_hybrid_workflow(insights)

# Compare metrics
print(f"Traditional: {traditional_result.metadata['execution_time']}s")
print(f"Hybrid: {hybrid_result.metadata['execution_time']}s")
print(f"Quality improvement: {hybrid_result.metadata['quality_score'] - traditional_result.metadata['quality_score']}")
```

## 🔧 Configuration

### Hybrid Orchestrator Config
```python
from writer_agents.code.HybridOrchestrator import HybridOrchestratorConfig

config = HybridOrchestratorConfig(
    max_iterations=3,
    enable_sk_planner=True,
    enable_quality_gates=True,
    auto_commit_threshold=0.85,
    exploration_rounds=2
)
```

### SK Configuration
```python
from writer_agents.code.sk_config import SKConfig

sk_config = SKConfig(
    model_name="gpt-4o",
    temperature=0.3,  # Lower for consistent legal writing
    max_tokens=4000,
    enable_memory=True
)
```

## 🚧 Extending the System

### Adding New Plugins

1. Create plugin class inheriting from `BaseSKPlugin`
2. Implement native and/or semantic functions
3. Register with plugin registry
4. Add to Hybrid Orchestrator

### Adding New Quality Gates

1. Define gate in `quality_gates.py`
2. Implement validation logic
3. Add to `QUALITY_GATES` list
4. Configure thresholds and requirements

### Custom Workflow Phases

1. Add new phase to `WorkflowPhase` enum
2. Implement execution logic in `HybridOrchestrator`
3. Update workflow router logic
4. Add phase transitions

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Not Found**
   ```bash
   export OPENAI_API_KEY="your-key"
   # OR create .openai_api_key.txt
   ```

2. **SK Plugin Registration Fails**
   - Check plugin initialization
   - Verify kernel is properly configured
   - Check function signatures

3. **Quality Gates Fail**
   - Review gate thresholds
   - Check document format
   - Verify evidence grounding

4. **AutoGen Integration Issues**
   - Check tool registration
   - Verify function signatures
   - Review context translation

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
result = await orchestrator.run_intelligent_workflow(insights)
```

## 📚 API Reference

### Core Classes

- `HybridOrchestrator`: Main orchestration controller
- `HybridOrchestratorConfig`: Configuration for hybrid workflow
- `WorkflowRouter`: Routes phases between AutoGen and SK
- `QualityGatePipeline`: Runs validation gates
- `AutoGenToSKBridge`: Translates between frameworks

### Plugin Classes

- `BaseSKPlugin`: Abstract base for SK plugins
- `PrivacyHarmPlugin`: Privacy harm drafting plugin
- `PluginRegistry`: Manages plugin registration

### Quality Gate Classes

- `QualityGate`: Configuration for validation gates
- `QualityGateRunner`: Executes validation pipeline
- `ValidationResult`: Results from gate validation

## 🤝 Contributing

1. Follow existing code patterns
2. Add tests for new functionality
3. Update documentation
4. Ensure backward compatibility

## 📄 License

Same license as parent project.

## 🆘 Support

For issues or questions:
1. Check this documentation
2. Review test examples
3. Check integration with existing systems
4. Verify OpenAI API key and dependencies
