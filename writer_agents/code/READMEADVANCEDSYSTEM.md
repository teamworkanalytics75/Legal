# Advanced Multi-Agent Writing System

This document describes the sophisticated multi-agent writing system with nested review processes, multi-order planning, and intelligent workflow selection.

## Overview

The advanced writing system builds upon the existing `writer_agents` framework to provide:

- **Multi-Order Planning**: Strategic, tactical, and operational planning levels
- **Nested Review Processes**: Multiple layers of review with different specializations
- **Research Integration**: Dedicated research agents for fact-checking and evidence gathering
- **Quality Assurance**: Multi-stage validation with different criteria
- **Adaptive Workflow**: Dynamic agent selection based on content complexity
- **BN Integration**: Seamless integration with Bayesian network analysis

## Architecture

### Core Components

1. **Advanced Agents** (`advanced_agents.py`)
   - Strategic, tactical, and operational planners
   - Research agents for fact-checking
   - Multi-level content reviewers
   - Technical and style reviewers
   - Quality assurance agents
   - Adaptive orchestrator

2. **Enhanced Orchestrator** (`enhanced_orchestrator.py`)
   - Intelligent workflow selection
   - Complexity analysis
   - Performance monitoring
   - Hybrid workflow support

3. **BN Integration** (`bn_integration.py`)
   - Evidence validation
   - Posterior analysis
   - Workflow recommendations
   - Integration metrics

## Usage

### Basic Advanced Workflow

```python
from writer_agents.advanced_agents import AdvancedAgentConfig, AdvancedWriterOrchestrator
from writer_agents.insights import CaseInsights, EvidenceItem, Posterior

# Configure the advanced system
config = AdvancedAgentConfig(
    max_review_rounds=3,
    enable_research_agents=True,
    enable_quality_gates=True,
    enable_adaptive_workflow=True,
)

# Initialize orchestrator
orchestrator = AdvancedWriterOrchestrator(config)

# Create case insights
insights = CaseInsights(
    reference_id="CASE-001",
    summary="Complex legal case requiring comprehensive analysis",
    posteriors=[
        Posterior(
            node_id="LegalSuccess_US",
            probabilities={"High": 0.3, "Moderate": 0.5, "Low": 0.2}
        )
    ],
    evidence=[
        EvidenceItem(node_id="OGC_Email_Apr18_2025", state="Sent"),
        EvidenceItem(node_id="PRC_Awareness", state="Direct")
    ],
    jurisdiction="US",
    case_style="Memorandum"
)

# Run the advanced workflow
result = await orchestrator.run_advanced_workflow(insights)

# Access results
print(f"Document: {result.edited_document}")
print(f"Sections: {len(result.sections)}")
print(f"Metadata: {result.metadata}")

# Clean up
await orchestrator.close()
```

### Enhanced Orchestrator with Intelligent Selection

```python
from writer_agents.enhanced_orchestrator import EnhancedOrchestratorConfig, EnhancedWriterOrchestrator

# Configure enhanced orchestrator
config = EnhancedOrchestratorConfig(
    use_advanced_workflow=True,
    complexity_threshold=0.7,
    enable_hybrid_mode=True,
    enable_performance_monitoring=True,
)

# Initialize orchestrator
orchestrator = EnhancedWriterOrchestrator(config)

# Get workflow recommendations
recommendations = orchestrator.get_workflow_recommendations(insights)
print(f"Recommended workflow: {recommendations['recommended_workflow']}")
print(f"Complexity score: {recommendations['complexity_score']}")

# Run intelligent workflow
result = await orchestrator.run_intelligent_workflow(insights)

# Access performance metrics
metrics = result.metadata.get("performance_metrics", {})
print(f"Execution time: {metrics.get('execution_time', 0):.2f}s")

await orchestrator.close()
```

### BN Integration

```python
from writer_agents.bn_integration import BNIntegrationConfig, BNWritingIntegrator

# Configure BN integration
config = BNIntegrationConfig(
    model_path=Path("path/to/bn_model.xdsl"),
    enable_pysmile=True,
    fallback_to_mock=True,
    use_advanced_workflow=True,
    enable_hybrid_mode=True,
)

# Initialize integrator
integrator = BNWritingIntegrator(config)

# Run BN-integrated workflow
result = await integrator.run_bn_writing_workflow(
    evidence={"OGC_Email_Apr18_2025": "Sent", "PRC_Awareness": "Direct"},
    summary="Complex legal case requiring comprehensive analysis",
    reference_id="BN-CASE-001",
    jurisdiction="US",
    case_style="Memorandum"
)

# Access results
print(f"Case insights: {result.case_insights}")
print(f"Writing result: {result.writing_result}")
print(f"Integration metrics: {result.integration_metrics}")

await integrator.close()
```

## Workflow Types

### Traditional Workflow
- **Use Case**: Simple cases with straightforward requirements
- **Features**: Standard planning, basic review, citation validation
- **Agents**: 5 agents (Planner, Writer, DoubleChecker, Editor, Stylist)
- **Time**: Baseline (2-5 minutes)
- **Quality**: Standard legal writing quality

### Hybrid Workflow
- **Use Case**: Moderate complexity cases requiring enhanced review
- **Features**: Traditional planning with advanced review processes
- **Agents**: 8 agents (includes multi-level reviewers)
- **Time**: 1.5-2x traditional (5-10 minutes)
- **Quality**: High-quality legal writing

### Advanced Workflow
- **Use Case**: Complex cases requiring comprehensive analysis
- **Features**: Multi-order planning, nested reviews, research agents, quality gates
- **Agents**: 12+ agents (full advanced system)
- **Time**: 2-4x traditional (10-20 minutes)
- **Quality**: Expert-level legal analysis

## Configuration Options

### AdvancedAgentConfig

```python
@dataclass
class AdvancedAgentConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    max_review_rounds: int = 3
    enable_research_agents: bool = True
    enable_quality_gates: bool = True
    enable_adaptive_workflow: bool = True
    review_levels: List[ReviewLevel] = [BASIC, INTERMEDIATE, ADVANCED]
    planning_orders: List[PlanningOrder] = [STRATEGIC, TACTICAL, OPERATIONAL]
```

### EnhancedOrchestratorConfig

```python
@dataclass
class EnhancedOrchestratorConfig:
    traditional_config: WriterOrchestratorConfig = field(default_factory=WriterOrchestratorConfig)
    advanced_config: AdvancedAgentConfig = field(default_factory=AdvancedAgentConfig)
    use_advanced_workflow: bool = True
    complexity_threshold: float = 0.7
    enable_hybrid_mode: bool = True
    max_parallel_agents: int = 3
    enable_performance_monitoring: bool = True
```

### BNIntegrationConfig

```python
@dataclass
class BNIntegrationConfig:
    model_path: Optional[Path] = None
    enable_pysmile: bool = True
    fallback_to_mock: bool = True
    use_advanced_workflow: bool = True
    complexity_threshold: float = 0.7
    enable_hybrid_mode: bool = True
    max_parallel_inference: int = 3
    enable_evidence_validation: bool = True
    enable_posterior_analysis: bool = True
```

## Review Levels

### Basic Review
- Grammar and spelling
- Obvious factual errors
- Basic structure validation

### Intermediate Review
- Logical flow and argumentation
- Citation accuracy
- Content coherence

### Advanced Review
- Legal reasoning quality
- Precedent alignment
- Strategic positioning

### Expert Review
- Nuanced legal analysis
- Rhetorical effectiveness
- Expert-level insights

## Planning Orders

### Strategic Planning
- High-level objectives and goals
- Key themes and arguments
- Document structure recommendations
- Quality standards and success criteria

### Tactical Planning
- Section breakdown and organization
- Inter-section dependencies
- Evidence allocation and placement
- Citation strategy

### Operational Planning
- Detailed section specifications
- Paragraph-level organization
- Specific evidence requirements
- Writing guidelines and constraints

## Quality Gates

The system includes multiple quality gates:

1. **Content Accuracy**: Factual accuracy and citation accuracy
2. **Legal Reasoning**: Logical flow and precedent alignment
3. **Style Consistency**: Tone consistency and formatting
4. **Completeness**: Section completeness and requirement coverage

Each gate has configurable thresholds and can be made required or optional.

## Performance Monitoring

The system includes comprehensive performance monitoring:

- Execution time tracking
- Quality score measurement
- Review round counting
- Agent interaction tracking
- Error monitoring

## Error Handling

The system includes robust error handling:

- Evidence validation with detailed error messages
- Graceful fallback to mock data when BN inference fails
- Workflow recovery mechanisms
- Comprehensive logging

## Testing

Run the comprehensive test suite:

```bash
python writer_agents/test_advanced_system.py
```

The test suite includes:
- Basic advanced workflow testing
- Enhanced orchestrator testing
- BN integration testing
- Workflow comparison testing
- Error handling testing
- Performance analysis testing

## Demo Scripts

### Basic Demo
```bash
python writer_agents/demo_advanced_system.py
```

### Integration with Existing System
```python
# In your existing WizardWeb1.1.4_STABLE.py
from writer_agents.bn_integration import BNWritingIntegrator, BNIntegrationConfig

# Configure and use the advanced system
config = BNIntegrationConfig(
    model_path=MODEL_PATH,
    use_advanced_workflow=True,
    enable_hybrid_mode=True,
)

integrator = BNWritingIntegrator(config)

# Use in your existing workflow
result = await integrator.run_bn_writing_workflow(
    evidence=evidence,
    summary=summary,
    reference_id=reference_id
)
```

## Best Practices

1. **Start Simple**: Begin with traditional workflow and upgrade as needed
2. **Monitor Performance**: Use performance monitoring to optimize workflows
3. **Validate Evidence**: Always validate evidence before BN inference
4. **Use Hybrid Mode**: For moderate complexity cases, hybrid mode provides good balance
5. **Configure Quality Gates**: Set appropriate thresholds for your use case
6. **Handle Errors Gracefully**: Always include error handling and fallback mechanisms

## Troubleshooting

### Common Issues

1. **PySMILE Not Available**: The system will fallback to mock data automatically
2. **AutoGen Import Errors**: Ensure `autogen-agentchat` and `autogen-ext` are installed
3. **Memory Issues**: Reduce `max_parallel_agents` for large documents
4. **Slow Performance**: Consider using hybrid mode instead of full advanced workflow

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned enhancements include:

- Machine learning-based workflow optimization
- Custom agent development framework
- Advanced caching mechanisms
- Real-time collaboration features
- Integration with external legal databases

## Contributing

When contributing to the advanced system:

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility with existing workflows
5. Include performance benchmarks for new features
