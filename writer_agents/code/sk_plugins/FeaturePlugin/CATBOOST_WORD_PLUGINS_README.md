# CatBoost Word Monitor Plugins

## Overview

This directory contains 17 modular SK plugins dedicated to monitoring the top CatBoost success signal words identified from your lawsuit documents. Each plugin tracks word frequency, context, and usage patterns to help calibrate with the wider network of plugins (sentences, paragraphs, arguments, etc.).

## Generated Plugins

### Base Plugin
- **`catboost_word_monitor_plugin.py`** - Base class for all word monitoring plugins

### Word-Specific Plugins (17 total)

| Plugin | Word | Category | Success Rate |
|--------|------|----------|--------------|
| `OrderWordMonitorPlugin` | order | Motion Language | 50% success |
| `HarmWordMonitorPlugin` | harm | Endangerment Implicit | Good |
| `SafetyWordMonitorPlugin` | safety | Endangerment Implicit | Good |
| `ImmediateWordMonitorPlugin` | immediate | Endangerment Implicit | Good |
| `PseudonymWordMonitorPlugin` | pseudonym | Motion Language | 50% success |
| `RiskWordMonitorPlugin` | risk | Endangerment Implicit | Good |
| `SecurityWordMonitorPlugin` | security | National Security | 45.3% success |
| `SeriousWordMonitorPlugin` | serious | Endangerment Implicit | Good |
| `SealedWordMonitorPlugin` | sealed | Motion Language | 50% success |
| `MotionWordMonitorPlugin` | motion | Motion Language | 50% success |
| `CitizenWordMonitorPlugin` | citizen | US Citizen Endangerment | **83.3% success** ⭐ |
| `CompleteWordMonitorPlugin` | complete | Thoroughness | Word count #1 predictor |
| `BodilyWordMonitorPlugin` | bodily | Endangerment Implicit | Good |
| `ThreatWordMonitorPlugin` | threat | Endangerment Implicit | Good |
| `ImpoundWordMonitorPlugin` | impound | Motion Language | 50% success |
| `ProtectiveWordMonitorPlugin` | protective | Motion Language | 50% success |
| `NationalWordMonitorPlugin` | national | National Security | 45.3% success |

## Features

Each plugin provides:

1. **Word Frequency Tracking**
   - Counts occurrences of the monitored word
   - Calculates frequency percentage
   - Tracks word position in document

2. **Context Analysis**
   - Extracts sentence context for each occurrence
   - Captures surrounding text (50 chars before/after)
   - Identifies usage patterns

3. **Validation**
   - Validates word usage against configurable thresholds
   - Provides success/failure status
   - Calculates scores based on usage

4. **Integration Ready**
   - Designed to integrate with sentence/paragraph/argument plugins
   - Supports ChromaDB queries for similar usage patterns
   - Compatible with memory store for learning

## Usage Example

```python
from writer_agents.code.sk_plugins.FeaturePlugin import CitizenWordMonitorPlugin

# Initialize plugin
plugin = CitizenWordMonitorPlugin(
    kernel=kernel,
    chroma_store=chroma_store,
    rules_dir=rules_dir,
    memory_store=memory_store
)

# Validate document
result = await plugin.validate(motion_text)

# Access results
print(f"Word count: {result.value['count']}")
print(f"Frequency: {result.value['frequency_percentage']:.2f}%")
print(f"Success: {result.success}")
```

## Configuration

Each plugin can be configured via rules files in `writer_agents/code/sk_plugins/rules/`:

```json
{
  "feature_name": "catboost_word_citizen",
  "validation_criteria": {
    "min_mentions": 1,
    "max_mentions": 50
  },
  "threshold": 0.7
}
```

## Future Integration

These plugins are designed to integrate with:

- **Sentence plugins** - Track word usage per sentence
- **Paragraph plugins** - Monitor word density per paragraph
- **Argument plugins** - Ensure words appear in key arguments
- **CatBoost analysis** - Feed data back into ML model calibration
- **RefinementLoop** - Coordinate with other feature plugins

## Generator Script

To regenerate or add new word plugins, use:

```bash
python3 writer_agents/code/sk_plugins/FeaturePlugin/generate_catboost_word_plugins.py
```

This will generate all plugin files and provide import statements for `__init__.py`.

## Notes

- All plugins follow the `BaseFeaturePlugin` pattern
- Word matching is case-insensitive and uses whole-word boundaries
- Context extraction is limited to 10 occurrences for performance
- Plugins are registered in `FeaturePlugin/__init__.py`

