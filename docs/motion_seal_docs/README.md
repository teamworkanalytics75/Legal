# Motion to Seal Pipeline Documentation for ChatGPT

## Purpose

This folder contains all documentation about the motion to seal pipeline so ChatGPT can understand the system and provide quality feedback on generated drafts.

## Files Included

1. **README.md** (this file) - Overview
2. **MOTION_TO_SEAL_PIPELINE_INTEGRATION_ANALYSIS.md** - Complete pipeline architecture
3. **MASTER_PLAN.md** - Project master plan
4. **ROOT_CAUSE_ANALYSIS_MOTION_ERRORS.md** - Problem analysis
5. **motion_workflow_recovery_complete.md** - Recovery work completed
6. **quality_gates.py** - Quality gate definitions
7. **personal_facts_verifier.py** - Fact validation code
8. **fact_payload_utils.py** - Fact enforcement utilities
9. **generate_optimized_motion.py** - Main pipeline script
10. **12_CRITICAL_SEAL_FACTS.md** - The 12 mandatory facts

## Quick Start for ChatGPT

When reviewing a motion draft:

1. **Check Fact Coverage**: Verify all 12 critical facts are present (see `12_CRITICAL_SEAL_FACTS.md`)
2. **Check Evidence Citations**: Look for `[Node:State]` format (target: 80% coverage)
3. **Check Quality Gates**: See `quality_gates.py` for validation thresholds
4. **Check Structure**: Introduction, Analysis, Conclusion sections required
5. **Check Legal Quality**: Argument coherence, legal accuracy, professional tone

## Key System Components

- **WorkflowOrchestrator.py**: Main conductor (not included, but referenced)
- **quality_gates.py**: Validation thresholds and requirements
- **personal_facts_verifier.py**: Fact pattern matching
- **fact_payload_utils.py**: Fact enforcement logic
- **generate_optimized_motion.py**: Main entry point

## Current Status

- ✅ Fact enforcement: 100% coverage achieved
- 🟡 Evidence grounding: 20% coverage (needs improvement to 80%)
- ✅ Quality gates: Fully functional
- ✅ Hard stop mechanism: Blocks commit on failures

