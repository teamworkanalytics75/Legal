# Agent 2 – Stricter Filtering & Fallback Handling

**Source:** [AGENT_PLAN_2025-11-15.md](../AGENT_PLAN_2025-11-15.md#agent-2-stricter-filtering--fallback-handling)

## Goal

Fix fallback behavior to fail explicitly when no personal facts are found (no silent fallback to generic evidence).

## Context

- Current issue: Lines 728-732 in `CaseFactsProvider.filter_evidence_for_lawsuit()` fall back to all evidence if nothing matches
- Need: Fail fast with clear error messages when personal corpus missing
- Personal corpus must be validated before filtering

## Deliverables

1. **Modify** `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py`
   - Change fallback behavior (lines 728-732) to raise exception or return empty with warning
   - Add `strict_filtering: bool = True` configuration flag
   - When strict and no personal corpus facts found, log error and return empty list
   - Add validation that personal corpus is loaded before filtering

2. **Update** `writer_agents/code/WorkflowOrchestrator.py`
   - Update `_filter_lawsuit_evidence()` to check if filtered evidence is empty
   - Log warning/error if filtering removed everything but personal corpus exists
   - Add early validation in workflow initialization
   - Update `_initialize_case_facts_provider()` to verify:
     - Personal corpus directory exists and has files
     - `case_insights.json` exists or can be generated
     - Fail fast with clear error message if personal facts unavailable

3. **Update** `writer_agents/code/WorkflowStrategyExecutor.py`
   - Same updates as WorkflowOrchestrator for `_filter_lawsuit_evidence()`

## Key Files

- `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py`
- `writer_agents/code/WorkflowOrchestrator.py`
- `writer_agents/code/WorkflowStrategyExecutor.py`

## Success Criteria

- ✅ System fails explicitly when personal corpus missing
- ✅ Clear error messages guide user to fix the issue
- ✅ No silent fallback to generic evidence
- ✅ Validation runs before filtering attempts

## Dependencies

- Agent 1 (optional, but helpful for validation)

## Validation

```bash
# Test with missing corpus (should fail fast)
rm -rf case_law_data/tmp_corpus
python writer_agents/scripts/generate_optimized_motion.py --dry-run
# Should show clear error about missing personal corpus

# Test with valid corpus (should work)
python writer_agents/scripts/validate_case_facts_provider.py
# Should show personal corpus facts loaded
```

## Fail-Fast Expectations

- If `case_law_data/tmp_corpus/` doesn't exist → Error: "Personal corpus directory not found"
- If `case_insights.json` missing → Error: "Case insights not found. Run build_case_insights.py"
- If filtering returns empty but corpus exists → Warning: "No lawsuit-specific evidence found despite personal corpus"
