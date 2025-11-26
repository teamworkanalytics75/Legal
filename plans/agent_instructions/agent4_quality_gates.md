# Agent 4 – Post-Generation Validation Integration

**Source:** [AGENT_PLAN_2025-11-15.md](../AGENT_PLAN_2025-11-15.md#agent-4-post-generation-validation-integration)

## Goal

Integrate personal facts verification into workflow quality gates so validation fails if critical facts missing.

## Context

- Uses Agent 1's `personal_facts_verifier` function
- Integrates into existing `QualityGatePipeline`
- Must run after motion generation in validation phase
- Should reject refinements that remove personal fact references

## Deliverables

1. **Integrate into** `QualityGatePipeline`:
   - Add `_validate_personal_facts_coverage()` method
   - Call after motion generation in validation phase
   - Include verification results in quality gate scores

2. **Update** `WorkflowStrategyExecutor._execute_validation_phase()`:
   - Run personal facts verification
   - Include results in validation output
   - Fail validation if critical facts missing

3. **Update** `WorkflowOrchestrator._execute_refinement_phase()`:
   - Check personal facts coverage before accepting refinement
   - Reject refinements that remove personal fact references

4. **Update quality gate scoring:**
   - Add "personal_facts_coverage" metric
   - Weight it heavily in overall quality score
   - Log warnings when coverage is low

5. **Add validation logging:**
   - Log which personal facts were found in motion
   - Log which facts are missing
   - Include in validation report

## Key Files

- `writer_agents/code/WorkflowStrategyExecutor.py`
- `writer_agents/code/WorkflowOrchestrator.py`
- `writer_agents/code/QualityGatePipeline` (if separate file, otherwise in above)

## Success Criteria

- ✅ Quality gates check personal facts coverage
- ✅ Validation fails if critical facts missing
- ✅ Clear logging shows what facts were found/missing
- ✅ Refinements that remove facts are rejected

## Dependencies

- **Agent 1** (requires `personal_facts_verifier` function)
- **Agent 3** (end-to-end test validates integration works)

## Integration Points

- `QualityGatePipeline._validate_personal_facts_coverage()` → called from validation phase
- `WorkflowStrategyExecutor._execute_validation_phase()` → includes verification results
- `WorkflowOrchestrator._execute_refinement_phase()` → checks coverage before accepting

## Metrics to Log

- `personal_facts_coverage`: float (0.0-1.0)
- `facts_found`: List[str] (which facts were detected)
- `facts_missing`: List[str] (which facts are missing)
- `critical_facts_missing`: bool (should fail validation if true)

## Validation

```bash
# Generate motion and check validation logs
python writer_agents/scripts/generate_optimized_motion.py \
  --case-summary "Motion to seal" \
  --enable-google-docs

# Check logs for personal_facts_coverage
grep -i "personal_facts_coverage" logs/*.log

# Run end-to-end test (from Agent 3)
pytest tests/test_motion_personal_facts_e2e.py -v
```
