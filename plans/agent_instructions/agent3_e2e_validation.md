# Agent 3 – End-to-End Test & Validation

**Source:** [AGENT_PLAN_2025-11-15.md](../AGENT_PLAN_2025-11-15.md#agent-3-end-to-end-test--validation)

## Goal

Create comprehensive end-to-end test that generates and validates a motion uses personal facts.

## Context

- Uses Agent 1's `personal_facts_verifier` function
- Uses Agent 2's stricter filtering (must be complete first)
- Tests full workflow from corpus → motion generation → validation

## Deliverables

1. **Create** `tests/test_motion_personal_facts_e2e.py`
   - Test that generates full motion using workflow
   - Verifies motion references personal corpus facts
   - Verifies motion doesn't include generic placeholder content
   - Verifies motion mentions HK Statement, OGC emails, specific dates

2. **Create test fixtures:**
   - Real personal corpus directory with sample documents (not mocks)
   - Real `case_insights.json` with personal facts
   - Real `truths.db` database
   - Expected fact mentions list

3. **Add integration test:**
   - Runs `generate_optimized_motion.py` with test corpus
   - Validates output motion using `personal_facts_verifier` (from Agent 1)
   - Checks that all key facts are mentioned
   - Verifies no generic "example" content

4. **Extend** `tests/test_case_facts_provider.py`:
   - Test that filtering works with real personal corpus
   - Test that filtering fails appropriately when corpus missing

## Key Files

- `tests/test_motion_personal_facts_e2e.py` (new)
- `tests/test_case_facts_provider.py` (extend)

## Success Criteria

- ✅ End-to-end test generates motion and validates it
- ✅ Test catches when motion doesn't use personal facts
- ✅ Test passes with valid personal corpus
- ✅ Test uses real data structures (not mocks)

## Dependencies

- **Agent 1** (requires `personal_facts_verifier` function)
- **Agent 2** (requires stricter filtering to be reliable)

## Validation

```bash
# Run end-to-end test
pytest tests/test_motion_personal_facts_e2e.py -v

# Run extended case facts provider tests
pytest tests/test_case_facts_provider.py -v

# Full test suite
pytest tests/test_motion_personal_facts_e2e.py tests/test_case_facts_provider.py -v
```

## Test Fixtures (Real Data)

- Create real `.txt` files in `tmp_path / "tmp_corpus"`
- Create real JSON file for `case_insights.json`
- Create real SQLite database for `truths.db`
- Use actual `CaseFactsProvider` class (not mocked)
