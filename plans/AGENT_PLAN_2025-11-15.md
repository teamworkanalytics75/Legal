# 📋 Agent Plan: Ensure Motion Uses Only Personal Facts

**Date:** 2025-11-15  
**Goal:** Complete the remaining 10-15% to ensure generated motions are filtered for facts about the user's lawsuit

**Current Status:** 85-90% complete
- ✅ Personal corpus loading works
- ✅ Evidence filtering implemented
- ✅ Prompt injection working
- ⚠️ Missing: Content verification
- ⚠️ Missing: Stricter fallback handling
- ⚠️ Missing: End-to-end validation
- ⚠️ Missing: Post-generation fact checking

---

## 🎯 Agent 1: Content Verification Function

**Objective:** Create a function that verifies generated motions actually reference personal corpus facts

**Tasks:**
1. Create `writer_agents/code/validation/personal_facts_verifier.py`
2. Implement `verify_motion_uses_personal_facts(motion_text: str, personal_corpus_facts: Dict) -> Tuple[bool, List[str], Dict[str, Any]]`
   - Check for mentions of HK Statement key facts
   - Check for references to OGC emails (dates, recipients, subjects)
   - Check for specific allegations (defamation, privacy leaks, retaliation)
   - Check for timeline events (June 2025 arrests, April 2025 OGC emails)
   - Return: (is_valid, missing_facts, verification_details)
3. Add keyword/phrase matching for personal corpus documents:
   - "Hong Kong Statement of Claim" or "HK Statement"
   - "OGC" or "Office of General Counsel"
   - Specific dates: "April 7, 2025", "April 18, 2025", "June 2, 2025", "June 4, 2025"
   - Key allegations: "defamation", "privacy breach", "retaliation", "harassment"
4. Create unit tests in `tests/test_personal_facts_verifier.py`

**Files to Create/Modify:**
- `writer_agents/code/validation/personal_facts_verifier.py` (new)
- `tests/test_personal_facts_verifier.py` (new)

**Success Criteria:**
- Function can detect when motion mentions personal facts
- Function can identify missing key facts
- Unit tests pass with sample motions

---

## 🎯 Agent 2: Stricter Filtering & Fallback Handling

**Objective:** Fix the fallback behavior to fail explicitly when no personal facts are found

**Tasks:**
1. Modify `CaseFactsProvider.filter_evidence_for_lawsuit()` in `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py`
   - Change fallback behavior (lines 728-732) to raise exception or return empty with warning
   - Add configuration flag: `strict_filtering: bool = True`
   - When strict and no personal corpus facts found, log error and return empty list
   - Add validation that personal corpus is loaded before filtering
2. Update `_filter_lawsuit_evidence()` in `WorkflowOrchestrator.py` and `WorkflowStrategyExecutor.py`
   - Check if filtered evidence is empty and personal corpus should be available
   - Log warning/error if filtering removed everything but personal corpus exists
   - Add early validation in workflow initialization
3. Add validation in `WorkflowOrchestrator._initialize_case_facts_provider()`
   - Verify personal corpus directory exists and has files
   - Verify `case_insights.json` exists or can be generated
   - Fail fast with clear error message if personal facts unavailable

**Files to Modify:**
- `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py`
- `writer_agents/code/WorkflowOrchestrator.py`
- `writer_agents/code/WorkflowStrategyExecutor.py`

**Success Criteria:**
- System fails explicitly when personal corpus missing
- Clear error messages guide user to fix the issue
- No silent fallback to generic evidence

---

## 🎯 Agent 3: End-to-End Test & Validation

**Objective:** Create comprehensive end-to-end test that generates and validates a motion

**Tasks:**
1. Create `tests/test_motion_personal_facts_e2e.py`
   - Test that generates a full motion using the workflow
   - Verifies motion references personal corpus facts
   - Verifies motion doesn't include generic placeholder content
   - Verifies motion mentions HK Statement, OGC emails, specific dates
2. Create test fixtures:
   - Mock personal corpus directory with sample documents
   - Sample `case_insights.json` with personal facts
   - Expected fact mentions list
3. Add integration test that:
   - Runs `generate_optimized_motion.py` with test corpus
   - Validates output motion using `personal_facts_verifier`
   - Checks that all key facts are mentioned
   - Verifies no generic "example" content
4. Add test to `tests/test_case_facts_provider.py`:
   - Test that filtering works with real personal corpus
   - Test that filtering fails appropriately when corpus missing

**Files to Create/Modify:**
- `tests/test_motion_personal_facts_e2e.py` (new)
- `tests/test_case_facts_provider.py` (extend)

**Success Criteria:**
- End-to-end test generates motion and validates it
- Test catches when motion doesn't use personal facts
- Test passes with valid personal corpus

---

## 🎯 Agent 4: Post-Generation Validation Integration

**Objective:** Integrate personal facts verification into workflow quality gates

**Tasks:**
1. Integrate `personal_facts_verifier` into `QualityGatePipeline`:
   - Add `_validate_personal_facts_coverage()` method
   - Call after motion generation in validation phase
   - Include verification results in quality gate scores
2. Update `WorkflowStrategyExecutor._execute_validation_phase()`:
   - Run personal facts verification
   - Include results in validation output
   - Fail validation if critical facts missing
3. Add validation to `WorkflowOrchestrator._execute_refinement_phase()`:
   - Check personal facts coverage before accepting refinement
   - Reject refinements that remove personal fact references
4. Update quality gate scoring:
   - Add "personal_facts_coverage" metric
   - Weight it heavily in overall quality score
   - Log warnings when coverage is low
5. Add validation logging:
   - Log which personal facts were found in motion
   - Log which facts are missing
   - Include in validation report

**Files to Modify:**
- `writer_agents/code/WorkflowStrategyExecutor.py`
- `writer_agents/code/WorkflowOrchestrator.py`
- `writer_agents/code/QualityGatePipeline` (if separate file)

**Success Criteria:**
- Quality gates check personal facts coverage
- Validation fails if critical facts missing
- Clear logging shows what facts were found/missing

---

## 📊 Success Metrics

**Overall Goal:** 100% confidence that generated motions use only personal facts

**Metrics:**
1. ✅ Content verification function detects personal fact mentions
2. ✅ System fails explicitly when personal corpus missing
3. ✅ End-to-end test validates full motion generation
4. ✅ Quality gates enforce personal facts coverage
5. ✅ Generated motions reference HK Statement, OGC emails, specific dates
6. ✅ Generated motions exclude generic placeholder content

**Validation:**
- Run end-to-end test: `pytest tests/test_motion_personal_facts_e2e.py -v`
- Generate test motion and verify it mentions personal facts
- Check logs show personal facts verification results

---

## 🔄 Dependencies

- **Agent 1** → **Agent 3, 4** (verification function needed for tests and integration)
- **Agent 2** → **Agent 3** (stricter filtering needed for reliable tests)
- **Agent 3** → **Agent 4** (end-to-end test validates integration works)

**Recommended Order:**
1. Agent 1 (build verification function)
2. Agent 2 (fix fallback handling)
3. Agent 3 (create end-to-end test)
4. Agent 4 (integrate into workflow)

---

## 📝 Notes

- Personal corpus location: `case_law_data/tmp_corpus/`
- Case insights location: `writer_agents/outputs/case_insights.json`
- Key documents: HK Statement, OGC emails, Harvard correspondence
- Key dates: April 7/18, 2025 (OGC emails), June 2/4, 2025 (HK Statement/arrests)

