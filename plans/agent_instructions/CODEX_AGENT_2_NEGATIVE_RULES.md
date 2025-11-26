# Codex Agent 2: Negative Fact Rules Testing & Validation

**Workstream:** False Fact Prevention System  
**Status:** Ready to start  
**Dependencies:** None (can use unit tests independently)

---

## 🎯 Objective

Test negative fact detection in isolation and verify that violations are caught and reported correctly.

---

## 📋 Tasks

### Task 1: Run Existing Unit Tests
**File:** `tests/test_personal_facts_verifier.py`

```bash
pytest tests/test_personal_facts_verifier.py::test_negative_rule_detects_citizenship_violation -v
```

**Expected:** Test passes, showing violation is detected

### Task 2: Create Test Motion with Violation
**File:** Create test script or add to existing test file

Test that `verify_motion_uses_personal_facts()` returns violations:

```python
from writer_agents.code.validation.personal_facts_verifier import verify_motion_uses_personal_facts

test_motion = "The plaintiff, a PRC citizen, seeks protection from retaliation."

is_valid, missing, violations, details = verify_motion_uses_personal_facts(test_motion, {})

assert is_valid is False
assert violations == ["not_prc_citizen"]
assert details["violations"] == ["not_prc_citizen"]
```

### Task 3: Test Orchestrator Integration
**File:** `writer_agents/code/WorkflowOrchestrator.py`

Verify `_run_personal_facts_verifier()` includes violations in result:

```python
from writer_agents.code import WorkflowOrchestrator

motion = "The plaintiff, a PRC citizen, seeks protection."
context = {"source_docs_dir": Path("case_law_data/lawsuit_source_documents")}

result = WorkflowOrchestrator._run_personal_facts_verifier(motion, context=context)

assert result is not None
assert result["has_violations"] is True
assert result["violations"]  # Should contain violation entries
assert result["is_valid"] is False
```

### Task 4: Test Quality Gate Rejection
**File:** `writer_agents/code/WorkflowOrchestrator.py` or test file

Verify quality gate logic checks `has_violations` flag:

```python
# Test that quality gate rejects drafts with violations
# Check that score is forced to 0.0 when has_violations=True
# Check that warnings are appended
```

### Task 5: Run Full Test Suite
```bash
pytest tests/test_personal_facts_verifier.py -v
```

**Expected:** All tests pass, including negative rule tests

---

## ✅ Success Criteria

- [ ] Unit tests pass for negative rule detection
- [ ] `verify_motion_uses_personal_facts()` returns `violations=["not_prc_citizen"]` for test motion
- [ ] Orchestrator's `_run_personal_facts_verifier()` includes violations in result dict
- [ ] Quality gate logic checks `has_violations` flag
- [ ] Quality gate sets score to 0.0 when violations detected

---

## 📁 Key Files

- `writer_agents/code/validation/personal_facts_verifier.py` - Contains `NEGATIVE_FACT_RULES`
- `tests/test_personal_facts_verifier.py` - Unit tests
- `writer_agents/code/WorkflowOrchestrator.py` - `_run_personal_facts_verifier()` method
- `writer_agents/code/WorkflowStrategyExecutor.py` - Mirror implementation

---

## 🔍 What to Check

1. **Negative rules defined:** `NEGATIVE_FACT_RULES` tuple in `personal_facts_verifier.py`
2. **Rule matching works:** Rules detect "PRC citizen" patterns
3. **Violations returned:** Function returns violations list
4. **Orchestrator integration:** `_run_personal_facts_verifier()` handles violations
5. **Quality gate:** Gates check `has_violations` and reject drafts

---

## 📝 Test Cases

### Test Case 1: Direct Violation
```python
motion = "The plaintiff, a PRC citizen, seeks protection."
# Expected: violations=["not_prc_citizen"]
```

### Test Case 2: No Violation
```python
motion = "The plaintiff, a US citizen, seeks protection."
# Expected: violations=[]
```

### Test Case 3: Multiple Violations
```python
motion = "PRC citizen files in District of Hong Kong federal court."
# Expected: violations=["not_prc_citizen", "not_wrong_court_location"]
```

---

## 📝 Notes

- You can work independently - no need for Agent 1's database (can use empty dict for testing)
- Focus on testing the validation logic, not database integration
- Report any test failures or missing functionality

---

## 🚀 When Complete

Mark tasks as complete and report:
- Test results (all passing?)
- Any test failures
- Issues with orchestrator integration
- Recommendations for improvements

