# Codex Agent 3: Contradiction Detection Testing & Integration

**Workstream:** False Fact Prevention System  
**Status:** Ready to start  
**Dependencies:** None (can use mock fact_registry for testing)

---

## 🎯 Objective

Test `ContradictionDetector` in isolation and verify it flags contradictions correctly, then verify orchestrator integration.

---

## 📋 Tasks

### Task 1: Run Existing Unit Tests
**File:** `tests/test_contradiction_detector.py`

```bash
pytest tests/test_contradiction_detector.py -v
```

**Expected:** All tests pass

### Task 2: Test with Mock Fact Registry
**File:** Create test or add to existing test file

Test that `ContradictionDetector` works with in-memory fact registry:

```python
from writer_agents.code.validation.contradiction_detector import ContradictionDetector

# Use mock fact_registry (no database needed)
detector = ContradictionDetector(fact_registry={"citizenship": "US citizen"})

motion = "The plaintiff is a PRC citizen seeking protection."
contradictions = detector.detect_contradictions(motion)

assert contradictions
assert contradictions[0]["contradiction_type"] == "DIRECT_CONTRADICTION"
assert contradictions[0]["severity"] == "critical"
assert contradictions[0]["fact_type"] == "citizenship"
```

### Task 3: Test No Contradiction Case
Test that no contradiction is reported when claim matches fact:

```python
detector = ContradictionDetector(fact_registry={"citizenship": "US citizen"})

motion = "The plaintiff is a United States citizen seeking protection."
contradictions = detector.detect_contradictions(motion)

assert contradictions == []  # No contradiction
```

### Task 4: Test Orchestrator Integration
**File:** `tests/test_workflow_personal_facts_gate.py` or create new test

Verify orchestrator includes contradictions in validation result:

```python
from writer_agents.code import WorkflowOrchestrator

# Stub the detector to return a contradiction
motion = "The plaintiff is a PRC citizen."
context = {"source_docs_dir": Path("case_law_data/lawsuit_source_documents")}

result = WorkflowOrchestrator._run_personal_facts_verifier(motion, context=context)

assert result is not None
assert result["has_contradictions"] is True
assert result["contradictions"]  # Should contain contradiction entries
assert result["is_valid"] is False
```

### Task 5: Test Quality Gate with Contradictions
Verify quality gate checks `has_contradictions` flag:

```python
# Test that quality gate rejects drafts with contradictions
# Check that score is forced to 0.0 when has_contradictions=True
# Check that warnings are appended
```

### Task 6: Run Full Test Suite
```bash
pytest tests/test_contradiction_detector.py tests/test_workflow_personal_facts_gate.py -v
```

**Expected:** All tests pass

---

## ✅ Success Criteria

- [ ] Unit tests pass for contradiction detection
- [ ] ContradictionDetector flags citizenship contradictions correctly
- [ ] Contradiction report includes: claim, type, severity, location
- [ ] Orchestrator includes contradictions in validation result
- [ ] Quality gate checks `has_contradictions` flag
- [ ] Quality gate sets score to 0.0 when contradictions detected

---

## 📁 Key Files

- `writer_agents/code/validation/contradiction_detector.py` - `ContradictionDetector` class
- `tests/test_contradiction_detector.py` - Unit tests
- `tests/test_workflow_personal_facts_gate.py` - Orchestrator integration test
- `writer_agents/code/WorkflowOrchestrator.py` - Integration point

---

## 🔍 What to Check

1. **Detector works:** Can detect contradictions with mock fact_registry
2. **Contradiction types:** DIRECT_CONTRADICTION, INFERENCE, HALLUCINATION
3. **Severity levels:** critical, warning
4. **Orchestrator integration:** `_run_personal_facts_verifier()` includes contradictions
5. **Quality gate:** Gates check `has_contradictions` and reject drafts

---

## 📝 Test Cases

### Test Case 1: Direct Contradiction
```python
fact_registry = {"citizenship": "US citizen"}
motion = "The plaintiff is a PRC citizen."
# Expected: contradiction with type="DIRECT_CONTRADICTION", severity="critical"
```

### Test Case 2: Inference (No Fact Registry)
```python
fact_registry = {}
motion = "Because their home country of PRC cannot protect them..."
# Expected: contradiction with type="INFERENCE", severity="warning"
```

### Test Case 3: No Contradiction
```python
fact_registry = {"citizenship": "US citizen"}
motion = "The plaintiff is a United States citizen."
# Expected: no contradictions
```

---

## 📝 Notes

- You can work independently - use mock fact_registry (in-memory dict) for testing
- No need to wait for Agent 1's database work
- Focus on testing the detection logic, not database loading
- Report any test failures or missing functionality

---

## 🚀 When Complete

Mark tasks as complete and report:
- Test results (all passing?)
- Any test failures
- Issues with orchestrator integration
- Contradiction detection accuracy
- Recommendations for improvements

