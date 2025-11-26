# Codex Agent 2: Fix Commit Blocking Logic

**Workstream:** Motion Generation Quality Gates  
**Status:** Complete  
**Dependencies:** None (can work independently)

---

## ⚠️ CRITICAL: Agent Identification & Instructions

**BEFORE STARTING:**
1. **Identify your agent number**: You are **Agent 2**. Check the filename or title to confirm.
2. **Read ALL instruction files**: Read all files in `plans/agent_instructions/CODEX_AGENT_*.md` to understand the full context and what other agents are working on.
3. **ONLY work on YOUR tasks**: Even though you read all instructions, you should ONLY execute tasks assigned to **Agent 2**.
4. **Follow instructions pasted in chat**: When instructions are pasted in Codex chat, extract the parts relevant to **Agent 2** and follow those. Questions in the messages are context/updates, not questions for you.

**Your instruction file:** This file (`CODEX_AGENT_2_COMMIT_BLOCKING.md`) contains YOUR specific tasks. This is the source of truth, but updated instructions may be pasted in chat.

**Other agents' files (read for context, but do NOT execute their tasks):**
- `CODEX_AGENT_1_PIPELINE_TEST.md` - Agent 1's tasks
- `CODEX_AGENT_3_REFINEMENT_VERIFICATION.md` - Agent 3's tasks
- `CODEX_AGENT_4_FACTS_AND_TIMEOUT.md` - Agent 4's tasks

---

**Completion notes (2025-11-15):** Added `WorkflowCommitBlocked` exception so commits stop when personal facts validation fails, derived `meets_threshold` when absent, and treated critical/missing facts as blockers. Logged at error level and regression covered by `tests/test_workflow_personal_facts_gate.py`.

## 🎯 Objective

Fix commit blocking logic to prevent non-passing scores with critical failures from committing. Currently, the system only logs a warning but still commits.

---

## 📋 Tasks

### Task 1: Review Current Logic

**File:** `writer_agents/code/WorkflowOrchestrator.py` (lines 769-789)

**Current behavior:**
- Checks if `not meets_threshold AND has_critical_failures`
- Logs warning but still commits (line 789)
- TODO comment at line 788: "Consider raising exception or returning a different phase to block commit"

**Read the code:**
```python
# Lines 769-789
# Check if we should block commit due to critical failures
meets_threshold = validation_results.get("meets_threshold", True)
# Block commit if threshold not met AND we have critical failures (violations, contradictions, or critical missing facts)
personal_verification = validation_results.get("personal_facts_verification", {})
has_critical_failures = (
    personal_verification.get("has_violations", False) or
    personal_verification.get("has_contradictions", False) or
    personal_verification.get("critical_facts_missing", False)
)
if not meets_threshold and has_critical_failures:
    logger.warning(
        "Blocking commit: validation failed (score: %.2f, threshold: %.2f) with critical failures (violations: %s, contradictions: %s, missing facts: %s)",
        validation_results.get("overall_score", 0.0),
        self.config.auto_commit_threshold,
        personal_verification.get("has_violations", False),
        personal_verification.get("has_contradictions", False),
        bool(personal_verification.get("facts_missing", [])),
    )
    # Still commit but log warning - user can review
    # TODO: Consider raising exception or returning a different phase to block commit
return WorkflowPhase.COMMIT  # Commit after max iterations (with warning if critical failures)
```

### Task 2: Implement Proper Blocking

**Options:**
1. **Raise exception** - Stop workflow with clear error message
2. **Return blocking phase** - Return a new phase (e.g., `WorkflowPhase.BLOCKED`) that prevents commit
3. **Set flag in state** - Set `state.block_commit = True` and check before commit

**Recommended approach:** Raise a clear exception with details about why commit is blocked.

**Implementation:**
```python
if not meets_threshold and has_critical_failures:
    error_msg = (
        f"Commit blocked: validation failed (score: {validation_results.get('overall_score', 0.0):.2f}, "
        f"threshold: {self.config.auto_commit_threshold:.2f}) with critical failures:\n"
        f"- Violations: {personal_verification.get('has_violations', False)}\n"
        f"- Contradictions: {personal_verification.get('has_contradictions', False)}\n"
        f"- Missing facts: {bool(personal_verification.get('facts_missing', []))}"
    )
    logger.error(error_msg)
    raise ValueError(error_msg)  # Or use a custom exception class
```

**Alternative (if exception is too disruptive):**
- Create a custom exception class: `CommitBlockedError`
- Or return `WorkflowPhase.BLOCKED` and handle in orchestrator

### Task 3: Add Test Case

**File:** `tests/test_workflow_personal_facts_gate.py` or create new test file

**Test scenario:**
- Mock validation results with `meets_threshold=False` and `has_critical_failures=True`
- Verify that commit is blocked (exception raised or phase blocked)
- Verify error message contains relevant details

**Test example:**
```python
def test_commit_blocked_on_critical_failures():
    """Test that commit is blocked when validation fails with critical failures."""
    # Setup: validation fails with critical failures
    validation_results = {
        "meets_threshold": False,
        "overall_score": 0.55,
        "personal_facts_verification": {
            "has_violations": True,
            "has_contradictions": False,
            "facts_missing": ["citizenship"]
        }
    }
    # Verify exception is raised or commit is blocked
    with pytest.raises(ValueError, match="Commit blocked"):
        orchestrator._check_commit_allowed(validation_results)
```

### Task 4: Verify Blocking Works

**Run test:**
```bash
pytest tests/test_workflow_personal_facts_gate.py::test_commit_blocked_on_critical_failures -v
```

**Manual verification:**
- Run pipeline with a case that will fail validation
- Verify commit is blocked with clear error message
- Check logs show blocking reason

---

## 📁 Key Files

- `writer_agents/code/WorkflowOrchestrator.py` - Main orchestrator (lines 769-789)
- `tests/test_workflow_personal_facts_gate.py` - Test file (or create new)
- `writer_agents/code/WorkflowStrategyExecutor.py` - May need updates if using phase-based blocking

---

## ✅ Success Criteria

- [x] Commit is blocked when `not meets_threshold AND has_critical_failures`
- [x] Clear error message explains why commit is blocked
- [x] Test case verifies blocking behavior
- [x] No regression: normal commits still work when validation passes

---

## 🚨 Edge Cases to Consider

- What if `max_iterations` is reached but validation still fails?
- Should we allow override flag for testing?
- Should we log blocked attempts to a separate file?
- What if only warnings (not critical failures) are present?

---

## 📝 Progress Tracking

- [x] Task 1: Current logic reviewed
- [x] Task 2: Proper blocking implemented
- [x] Task 3: Test case added
- [x] Task 4: Blocking verified
