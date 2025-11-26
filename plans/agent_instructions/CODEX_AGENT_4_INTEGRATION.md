# Codex Agent 4: End-to-End Integration & Quality Gates

**Workstream:** False Fact Prevention System  
**Status:** Ready to start  
**Dependencies:** Agents 1-3 should complete first (but can verify code independently)

---

## 🎯 Objective

Verify full pipeline integration, quality gates, and prompt assembly. Ensure the complete system works end-to-end.

---

## 📋 Tasks

### Task 1: Verify Prompt Assembly
**File:** `writer_agents/code/WorkflowOrchestrator.py` (around line 3537)

Check that the prompt includes explicit prohibitions:

```python
# Search for these strings in the prompt:
# - "⚠️ CRITICAL: You MUST use ONLY the facts provided below. DO NOT invent..."
# - "NEVER describe, infer, or assume ANY fact unless it appears verbatim in STRUCTURED FACTS"
# - "DO NOT infer facts from context clues, safety risks, foreign government connections..."
```

**Command:**
```bash
grep -A 5 "DO NOT invent" writer_agents/code/WorkflowOrchestrator.py
grep -A 3 "NEVER describe.*infer" writer_agents/code/WorkflowOrchestrator.py
```

**Expected:** All prohibition statements present in prompt assembly

### Task 2: Verify WorkflowStrategyExecutor Prompt
**File:** `writer_agents/code/WorkflowStrategyExecutor.py` (around line 3777)

Check that SK-based path also has matching instructions:

```bash
grep -A 5 "DO NOT generate" writer_agents/code/WorkflowStrategyExecutor.py
```

**Expected:** Similar prohibition statements for SK path

### Task 3: Verify Quality Gate Logic - WorkflowOrchestrator
**File:** `writer_agents/code/WorkflowOrchestrator.py` (around line 5365)

Check that quality gates check violations and contradictions:

```python
# Look for:
# - if summary["has_violations"]: validation_results["meets_threshold"] = False
# - if summary["has_contradictions"]: validation_results["meets_threshold"] = False
# - Warnings appended when violations/contradictions detected
```

**Command:**
```bash
grep -A 5 "has_violations" writer_agents/code/WorkflowOrchestrator.py | head -20
grep -A 5 "has_contradictions" writer_agents/code/WorkflowOrchestrator.py | head -20
```

### Task 4: Verify Quality Gate Logic - WorkflowStrategyExecutor
**File:** `writer_agents/code/WorkflowStrategyExecutor.py` (around line 4754)

Check that both orchestrators have matching logic:

```bash
grep -A 5 "has_violations" writer_agents/code/WorkflowStrategyExecutor.py | head -20
grep -A 5 "has_contradictions" writer_agents/code/WorkflowStrategyExecutor.py | head -20
```

**Expected:** Same quality gate logic in both orchestrators

### Task 5: Verify Score Forcing to 0.0
**File:** `writer_agents/code/WorkflowOrchestrator.py` (around line 1193)

Check that score is forced to 0.0 when violations/contradictions present:

```python
# Look for:
# score = coverage
# if has_violations or has_contradictions:
#     score = 0.0
```

**Command:**
```bash
grep -B 2 -A 2 "score = 0.0" writer_agents/code/WorkflowOrchestrator.py
```

### Task 6: Test Full Pipeline (Optional - Requires LLM)
**File:** `writer_agents/scripts/generate_optimized_motion.py`

If Ollama is available, run full pipeline:

```bash
python writer_agents/scripts/generate_optimized_motion.py \
    --case-summary "Motion to seal and pseudonym for Section 1782 discovery case" \
    --max-iterations 3
```

**Expected:**
- No errors during execution
- Fact registry facts appear in structured facts
- If false fact is generated, it's caught by validation
- Quality gates work correctly

**Note:** This requires LLM infrastructure. If not available, verify code is correct instead.

### Task 7: Verify Fact Registry Integration in Pipeline
Check logs or code to confirm fact registry data flows:

```python
# Verify CaseFactsProvider loads fact_registry
# Verify facts appear in STRUCTURED FACTS block
# Verify ContradictionDetector can read from database
```

---

## ✅ Success Criteria

- [ ] Prompt includes "DO NOT invent, assume, or create any facts"
- [ ] Prompt includes "NEVER describe, infer, or assume ANY fact unless it appears verbatim in STRUCTURED FACTS"
- [ ] Quality gate rejects drafts with `has_violations=True`
- [ ] Quality gate rejects drafts with `has_contradictions=True`
- [ ] Score forced to 0.0 when violations/contradictions detected
- [ ] Both orchestrators have matching quality gate logic
- [ ] Fact registry data flows into structured facts (if Agent 1 complete)
- [ ] Full pipeline runs without errors (if LLM available)

---

## 📁 Key Files

- `writer_agents/code/WorkflowOrchestrator.py` - Main orchestrator (prompt assembly, quality gates)
- `writer_agents/code/WorkflowStrategyExecutor.py` - Strategy executor (quality gates)
- `writer_agents/scripts/generate_optimized_motion.py` - Full pipeline script
- `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py` - Fact provider

---

## 🔍 What to Check

1. **Prompt prohibitions:** All explicit "DO NOT" statements present
2. **Quality gate checks:** Both `has_violations` and `has_contradictions` checked
3. **Score forcing:** Score set to 0.0 when flags raised
4. **Warning messages:** Appropriate warnings appended
5. **Consistency:** Both orchestrators have same logic
6. **Integration:** Fact registry data flows through pipeline

---

## 📝 Verification Checklist

### Prompt Assembly
- [ ] WorkflowOrchestrator prompt has explicit prohibitions
- [ ] WorkflowStrategyExecutor prompt has matching instructions
- [ ] Prohibitions are clear and unambiguous

### Quality Gates
- [ ] WorkflowOrchestrator checks `has_violations`
- [ ] WorkflowOrchestrator checks `has_contradictions`
- [ ] WorkflowStrategyExecutor checks `has_violations`
- [ ] WorkflowStrategyExecutor checks `has_contradictions`
- [ ] Both set `meets_threshold = False` when flags raised
- [ ] Both append warnings

### Score Logic
- [ ] Score forced to 0.0 when violations detected
- [ ] Score forced to 0.0 when contradictions detected
- [ ] Score calculation happens after violation/contradiction checks

---

## 📝 Notes

- You can verify code independently (no need to wait for other agents)
- Focus on code correctness, not runtime testing (unless LLM available)
- Report any inconsistencies between orchestrators
- Document any missing quality gate checks

---

## 🚀 When Complete

Mark tasks as complete and report:
- Prompt assembly verification results
- Quality gate logic verification
- Any inconsistencies found
- Code correctness status
- Runtime test results (if LLM available)
- Recommendations for improvements

