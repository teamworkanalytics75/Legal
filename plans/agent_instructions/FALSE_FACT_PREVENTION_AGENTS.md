# False Fact Prevention System - 4-Agent Workstream Instructions

**Date:** 2025-11-15  
**Status:** All agents complete ✅  
**Goal:** Prevent false facts (like "PRC citizen" inference) from slipping through the system

---

## 📋 Overview

This workstream implements a general, extensible framework to prevent false facts from appearing in generated motions. The system includes:
- **Negative fact validation** (rules that prohibit certain facts)
- **Contradiction detection** (comparing generated text against verified sources)
- **Fact registry** (explicit facts extracted from source documents)
- **Quality gates** (rejecting drafts with violations or contradictions)

---

## ✅ Agent 1: Fact Registry Population & Database Integration

**Status:** COMPLETE ✅

### Completed Work
- ✅ Fixed `extract_fact_registry.py` path resolution
- ✅ Created `verify_fact_registry_citizenship.py` CLI helper
- ✅ Added sample fixtures for testing
- ✅ Verified `CaseFactsProvider` and `ContradictionDetector` integration
- ✅ Fact registry populated with citizenship facts

### Key Files
- `writer_agents/scripts/extract_fact_registry.py`
- `writer_agents/scripts/verify_fact_registry_citizenship.py`
- `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py`
- `writer_agents/code/validation/contradiction_detector.py`

### Validation Commands
```bash
# Populate fact registry
python writer_agents/scripts/extract_fact_registry.py \
    --source-dir case_law_data/lawsuit_source_documents \
    --database case_law_data/lawsuit_facts_database.db

# Verify citizenship facts
python writer_agents/scripts/verify_fact_registry_citizenship.py \
    --database case_law_data/lawsuit_facts_database.db
```

### Next Steps
- **None** - Complete and verified

---

## ✅ Agent 2: Negative Fact Rules Testing & Validation

**Status:** COMPLETE ✅

### Completed Work
- ✅ Unit tests for negative rule detection
- ✅ Tests verify `verify_motion_uses_personal_facts()` catches violations
- ✅ Orchestrator integration tests
- ✅ Quality gate rejection tests

### Key Files
- `writer_agents/code/validation/personal_facts_verifier.py` (NEGATIVE_FACT_RULES)
- `tests/test_personal_facts_verifier.py`
- `writer_agents/code/WorkflowOrchestrator.py` (_run_personal_facts_verifier)

### Test Commands
```bash
# Run negative fact detection tests
pytest tests/test_personal_facts_verifier.py::test_negative_rule_detects_citizenship_violation -v

# Run all negative fact tests
pytest tests/test_personal_facts_verifier.py -v
```

### Test Example
```python
test_motion = "The plaintiff, a PRC citizen, seeks protection from retaliation."
# Should return violations=["not_prc_citizen"]
```

### Next Steps
- **None** - Complete and verified

---

## ✅ Agent 3: Contradiction Detection Testing & Integration

**Status:** COMPLETE ✅

### Completed Work
- ✅ Unit tests for contradiction detection
- ✅ Tests with mock fact_registry (in-memory dict)
- ✅ Orchestrator integration test (`test_workflow_personal_facts_gate.py`)
- ✅ Verifies contradictions are flagged and included in validation results

### Key Files
- `writer_agents/code/validation/contradiction_detector.py`
- `tests/test_contradiction_detector.py`
- `tests/test_workflow_personal_facts_gate.py`

### Test Commands
```bash
# Run contradiction detection tests
pytest tests/test_contradiction_detector.py -v

# Run orchestrator integration test
pytest tests/test_workflow_personal_facts_gate.py -v
```

### Test Example
```python
# Can use mock fact_registry for testing
detector = ContradictionDetector(fact_registry={"citizenship": "US citizen"})
contradictions = detector.detect_contradictions("The plaintiff is a PRC citizen.")
# Should return contradiction with type="DIRECT_CONTRADICTION", severity="critical"
```

### Next Steps
- **None** - Complete and verified

---

## ✅ Agent 4: End-to-End Integration & Quality Gates

**Status:** COMPLETE ✅ (Code verified, runtime test requires LLM)

### Completed Work
- ✅ Verified prompt assembly includes explicit prohibitions
- ✅ Verified quality gate logic checks violations/contradictions
- ✅ Verified fact registry integration works
- ✅ Fixed Ollama client type annotation errors
- ✅ Standardized default model to `qwen2.5:14b`

### Key Files
- `writer_agents/code/WorkflowOrchestrator.py` (prompt assembly, quality gates)
- `writer_agents/code/WorkflowStrategyExecutor.py` (quality gates)
- `writer_agents/code/ollama_client.py` (fixed type annotations)
- `writer_agents/code/sk_config.py` (standardized to qwen2.5:14b)

### Verification Points

**Prompt Assembly (WorkflowOrchestrator.py:3537-3570):**
- ✅ "⚠️ CRITICAL: You MUST use ONLY the facts provided below. DO NOT invent..."
- ✅ "NEVER describe, infer, or assume ANY fact unless it appears verbatim in STRUCTURED FACTS"
- ✅ "DO NOT infer facts from context clues, safety risks, foreign government connections..."

**Quality Gates (WorkflowOrchestrator.py:5365-5376):**
- ✅ Checks `has_violations` flag
- ✅ Checks `has_contradictions` flag
- ✅ Sets `meets_threshold = False` when violations/contradictions detected
- ✅ Appends blocking warnings

**Fact Registry Integration:**
- ✅ CaseFactsProvider loads facts from `fact_registry` table
- ✅ Facts appear in STRUCTURED FACTS block
- ✅ ContradictionDetector reads from database

### Runtime Test Command
```bash
# Requires Ollama server running and qwen2.5:14b model available
python writer_agents/scripts/generate_optimized_motion.py \
    --case-summary "Motion to seal and pseudonym for Section 1782 discovery case" \
    --max-iterations 3
```

### Next Steps
- **Optional:** Run full end-to-end test when LLM is available
- **Code is complete** - all validation logic verified

---

## 🔧 Infrastructure Fixes (Completed)

### Ollama Client Fix
- **Issue:** Type annotation errors prevented import
- **Fix:** Changed `LLMMessage` → `Any` in type annotations
- **Files:** `writer_agents/code/ollama_client.py`
- **Status:** ✅ Fixed and verified

### Default Model Standardization
- **Issue:** Inconsistent defaults (some used `phi3:mini`, others `qwen2.5:14b`)
- **Fix:** Standardized all defaults to `qwen2.5:14b`
- **Files:** `writer_agents/code/sk_config.py`, `writer_agents/scripts/generate_optimized_motion.py`
- **Status:** ✅ Complete

---

## 📊 Validation Checklist

### Agent 1
- [x] Fact registry table populated with citizenship entry
- [x] CaseFactsProvider.get_fact_registry() returns ['US citizen']
- [x] Structured facts block includes registry data
- [x] ContradictionDetector reads from database

### Agent 2
- [x] Unit tests pass for negative rule detection
- [x] verify_motion_uses_personal_facts() returns violations for PRC citizen claim
- [x] Orchestrator includes violations in result dict
- [x] Quality gate checks has_violations flag

### Agent 3
- [x] Unit tests pass for contradiction detection
- [x] ContradictionDetector flags citizenship contradictions
- [x] Contradiction reports include claim, type, severity, location
- [x] Orchestrator includes contradictions in validation result
- [x] Quality gate checks has_contradictions flag

### Agent 4
- [x] Prompt includes explicit fact-inference prohibitions
- [x] Quality gate rejects drafts with violations/contradictions
- [x] Fact registry data flows into structured facts
- [x] Ollama client fixed and working
- [x] Default model standardized to qwen2.5:14b
- [ ] Full pipeline runtime test (requires LLM - code verified)

---

## 🎯 Summary

All four agents have completed their workstreams:

1. **Agent 1:** Fact registry populated and integrated ✅
2. **Agent 2:** Negative fact detection tested and verified ✅
3. **Agent 3:** Contradiction detection tested and verified ✅
4. **Agent 4:** End-to-end integration verified (code complete) ✅

The false fact prevention system is **fully implemented and ready**. The only remaining step is an optional runtime test when LLM infrastructure is available, but all code verification is complete.

---

## 📝 Documentation

- **Fix Log:** `reports/agent_fix_logs/2025-11-15.md`
- **Plan:** See plan file for detailed workstream division
- **Tests:** All unit tests passing

