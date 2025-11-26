# Codex Agents Overview - False Fact Prevention System

**Date:** 2025-11-15  
**Goal:** 4 parallel Codex agents working on different parts of the false fact prevention system

---

## 🎯 Workstream Division

### Agent 1: Fact Registry Population & Database Integration
**File:** `CODEX_AGENT_1_FACT_REGISTRY.md`  
**Focus:** Database population, CaseFactsProvider integration, ContradictionDetector integration  
**Dependencies:** None (can work independently)  
**Status:** Ready to start

### Agent 2: Negative Fact Rules Testing & Validation
**File:** `CODEX_AGENT_2_NEGATIVE_RULES.md`  
**Focus:** Unit tests, violation detection, orchestrator integration, quality gates  
**Dependencies:** None (can use unit tests independently)  
**Status:** Ready to start

### Agent 3: Contradiction Detection Testing & Integration
**File:** `CODEX_AGENT_3_CONTRADICTION_DETECTION.md`  
**Focus:** Unit tests, contradiction detection, orchestrator integration, quality gates  
**Dependencies:** None (can use mock fact_registry)  
**Status:** Ready to start

### Agent 4: End-to-End Integration & Quality Gates
**File:** `CODEX_AGENT_4_INTEGRATION.md`  
**Focus:** Prompt assembly verification, quality gate logic, full pipeline testing  
**Dependencies:** Agents 1-3 (but can verify code independently)  
**Status:** Ready to start

---

## 🔄 Coordination

### Parallel Work
- **Agents 1, 2, 3** can work completely in parallel
- **Agent 4** can verify code independently, but full runtime test benefits from Agents 1-3 completion

### No Conflicts
- Agent 1: Database/scripts (no code conflicts)
- Agent 2: Tests only (no code conflicts)
- Agent 3: Tests only (no code conflicts)
- Agent 4: Verification only (read-only checks)

### Shared Files (Read-Only)
All agents may read:
- `writer_agents/code/validation/personal_facts_verifier.py`
- `writer_agents/code/validation/contradiction_detector.py`
- `writer_agents/code/WorkflowOrchestrator.py`
- `writer_agents/code/WorkflowStrategyExecutor.py`

**No conflicts expected** - each agent works on different aspects.

---

## 📋 Quick Start

Each agent should:
1. Read their specific instruction file
2. Work through tasks in order
3. Mark tasks complete as they finish
4. Report results when done

---

## ✅ Success Criteria (All Agents)

When all agents complete:
- [ ] Fact registry populated and integrated
- [ ] Negative fact rules tested and verified
- [ ] Contradiction detection tested and verified
- [ ] Quality gates verified and working
- [ ] Prompt assembly verified
- [ ] Full pipeline tested (if LLM available)

---

## 📝 Reporting

Each agent should report:
- Tasks completed
- Test results
- Issues encountered
- Recommendations

---

## 🚀 Next Steps

1. Each Codex agent reads their instruction file
2. Work through tasks independently
3. Report completion status
4. Agent 4 coordinates final integration verification

