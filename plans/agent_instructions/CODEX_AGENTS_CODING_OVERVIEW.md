# Codex Agents Coding Instructions - Overview

**Goal:** 4 Codex agents writing code in parallel to implement false fact prevention system

---

## 🎯 Workstream Division

### Agent 1: Fact Registry Extraction & Integration
**File:** `CODEX_AGENT_1_WRITE_FACT_REGISTRY.md`  
**Writes:**
- `extract_fact_registry.py` script
- `CaseFactsProvider.get_fact_registry()` method
- `ContradictionDetector._load_fact_registry()` method
- Verification helper script

**No conflicts** - Creates new files and adds methods to existing classes

---

### Agent 2: Negative Fact Rules
**File:** `CODEX_AGENT_2_WRITE_NEGATIVE_RULES.md`  
**Writes:**
- Extends `FactRule` dataclass
- Creates `NEGATIVE_FACT_RULES` tuple
- Updates `verify_motion_uses_personal_facts()` function
- Unit tests

**No conflicts** - Extends existing code, adds new functionality

---

### Agent 3: Contradiction Detection
**File:** `CODEX_AGENT_3_WRITE_CONTRADICTION_DETECTOR.md`  
**Writes:**
- `contradiction_detector.py` (new file)
- `ContradictionDetector` class
- Citizenship validator
- Unit tests

**No conflicts** - Creates entirely new file

---

### Agent 4: Integration & Quality Gates
**File:** `CODEX_AGENT_4_WRITE_INTEGRATION.md`  
**Writes:**
- Updates `_run_personal_facts_verifier()` in both orchestrators
- Updates quality gate logic
- Updates prompt assembly
- Updates refinement guard

**Potential conflicts** - Modifies same methods as other agents might touch  
**Coordination:** Agent 4 should coordinate or work after Agents 1-3

---

## 🔄 Coordination Strategy

### Parallel Work (No Conflicts)
- **Agents 1, 2, 3** can work completely in parallel
  - Agent 1: New scripts + new methods
  - Agent 2: Extends existing classes
  - Agent 3: New file entirely

### Sequential Work (Coordination Needed)
- **Agent 4** should work after Agents 1-3 complete
  - Depends on Agent 1's `get_fact_registry()` method
  - Depends on Agent 2's negative rules
  - Depends on Agent 3's `ContradictionDetector` class

---

## 📋 Each Agent Should

1. Read their instruction file
2. Write the code as specified
3. Follow existing code style (4-space indent, type hints, logging)
4. Add appropriate error handling
5. Test their code works
6. Report completion

---

## ✅ Success Criteria (All Agents)

When all complete:
- [ ] Fact registry extraction script works
- [ ] Negative rules detect violations
- [ ] Contradiction detector flags contradictions
- [ ] Quality gates reject drafts with violations/contradictions
- [ ] Prompts include explicit prohibitions
- [ ] All unit tests pass

---

## 🚀 Start Here

Each Codex agent:
1. Read your specific instruction file (`CODEX_AGENT_X_WRITE_*.md`)
2. Write the code as specified
3. Test your code
4. Report when done

**Agent 4:** Wait for Agents 1-3 to complete, then integrate everything.

