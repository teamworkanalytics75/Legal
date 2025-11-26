# Codex Agent 4: Write Quality Gate Integration & Prompt Assembly Code

**Your Task:** Write code to integrate negative fact validation and contradiction detection into the workflow orchestrators and update prompt assembly.

---

## 🎯 What to Build

### 1. Update _run_personal_facts_verifier() in WorkflowOrchestrator

**File to modify:** `writer_agents/code/WorkflowOrchestrator.py`

**Requirements:**
- Modify `_run_personal_facts_verifier()` to call both negative validation and contradiction detection
- Import `ContradictionDetector` (handle ImportError gracefully)
- Call `verify_motion_uses_personal_facts()` with negative rules
- Call `ContradictionDetector.detect_contradictions()` if available
- Merge violations and contradictions into result dict
- Set `has_violations` and `has_contradictions` flags
- Force score to 0.0 if violations or contradictions present

**Example:**
```python
def _run_personal_facts_verifier(
    document: str,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    # ... existing code to get personal facts ...
    
    # Call verify_motion_uses_personal_facts with negative rules
    is_valid, missing, violations, details = verify_motion_uses_personal_facts(
        document,
        personal_corpus_facts,
        negative_rules=NEGATIVE_FACT_RULES,  # NEW
    )
    
    # NEW: Contradiction detection
    contradictions = []
    if ContradictionDetector is not None:
        try:
            source_docs_dir = context.get("source_docs_dir") or Path("case_law_data/lawsuit_source_documents")
            lawsuit_facts_db = context.get("lawsuit_facts_db") or Path("case_law_data/lawsuit_facts_database.db")
            
            detector = ContradictionDetector(
                source_docs_dir=source_docs_dir,
                lawsuit_facts_db=lawsuit_facts_db,
            )
            contradictions = detector.detect_contradictions(document)
        except Exception as exc:
            logger.warning(f"Contradiction detection failed: {exc}")
    
    # Merge results
    has_violations = bool(violations)
    has_contradictions = bool(contradictions)
    overall_valid = is_valid and not has_violations and not has_contradictions
    
    result = {
        "is_valid": overall_valid,
        "missing_facts": missing,
        "violations": violations,  # NEW
        "contradictions": contradictions,  # NEW
        "has_violations": has_violations,  # NEW
        "has_contradictions": has_contradictions,  # NEW
        "score": coverage if overall_valid else 0.0,  # Force to 0.0 if violations/contradictions
        # ... other fields ...
    }
    
    return result
```

### 2. Update Quality Gate Logic in WorkflowOrchestrator

**File to modify:** `writer_agents/code/WorkflowOrchestrator.py` (around line 5325-5376)

**Requirements:**
- In quality gate execution, check `has_violations` and `has_contradictions` flags
- Set `validation_results["meets_threshold"] = False` if either flag is True
- Append warnings when violations or contradictions detected
- Log violations and contradictions

**Example:**
```python
# In quality gate execution
if summary["has_violations"]:
    warnings = validation_results.setdefault("warnings", [])
    warnings.append("Prohibited or false facts detected; remove hallucinated content.")
    validation_results["meets_threshold"] = False
    logger.warning(
        "[FACTS] Violations detected: %s",
        ", ".join(v.get("name", "unknown") for v in summary["violations"][:4]) or "unspecified",
    )

if summary["has_contradictions"]:
    warnings = validation_results.setdefault("warnings", [])
    warnings.append("Contradictions with verified facts detected; revise statements.")
    validation_results["meets_threshold"] = False
    logger.warning(
        "[FACTS] Contradictions detected: %s",
        ", ".join(c.get("claim", "unknown")[:50] for c in summary["contradictions"][:4]) or "unspecified",
    )
```

### 3. Update Prompt Assembly in WorkflowOrchestrator

**File to modify:** `writer_agents/code/WorkflowOrchestrator.py` (around line 3537)

**Requirements:**
- Add explicit prohibitions to the AutoGen prompt
- Include warnings about not inferring facts
- Reference STRUCTURED FACTS section explicitly

**Example:**
```python
# In _assemble_autogen_prompt() method
prompt = f"""
⚠️ CRITICAL: You MUST use ONLY the facts provided below. DO NOT invent, assume, or create any facts, names, dates, or case details that are not explicitly stated in the STRUCTURED FACTS and EVIDENCE sections below.

STRUCTURED FACTS (USE THESE EXACT FACTS - DO NOT INVENT ANYTHING):
{structured_facts_block}

- NEVER describe, infer, or assume ANY fact (citizenship, nationality, dates, locations, relationships, employment, etc.) unless it appears verbatim in the STRUCTURED FACTS section.
- If a fact is not listed in STRUCTURED FACTS, do NOT mention it. Do not speculate, infer, or "fill in" missing information.
- DO NOT infer facts from context clues, safety risks, foreign government connections, or other hints.
- DO NOT invent case names, parties, dates, or facts. Use ONLY what is provided in STRUCTURED FACTS and EVIDENCE above.

{rest_of_prompt}
"""
```

### 4. Mirror Changes in WorkflowStrategyExecutor

**File to modify:** `writer_agents/code/WorkflowStrategyExecutor.py`

**Requirements:**
- Apply same changes to `_run_personal_facts_verifier()` method
- Apply same quality gate logic
- Add similar prompt prohibitions for SK-based path

### 5. Update Refinement Guard

**File to modify:** `writer_agents/code/WorkflowOrchestrator.py` (in refinement phase)

**Requirements:**
- Check if refined draft introduces new violations or contradictions
- Reject refinement if it adds violations/contradictions that weren't in baseline
- Log rejection reason

**Example:**
```python
# In refinement phase
if refined_personal.get("has_violations") and not baseline_personal.get("has_violations"):
    revert_reason = "introduced prohibited facts"
elif refined_personal.get("has_contradictions") and not baseline_personal.get("has_contradictions"):
    revert_reason = "introduced contradictions"
```

---

## 📁 Files to Create/Modify

1. **Modify:** `writer_agents/code/WorkflowOrchestrator.py`
   - Update `_run_personal_facts_verifier()`
   - Update quality gate logic
   - Update prompt assembly
   - Update refinement guard

2. **Modify:** `writer_agents/code/WorkflowStrategyExecutor.py`
   - Mirror all changes from WorkflowOrchestrator

---

## ✅ Success Criteria

- [ ] `_run_personal_facts_verifier()` calls negative validation and contradiction detection
- [ ] Quality gates check `has_violations` and `has_contradictions` flags
- [ ] Quality gates set `meets_threshold = False` when violations/contradictions detected
- [ ] Prompt includes explicit prohibitions against inferring facts
- [ ] Refinement guard rejects drafts that introduce violations/contradictions
- [ ] Both orchestrators have matching logic

---

## 🔍 Reference Files

- Look at existing `_run_personal_facts_verifier()` implementation
- Look at existing quality gate logic
- Look at existing prompt assembly code
- Follow existing error handling patterns

---

## 📝 Notes

- Handle ImportError gracefully if ContradictionDetector not available
- Use fallback paths for source_docs_dir (check tmp_corpus if lawsuit_source_documents doesn't exist)
- Make prompt prohibitions clear and unambiguous
- Log all violations and contradictions for debugging
- Ensure both orchestrators stay in sync

