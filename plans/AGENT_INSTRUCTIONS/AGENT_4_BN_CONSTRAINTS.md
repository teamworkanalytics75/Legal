# Agent 4 Instructions: BN Structural Constraints (Module #3)

## Status: READY TO START

**No dependencies** - Can begin immediately.

---

## Task

Build `writer_agents/code/bn_structural_constraints.py` to enforce legal causation constraints on BN structure.

---

## Key Requirements

1. **Hard Constraints (Edges that MUST exist)**
   - "PRC political risk → Plaintiff physical risk" (mandatory)
   - "Harvard statements → Media amplification" (if statements exist)
   - "Media amplification → PRC awareness" (if media exists)

2. **Forbidden Edges (Edges that MUST NOT exist)**
   - "Plaintiff's career history → PRC crackdowns" (forbidden)
   - "Club bylaws rewrite → PRC response" (forbidden)
   - Any edge that violates legal causation logic

3. **Optional Edges (Learned from data)**
   - All other edges are optional
   - Validated against data
   - Can be added if evidence supports

4. **Domain-Specific Rules**
   - Legal causation patterns
   - Temporal constraints (cause must precede effect)
   - Institutional pathway validation

---

## Build On Existing Code

Reference:
- `writer_agents/code/BuildBnStructureFromKg.py` - Has `ensure_acyclic()` and structure building
- `nlp_analysis/code/pipeline.py` - Has `extract_bayesian_network()` method

**Add constraint validation** to existing BN structure building.

---

## Input/Output

**Input**: 
- BN structure (nodes, edges) from `BuildBnStructureFromKg.py`
- Facts from `bn_node_lookup.csv`
- Constraint rules (hard constraints, forbidden edges)

**Output**:
- Constrained BN structure (validated edges only)
- Report of removed edges (forbidden)
- Report of added edges (hard constraints)
- Validation errors/warnings

---

## Integration Point

Modify `writer_agents/code/BuildBnStructureFromKg.py`:
- Add constraint application after structure building (around line ~404-437)
- Call `bn_structural_constraints.apply_constraints()` before returning structure
- Validate edges against constraint rules

---

## Success Criteria

- BN structure respects legal causation constraints
- No impossible edges in final structure
- Hard constraints are enforced
- Forbidden edges are removed
- Works with existing BN building pipeline

---

## Files to Create

1. `writer_agents/code/bn_structural_constraints.py` - Main module
2. `writer_agents/code/tests/test_bn_structural_constraints.py` - Unit tests

---

## Testing

Test with:
- Sample BN structure from existing pipeline
- Validate constraint rules are applied correctly
- Check that impossible edges are removed

---

**Start building now!**

