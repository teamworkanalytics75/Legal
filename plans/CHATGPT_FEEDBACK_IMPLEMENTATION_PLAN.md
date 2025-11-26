# ChatGPT Feedback Implementation Plan

## Summary

ChatGPT identified 5 critical missing components that will radically increase accuracy and show courts/investigators the causal structure more clearly.

---

## ✅ What We Already Have

- ✅ ML-based fact extraction (NER + OpenIE + temporal parsing)
- ✅ DAGs for legal causation
- ✅ Bayesian networks
- ✅ Multi-agent writing + reasoning workflow
- ✅ SQL + LangChain for evidence grounding
- ✅ Basic confidence scoring (0.0-1.0)
- ✅ Basic entity merging (CoreferenceResolver, KnowledgeGraph)
- ✅ Causal inference detection (CausalInference.py)

---

## ❗ What We're Missing (5 Critical Modules)

### 1. Multi-Layer FACT CONFIDENCE FILTER
**Status**: ❌ Missing  
**Priority**: 🔥 HIGH  
**What it does**: Stratifies facts into tiers (Direct evidence, Strong inference, Weak inference, Unverifiable garbage)

### 2. CAUSAL-SALIENCE FILTER
**Status**: ❌ Missing  
**Priority**: 🔥 HIGH  
**What it does**: Filters facts to only those causally relevant to HARVARD → PRC → HARM pathway

### 3. STRUCTURAL PRIOR for BN
**Status**: ❌ Missing  
**Priority**: 🔥 HIGH  
**What it does**: Hard constraints (must exist), forbidden edges (must not exist), optional edges (learned)

### 4. COUNTERFACTUAL ENGINE
**Status**: ❌ Missing  
**Priority**: 🔥 HIGH  
**What it does**: "What if" queries to demonstrate foreseeability, negligence, hazard propagation

### 5. FALSE-POSITIVE REMOVAL MODULE (Entity Canonicalization)
**Status**: ⚠️ Partial (basic merging exists)  
**Priority**: 🔥 HIGH  
**What it does**: Cross-doc entity disambiguation, bilingual name canonicalization, identity-role resolution

---

## Implementation Plan

### Phase 1: Confidence Filter (Module #1)
**Location**: `fact_engine/confidence_filter.py`  
**Integration**: After truth table conversion, before BN ingestion

**Features**:
- Tier 1: Direct evidence (quoted statements, official documents)
- Tier 2: Strong inference (temporal proximity, multiple sources)
- Tier 3: Weak inference (single source, low confidence)
- Tier 4: Unverifiable garbage (remove)

**Input**: Truth table facts  
**Output**: Tiered facts with confidence scores

---

### Phase 2: Causal Salience Filter (Module #2)
**Location**: `fact_engine/causal_salience_filter.py`  
**Integration**: After confidence filter, before DAG/BN building

**Features**:
- Filters facts relevant to HARVARD → PRC → HARM pathway
- Timing-based salience (temporal proximity to key events)
- Pathway-based salience (institutional channels)
- Foreseeability-based salience (risk propagation)

**Input**: Tiered facts  
**Output**: Causally relevant facts only

---

### Phase 3: Structural Constraints for BN (Module #3)
**Location**: `writer_agents/code/bn_structural_constraints.py`  
**Integration**: During BN structure building

**Features**:
- Hard constraints (edges that must exist)
- Forbidden edges (edges that must not exist)
- Optional edges (learned from data)
- Domain-specific rules (legal causation patterns)

**Input**: BN structure + facts  
**Output**: Constrained BN structure

---

### Phase 4: Counterfactual Engine (Module #4)
**Location**: `fact_engine/counterfactual_engine.py`  
**Integration**: After BN is built, for querying

**Features**:
- "What if" queries (e.g., "What if Harvard never sent the 19 April 2019 clarification?")
- Alternative timeline generation
- Probability comparison (baseline vs. counterfactual)
- "But for" causation analysis

**Input**: BN + query  
**Output**: Counterfactual analysis results

---

### Phase 5: Entity Canonicalization (Module #5)
**Location**: `fact_engine/entity_canonicalizer.py`  
**Integration**: After extraction, before truth table conversion

**Features**:
- Cross-doc entity disambiguation (e.g., multiple "Wang")
- PRC/Harvard bilingual name canonicalization
- Fuzzy location normalization
- Identity-role resolution ("Wang = Vice President of HCS")
- Duplicate identity merging

**Input**: Raw extracted entities  
**Output**: Canonicalized entities

---

## Implementation Order

1. **Module #5** (Entity Canonicalization) - Foundation for everything else
2. **Module #1** (Confidence Filter) - Prunes noise early
3. **Module #2** (Causal Salience Filter) - Focuses on relevant facts
4. **Module #3** (Structural Constraints) - Ensures valid BN structure
5. **Module #4** (Counterfactual Engine) - Legal demonstration tool

---

## Integration Points

```
Raw Extraction
    ↓
[Module #5: Entity Canonicalization]
    ↓
Truth Table Conversion
    ↓
[Module #1: Confidence Filter]
    ↓
[Module #2: Causal Salience Filter]
    ↓
[Module #3: Structural Constraints]
    ↓
BN/DAG Building
    ↓
[Module #4: Counterfactual Engine]
    ↓
Legal Analysis
```

---

## Success Criteria

- [ ] Facts stratified into 4 tiers (Tier 1-4)
- [ ] Only causally relevant facts reach BN/DAG
- [ ] BN structure respects legal causation constraints
- [ ] Counterfactual queries return meaningful results
- [ ] Entity canonicalization reduces duplicate nodes by 50%+

---

## Next Steps

1. Review existing code (confidence scoring, entity merging, causal detection)
2. Design module interfaces
3. Implement Module #5 (Entity Canonicalization)
4. Implement Module #1 (Confidence Filter)
5. Implement Module #2 (Causal Salience Filter)
6. Implement Module #3 (Structural Constraints)
7. Implement Module #4 (Counterfactual Engine)
8. Integration testing
9. Performance optimization

---

**Status**: Ready to implement  
**Priority**: 🔥 CRITICAL  
**Estimated Time**: 2-3 days for all 5 modules

