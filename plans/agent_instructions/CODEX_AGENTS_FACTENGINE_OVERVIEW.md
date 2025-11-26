# Codex Agents Overview - FactEngine Module Build

**Date:** 2025-11-15  
**Goal:** 4 parallel Codex agents building the FactEngine module for canonical fact management, ML importance analysis, and auto-discovery of missing facts

---

## 🎯 Workstream Division

### Agent 1: Schema & Data Loading Foundation
**File:** `CODEX_AGENT_1_FACTENGINE_SCHEMA_LOADER.md`  
**Focus:** Create Pydantic schemas, CSV loaders, and canonical truth table builder  
**Dependencies:** None (foundation layer)  
**Status:** ✅ Complete — fact_engine/ package created with schema.py, loader.py, builder.py, __init__.py; convert_to_truth_table.py and template registry added

**Deliverables:**
- `fact_engine/schema.py` - Fact, FactImportance, BNMapping models
- `fact_engine/loader.py` - load_raw_facts(), load_truth_facts()
- `fact_engine/builder.py` - build_canonical_truth_table() with cleaning/normalization

---

### Agent 2: ML Importance Model (CatBoost + SHAP)
**File:** `CODEX_AGENT_2_FACTENGINE_ML_IMPORTANCE.md`  
**Focus:** Wrap CatBoost with SHAP for fact importance scoring and feature analysis  
**Dependencies:** Agent 1 (needs Fact schema)  
**Status:** ✅ Complete — ml_importance.py implemented with TF-IDF, categorical, keyword, and optional KG features; CatBoost training/prediction, SHAP explanations, heuristic labeling, and CSV-based training data loading

**Deliverables:**
- `fact_engine/ml_importance.py` - FactImportanceModel class
- Feature engineering (TF-IDF, binary flags, graph features)
- SHAP integration for explainability
- Model training and persistence

---

### Agent 3: Evidence Querying (SQL + Embeddings + KG)
**File:** `CODEX_AGENT_3_FACTENGINE_EVIDENCE_QUERY.md`  
**Focus:** Create wrappers for LangChain SQL, EmbeddingRetriever, and Knowledge Graph queries  
**Dependencies:** None (can work independently, just needs to know interfaces)  
**Status:** ✅ Complete — evidence_query.py implemented with fully-guarded wrappers: query_db_for_pattern(), search_similar_passages(), load_kg()/query_kg_for_entity(), build_query_from_hypothesis(), and find_evidence_for_hypothesis() that blends SQL, embeddings, and KG results

**Deliverables:**
- `fact_engine/evidence_query.py` - Query wrappers
- `query_db_for_pattern()` - LangChainSQLAgent wrapper
- `search_similar_passages()` - EmbeddingRetriever wrapper
- `query_kg_for_entity()` - KnowledgeGraph wrapper
- `find_evidence_for_hypothesis()` - High-level orchestrator

---

### Agent 4: Auto-Promotion & Integration
**File:** `CODEX_AGENT_4_FACTENGINE_AUTO_PROMOTE.md`  
**Focus:** Auto-discovery logic, deduplication, entrypoint script, and agent integration  
**Dependencies:** Agents 1-3 (needs all components)  
**Status:** ✅ Complete — auto_promote.py, run_fact_engine.py, helpers.py implemented with guardrails; all dependencies now available for full end-to-end pipeline

**Deliverables:**
- `fact_engine/auto_promote.py` - find_feature_gaps(), propose_new_facts_from_hypothesis(), deduplicate_facts(), auto_promote_missing_facts()
- `fact_engine/run_fact_engine.py` - Main entrypoint orchestrating full pipeline
- `get_top_facts_for_sealing()` helper function
- Integration with FactExtractorAgent or new FactSupervisorAgent

---

## 🔄 Coordination

### Parallel Work
- **Agent 1** can work completely independently (foundation)
- **Agent 2** can start once Agent 1's `schema.py` exists (just needs Fact model)
- **Agent 3** can work completely independently (just needs to know function signatures)
- **Agent 4** can design interfaces independently, but full implementation needs Agents 1-3

### No Conflicts
- Agent 1: New package structure + schema + loaders (no conflicts)
- Agent 2: New ML module (no conflicts)
- Agent 3: New query module (no conflicts)
- Agent 4: New auto-promote + entrypoint (no conflicts)

### Shared Files (Read-Only)
All agents may read:
- `case_law_data/facts_ranked_for_sealing.csv`
- `case_law_data/facts_truth_table.csv`
- `case_law_data/facts_knowledge_graph.json`
- `writer_agents/code/LangchainIntegration.py`
- `writer_agents/code/sk_plugins/FeaturePlugin/EmbeddingRetriever.py`
- `nlp_analysis/code/pipeline.py`
- `nlp_analysis/code/KnowledgeGraph.py`

**No conflicts expected** - each agent works on different modules.

---

## 📋 Each Agent Should

1. Read their instruction file (`CODEX_AGENT_N_FACTENGINE_*.md`)
2. Read this overview for context
3. Read the main plan document for full requirements
4. **ONLY work on their assigned tasks**
5. Create files in `fact_engine/` package (root level, parallel to `writer_agents/`)

## ⚠️ Priority & Critical Rules

**Implementation Priority:**
1. Make `fact_engine/schema.py`, `loader.py`, `builder.py` compile and run against existing CSVs
2. Then add `ml_importance.py` using existing CatBoost/SHAP scripts
3. Only after that, wire `evidence_query.py` and `auto_promote.py`

**Critical Rules:**
- **NEVER create a fact without anchoring it to an actual `source_excerpt` from the corpus**
- Builder only cleans existing facts - never invents new ones
- Auto-promoted facts go to separate CSV for manual review
- SHAP is optional - pipeline should work with just CatBoost feature importances if SHAP unavailable

---

## 🎯 Success Criteria

- Agent 1: ✅ Complete — Can load CSVs and build clean canonical truth table
  - ✅ Schema and loader implemented
  - ✅ Builder with scaffolding filters and heuristics
  - ✅ Template registry and conversion pipeline
  - ✅ Enhanced schema.py with Field metadata
  - ✅ Rebuilt builder.py with stronger guardrails and legal-label separation
- Agent 2: ✅ Complete — Can train CatBoost model and get SHAP explanations for facts
  - ✅ ml_importance.py with TF-IDF, categorical, keyword, and KG features
  - ✅ CatBoost training/prediction and SHAP explanations
  - ✅ CSV-based training data loading with heuristic fallback
- Agent 3: ✅ Complete — Can query corpus via SQL, embeddings, and KG
  - ✅ Fully-guarded wrappers for LangChain SQL, EmbeddingRetriever, and KG
  - ✅ find_evidence_for_hypothesis() blends all three sources
  - ✅ Graceful degradation when dependencies unavailable
- Agent 4: ✅ Complete — Auto-promotion logic implemented; full end-to-end pipeline ready
  - ✅ SHAP-driven gap detection and hypothesis → evidence retrieval
  - ✅ Fuzzy deduplication and safety/public-exposure tagging
  - ✅ CLI entrypoint with --rebuild, --train, --promote flags
  - ✅ All dependencies now available for full pipeline execution

---

## 📁 Output Files

All agents write to:
- `fact_engine/` package (new directory at repo root)
- Output CSVs: `case_law_data/facts_truth_table_v2.csv`, `case_law_data/facts_sealing_index.csv`
- Model: `case_law_data/models/fact_importance_model.cbm`
