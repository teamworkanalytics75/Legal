# Final Facts Database Integration Plan - 4 Parallel Codex Sessions

## Overview

Integrate the final litigation-ready facts database (`case_law_data/top_1000_facts_for_chatgpt_final.csv` - 605 facts) into the motion to seal pipeline. The pipeline currently expects SQLite database (`lawsuit_facts_database.db`) and knowledge graph (`facts_knowledge_graph.json`).

## Execution Strategy

**4 Parallel Codex Sessions** working independently:
- **Session 1**: Database Conversion (can start immediately)
- **Session 2**: KG Builder (depends on Session 1)
- **Session 3**: CaseFactsProvider (can start after Session 1)
- **Session 4**: Testing (depends on Sessions 1-3)

---

## Task 1: CSV to SQLite Database Conversion Script

**Session**: Codex Session #1  
**File**: `scripts/import_final_facts_to_database.py`  
**Status**: ⏳ In Progress (Session 1 working on this)

### Objective
Convert final CSV to SQLite `fact_registry` table format

### Key Requirements
- Read `case_law_data/top_1000_facts_for_chatgpt_final.csv`
- Map CSV columns to `fact_registry` schema:
  - `factid` → `fact_id` (PRIMARY KEY)
  - `proposition` → `fact_value` (and also `description`)
  - `eventtype` → `fact_type` (with fallback to infer from `subject`/`actorrole` if needed)
  - `evidencetype` → `source_doc` (or create structured source)
  - `truthstatus` → store in metadata or new column
  - `causal_salience_score` → store in metadata
  - `eventdate`, `eventlocation` → store in metadata
- Create/update `case_law_data/lawsuit_facts_database.db`
- Preserve all metadata fields (TruthStatus, EvidenceType, CausalSalienceScore, etc.)
- Handle existing database (backup or merge strategy)
- Validate data integrity after import

### Reference Schema
From `extract_fact_registry.py:417-425`:
```python
CREATE TABLE IF NOT EXISTS fact_registry (
    fact_id TEXT PRIMARY KEY,
    fact_type TEXT,
    fact_value TEXT,
    description TEXT,
    source_doc TEXT,
    extraction_method TEXT,
    confidence REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

### Deliverables
- Conversion script with error handling
- Data validation checks
- Backup mechanism for existing database

---

## Task 2: Knowledge Graph Builder Updates

**Session**: Codex Session #2  
**Files**: 
- `writer_agents/scripts/build_fact_knowledge_graph.py`
- `writer_agents/scripts/map_facts_to_kg.py` (if needed)

**Status**: ⏳ Waiting for Task 1

### Objective
Update knowledge graph builder to work with final facts database

### Key Requirements
- Update `build_fact_knowledge_graph()` to accept database path as input
- Load facts from SQLite `fact_registry` table (after Task 1 completes)
- Extract entities and relations from `fact_value` (proposition) text
- Map fact metadata (Subject, ActorRole, EventType) to graph nodes
- Preserve causal relationships (CausalSalienceScore, CausalSalienceReason)
- Update `map_facts_to_knowledge_graph()` to handle new fact structure
- Ensure backward compatibility with source document extraction

### Reference Implementation
From `build_fact_knowledge_graph.py:55-100`:
- Uses `EntityRelationExtractor` or `LawsuitEntityExtractor`
- Processes text documents to extract entities/relations
- Needs to process `fact_value` (proposition) text instead

### Deliverables
- Updated `build_fact_knowledge_graph.py` with database input option
- Script to rebuild KG from final facts: `scripts/rebuild_kg_from_final_facts.py`
- Validation that KG contains entities from final facts

---

## Task 3: CaseFactsProvider Integration Updates

**Session**: Codex Session #3  
**File**: `writer_agents/code/sk_plugins/FeaturePlugin/CaseFactsProvider.py`  
**Status**: ⏳ Waiting for Task 1

### Objective
Ensure CaseFactsProvider correctly loads and queries final facts

### Key Requirements
- Verify `_load_facts_from_database()` method works with new database structure
- Ensure `get_fact_block()` can retrieve facts by FactID
- Update fact querying to use `fact_value` (proposition) as primary content
- Preserve access to metadata (TruthStatus, EvidenceType, etc.)
- Test backward compatibility with existing fact loading paths
- Ensure `_lawsuit_facts_cache` properly loads from new database

### Reference Methods
From `CaseFactsProvider.py`:
- `_load_facts_from_database()` - loads from SQLite
- `get_fact_block(key)` - retrieves fact by key
- `_lawsuit_facts_cache` - caches loaded facts

### Deliverables
- Updated CaseFactsProvider (if needed) or verification it works as-is
- Test script: `scripts/test_case_facts_provider_final_facts.py`
- Documentation of any API changes

---

## Task 4: Integration Testing & Documentation

**Session**: Codex Session #4  
**Files**: 
- `tests/test_final_facts_integration.py` (new)
- `docs/FINAL_FACTS_PIPELINE_INTEGRATION.md` (new)
- Update `docs/MASTER_FACTS_DATABASE_ARCHITECTURE.md`

**Status**: ⏳ Waiting for Tasks 1-3

### Objective
Comprehensive testing and documentation of integration

### Key Requirements
- End-to-end test: CSV → Database → KG → CaseFactsProvider → Motion Generation
- Verify fact count matches (605 facts)
- Test fact retrieval by FactID
- Test knowledge graph queries
- Test motion generation uses final facts
- Update architecture documentation
- Create integration guide with step-by-step instructions

### Test Scenarios
1. Import final CSV to database
2. Verify all 605 facts imported correctly
3. Rebuild knowledge graph from database
4. Query facts via CaseFactsProvider
5. Generate test motion and verify it references final facts
6. Validate fact metadata preserved (TruthStatus, EvidenceType, etc.)

### Deliverables
- Comprehensive test suite
- Integration documentation
- Updated architecture docs
- Troubleshooting guide

---

## Execution Order & Dependencies

```
Session 1 (Database Conversion)
    ↓
    ├─→ Session 2 (KG Builder) - starts after Session 1
    └─→ Session 3 (CaseFactsProvider) - can start after Session 1
            ↓
            ↓
Session 4 (Testing) - starts after Sessions 1-3 complete
```

**Parallel Execution Strategy**:
- **Session 1**: Task 1 (Database Conversion) - Can start immediately
- **Session 2**: Task 2 (KG Builder) - Starts after Task 1 completes
- **Session 3**: Task 3 (CaseFactsProvider) - Can start after Task 1 completes
- **Session 4**: Task 4 (Testing) - Starts after Tasks 1-3 complete

---

## Success Criteria

- ✅ All 605 facts from final CSV imported to SQLite database
- ✅ Knowledge graph built from final facts with entities/relations extracted
- ✅ CaseFactsProvider successfully loads and queries final facts
- ✅ Motion generation pipeline uses final facts without errors
- ✅ All tests pass
- ✅ Documentation complete and accurate

---

## Input Files

- **Final CSV**: `case_law_data/top_1000_facts_for_chatgpt_final.csv` (605 facts)
- **Existing Database**: `case_law_data/lawsuit_facts_database.db` (may exist)
- **Existing KG**: `case_law_data/facts_knowledge_graph.json` (may exist)

## Output Files

- **Updated Database**: `case_law_data/lawsuit_facts_database.db`
- **Updated KG**: `case_law_data/facts_knowledge_graph.json`
- **Test Scripts**: `tests/test_final_facts_integration.py`
- **Documentation**: `docs/FINAL_FACTS_PIPELINE_INTEGRATION.md`

---

## Quick Start Commands

### Session 1 (Database Conversion)
```bash
# Create conversion script
# Run: python scripts/import_final_facts_to_database.py
```

### Session 2 (KG Builder)
```bash
# After Session 1 completes
# Run: python scripts/rebuild_kg_from_final_facts.py
```

### Session 3 (CaseFactsProvider)
```bash
# After Session 1 completes
# Run: python scripts/test_case_facts_provider_final_facts.py
```

### Session 4 (Testing)
```bash
# After Sessions 1-3 complete
# Run: pytest tests/test_final_facts_integration.py -v
```

---

## Notes

- Session 1 is currently in progress (Codex session working on `import_final_facts_to_database.py`)
- All sessions should coordinate through this plan document
- Each session should update status as tasks complete
- Final verification requires all 4 sessions to complete successfully

