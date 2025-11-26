# Additional Tasks for Agents 1-4: Master Facts Database & Enhancements

**Status:** Ready for parallel work  
**Timing:** Execute while Agent 5 runs pipeline tests  
**Goal:** Enhance master facts database and improve system quality

---

## 🎯 Overview

While Agent 5 executes the end-to-end pipeline, Agents 1-4 can work on additional enhancements that don't conflict with testing. These tasks improve the master facts database, add quality improvements, and prepare for production use.

---

## Agent 1: Master Facts Database Integration

### Task 1.1: Sync Fact Registry to Master Database ✅

**Goal:** Ensure `fact_registry` table is synced with the master `document_facts` table in `build_truth_database.py`

**File:** `writer_agents/scripts/sync_fact_registry_to_master.py` (create new)

**What to do:**
- Read facts from `fact_registry` table (new hierarchical structure)
- Map to `document_facts` table structure
- Update `build_truth_database.py` to also read from `fact_registry`
- Ensure both databases stay in sync

**Status:** Script implemented and wired into `build_truth_database.py` (`--sync-fact-registry` flag). Manual CLI sync writes hierarchical facts into `document_facts`.

**Code structure:**
```python
def sync_fact_registry_to_document_facts(
    fact_registry_db: Path,
    master_db: Path
) -> None:
    """Sync hierarchical fact_registry to document_facts table."""
    # Read from fact_registry
    # Map fact_type + fact_value + description to document_facts
    # Insert/update master database
    pass
```

### Task 1.2: Enhance Master Database Schema ✅

**Goal:** Add support for hierarchical fact types in master database

**File:** `writer_agents/scripts/build_truth_database.py` (update)

**What to do:**
- Add `description` field to `document_facts` table (if missing)
- Support hierarchical fact types (date, allegation, etc.)
- Add fact type hierarchy metadata
- Update fact insertion logic

**Status:** `document_facts` now tracks `fact_category`, `fact_value`, and `description` columns with automatic migrations; legacy JSON imports populate categories using key prefixes.

### Task 1.3: Create Master Database Query Interface ✅

**Goal:** Create unified query interface for both fact_registry and document_facts

**File:** `writer_agents/scripts/query_master_facts.py` (create new)

**What to do:**
- Query both `fact_registry` and `document_facts` tables
- Provide unified API for fact lookups
- Support hierarchical fact type queries
- Return facts with source attribution

**Status:** `writer_agents/scripts/query_master_facts.py` allows combined queries (filters by fact type/category/source) and prints merged registry+master results.

---

## Agent 2: Enhanced Entity Extraction & Graph Quality

### Task 2.1: Improve Entity Extraction Patterns

**Goal:** Add more lawsuit-specific entity patterns for better extraction

**File:** `writer_agents/code/validation/lawsuit_entity_extractor.py` (enhance)

**What to do:**
- Add patterns for legal document types (motions, complaints, exhibits)
- Improve date extraction (support more formats)
- Add organization alias expansion (Harvard OGC → Office of General Counsel)
- Add person name extraction (plaintiff, defendants)
- Add case citation patterns

### Task 2.2: Enhance Relationship Extraction

**Goal:** Extract more relationship types from documents

**File:** `writer_agents/scripts/map_fact_relationships.py` (enhance)

**What to do:**
- Add temporal relationships (before, after, during)
- Add causal relationships (caused_by, resulted_in)
- Add document relationships (references, cites, responds_to)
- Add organizational relationships (affiliated_with, employed_by)
- Improve confidence scoring for relationships

### Task 2.3: Graph Quality Metrics & Validation

**Goal:** Add quality checks and metrics for Knowledge Graph

**File:** `writer_agents/scripts/validate_knowledge_graph.py` (create new)

**What to do:**
- Check for orphaned nodes (no relationships)
- Validate relationship types are correct
- Check for duplicate entities
- Verify fact nodes link to source documents
- Generate quality report

---

## Agent 3: Query Interface Enhancements

### Task 3.1: Add Advanced Query Methods

**Goal:** Add more sophisticated query capabilities

**File:** `writer_agents/code/validation/fact_graph_query.py` (enhance)

**What to do:**
- Add `find_facts_by_date_range(start_date, end_date)`
- Add `find_facts_by_source_document(doc_path)`
- Add `get_fact_timeline()` - chronological ordering
- Add `find_contradictory_facts(fact_type, fact_value)` - detect conflicts
- Add `get_fact_chain(entity1, entity2)` - find connection paths

### Task 3.2: Improve Semantic Search

**Goal:** Enhance semantic search accuracy and performance

**File:** `writer_agents/code/validation/fact_graph_query.py` (enhance)

**What to do:**
- Add synonym expansion (OGC = Office of General Counsel)
- Improve fuzzy matching algorithm
- Add context-aware search (consider relationships)
- Cache embedding computations
- Add search result ranking

### Task 3.3: Query Performance Optimization

**Goal:** Optimize query performance for large graphs

**File:** `writer_agents/code/validation/fact_graph_query.py` (enhance)

**What to do:**
- Add query result caching (LRU cache)
- Implement lazy loading for large graphs
- Add query timeout handling
- Optimize graph traversal algorithms
- Add query performance metrics

---

## Agent 4: Validator Enhancements & Quality Improvements

### Task 4.1: Enhanced Contradiction Detection

**Goal:** Improve contradiction detection using graph relationships

**File:** `writer_agents/code/validation/contradiction_detector.py` (enhance)

**What to do:**
- Use graph relationships to detect indirect contradictions
- Check temporal contradictions (date conflicts)
- Detect logical contradictions (A → B but motion says B → A)
- Add contradiction severity scoring
- Improve contradiction evidence collection

### Task 4.2: Fact Coverage Analysis

**Goal:** Analyze which facts are covered in motions

**File:** `writer_agents/code/validation/personal_facts_verifier.py` (enhance)

**What to do:**
- Generate fact coverage report (which facts present/missing)
- Calculate coverage percentage by fact type
- Identify critical missing facts
- Suggest facts to add based on graph relationships
- Track fact coverage over time

### Task 4.3: Validator Performance & Logging

**Goal:** Improve validator performance and observability

**File:** `writer_agents/code/validation/` (enhance both validators)

**What to do:**
- Add detailed logging for graph queries
- Add performance metrics (query time, cache hits)
- Add validation result caching
- Improve error messages
- Add validation statistics tracking

---

## Cross-Agent Collaboration Tasks

### Task X.1: Master Facts Database Consolidation

**Goal:** Create unified master facts database from all sources

**File:** `writer_agents/scripts/consolidate_master_facts.py` (create new)

**What to do:**
- Read from `fact_registry` table
- Read from `document_facts` table
- Read from `lawsuit_facts_extracted.json`
- Read from Knowledge Graph
- Merge and deduplicate
- Create unified master database
- Generate consolidation report

**Agents involved:** All 4 agents (coordinate)

### Task X.2: Fact Quality Scoring System

**Goal:** Score facts by quality (confidence, source, verification)

**File:** `writer_agents/scripts/score_fact_quality.py` (create new)

**What to do:**
- Calculate quality scores for each fact
- Consider: source reliability, extraction confidence, graph support
- Flag low-quality facts for review
- Generate quality report
- Update fact_registry with quality scores

**Agents involved:** Agents 1, 2, 3

### Task X.3: Documentation & Examples

**Goal:** Create comprehensive documentation for the fact system

**Files:** Create new documentation files

**What to do:**
- Document hierarchical fact types
- Document Knowledge Graph structure
- Create usage examples
- Document query interface
- Create troubleshooting guide

**Agents involved:** All 4 agents (each documents their area)

---

## Priority Recommendations

### High Priority (Do First)
1. **Agent 1: Task 1.1** - Sync fact_registry to master database
2. **Agent 2: Task 2.1** - Improve entity extraction patterns
3. **Agent 3: Task 3.1** - Add advanced query methods
4. **Agent 4: Task 4.2** - Fact coverage analysis

### Medium Priority
5. **Agent 1: Task 1.2** - Enhance master database schema
6. **Agent 2: Task 2.2** - Enhance relationship extraction
7. **Agent 3: Task 3.2** - Improve semantic search
8. **Agent 4: Task 4.1** - Enhanced contradiction detection

### Low Priority (Nice to Have)
9. **Cross-Agent: Task X.1** - Master facts consolidation
10. **Cross-Agent: Task X.2** - Fact quality scoring
11. **Cross-Agent: Task X.3** - Documentation

---

## Success Criteria

### Agent 1
- [ ] Fact registry synced to master database
- [ ] Master database supports hierarchical types
- [ ] Unified query interface works

### Agent 2
- [ ] More entity patterns added
- [ ] More relationship types extracted
- [ ] Graph quality validation works

### Agent 3
- [ ] Advanced query methods implemented
- [ ] Semantic search improved
- [ ] Query performance optimized

### Agent 4
- [ ] Contradiction detection enhanced
- [ ] Fact coverage analysis works
- [ ] Validator performance improved

---

## Coordination Notes

- **No conflicts with Agent 5**: All tasks work on separate files or enhancements
- **Can work in parallel**: Each agent has independent tasks
- **Share results**: Update shared documentation as you complete tasks
- **Test independently**: Each task should be testable on its own

---

## Reporting

When completing tasks, report:
- Task completed
- Files created/modified
- Test results
- Any issues encountered
- Recommendations for next steps
