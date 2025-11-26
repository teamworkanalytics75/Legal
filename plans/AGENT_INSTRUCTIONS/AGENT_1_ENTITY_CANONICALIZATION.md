# Agent 1 Instructions: Entity Canonicalization

## Status: READY TO START

**No dependencies** - Can begin immediately.

---

## Task

Build `fact_engine/entity_canonicalizer.py` to canonicalize entities across documents, reducing duplicate nodes and improving fact linking.

---

## Key Requirements

1. **Cross-document entity disambiguation**
   - Handle multiple "Wang" mentions across different documents
   - Use context (roles, organizations, dates) to distinguish entities
   - Example: "Wang" in Harvard context vs. "Wang" in PRC context

2. **PRC/Harvard bilingual name canonicalization**
   - Map Chinese names to English variants
   - Handle transliteration variations (e.g., "Xi Jinping" vs. "Xi Mingze")
   - Normalize organization names (e.g., "Harvard" vs. "Harvard University")

3. **Fuzzy location normalization**
   - Normalize location variants (e.g., "PRC", "China", "People's Republic of China")
   - Handle city/country relationships (e.g., "Shanghai, China" → "Shanghai" + "PRC")

4. **Identity-role resolution**
   - Link entities to roles (e.g., "Wang = Vice President of HCS")
   - Extract role information from context
   - Merge entities with same role in same organization

5. **Duplicate identity merging**
   - Merge entities that refer to the same person/org
   - Use similarity thresholds (fuzzy matching)
   - Preserve all source document references

---

## Build On Existing Code

Reference these existing modules:
- `nlp_analysis/code/CoreferenceResolver.py` - Has `find_entity_clusters()` method
- `nlp_analysis/code/KnowledgeGraph.py` - Has `merge_similar_entities()` method

**Extend** these to work across multiple documents, not just within a single document.

---

## Input/Output

**Input**: 
- Raw extracted entities from `fact_registry` database
- Or: CSV with entity columns (fact_value, source_doc, fact_type)

**Output**:
- Canonical entity mapping (dict: {original_entity: canonical_entity})
- Updated fact entries with canonicalized entities
- Statistics: number of duplicates merged, reduction percentage

---

## Integration Point

Modify `writer_agents/scripts/convert_to_truth_table.py`:
- Add call to canonicalizer before processing facts
- Apply canonicalization to `fact_value` column
- Preserve original values in metadata

---

## Success Criteria

- Reduces duplicate entity nodes by 50%+
- Handles bilingual name variants correctly
- Preserves all source document references
- Works with existing fact extraction pipeline

---

## Files to Create

1. `fact_engine/entity_canonicalizer.py` - Main module
2. `fact_engine/tests/test_entity_canonicalizer.py` - Unit tests

---

## Testing

Test with sample data from:
- `case_law_data/facts_truth_table_v2.csv`
- Or: Query `case_law_data/lawsuit_facts_database.db` fact_registry table

---

**Start building now!**

