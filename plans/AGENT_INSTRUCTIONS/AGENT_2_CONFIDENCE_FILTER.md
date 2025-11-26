# Agent 2 Instructions: Confidence Filter

## Status: READY TO START

**No dependencies** - Can begin immediately.

---

## Task

Build `fact_engine/confidence_filter.py` to stratify facts into 4 confidence tiers and remove unverifiable garbage.

---

## Key Requirements

1. **Tier 1: Direct Evidence**
   - Quoted statements from documents
   - Official documents (court filings, certified translations)
   - High confidence extraction methods (TEMPLATE with confidence >0.8)
   - Multiple source documents confirming same fact

2. **Tier 2: Strong Inference**
   - Temporal proximity to key events
   - Multiple sources (2+ documents)
   - Medium-high confidence (0.6-0.8)
   - Well-structured propositions (complete S-V-O)

3. **Tier 3: Weak Inference**
   - Single source document
   - Low confidence (0.5-0.6)
   - Inferred relationships
   - Fragments that were converted to propositions

4. **Tier 4: Unverifiable Garbage (REMOVE)**
   - Contradictory facts (same entity, different values)
   - Fragments that couldn't be converted
   - Noise patterns (repeated "====", punctuation-only)
   - Facts with confidence <0.5

---

## Build On Existing Code

Reference:
- `writer_agents/scripts/extract_facts_ml_enhanced.py` - Has confidence scores in FactEntry
- `writer_agents/scripts/convert_to_truth_table.py` - Has `ExtractionConfidence` column

**Use existing confidence scores** and add tiering logic.

---

## Input/Output

**Input**: 
- `facts_truth_table.csv` (from convert_to_truth_table.py)
- Columns: FactID, Proposition, ExtractionMethod, ExtractionConfidence, SourceDocument, etc.

**Output**:
- Same CSV format with added `confidence_tier` column (1, 2, 3, or removed)
- Statistics: tier distribution, number removed

---

## Integration Point

Modify `fact_engine/run_fact_engine.py`:
- Add filter step after truth table loading (around line ~132)
- Filter out Tier 4 facts before promoting/processing
- Pass tiered facts to next stage

---

## Success Criteria

- Facts stratified into 4 tiers
- Tier 4 facts removed (unverifiable garbage)
- Confidence scores normalized and validated
- Works with existing truth table format

---

## Files to Create

1. `fact_engine/confidence_filter.py` - Main module
2. `fact_engine/tests/test_confidence_filter.py` - Unit tests

---

## Testing

Test with:
- `case_law_data/facts_truth_table_v2.csv`
- Validate tier distribution is reasonable
- Check that Tier 4 facts are actually garbage

---

**Start building now!**

