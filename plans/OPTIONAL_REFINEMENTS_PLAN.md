# Optional Refinements Plan - 4 Parallel Codex Sessions

## Overview

All 5 core modules are complete and validated. This plan divides optional refinements into parallel work for **4 separate Codex terminal sessions** to improve quality, coverage, and integration.

**Note**: "Agent 1-4" refers to 4 separate terminal sessions, each with a Codex `/init` command. Each session will work on a different part of the plan in parallel.

---

## Session Assignments

### Session 1 (Codex Terminal #1): Subject Extraction & Canonicalization Enhancement
**Goal**: Reach 50%+ duplicate reduction by enriching Subject column

### Session 2 (Codex Terminal #2): Filter Integration & Auto-Promotion Wiring
**Goal**: Wire filtered outputs (tiered/salient) into BN and auto-promotion pipeline

### Session 3 (Codex Terminal #3): Canon Coverage Recovery
**Goal**: Recover 6 unmatched ChatGPT canon facts via synonym expansion and manual anchors

### Session 4 (Codex Terminal #4): BN Enrichment & Counterfactual Fidelity
**Goal**: Rebuild BN from richer fact set for higher-fidelity counterfactuals

---

## Session 1: Subject Extraction & Canonicalization Enhancement

### Current Status
- ✅ Canonicalizer implemented and integrated
- ✅ Location duplicates: 25% reduction (4→3)
- ❌ Subject column empty → bilingual-name canonicalization can't engage
- ❌ Duplicate reduction < 50% target

### Objective
Enrich Subject column during extraction so canonicalization can merge real entity names and reach 50%+ duplicate reduction.

### Tasks

#### Task 1.1: Analyze Current Subject Population
**Files**: `case_law_data/facts_truth_table_v3_canonicalized.csv`
- Count facts with non-empty Subject values
- Identify patterns in SourceExcerpt that could yield subjects
- Document Subject population rate (target: >50% populated)

**Output**: Analysis report in `reports/analysis_outputs/subject_population_analysis.md`

#### Task 1.2: Enhance Subject Extraction in convert_to_truth_table.py
**File**: `writer_agents/scripts/convert_to_truth_table.py`

**Enhancements**:
1. Parse SourceExcerpt for email headers:
   - Extract "From:" patterns → populate Subject
   - Extract "To:" patterns → populate Object (if missing)
   - Handle email signature patterns

2. Extract subject from sentence structure:
   - Use S-V-O patterns from existing templates
   - Leverage NER output to identify subject entities
   - Parse communication templates (e.g., "X said Y" → Subject = X)

3. Use templates to infer subjects:
   - Communication templates: extract speaker/actor
   - Legal templates: extract plaintiff/defendant
   - Risk templates: extract affected party

**Code Location**: 
- Function: `build_fragment_proposition()` (around line 270)
- Function: `apply_templates()` (around line 112)
- Add new function: `extract_subject_from_excerpt(excerpt: str) -> str`

#### Task 1.3: Expand NAME_ALIASES in entity_canonicalizer.py
**File**: `fact_engine/entity_canonicalizer.py`

**Enhancements**:
1. Add more bilingual name variants:
   - Chinese/English name pairs (e.g., "Wang Wei" ↔ "王伟")
   - Common transliterations
   - Abbreviations and nicknames

2. Add role-based aliases:
   - "Vice President" → infer person name from context
   - "Harvard OGC" → canonicalize to "Harvard Office of General Counsel"
   - Organization abbreviations → full names

3. Add fuzzy matching for name variations:
   - Handle typos and spelling variations
   - Handle title variations (Dr., Mr., Ms., etc.)

**Code Location**: 
- Variable: `NAME_ALIASES` (around line 15-50)
- Function: `_canonicalize_name()` (around line 80-120)

#### Task 1.4: Add Automated Reduction Metric
**File**: `fact_engine/entity_canonicalizer.py` or new validation script

**Enhancements**:
1. Create JSON summary of canonicalization results:
   - Before/after entity counts
   - Duplicate reduction percentage
   - Merged entity examples
   - Subject population rate

2. Add validation failure if reduction < 50%:
   - Raise warning/error if target not met
   - Log detailed metrics for debugging

**Output**: `reports/analysis_outputs/canonicalization_metrics.json`

#### Task 1.5: Re-run Validation
1. Regenerate truth table with enriched Subject:
   ```bash
   python writer_agents/scripts/convert_to_truth_table.py \
     --input case_law_data/facts_ranked_for_sealing.csv \
     --output case_law_data/facts_truth_table_v4_enriched.csv
   ```

2. Re-measure duplicate reduction:
   - Compare entity counts before/after canonicalization
   - Calculate reduction percentage
   - Target: 50%+ reduction

3. Update validation report:
   - Document Subject enrichment approach
   - Show before/after Subject population
   - Re-measure duplicate reduction with enriched data

**Output**: Updated `reports/analysis_outputs/canonicalization_validation.md`

### Success Criteria
- ✅ Subject column populated for >50% of facts
- ✅ Bilingual-name canonicalization engages
- ✅ 50%+ duplicate reduction achieved
- ✅ Automated metrics JSON generated
- ✅ Updated validation report

### Dependencies
- None (can work in parallel)

### Estimated Time
2-3 hours

---

## Session 2: Filter Integration & Auto-Promotion Wiring

### Current Status
- ✅ Confidence filter outputs `facts_truth_table_tiered.csv`
- ✅ Salience filter outputs `facts_truth_table_salient.csv`
- ❌ Filtered outputs not wired into BN construction
- ❌ Auto-promotion doesn't use filtered facts

### Objective
Wire filtered outputs (tiered/salient) into BN construction and auto-promotion pipeline so downstream systems use high-quality, causally relevant facts.

### Tasks

#### Task 2.1: Update BN Construction to Use Filtered Facts
**File**: `writer_agents/code/BuildBnStructureFromKg.py`

**Enhancements**:
1. Add CLI flag to select fact source:
   - `--facts-source {raw|tiered|salient}` (default: raw)
   - Load from appropriate CSV based on flag

2. Update fact loading logic:
   - If `--facts-source tiered`: load `facts_truth_table_tiered.csv`
   - If `--facts-source salient`: load `facts_truth_table_salient.csv`
   - Preserve existing raw fact loading as fallback

3. Preserve tier/salience metadata:
   - Pass ConfidenceTier and CausalSalienceScore to BN nodes
   - Use tier/salience as node attributes for visualization

**Code Location**: 
- Function: `load_facts()` or similar (around line 100-150)
- CLI argument parsing (around line 50-80)

#### Task 2.2: Update Auto-Promotion to Use Filtered Facts
**File**: `fact_engine/auto_promote.py`

**Enhancements**:
1. Add fact source selection:
   - Parameter: `fact_source: str = "salient"` (default to salient)
   - Load from `facts_truth_table_salient.csv` by default

2. Prioritize high-tier facts:
   - Prefer Tier 1 facts for promotion
   - Use salience scores to rank promotion candidates
   - Skip Tier 4 facts (already filtered out)

3. Update promotion logic:
   - Use ConfidenceTier and CausalSalienceScore in scoring
   - Weight high-tier/high-salience facts more heavily

**Code Location**: 
- Function: `auto_promote_missing_facts()` (around line 50-100)
- Function: `_load_facts_for_promotion()` (new or existing)

#### Task 2.3: Update Fact Engine CLI to Chain Filters
**File**: `fact_engine/run_fact_engine.py`

**Enhancements**:
1. Add `--use-filtered-facts` flag:
   - If set, use `facts_truth_table_salient.csv` for downstream tasks
   - Apply to BN construction, auto-promotion, counterfactuals

2. Update pipeline flow:
   - Run confidence filter → tiered CSV
   - Run salience filter → salient CSV
   - Use salient CSV for all downstream tasks (if flag set)

**Code Location**: 
- Function: `_promote_facts()` (around line 200-250)
- Function: `_build_bn_structure()` (around line 300-350)
- CLI argument parsing (around line 50-100)

#### Task 2.4: Create Integration Test
**File**: `tests_integration/test_filter_integration.py` (new)

**Test Cases**:
1. BN construction with tiered facts:
   - Verify tiered CSV loads correctly
   - Verify BN nodes include tier metadata
   - Verify node count matches tiered fact count

2. BN construction with salient facts:
   - Verify salient CSV loads correctly
   - Verify BN nodes include salience scores
   - Verify node count matches salient fact count

3. Auto-promotion with filtered facts:
   - Verify promotion uses salient facts
   - Verify Tier 1 facts prioritized
   - Verify Tier 4 facts excluded

**Output**: Integration test file

#### Task 2.5: Update Documentation
**Files**: 
- `docs/FACT_ENGINE_USAGE.md` (or create if missing)
- `fact_engine/run_fact_engine.py` (docstrings)

**Documentation**:
1. Document `--facts-source` flag usage
2. Document `--use-filtered-facts` flag usage
3. Explain when to use tiered vs salient facts
4. Provide examples of chained filter pipeline

### Success Criteria
- ✅ BN construction can use tiered/salient facts
- ✅ Auto-promotion uses filtered facts by default
- ✅ Fact engine CLI chains filters correctly
- ✅ Integration tests pass
- ✅ Documentation updated

### Dependencies
- None (can work in parallel, but benefits from Agent 1's enriched facts)

### Estimated Time
2-3 hours

---

## Session 3: Canon Coverage Recovery

### Current Status
- ✅ 96/102 ChatGPT canon facts preserved (94% coverage)
- ❌ 6 unmatched canon items need recovery
- ❌ Synonym expansion not applied to canon matching

### Objective
Recover 6 unmatched ChatGPT canon facts via synonym expansion, manual anchors, and improved matching logic.

### Tasks

#### Task 3.1: Identify 6 Unmatched Canon Items
**Files**: 
- `case_law_data/chatgpt_facts_list.csv`
- `case_law_data/facts_truth_table_salient.csv`
- `reports/analysis_outputs/salience_filter_validation.md`

**Steps**:
1. Load ChatGPT canon list (102 facts)
2. Load salient facts (3,345 facts)
3. Run matching logic to identify 6 unmatched items
4. Document each unmatched item:
   - Fact sentence from canon
   - Fact type
   - Source document
   - Why it didn't match (analysis)

**Output**: `reports/analysis_outputs/unmatched_canon_items.md`

#### Task 3.2: Expand Synonym Lists for Matching
**File**: `fact_engine/tests/utils.py` or `fact_engine/canon_guided_extraction.py`

**Enhancements**:
1. Add domain-specific synonyms:
   - Legal terms (e.g., "defamation" ↔ "libel" ↔ "slander")
   - Risk terms (e.g., "torture" ↔ "abuse" ↔ "mistreatment")
   - Organization names (e.g., "PRC" ↔ "China" ↔ "People's Republic of China")

2. Add temporal synonyms:
   - Date formats (e.g., "April 2019" ↔ "2019-04" ↔ "Spring 2019")
   - Relative time (e.g., "later" ↔ "subsequently" ↔ "afterward")

3. Add entity synonyms:
   - Person name variations
   - Organization abbreviations
   - Location variations

**Code Location**: 
- Function: `fuzzy_recovery_match()` (around line 50-150)
- Variable: `target_keywords` or similar (around line 20-40)

#### Task 3.3: Create Manual Anchors for Critical Facts
**File**: `fact_engine/canon_guided_extraction.py` or new file `fact_engine/manual_canon_anchors.py`

**Enhancements**:
1. Create manual anchor mapping:
   - Map each unmatched canon fact to a search pattern
   - Include multiple search strategies (keywords, phrases, regex)
   - Include source document hints

2. Implement anchor matching:
   - Search truth table for anchor patterns
   - Use fuzzy matching with lower threshold for anchors
   - Log matches for validation

**Code Location**: New file or extend `canon_guided_extraction.py`

**Example Format**:
```python
MANUAL_ANCHORS = [
    {
        "canon_fact": "Harvard published Statement 1 on April 19, 2019",
        "search_patterns": [
            "Harvard.*Statement.*April.*2019",
            "April 19.*2019.*Harvard",
            "Statement 1.*published"
        ],
        "source_hints": ["statement", "april", "2019"]
    },
    # ... 5 more
]
```

#### Task 3.4: Update Recovery Checker with Enhanced Matching
**File**: `fact_engine/scripts/audit_fact_recovery.py`

**Enhancements**:
1. Integrate synonym expansion:
   - Use expanded synonym lists in matching
   - Apply synonym substitution before fuzzy matching

2. Integrate manual anchors:
   - Check manual anchors for unmatched items
   - Use anchor patterns as fallback matching strategy

3. Improve matching threshold:
   - Lower threshold for manual anchors (e.g., 60% instead of 75%)
   - Use multiple matching strategies (synonym + fuzzy + anchor)

**Code Location**: 
- Function: `find_best_match()` (around line 100-200)
- Function: `categorize_fact()` (around line 50-100)

#### Task 3.5: Re-run Recovery Check and Validate
1. Run recovery checker with enhanced matching:
   ```bash
   python -m fact_engine.scripts.audit_fact_recovery \
     --canon case_law_data/chatgpt_facts_list.csv \
     --truth-table case_law_data/facts_truth_table_salient.csv \
     --output reports/analysis_outputs/recovery_check_v2.md
   ```

2. Validate recovery:
   - Target: 98/102 or better (96%+ coverage)
   - Document which items were recovered
   - Document which items remain unmatched (if any)

3. Update validation report:
   - Document synonym expansion approach
   - Document manual anchors
   - Show before/after recovery rates

**Output**: Updated `reports/analysis_outputs/salience_filter_validation.md`

### Success Criteria
- ✅ 6 unmatched items identified and documented
- ✅ Synonym lists expanded for domain terms
- ✅ Manual anchors created for critical facts
- ✅ Recovery checker uses enhanced matching
- ✅ Recovery rate improved to 98/102 or better (96%+)
- ✅ Updated validation report

### Dependencies
- None (can work in parallel)

### Estimated Time
2-3 hours

---

## Session 4: BN Enrichment & Counterfactual Fidelity

### Current Status
- ✅ BN structure generated with constraints (4 nodes, 2 edges)
- ✅ Counterfactual engine functional (deltas calculated)
- ❌ BN is small (miniature KG) → limited counterfactual fidelity
- ❌ BN built from old KG, not enriched facts

### Objective
Rebuild BN from richer fact set (canonicalized + filtered) to enable higher-fidelity counterfactuals with more nodes, edges, and realistic causal pathways.

### Tasks

#### Task 4.1: Regenerate KG from Enriched Facts
**Files**: 
- `case_law_data/facts_truth_table_v3_canonicalized.csv` (or v4 if Agent 1 completes)
- `case_law_data/facts_truth_table_salient.csv`
- KG generation scripts (likely in `writer_agents/code/` or `nlp_analysis/`)

**Steps**:
1. Identify KG generation script:
   - Search for scripts that create `entities_all.json` and `facts_cooccurrence_graph.pickle`
   - Likely in `writer_agents/code/` or `nlp_analysis/`

2. Regenerate KG from salient facts:
   - Use `facts_truth_table_salient.csv` as input
   - Include canonicalized entities (from Agent 1's work)
   - Generate new `entities_all.json` and `facts_cooccurrence_graph.pickle`

3. Validate KG quality:
   - Count entities (target: >50 entities)
   - Count co-occurrence edges (target: >100 edges)
   - Verify entity canonicalization applied

**Output**: 
- `case_law_data/entities_all_enriched.json`
- `case_law_data/facts_cooccurrence_graph_enriched.pickle`

#### Task 4.2: Rebuild BN Structure from Enriched KG
**File**: `writer_agents/code/BuildBnStructureFromKg.py`

**Steps**:
1. Update BN construction to use enriched KG:
   - Load `entities_all_enriched.json`
   - Load `facts_cooccurrence_graph_enriched.pickle`
   - Apply structural constraints (already integrated)

2. Generate new BN structure:
   ```bash
   python -m writer_agents.code.BuildBnStructureFromKg \
     --entities case_law_data/entities_all_enriched.json \
     --graph case_law_data/facts_cooccurrence_graph_enriched.pickle \
     --output case_law_data/bn_structure/bn_structure_enriched.pkl
   ```

3. Validate BN structure:
   - Count nodes (target: >20 nodes)
   - Count edges (target: >30 edges)
   - Verify constraints applied correctly
   - Verify no forbidden edges present

**Output**: 
- `case_law_data/bn_structure/bn_structure_enriched.pkl`
- Validation report: `reports/analysis_outputs/bn_enrichment_validation.md`

#### Task 4.3: Update Counterfactual Engine for Enriched BN
**File**: `fact_engine/counterfactual_engine.py`

**Enhancements**:
1. Add support for enriched BN:
   - Auto-detect enriched BN file if available
   - Fallback to original BN if enriched not found
   - Log which BN version is used

2. Improve intervention handling:
   - Support multi-node interventions
   - Support conditional interventions (if-then)
   - Support temporal interventions (before/after events)

**Code Location**: 
- Function: `load_bn_structure()` (around line 50-100)
- Function: `run_counterfactual()` (around line 150-220)

#### Task 4.4: Create Enhanced Counterfactual Scenarios
**File**: `case_law_data/counterfactual_scenarios/` (existing directory)

**New Scenarios**:
1. **Scenario 4: No Harvard Statement (April 2019)**:
   - Intervention: Harvard never published Statement 1
   - Query: Impact on PRC visibility, WeChat articles, harm to plaintiff
   - Expected: Large negative delta for PRC visibility

2. **Scenario 5: Early OGC Response**:
   - Intervention: OGC responded to emails within 24 hours
   - Query: Impact on plaintiff risk, PRC actions
   - Expected: Negative delta for plaintiff harm

3. **Scenario 6: No WeChat Articles**:
   - Intervention: WeChat articles never published
   - Query: Impact on PRC visibility, plaintiff harm
   - Expected: Negative delta for PRC visibility and harm

**Output**: 
- `case_law_data/counterfactual_scenarios/scenario_4.json`
- `case_law_data/counterfactual_scenarios/scenario_5.json`
- `case_law_data/counterfactual_scenarios/scenario_6.json`

#### Task 4.5: Run Enhanced Counterfactuals and Validate
1. Run counterfactuals with enriched BN:
   ```bash
   python -m fact_engine.run_fact_engine \
     --counterfactual case_law_data/counterfactual_scenarios/scenario_4.json \
     --bn-structure case_law_data/bn_structure/bn_structure_enriched.pkl \
     --output reports/analysis_outputs/counterfactual_scenario_4.json
   ```

2. Validate counterfactual fidelity:
   - Compare deltas: enriched BN vs original BN
   - Verify deltas are larger/more realistic (target: >0.2 for key nodes)
   - Verify no errors or fallbacks

3. Update validation report:
   - Document enriched BN structure (nodes/edges)
   - Document counterfactual results
   - Compare fidelity: enriched vs original

**Output**: 
- `reports/analysis_outputs/counterfactual_scenario_4.json`
- `reports/analysis_outputs/counterfactual_scenario_5.json`
- `reports/analysis_outputs/counterfactual_scenario_6.json`
- Updated `reports/analysis_outputs/counterfactual_validation.md`

### Success Criteria
- ✅ KG regenerated from enriched facts (>50 entities, >100 edges)
- ✅ BN structure rebuilt (>20 nodes, >30 edges)
- ✅ Counterfactual engine uses enriched BN
- ✅ 3 new counterfactual scenarios created and run
- ✅ Counterfactual deltas improved (>0.2 for key nodes)
- ✅ Updated validation report

### Dependencies
- Benefits from Agent 1's canonicalized facts (but can proceed with v3)
- Benefits from Agent 2's filtered facts (but can proceed with existing salient CSV)

### Estimated Time
3-4 hours

---

## Parallel Execution Plan

### Phase 1: Independent Work (All Sessions)
All Codex sessions can work in parallel on their independent tasks:

- **Session 1**: Subject extraction analysis + enhancement
- **Session 2**: Filter integration analysis + wiring
- **Session 3**: Unmatched canon identification + synonym expansion
- **Session 4**: KG regeneration + BN rebuilding

### Phase 2: Integration (Sequential)
After Phase 1, sessions may need to coordinate:

1. **Session 1** → **Session 4**: Enriched facts (v4) available for KG regeneration
2. **Session 2** → **Session 4**: Filtered facts integration confirmed
3. **Session 3** → **Session 2**: Recovery improvements may affect auto-promotion

### Phase 3: Validation (All Sessions)
All sessions validate their work independently:

- **Session 1**: Re-run canonicalization validation
- **Session 2**: Run integration tests
- **Session 3**: Re-run recovery check
- **Session 4**: Run enhanced counterfactuals

---

## Success Metrics

### Session 1
- Subject population: >50% of facts
- Duplicate reduction: 50%+ (from current 25%)
- Automated metrics JSON generated

### Session 2
- BN construction uses filtered facts
- Auto-promotion uses filtered facts
- Integration tests pass

### Session 3
- Recovery rate: 98/102 or better (96%+)
- 6 unmatched items recovered or documented
- Enhanced matching logic implemented

### Session 4
- Enriched BN: >20 nodes, >30 edges
- Counterfactual deltas: >0.2 for key nodes
- 3 new scenarios created and validated

---

## Files to Create/Modify

### Session 1
- Modify: `writer_agents/scripts/convert_to_truth_table.py`
- Modify: `fact_engine/entity_canonicalizer.py`
- Create: `reports/analysis_outputs/subject_population_analysis.md`
- Create: `reports/analysis_outputs/canonicalization_metrics.json`
- Update: `reports/analysis_outputs/canonicalization_validation.md`

### Session 2
- Modify: `writer_agents/code/BuildBnStructureFromKg.py`
- Modify: `fact_engine/auto_promote.py`
- Modify: `fact_engine/run_fact_engine.py`
- Create: `tests_integration/test_filter_integration.py`
- Create/Update: `docs/FACT_ENGINE_USAGE.md`

### Session 3
- Modify: `fact_engine/tests/utils.py` or `fact_engine/canon_guided_extraction.py`
- Create: `fact_engine/manual_canon_anchors.py` (optional)
- Modify: `fact_engine/scripts/audit_fact_recovery.py`
- Create: `reports/analysis_outputs/unmatched_canon_items.md`
- Update: `reports/analysis_outputs/salience_filter_validation.md`

### Session 4
- Modify: KG generation scripts (TBD location)
- Modify: `writer_agents/code/BuildBnStructureFromKg.py`
- Modify: `fact_engine/counterfactual_engine.py`
- Create: `case_law_data/counterfactual_scenarios/scenario_4.json`
- Create: `case_law_data/counterfactual_scenarios/scenario_5.json`
- Create: `case_law_data/counterfactual_scenarios/scenario_6.json`
- Create: `reports/analysis_outputs/bn_enrichment_validation.md`
- Update: `reports/analysis_outputs/counterfactual_validation.md`

---

## Timeline Estimate

- **Phase 1 (Independent)**: 2-4 hours (parallel)
- **Phase 2 (Integration)**: 1-2 hours (sequential)
- **Phase 3 (Validation)**: 1-2 hours (parallel)

**Total**: 4-8 hours (depending on agent speed and complexity)

---

## Notes

- All Codex sessions can start immediately (no blocking dependencies)
- Session 4 can proceed with existing enriched facts (v3) if Session 1 not complete
- Session 2 can proceed with existing filtered CSVs
- Session 3 is fully independent
- Coordinate in Phase 2 if integration needed

## How to Use This Plan

1. Open 4 separate terminal sessions (or Codex windows)
2. In each session, run `/init` with the appropriate focus:
   - Session 1: `/init` then paste instructions from `OPTIONAL_REFINEMENTS_INSTRUCTIONS.md` section "FOR AGENT 1"
   - Session 2: `/init` then paste instructions from `OPTIONAL_REFINEMENTS_INSTRUCTIONS.md` section "FOR AGENT 2"
   - Session 3: `/init` then paste instructions from `OPTIONAL_REFINEMENTS_INSTRUCTIONS.md` section "FOR AGENT 3"
   - Session 4: `/init` then paste instructions from `OPTIONAL_REFINEMENTS_INSTRUCTIONS.md` section "FOR AGENT 4"
3. Each session works independently in parallel
4. Report back when each session completes their tasks

---

## Last Updated
2025-01-XX - Plan created for optional refinements

