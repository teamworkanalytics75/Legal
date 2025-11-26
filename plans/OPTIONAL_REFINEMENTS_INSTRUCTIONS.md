# Optional Refinements - Codex Session Instructions

**How to Use**: Open 4 separate Codex terminal sessions. In each session, run `/init` then copy and paste the entire section for that session number below.

**Note**: "Agent 1-4" refers to 4 separate Codex terminal sessions, not agents within the codebase.

---

## FOR SESSION 1 (Codex Terminal #1)

```
You are working on Session 1: Subject Extraction & Canonicalization Enhancement

Copy this entire block and paste it into a Codex terminal session after running /init

STATUS UPDATE:
✅ Your core validation is COMPLETE!
- Canonicalizer implemented and integrated
- Location duplicates: 25% reduction (4→3)
- Gap documented: Subject column empty → bilingual-name canonicalization can't engage
- Duplicate reduction < 50% target

YOUR OPTIONAL TASK: Improve Subject Extraction to Reach 50%+ Duplicate Reduction

This is OPTIONAL work to improve the canonicalization metric. Your core work is complete.

Objective: Enrich Subject column during extraction so canonicalization can merge real entity names and reach 50%+ duplicate reduction.

Steps:

1. Analyze Current Subject Population:
   - Load: case_law_data/facts_truth_table_v3_canonicalized.csv
   - Count facts with non-empty Subject values
   - Identify patterns in SourceExcerpt that could yield subjects
   - Document Subject population rate (target: >50% populated)
   - Output: reports/analysis_outputs/subject_population_analysis.md

2. Enhance Subject Extraction in convert_to_truth_table.py:
   - File: writer_agents/scripts/convert_to_truth_table.py
   - Parse SourceExcerpt for email headers ("From:" → Subject, "To:" → Object)
   - Extract subject from sentence structure (S-V-O patterns, NER output)
   - Use templates to infer subjects (communication templates → speaker, legal templates → plaintiff/defendant)
   - Add new function: extract_subject_from_excerpt(excerpt: str) -> str
   - Code locations: build_fragment_proposition() (~line 270), apply_templates() (~line 112)

3. Expand NAME_ALIASES in entity_canonicalizer.py:
   - File: fact_engine/entity_canonicalizer.py
   - Add more bilingual name variants (Chinese/English pairs, transliterations)
   - Add role-based aliases ("Vice President" → person name, "Harvard OGC" → full name)
   - Add fuzzy matching for name variations (typos, title variations)
   - Code locations: NAME_ALIASES (~line 15-50), _canonicalize_name() (~line 80-120)

4. Add Automated Reduction Metric:
   - Create JSON summary of canonicalization results (before/after entity counts, duplicate reduction %, merged entity examples, Subject population rate)
   - Add validation failure if reduction < 50% (raise warning/error, log detailed metrics)
   - Output: reports/analysis_outputs/canonicalization_metrics.json

5. Re-run Validation:
   - Regenerate truth table with enriched Subject:
     python writer_agents/scripts/convert_to_truth_table.py \
       --input case_law_data/facts_ranked_for_sealing.csv \
       --output case_law_data/facts_truth_table_v4_enriched.csv
   - Re-measure duplicate reduction (compare entity counts, calculate %, target: 50%+)
   - Update validation report: reports/analysis_outputs/canonicalization_validation.md

Success Criteria:
- ✅ Subject column populated for >50% of facts
- ✅ Bilingual-name canonicalization engages
- ✅ 50%+ duplicate reduction achieved
- ✅ Automated metrics JSON generated
- ✅ Updated validation report

Report back with:
- Subject population before/after enrichment
- Duplicate reduction percentage (target: 50%+)
- Examples of merged entities
- Link to updated validation report
```

---

## FOR SESSION 2 (Codex Terminal #2)

```
You are working on Session 2: Filter Integration & Auto-Promotion Wiring

Copy this entire block and paste it into a Codex terminal session after running /init

STATUS UPDATE:
✅ Your core validation is COMPLETE!
- Confidence filter outputs facts_truth_table_tiered.csv
- Salience filter outputs facts_truth_table_salient.csv
- Filter stack validated end-to-end
- Gap: Filtered outputs not wired into BN construction and auto-promotion

YOUR OPTIONAL TASK: Wire Filtered Outputs into BN and Auto-Promotion Pipeline

This is OPTIONAL work to improve integration. Your core work is complete.

Objective: Wire filtered outputs (tiered/salient) into BN construction and auto-promotion pipeline so downstream systems use high-quality, causally relevant facts.

Steps:

1. Update BN Construction to Use Filtered Facts:
   - File: writer_agents/code/BuildBnStructureFromKg.py
   - Add CLI flag: --facts-source {raw|tiered|salient} (default: raw)
   - Update fact loading logic to load from appropriate CSV based on flag
   - Preserve tier/salience metadata (ConfidenceTier, CausalSalienceScore) as BN node attributes
   - Code locations: load_facts() (~line 100-150), CLI parsing (~line 50-80)

2. Update Auto-Promotion to Use Filtered Facts:
   - File: fact_engine/auto_promote.py
   - Add parameter: fact_source: str = "salient" (default to salient)
   - Load from facts_truth_table_salient.csv by default
   - Prioritize high-tier facts (prefer Tier 1, use salience scores to rank)
   - Skip Tier 4 facts (already filtered out)
   - Use ConfidenceTier and CausalSalienceScore in promotion scoring
   - Code locations: auto_promote_missing_facts() (~line 50-100), add _load_facts_for_promotion() if needed

3. Update Fact Engine CLI to Chain Filters:
   - File: fact_engine/run_fact_engine.py
   - Add --use-filtered-facts flag (if set, use facts_truth_table_salient.csv for downstream tasks)
   - Update pipeline flow: confidence filter → tiered CSV → salience filter → salient CSV → use salient CSV for all downstream tasks (if flag set)
   - Code locations: _promote_facts() (~line 200-250), _build_bn_structure() (~line 300-350), CLI parsing (~line 50-100)

4. Create Integration Test:
   - File: tests_integration/test_filter_integration.py (new)
   - Test BN construction with tiered facts (verify tiered CSV loads, BN nodes include tier metadata, node count matches)
   - Test BN construction with salient facts (verify salient CSV loads, BN nodes include salience scores, node count matches)
   - Test auto-promotion with filtered facts (verify promotion uses salient facts, Tier 1 prioritized, Tier 4 excluded)

5. Update Documentation:
   - Files: docs/FACT_ENGINE_USAGE.md (create if missing), fact_engine/run_fact_engine.py (docstrings)
   - Document --facts-source flag usage
   - Document --use-filtered-facts flag usage
   - Explain when to use tiered vs salient facts
   - Provide examples of chained filter pipeline

Success Criteria:
- ✅ BN construction can use tiered/salient facts
- ✅ Auto-promotion uses filtered facts by default
- ✅ Fact engine CLI chains filters correctly
- ✅ Integration tests pass
- ✅ Documentation updated

Report back with:
- BN construction test results (tiered/salient)
- Auto-promotion test results
- Integration test results
- Links to updated documentation
```

---

## FOR SESSION 3 (Codex Terminal #3)

```
You are working on Session 3: Canon Coverage Recovery

Copy this entire block and paste it into a Codex terminal session after running /init

STATUS UPDATE:
✅ Your core validation is COMPLETE!
- Salience filter validated: 39.6% reduction, 96/102 facts preserved (94% coverage)
- All pathways present
- Gap: 6 unmatched ChatGPT canon facts need recovery

YOUR OPTIONAL TASK: Recover 6 Unmatched Canon Facts

This is OPTIONAL work to improve coverage. Your core work is complete.

Objective: Recover 6 unmatched ChatGPT canon facts via synonym expansion, manual anchors, and improved matching logic.

Steps:

1. Identify 6 Unmatched Canon Items:
   - Load: case_law_data/chatgpt_facts_list.csv (102 facts)
   - Load: case_law_data/facts_truth_table_salient.csv (3,345 facts)
   - Run matching logic to identify 6 unmatched items
   - Document each unmatched item (fact sentence, fact type, source document, why it didn't match)
   - Output: reports/analysis_outputs/unmatched_canon_items.md

2. Expand Synonym Lists for Matching:
   - File: fact_engine/tests/utils.py or fact_engine/canon_guided_extraction.py
   - Add domain-specific synonyms (legal terms: "defamation" ↔ "libel" ↔ "slander", risk terms: "torture" ↔ "abuse", org names: "PRC" ↔ "China")
   - Add temporal synonyms (date formats: "April 2019" ↔ "2019-04", relative time: "later" ↔ "subsequently")
   - Add entity synonyms (person name variations, org abbreviations, location variations)
   - Code locations: fuzzy_recovery_match() (~line 50-150), target_keywords (~line 20-40)

3. Create Manual Anchors for Critical Facts:
   - File: fact_engine/canon_guided_extraction.py or new file fact_engine/manual_canon_anchors.py
   - Create manual anchor mapping (map each unmatched canon fact to search patterns, include multiple strategies, include source document hints)
   - Implement anchor matching (search truth table for anchor patterns, use fuzzy matching with lower threshold, log matches)
   - Example format: MANUAL_ANCHORS = [{"canon_fact": "...", "search_patterns": [...], "source_hints": [...]}, ...]

4. Update Recovery Checker with Enhanced Matching:
   - File: fact_engine/scripts/audit_fact_recovery.py
   - Integrate synonym expansion (use expanded synonym lists, apply synonym substitution before fuzzy matching)
   - Integrate manual anchors (check manual anchors for unmatched items, use anchor patterns as fallback)
   - Improve matching threshold (lower threshold for manual anchors: 60% instead of 75%, use multiple strategies)
   - Code locations: find_best_match() (~line 100-200), categorize_fact() (~line 50-100)

5. Re-run Recovery Check and Validate:
   - Run recovery checker:
     python -m fact_engine.scripts.audit_fact_recovery \
       --canon case_law_data/chatgpt_facts_list.csv \
       --truth-table case_law_data/facts_truth_table_salient.csv \
       --output reports/analysis_outputs/recovery_check_v2.md
   - Validate recovery (target: 98/102 or better, 96%+ coverage)
   - Document which items were recovered, which remain unmatched (if any)
   - Update validation report: reports/analysis_outputs/salience_filter_validation.md

Success Criteria:
- ✅ 6 unmatched items identified and documented
- ✅ Synonym lists expanded for domain terms
- ✅ Manual anchors created for critical facts
- ✅ Recovery checker uses enhanced matching
- ✅ Recovery rate improved to 98/102 or better (96%+)
- ✅ Updated validation report

Report back with:
- List of 6 unmatched items (with analysis)
- Synonym expansion details
- Manual anchors created
- Recovery rate before/after (target: 98/102)
- Link to updated validation report
```

---

## FOR SESSION 4 (Codex Terminal #4)

```
You are working on Session 4: BN Enrichment & Counterfactual Fidelity

Copy this entire block and paste it into a Codex terminal session after running /init

STATUS UPDATE:
✅ Your core validation is COMPLETE!
- BN structure generated with constraints (4 nodes, 2 edges)
- Counterfactual engine functional (deltas calculated, e.g., ΔUnited = +0.61)
- Gap: BN is small (miniature KG) → limited counterfactual fidelity

YOUR OPTIONAL TASK: Rebuild BN from Richer Fact Set

This is OPTIONAL work to improve counterfactual fidelity. Your core work is complete.

Objective: Rebuild BN from richer fact set (canonicalized + filtered) to enable higher-fidelity counterfactuals with more nodes, edges, and realistic causal pathways.

Steps:

1. Regenerate KG from Enriched Facts:
   - Identify KG generation script (search for scripts creating entities_all.json and facts_cooccurrence_graph.pickle, likely in writer_agents/code/ or nlp_analysis/)
   - Use facts_truth_table_salient.csv as input (or v4 if Agent 1 completes)
   - Include canonicalized entities (from Agent 1's work)
   - Generate new entities_all_enriched.json and facts_cooccurrence_graph_enriched.pickle
   - Validate KG quality (target: >50 entities, >100 edges, verify canonicalization applied)
   - Output: case_law_data/entities_all_enriched.json, case_law_data/facts_cooccurrence_graph_enriched.pickle

2. Rebuild BN Structure from Enriched KG:
   - File: writer_agents/code/BuildBnStructureFromKg.py
   - Update BN construction to use enriched KG (load entities_all_enriched.json, load facts_cooccurrence_graph_enriched.pickle, apply structural constraints)
   - Generate new BN structure:
     python -m writer_agents.code.BuildBnStructureFromKg \
       --entities case_law_data/entities_all_enriched.json \
       --graph case_law_data/facts_cooccurrence_graph_enriched.pickle \
       --output case_law_data/bn_structure/bn_structure_enriched.pkl
   - Validate BN structure (target: >20 nodes, >30 edges, verify constraints applied, verify no forbidden edges)
   - Output: case_law_data/bn_structure/bn_structure_enriched.pkl, reports/analysis_outputs/bn_enrichment_validation.md

3. Update Counterfactual Engine for Enriched BN:
   - File: fact_engine/counterfactual_engine.py
   - Add support for enriched BN (auto-detect enriched BN file if available, fallback to original if not found, log which version used)
   - Improve intervention handling (support multi-node interventions, conditional interventions if-then, temporal interventions before/after)
   - Code locations: load_bn_structure() (~line 50-100), run_counterfactual() (~line 150-220)

4. Create Enhanced Counterfactual Scenarios:
   - Directory: case_law_data/counterfactual_scenarios/
   - Scenario 4: No Harvard Statement (April 2019) - Intervention: Harvard never published Statement 1, Query: Impact on PRC visibility/WeChat/harm, Expected: Large negative delta for PRC visibility
   - Scenario 5: Early OGC Response - Intervention: OGC responded within 24 hours, Query: Impact on plaintiff risk/PRC actions, Expected: Negative delta for plaintiff harm
   - Scenario 6: No WeChat Articles - Intervention: WeChat articles never published, Query: Impact on PRC visibility/plaintiff harm, Expected: Negative delta for PRC visibility and harm
   - Output: scenario_4.json, scenario_5.json, scenario_6.json

5. Run Enhanced Counterfactuals and Validate:
   - Run counterfactuals:
     python -m fact_engine.run_fact_engine \
       --counterfactual case_law_data/counterfactual_scenarios/scenario_4.json \
       --bn-structure case_law_data/bn_structure/bn_structure_enriched.pkl \
       --output reports/analysis_outputs/counterfactual_scenario_4.json
   - Validate counterfactual fidelity (compare deltas: enriched vs original, verify deltas larger/more realistic, target: >0.2 for key nodes, verify no errors)
   - Update validation report (document enriched BN structure, document counterfactual results, compare fidelity)
   - Output: counterfactual_scenario_4.json, counterfactual_scenario_5.json, counterfactual_scenario_6.json, updated counterfactual_validation.md

Success Criteria:
- ✅ KG regenerated from enriched facts (>50 entities, >100 edges)
- ✅ BN structure rebuilt (>20 nodes, >30 edges)
- ✅ Counterfactual engine uses enriched BN
- ✅ 3 new counterfactual scenarios created and run
- ✅ Counterfactual deltas improved (>0.2 for key nodes)
- ✅ Updated validation report

Report back with:
- Enriched KG metrics (entities, edges)
- Enriched BN metrics (nodes, edges)
- Counterfactual deltas (enriched vs original)
- Links to new scenario JSONs
- Link to updated validation report
```

---

## Status Summary for All Sessions

**Current Phase**: OPTIONAL REFINEMENTS

All core modules have completed validation. These are optional improvements to enhance quality, coverage, and integration.

**Parallel Execution**: All Codex sessions can work independently in Phase 1. Coordinate in Phase 2 if integration needed.

**Timeline**: 4-8 hours total (2-4 hours parallel, 1-2 hours integration, 1-2 hours validation)

## How to Use

1. Open 4 separate terminal/Codex sessions
2. In each session, run `/init`
3. Copy the appropriate section above (FOR SESSION 1, 2, 3, or 4) and paste it into that session
4. Each session works independently in parallel
5. Report back when each session completes their tasks

---

## Last Updated
2025-01-XX - Optional refinements plan created

