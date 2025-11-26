# Agent Instructions: Validation & Testing Phase

## Status Update

**All 4 agents have completed their implementations!** ✅

- Agent 1: Entity Canonicalization ✅
- Agent 2: Confidence Filter ✅  
- Agent 3: Causal Salience Filter ✅
- Agent 4: BN Constraints + Counterfactual Engine ✅

---

## Next Phase: Validation & Integration Testing

All modules are implemented and integrated. We need to validate they work end-to-end and measure success metrics.

---

## Available Work: Validation Tasks

### Task 1: Regenerate Truth Table with Canonicalization
**Assigned to**: Any available agent  
**Priority**: High  
**Estimated Time**: 15-30 minutes

**Objective**: Run the updated pipeline to generate truth table with canonicalization and measure duplicate reduction.

**Steps**:
1. Ensure database has facts: `case_law_data/lawsuit_facts_database.db`
2. Run truth table conversion with canonicalization:
   ```bash
   python writer_agents/scripts/convert_to_truth_table.py \
     --input case_law_data/lawsuit_facts_database.db \
     --output case_law_data/facts_truth_table_v3_canonicalized.csv
   ```
3. Analyze results:
   - Count unique `CanonicalSubject` values vs. raw `Subject` values
   - Count unique `CanonicalSpeaker` values vs. raw `Speaker` values
   - Calculate duplicate reduction percentage
   - **Target**: 50%+ reduction in duplicate entity nodes
4. Compare with previous truth table (`facts_truth_table_v2.csv`) to show improvement
5. Document findings in `reports/analysis_outputs/canonicalization_validation.md`

**Success Criteria**:
- Truth table generated successfully
- Canonical* columns populated
- 50%+ reduction in duplicate entities measured
- Report created with metrics

---

### Task 2: Run Fact Engine with Filters
**Assigned to**: Any available agent  
**Priority**: High  
**Estimated Time**: 20-40 minutes

**Objective**: Run fact engine through confidence and salience filters to generate tiered/salient CSVs.

**Steps**:
1. Ensure truth table exists (from Task 1 or use existing `facts_truth_table_v2.csv`)
2. Run fact engine rebuild:
   ```bash
   python -m fact_engine.run_fact_engine --rebuild --verbose
   ```
3. Verify outputs created:
   - `case_law_data/facts_truth_table_tiered.csv` (confidence filter output)
   - `case_law_data/facts_truth_table_salient.csv` (salience filter output)
4. Analyze tier distribution:
   - Count facts per tier (Tier 1, 2, 3, 4)
   - Verify Tier 4 facts are removed
   - Calculate percentage reduction at each stage
5. Analyze salience scores:
   - Count facts above/below salience threshold
   - Calculate 30-50% reduction target
   - Verify critical facts (from `chatgpt_facts_list.csv`) are preserved
6. Document findings in `reports/analysis_outputs/filter_validation.md`

**Success Criteria**:
- Tiered CSV generated with 4-tier distribution
- Salient CSV generated with filtered facts
- Tier 4 facts removed
- 30-50% reduction in salience filter (while preserving critical facts)
- Report created with metrics

---

### Task 3: Generate BN Structure with Constraints
**Assigned to**: Any available agent  
**Priority**: Medium  
**Estimated Time**: 30-60 minutes

**Objective**: Build BN structure with structural constraints applied and validate constraint compliance.

**Steps**:
1. Ensure entities and knowledge graph exist:
   - `case_law_data/entities.json` (or equivalent)
   - `case_law_data/facts_knowledge_graph.json`
2. Run BN structure builder:
   ```bash
   python writer_agents/code/BuildBnStructureFromKg.py \
     --entities case_law_data/entities.json \
     --graph case_law_data/facts_knowledge_graph.json \
     --output case_law_data/bn_structure.pkl
   ```
3. Verify constraints applied:
   - Check that required edges exist
   - Check that forbidden edges are absent
   - Validate domain-specific rules are followed
4. Analyze structure:
   - Count nodes and edges
   - Verify acyclic (no cycles)
   - Check parent-cap handling
5. Document findings in `reports/analysis_outputs/bn_constraints_validation.md`

**Success Criteria**:
- BN structure generated (`bn_structure.pkl`)
- Constraints validated (required edges present, forbidden edges absent)
- Structure is acyclic and valid
- Report created with validation results

---

### Task 4: Create Counterfactual Scenarios & Run Queries
**Assigned to**: Any available agent  
**Priority**: Medium  
**Estimated Time**: 30-45 minutes

**Objective**: Create scenario JSON files and run counterfactual queries to demonstrate functionality.

**Steps**:
1. Ensure BN structure exists (from Task 3 or use existing)
2. Create scenario JSON files in `case_law_data/counterfactual_scenarios/`:
   - `scenario_1_no_april_2019_statement.json`:
     ```json
     {
       "description": "What if Harvard never sent the 19 April 2019 clarification?",
       "interventions": [
         {"node": "COR_Harvard_2019_04_19", "value": false}
       ],
       "targets": ["HARM_Security", "HARM_ImmigrationRisk", "HUB_Harm"]
     }
     ```
   - `scenario_2_no_wechat_articles.json`:
     ```json
     {
       "description": "What if WeChat articles were never published?",
       "interventions": [
         {"node": "ART_Monkey_WeChat", "value": false},
         {"node": "ART_Resume_WeChat", "value": false}
       ],
       "targets": ["HARM_Security", "PRC_Visibility"]
     }
     ```
   - `scenario_3_ogc_responded.json`:
     ```json
     {
       "description": "What if OGC had responded to communications?",
       "interventions": [
         {"node": "COR_OGC_Response", "value": true}
       ],
       "targets": ["HARM_Security", "HUB_Lawsuit"]
     }
     ```
3. Run counterfactual queries:
   ```bash
   python -m fact_engine.run_fact_engine \
     --counterfactual case_law_data/counterfactual_scenarios/scenario_1_no_april_2019_statement.json \
     --bn-structure case_law_data/bn_structure.pkl
   ```
4. Verify outputs:
   - Check `reports/analysis_outputs/counterfactual_*.json` files created
   - Verify probability deltas are calculated
   - **Target**: Deltas >0.1 for meaningful counterfactuals
5. Document findings in `reports/analysis_outputs/counterfactual_validation.md`

**Success Criteria**:
- 3+ scenario JSON files created
- Counterfactual queries run successfully
- Probability deltas calculated (>0.1 for meaningful scenarios)
- Reports generated in `reports/analysis_outputs/`
- Report created with validation results

---

### Task 5: End-to-End Integration Test
**Assigned to**: Any available agent  
**Priority**: High  
**Estimated Time**: 45-60 minutes

**Objective**: Run complete pipeline end-to-end and validate all modules work together.

**Steps**:
1. Start from raw database: `case_law_data/lawsuit_facts_database.db`
2. Run complete pipeline:
   ```bash
   # Step 1: Convert with canonicalization
   python writer_agents/scripts/convert_to_truth_table.py \
     --input case_law_data/lawsuit_facts_database.db \
     --output case_law_data/facts_truth_table_v3_canonicalized.csv
   
   # Step 2: Run fact engine with filters
   python -m fact_engine.run_fact_engine --rebuild --verbose
   
   # Step 3: Build BN with constraints
   python writer_agents/code/BuildBnStructureFromKg.py \
     --entities case_law_data/entities.json \
     --graph case_law_data/facts_knowledge_graph.json \
     --output case_law_data/bn_structure.pkl
   
   # Step 4: Run counterfactual query
   python -m fact_engine.run_fact_engine \
     --counterfactual case_law_data/counterfactual_scenarios/scenario_1_no_april_2019_statement.json \
     --bn-structure case_law_data/bn_structure.pkl
   ```
3. Validate all outputs:
   - Truth table with canonical columns
   - Tiered CSV (4 tiers)
   - Salient CSV (filtered)
   - BN structure (constrained)
   - Counterfactual report
4. Measure all success metrics:
   - Entity duplicate reduction: 50%+ ✅/❌
   - 4-tier confidence output: ✅/❌
   - 30-50% causal pruning: ✅/❌
   - Constraint-compliant BN: ✅/❌
   - >0.1 counterfactual deltas: ✅/❌
5. Create comprehensive validation report: `reports/analysis_outputs/end_to_end_validation.md`

**Success Criteria**:
- Complete pipeline runs without errors
- All outputs generated successfully
- All success metrics measured and documented
- Comprehensive validation report created

---

## Priority Order

1. **Task 1** (Truth table with canonicalization) - Foundation for everything
2. **Task 2** (Fact engine filters) - Validates core filtering logic
3. **Task 5** (End-to-end test) - Validates integration
4. **Task 3** (BN structure) - Needed for counterfactuals
5. **Task 4** (Counterfactual scenarios) - Final validation

---

## Notes

- All tasks can be done in parallel by different agents
- Tasks 1 and 2 should be done first (foundation)
- Task 5 should be done last (requires all others)
- Document all findings in `reports/analysis_outputs/`

---

**Last Updated**: 2025-01-XX - Validation phase instructions

