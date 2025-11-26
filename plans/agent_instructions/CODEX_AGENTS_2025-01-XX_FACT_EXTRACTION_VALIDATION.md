# Codex Agents Plan — Fact Extraction Validation & Enhancement
**Date:** 2025-01-XX  
**Goal:** 4 Codex agents working in parallel to validate, test, and enhance the fact extraction pipeline

---

## 🎯 Context

**Recent Work:**
- ✅ Fixed OGC acknowledgment bug (OGC never responded - now correctly extracted)
- ✅ Added communication fact extraction (who said what, when, to whom)
- ✅ Implemented fuzzy file matching, multi-format support, relaxed filtering
- ✅ Enhanced fragment-to-proposition conversion
- ✅ Added micro-templates for legal/risk terms

**Current State:**
- Fact extraction pipeline: Working
- Communication templates: Added (9 new templates)
- Canon-guided extraction: Integrated
- **PENDING:** Unit tests, validation, pipeline re-run with new fixes

---

## 🔄 Agent Workstreams (Parallel, No Conflicts)

### **Agent 1: Unit Tests & Validation**
**Focus:** Test all recent fixes with comprehensive unit tests  
**Files to Create/Modify:**
- `tests/test_fact_extraction_fuzzy_matching.py` (new)
- `tests/test_fact_extraction_file_types.py` (new)
- `tests/test_fact_extraction_communication.py` (new)
- `tests/test_fact_extraction_filtering.py` (new)
- `tests/test_path_normalization.py` (new)

**Tasks:**
1. Test fuzzy file matching with various filename variations
2. Test multi-format file support (.txt, .md, .docx, .html, .pdf)
3. Test communication fact extraction (email headers, statements, non-responses)
4. Test relaxed filtering for legal/risk terms
5. Test path normalization across platforms
6. Test OGC non-response extraction (critical: should NOT extract false acknowledgments)

**Success Criteria:**
- All tests pass
- Edge cases covered (Unicode, special chars, Windows/Linux paths)
- OGC acknowledgment test verifies NO false positives

---

### **Agent 2: Pipeline Re-Run & Validation**
**Focus:** Re-run entire pipeline with new fixes and validate results  
**Files to Modify:**
- `writer_agents/scripts/extract_facts_ml_enhanced.py` (verify fixes)
- `writer_agents/scripts/convert_to_truth_table.py` (verify fixes)
- Create: `scripts/validate_fact_extraction_pipeline.py` (new validation script)

**Tasks:**
1. Re-run fact extraction on all source documents
2. Verify fuzzy matching links facts to correct files
3. Verify multi-format files are processed
4. Verify communication facts are extracted correctly
5. Verify OGC non-response facts (not false acknowledgments)
6. Generate validation report comparing before/after
7. Check fact count increase (should be 2,000+ facts)

**Success Criteria:**
- Pipeline runs without errors
- Fact count increases significantly
- All file types processed
- Communication facts present
- No false OGC acknowledgments

---

### **Agent 3: Canon-Guided Extraction Enhancement**
**Focus:** Enhance canon-guided extraction to derive MORE facts from ChatGPT list  
**Files to Modify:**
- `fact_engine/canon_guided_extraction.py` (enhance)
- `fact_engine/auto_promote.py` (verify integration)
- `fact_engine/run_fact_engine.py` (verify integration)

**Tasks:**
1. Enhance hypothesis generation from ChatGPT canon
2. Add pattern-based fact discovery (e.g., "if X said Y, search for related Z")
3. Add category-based fact expansion (e.g., "if defamation fact exists, search for related harm facts")
4. Improve communication fact derivation (who said what → search for related statements)
5. Add temporal fact derivation (if date X exists, search for events around that date)
6. Test canon-guided extraction generates 50+ new hypotheses

**Success Criteria:**
- Canon-guided extraction generates 50+ new fact hypotheses
- Hypotheses are relevant and actionable
- Integration with auto-promote works
- New facts discovered beyond ChatGPT list

---

### **Agent 4: Sealing Template & Fact Labeling**
**Focus:** Enhance sealing template and fact labeling workflow  
**Files to Modify:**
- `case_law_data/facts_labels_sealing_template.csv` (enhance)
- Create: `scripts/export_facts_for_sealing_labeling.py` (new)
- Create: `scripts/import_sealing_labels.py` (new)

**Tasks:**
1. Export all facts from truth table to sealing template format
2. Add metadata columns (fact_type, source_document, importance_score)
3. Create labeling workflow script (export → label → import)
4. Add validation for labeling (ensure all facts labeled)
5. Generate sealing recommendations based on fact types
6. Create summary report of facts requiring sealing

**Success Criteria:**
- All facts exported to sealing template
- Labeling workflow functional
- Sealing recommendations generated
- Summary report created

---

## 📋 Coordination Notes

**No Conflicts:**
- Agent 1: Creates new test files
- Agent 2: Runs pipeline (read-only on code, writes to database/CSV)
- Agent 3: Enhances existing extraction logic
- Agent 4: Works with CSV files and creates new scripts

**Shared Resources:**
- Database: `case_law_data/lawsuit_facts_database.db` (Agent 2 writes, others read)
- Truth table: `case_law_data/facts_truth_table_v2.csv` (Agent 2 writes, Agent 4 reads)
- Source documents: All agents may read (no conflicts)

**Sequencing:**
- Agents 1, 3, 4 can start immediately (parallel)
- Agent 2 should wait for Agent 1's tests to pass (or run in parallel and validate after)

---

## 🎯 Success Metrics

**Agent 1:**
- ✅ 5+ test files created
- ✅ 50+ test cases passing
- ✅ 100% coverage of recent fixes

**Agent 2:**
- ✅ Pipeline re-run successful
- ✅ Fact count: 2,000+ facts
- ✅ Validation report generated
- ✅ 0 false OGC acknowledgments

**Agent 3:**
- ✅ 50+ new hypotheses generated
- ✅ Canon-guided extraction enhanced
- ✅ New facts discovered

**Agent 4:**
- ✅ Sealing template complete
- ✅ Labeling workflow functional
- ✅ Summary report generated

---

## 🚀 Quick Start Commands

**Agent 1 (Tests):**
```bash
cd /home/serteamwork/projects/TheMatrix
python -m pytest tests/test_fact_extraction_*.py -v
```

**Agent 2 (Pipeline):**
```bash
cd /home/serteamwork/projects/TheMatrix
source .venv/bin/activate
python writer_agents/scripts/extract_facts_ml_enhanced.py \
    --source-dir case_law_data/lawsuit_source_documents \
    --database case_law_data/lawsuit_facts_database.db
python writer_agents/scripts/convert_to_truth_table.py
python scripts/validate_fact_extraction_pipeline.py
```

**Agent 3 (Canon Extraction):**
```bash
cd /home/serteamwork/projects/TheMatrix
source .venv/bin/activate
python fact_engine/canon_guided_extraction.py \
    --canon case_law_data/chatgpt_facts_list.csv \
    --output hypotheses.json
python fact_engine/run_fact_engine.py --rebuild
```

**Agent 4 (Sealing Template):**
```bash
cd /home/serteamwork/projects/TheMatrix
source .venv/bin/activate
python scripts/export_facts_for_sealing_labeling.py \
    --input case_law_data/facts_truth_table_v2.csv \
    --output case_law_data/facts_labels_sealing_template.csv
```

---

## 📝 Reporting

Each agent should report:
1. Tasks completed
2. Files created/modified
3. Test results (Agent 1) or validation results (Agent 2-4)
4. Issues encountered
5. Recommendations for next steps

---

## 🔗 Related Files

- Master Plan: `plans/MASTER_PLAN.md`
- Activity Digest: `reports/analysis_outputs/activity_digest.md`
- Bug Analysis: `case_law_data/BUG_ANALYSIS_MISSING_FACTS.md`
- Fact Extraction Config: `writer_agents/config/fact_extraction_templates.py`

