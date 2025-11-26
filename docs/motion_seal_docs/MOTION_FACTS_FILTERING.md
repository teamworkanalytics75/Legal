# Motion Facts Filtering - Quick Reference

## How It Works

The system now ensures generated motions reference your personal lawsuit facts:

1. **Lawsuit Source Documents** (`case_law_data/lawsuit_source_documents/`) — your lawsuit documents
2. **Lawsuit Facts Extracted** (`writer_agents/outputs/lawsuit_facts_extracted.json`) — extracted facts
3. **Lawsuit Facts Database** (`case_law_data/lawsuit_facts_database.db`) — structured facts database
4. **Verification** — checks motions mention HK Statement, OGC emails, key dates, and allegations
5. **Quality Gates** — validation fails if critical facts missing; refinements that drop facts are rejected

## Key Files

- Lawsuit source documents: `case_law_data/lawsuit_source_documents/`
- Lawsuit facts extracted: `writer_agents/outputs/lawsuit_facts_extracted.json`
- Lawsuit facts database: `case_law_data/lawsuit_facts_database.db`
- Verifier: `writer_agents/code/validation/personal_facts_verifier.py`
- Workflow hooks: `writer_agents/code/WorkflowOrchestrator.py`, `writer_agents/code/WorkflowStrategyExecutor.py`

## Usage

The workflow automatically:

- Loads lawsuit source documents and builds `lawsuit_facts_extracted.json`
- Builds structured database: `python writer_agents/scripts/build_truth_database.py`
- Filters evidence so only lawsuit-specific items remain
- Verifies each generated motion references personal facts
- Rejects drafts or refinements that fall back to generic content

## Troubleshooting

- Ensure `case_law_data/lawsuit_source_documents/` exists and contains `.txt` files
- Refresh insights if needed: `python writer_agents/scripts/build_case_insights.py`
- Rebuild database: `python writer_agents/scripts/build_truth_database.py`
- Validate end-to-end: `pytest tests/test_motion_personal_facts_e2e.py -v`
- Check logs for `personal_facts_coverage` entries when debugging validation failures

## Backward Compatibility

The system supports old names for backward compatibility:
- `case_law_data/tmp_corpus/` → `case_law_data/lawsuit_source_documents/`
- `writer_agents/outputs/case_insights.json` → `writer_agents/outputs/lawsuit_facts_extracted.json`
- `case_law_data/truths.db` → `case_law_data/lawsuit_facts_database.db`
