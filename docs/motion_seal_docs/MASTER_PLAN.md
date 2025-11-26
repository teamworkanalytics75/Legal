# Master Plan — Project Agenda

Purpose
-------
- Central source‑of‑truth for scope, priorities, and runbooks.
- Keep this in sync with `MASTER_PLAN.json` (JSON is authoritative).

Objectives
----------
- Reliable ML‑assisted drafting for §1782 and seal/pseudonym motions
- High‑coverage research via semantic/hybrid search + curated citations
- Calibrated, explainable models with clear thresholds and evaluation
- Accessible, reproducible workflows for all contributors

Top Priorities (P0)
-------------------
- Expand favorable labels by including opinions (not just motions)
- Adopt semantic/hybrid CourtListener search by default
- Retrain unified outline model with TF‑IDF + Legal‑BERT
- Add pre‑flight checker for required models/features

Task Board (Snapshot)
---------------------
- DATA‑001 — Expand favorable labels — completed — P0 — area: data
- SEARCH‑001 — Semantic/hybrid search default — completed — P0 — area: search
- MODEL‑001 — Retrain unified outline model — completed — P0 — area: modeling
- WORKFLOW‑001 — Pre‑flight checker — completed — P0 — area: workflow
- WORKFLOW‑002 — Fix case facts drop-out — completed — P0 — area: workflow
- WORKFLOW‑003 — Enable refinement loop by default — completed — P0 — area: workflow
- MODEL‑004 — Integrate personal corpus metrics — completed — P1 — area: modeling
- WORKFLOW‑004 — Fix semantic retrieval fallback — completed — P1 — area: workflow
- WORKFLOW‑005 — Fix feature_extractor path bug — completed — P1 — area: workflow
- WORKFLOW‑006 — Limit fact_block_ prefix (optional) — completed — P2 — area: workflow
- WORKFLOW‑007 — De-duplicate WorkflowStrategyConfig — completed — P2 — area: workflow
- WORKFLOW‑008 — Hardened CaseFactsProvider fallbacks — completed — P1 — area: workflow
- WORKFLOW‑010 — CaseFactsProvider enhancements — completed — P0 — area: workflow
- WORKFLOW‑011 — AutoGen bridge & exploration — completed — P0 — area: workflow
- WORKFLOW‑012 — Workflow phase updates — completed — P0 — area: workflow
- WORKFLOW‑013 — Factuality filter integration — completed — P1 — area: workflow
- MODEL‑002 — Calibration + thresholds — pending — P1 — area: modeling
- MODEL‑003 — Optuna HPO — pending — P1 — area: modeling
- DOCS‑001 — Add short runbooks — completed — P1 — area: docs
- WORKFLOW‑009 — Fix CaseFactsProvider path — completed — P1 — area: workflow
- WORKFLOW‑010 — CaseFactsProvider enhancements — completed — P0 — area: workflow
- WORKFLOW‑011 — AutoGen bridge & exploration phase — completed — P0 — area: workflow
- WORKFLOW‑012 — Workflow phase updates — completed — P0 — area: workflow
- WORKFLOW‑013 — Factuality filter integration — completed — P1 — area: workflow
- WORKFLOW‑014 — CaseFactsProvider caching & enrichment — completed — P0 — area: workflow
- WORKFLOW‑015 — Constraint system formatting — completed — P0 — area: workflow
- WORKFLOW‑016 — Prompt assembly & context flow — completed — P0 — area: workflow
- WORKFLOW‑017 — Fix fact enforcement timing — completed — P0 — area: workflow
- MLOPS‑001 — Tracking + nightly inventory — pending — P2 — area: mlops

Do This Next (Runbook)
----------------------
```
# 1) Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_upwork.txt
pip install -r Agents_1782_ML_Dataset/RequirementsMlSota.txt

# 2) Embeddings → features → retrain
python case_law_data/scripts/generate_bert_features.py
python Agents_1782_ML_Dataset/ml_system/pipelines/TrainingPipeline.py

# 3) Semantic search (validate coverage)
python case_law_data/scripts/semantic_search_harvard_lawsuit.py --category 1 --max-results 200
python case_law_data/scripts/analyze_semantic_search_results.py --input results/semantic_search_harvard_lawsuit_*.json

# 4) Motion generator sanity check
python writer_agents/scripts/generate_optimized_motion.py --help

# 5) Refresh inventories for reviewers
python scripts/refresh_model_inventory.py
```

Pointers
--------
- Docs index: `docs/for_agents/ML_Lawsuit_Analysis_Docs_Overview.md`
- Activity digest: `reports/analysis_outputs/activity_digest.md` (if present)
- Model inventory: `reports/analysis_outputs/model_artifacts_inventory.csv`
