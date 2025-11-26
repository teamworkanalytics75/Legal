MODEL-001 — Retrain Unified Outline Model (TF‑IDF + Legal‑BERT)

Summary
- Retrain outline model combining TF‑IDF + Legal‑BERT features
- Ensure `extract_perfect_outline_features` is included in validation paths

Prerequisites
- Python env with ML deps installed
- Features + embeddings generated

Action Box (copy/paste)
```
# 1) Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_upwork.txt
pip install -r Agents_1782_ML_Dataset/RequirementsMlSota.txt

# 2) Generate embeddings + unified features
python case_law_data/scripts/generate_bert_features.py --model legal-bert-base-uncased
python case_law_data/scripts/BuildUnifiedFeatures.py --include-outline

# 3) Retrain outline model (with validation utility present)
python case_law_data/scripts/retrain_unified_outline_model.py \
  --features-csv case_law_data/features/unified_features.csv \
  --target perfect_outline \
  --output-model case_law_data/models/catboost_outline_unified.cbm
```

Validation
- Verify model artifacts under `case_law_data/models/*.cbm`
- Confirm validation log mentions `extract_perfect_outline_features`

Deliverables
- Updated outline model `.cbm` and metrics report
- Short note in `Agents_1782_ML_Dataset/MLUPGRADEROADMAP.md`

Links
- Plan: MODEL-001
- Scripts: case_law_data/scripts/generate_bert_features.py
