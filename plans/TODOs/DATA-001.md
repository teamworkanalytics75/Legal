DATA-001 — Expand Favorable Labels (Include Opinions)

Summary
- Expand labeling to include opinions/orders referencing motions
- Regenerate queues; verify counts and coverage

Prerequisites
- `case_law_data/features/unified_features.csv`
- `case_law_data/unified_corpus.db`

Action Box (copy/paste)
```
# 1) Verify feature + DB artifacts
ls -l case_law_data/features/unified_features.csv
ls -l case_law_data/unified_corpus.db

# 2) Check current totals
python case_law_data/scripts/check_actual_seal_pseudonym_counts.py

# 3) Update outline model signals with opinion inclusion
python case_law_data/scripts/analyze_motion_types_and_update_outline_model.py --include-opinions

# 4) Regenerate labeling queues with opinions
python case_law_data/scripts/create_labeling_queue.py --include-opinions \
  --output-dir case_law_data/labeling_queues

# 5) Optional: restrict to doc types (example)
python case_law_data/scripts/create_labeling_queue.py --include-opinions \
  --doc-type opinion --doc-type order \
  --output-dir case_law_data/labeling_queues
```

Validation
- Open `case_law_data/labeling_queues/*.csv`; confirm opinion/order rows exist
- Re-run counts and snapshot deltas in `reports/analysis_outputs/`

Deliverables
- Updated queues under `case_law_data/labeling_queues/`
- Brief note in `Agents_1782_ML_Dataset/MLUPGRADEROADMAP.md` (data coverage)

Links
- Plan: DATA-001
- Scripts: case_law_data/scripts/create_labeling_queue.py

