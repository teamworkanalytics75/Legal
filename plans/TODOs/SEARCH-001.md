SEARCH-001 — Adopt Semantic/Hybrid CourtListener Search by Default

Summary
- Make semantic/hybrid search the default for Harvard lawsuit research
- Validate known failing queries; persist top-K with scores and rationales

Prerequisites
- CourtListener credentials (if required by client lib)
- Network access allowed

Action Box (copy/paste)
```
# 1) Run hybrid search for category 1 (Harvard/Academia)
python case_law_data/scripts/semantic_search_harvard_lawsuit.py \
  --category 1 --max-results 200

# 2) Analyze results and export report
python case_law_data/scripts/analyze_semantic_search_results.py \
  --input results/semantic_search_harvard_lawsuit_*.json \
  --top-k 50 --emit-report reports/analysis_outputs/semantic_report.md

# 3) Persist top-K with rationale snippets (if not already)
# (If needed, add a --emit-csv flag or extend analyzer to include rationale)

# 4) Optional: run across all categories
python case_law_data/scripts/semantic_search_harvard_lawsuit.py --category all --max-results 150
```

Validation
- Confirm report at `reports/analysis_outputs/semantic_report.md` with coverage stats
- Spot-check top-K JSON entries for meaningful semantic matches and rationale

Deliverables
- Updated JSON results under `results/`
- Report markdown: `reports/analysis_outputs/semantic_report.md`

Links
- Plan: SEARCH-001
- Scripts: case_law_data/scripts/semantic_search_harvard_lawsuit.py

