# Motion Type Analysis and Perfect Outline Model Update - Complete

## Summary

Successfully analyzed motions to seal, motions for pseudonym, and motions for both separately. Trained models with perfect outline features and updated the system to use these models.

## Motion Type Analysis Results

### Dataset Breakdown

| Motion Type | Total Cases | Favorable | Denied | Unclear |
|-------------|-------------|-----------|--------|---------|
| **Seal** | 233 | 11 (4.7%) | 19 (8.2%) | 203 (87.1%) |
| **Pseudonym** | 109 | 11 (10.1%) | 16 (14.7%) | 82 (75.2%) |
| **Both** | 27 | 4 (14.8%) | 3 (11.1%) | 20 (74.1%) |
| **Total** | 369 | 26 (7.0%) | 38 (10.3%) | 305 (82.7%) |

### Key Findings

**Seal Motions:**
- Most common type (233 cases)
- Lowest favorable rate (4.7%)
- Top features: `char_count`, `sentence_count`, `has_protective_measures`, `danger_safety_position`

**Pseudonym Motions:**
- 109 cases
- Highest favorable rate (10.1%)
- Top features: `danger_safety_position` (18.09 importance), `sentence_count`, `word_count`

**Both (Seal + Pseudonym):**
- Smallest sample (27 cases)
- Highest favorable rate (14.8%) but small sample size
- Model not trained due to insufficient data

## Model Performance

### Seal Model (`catboost_outline_seal.cbm`)
- **Accuracy:** 87.2%
- **Precision:** 16.7%
- **Recall:** 50.0%
- **F1 Score:** 25.0%
- **Training:** 186 cases, **Test:** 47 cases

**Top Features:**
1. `char_count` (28.47)
2. `sentence_count` (22.82)
3. `has_protective_measures` (19.07)
4. `paragraph_count` (9.53)
5. `danger_safety_position` (3.61)
6. `balancing_test_position` (3.51)
7. `enumeration_density` (2.58)

### Pseudonym Model (`catboost_outline_pseudonym.cbm`)
- **Accuracy:** 90.9%
- **Precision:** 50.0%
- **Recall:** 50.0%
- **F1 Score:** 50.0%
- **Training:** 87 cases, **Test:** 22 cases

**Top Features:**
1. `danger_safety_position` (18.09) ⭐ **CRITICAL**
2. `sentence_count` (17.48)
3. `word_count` (14.16)
4. `char_count` (13.52)
5. `enumeration_density` (4.92)
6. `balancing_test_position` (4.49)

### Unified Model (`catboost_outline_unified.cbm`)
- **Accuracy:** 78.7%
- **Precision:** 18.8%
- **Recall:** 60.0% ✅
- **F1 Score:** 28.6%
- **Training:** 295 cases, **Test:** 74 cases

**Top Features:**
1. `word_count` (14.82)
2. `danger_safety_position` (12.42) ⭐
3. `paragraph_count` (12.15)
4. `sentence_count` (11.96)
5. `enumeration_density` (9.35) ⭐
6. `motion_type_seal` (7.46) - Motion type matters!
7. `has_protective_measures` (4.07)
8. `balancing_test_position` (3.89)

## Perfect Outline Features Integrated

### Critical Features (from TOP_FEATURES_PERFECT_OUTLINE.md)

1. **`transition_legal_standard_to_factual_background`** (64.80 importance)
   - Legal Standard section immediately followed by Factual Background
   - **MOST IMPORTANT** feature for success
   - Found in: 0% of seal, 0% of pseudonym, 7.4% of both motions

2. **`has_bullet_points`** (31.56 importance)
   - Motion contains bullet points for enumeration
   - **SECOND MOST IMPORTANT** feature
   - Signals organization and thoroughness

3. **`enumeration_count`** and **`enumeration_density`**
   - Successful motions use 11.75 vs 6.18 enumeration instances
   - Higher density = more organized

4. **`danger_safety_position`** (3.36 importance)
   - Where Danger/Safety section appears
   - Earlier placement = better (position matters!)

5. **`balancing_test_position`** (0.23 importance)
   - Where Balancing Test section appears
   - Should be prominent, not buried

6. **Section-specific enumeration:**
   - `enumeration_in_danger_safety`
   - `enumeration_in_privacy_harm`
   - `enumeration_in_protective_measures`

## Model Files Created

1. **`catboost_outline_seal.cbm`** - Seal-only motions
2. **`catboost_outline_pseudonym.cbm`** - Pseudonym-only motions
3. **`catboost_outline_unified.cbm`** - All motion types with motion_type features ⭐ **PREFERRED**

## System Updates

### WorkflowOrchestrator Model Priority Updated

**New Priority Order:**
1. `catboost_outline_unified.cbm` ⭐ **PREFERRED** (includes perfect outline features + motion type)
2. `catboost_outline_both.cbm` (if available)
3. `catboost_motion_seal_pseudonym.cbm` (fallback)
4. `catboost_motion.cbm` (general fallback)
5. `section_1782_discovery_model.cbm` (1782-specific fallback)

### Feature Extraction

Created `extract_perfect_outline_features()` function that extracts:
- Legal Standard → Factual Background transition detection
- Enumeration features (count, density, depth)
- Section position features
- Section-specific enumeration counts
- Section presence indicators

## Key Insights

### For Seal Motions:
- **Length matters:** `char_count` and `sentence_count` are top features
- **Protective measures section is important** (19.07 importance)
- **Section positioning matters:** `danger_safety_position` and `balancing_test_position` are predictive

### For Pseudonym Motions:
- **Section positioning is CRITICAL:** `danger_safety_position` is #1 feature (18.09)
- **Length and structure matter:** `sentence_count`, `word_count`, `char_count` are top features
- **Enumeration density matters** (4.92 importance)

### For All Motions (Unified):
- **Motion type matters:** `motion_type_seal` has 7.46 importance
- **Perfect outline features are important:** `enumeration_density` (9.35), `danger_safety_position` (12.42)
- **Structure matters:** Word count, paragraph count, sentence count are top features

## Perfect Outline Requirements

Based on TOP_FEATURES_PERFECT_OUTLINE.md:

### Must-Have Structure:
1. **Introduction**
2. **Legal Standard** ← Must come first
3. **Factual Background** ← Must immediately follow Legal Standard (CRITICAL TRANSITION)
4. **Privacy Harm / Good Cause** (with enumeration)
5. **Danger/Safety Arguments** (place early, with enumeration)
6. **Public Interest Analysis**
7. **Balancing Test** (place prominently, with enumeration)
8. **Protective Measures** (with enumeration)
9. **Conclusion**

### Formatting Requirements:
- **Extensive bullet points** throughout (11+ instances)
- **Enumeration in key sections:** Danger/Safety, Privacy Harm, Protective Measures
- **Strategic section ordering:** Critical sections placed early

## ⚠️ Statistical Reliability Gap

### Dataset Limitations:
- **Total cases analyzed:** 369
- **Favorable cases:** 26 (7.0%) - **TOO FEW for reliable feature importance**
- **Need:** 220-440 positive cases (10-20 per feature)
- **Gap:** **194-414 more positive cases needed**

### Impact:
- ⚠️ Feature importance rankings may be unstable
- ⚠️ Exact rankings should not be relied upon
- ✅ Perfect outline features (from 628-motion dataset) are reliable
- ⚠️ Structural features are medium confidence

**See:** [STATISTICAL_RELIABILITY_ASSESSMENT.md](STATISTICAL_RELIABILITY_ASSESSMENT.md) for full analysis.

## Next Steps

1. ✅ Models trained with perfect outline features
2. ✅ WorkflowOrchestrator updated to use outline models
3. ✅ Integrate `extract_perfect_outline_features()` into `compute_draft_features()`
4. ⏳ **Collect more cases with clear outcomes** (especially positive cases)
5. ⏳ **Label unclear cases** (305 cases need outcome labels)
6. ⏳ **Re-run analysis** when we have 200+ positive cases
7. ⏳ Test workflow with actual motion to seal + pseudonym draft

## Files Created

1. `case_law_data/scripts/analyze_motion_types_and_update_outline_model.py` - Analysis script
2. `case_law_data/scripts/extract_perfect_outline_features.py` - Feature extraction module
3. `case_law_data/models/catboost_outline_seal.cbm` - Seal model
4. `case_law_data/models/catboost_outline_pseudonym.cbm` - Pseudonym model
5. `case_law_data/models/catboost_outline_unified.cbm` - Unified model ⭐
6. `case_law_data/analysis/motion_type_analysis_20251106_142421.json` - Analysis results
7. `MOTION_TYPE_ANALYSIS_AND_OUTLINE_MODEL_UPDATE_COMPLETE.md` - This file

## Files Modified

1. `writer_agents/code/WorkflowOrchestrator.py` - Updated model path priority

