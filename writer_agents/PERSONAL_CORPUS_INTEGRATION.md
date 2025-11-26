# Personal Corpus Integration Guide

This document explains how the writing workflow uses your personal corpus (`case_law_data/tmp_corpus/`) as the **single source of truth** for case facts.

## Overview

The workflow automatically discovers and uses documents from your personal corpus directory to build case insights. This ensures that every motion generation run uses the latest facts from your HK Statement, OGC emails, and other case documents.

## Corpus Structure

Your personal corpus lives at:
```
case_law_data/tmp_corpus/
```

This directory contains:
- **HK Statement of Claim** (`Exhibit 2 — Certified Statement of Claim (Hong Kong, 2 Jun 2025).txt`)
- **OGC Emails** (`3 Emails to Harvard_s OGC in 2025.txt`)
- **Harvard Correspondence** (`All 2019-2024 Harvard Correspondence.txt`)
- **Xi Slides** (`Exhibit 7 - The Two Xi Slides.txt`)
- **Vivien Chan Letters** (Exhibit 6-A, 6-B, 6-C)
- **Motion to Seal drafts** (`Initial Filings - Motion to Seal 1782.txt`)
- **Other exhibits and documents**

## Auto-Discovery

The `build_case_insights.py` script automatically discovers key documents using pattern matching:

- **HK Statement**: Matches `*Statement of Claim*`, `*HK Statement*`, `Exhibit 2*`, or `hk_statement*.txt`
- **OGC Emails**: Matches `*OGC*2025*`, `*3 Emails*`, or `*OGC*`
- **Harvard Correspondence**: Matches `*Harvard Correspondence*` or `*All 2019*`
- **Xi Slides**: Matches `*Xi Slides*` or `*Exhibit 7*`
- **Vivien Chan**: Matches `*Vivien Chan*`
- **Motion to Seal**: Matches `*Motion to Seal*` or `*Motion-to-Seal*`

The script also checks the root directory for `hk_statement_text.txt` as a fallback.

## Usage

### Option 1: Auto-Discovery (Recommended)

Simply run the workflow without specifying file paths. It will automatically discover files from the corpus:

```bash
python3 writer_agents/scripts/generate_optimized_motion.py \
  --case-summary "Motion to seal sensitive personal information" \
  --enable-google-docs \
  --master-draft-mode
```

The workflow will:
1. Auto-discover HK Statement and OGC emails from `case_law_data/tmp_corpus/`
2. Extract facts and build case insights
3. Use these facts throughout the drafting process

### Option 2: Explicit Corpus Directory

Specify the corpus directory explicitly:

```bash
python3 writer_agents/scripts/generate_optimized_motion.py \
  --case-summary "Motion to seal sensitive personal information" \
  --corpus-dir case_law_data/tmp_corpus \
  --enable-google-docs \
  --master-draft-mode
```

### Option 3: Pre-Built Case Insights

Build case insights once and reuse:

```bash
# Build insights from corpus
python3 writer_agents/scripts/build_case_insights.py \
  --corpus-dir case_law_data/tmp_corpus \
  --output writer_agents/outputs/case_insights.json

# Use pre-built insights
python3 writer_agents/scripts/generate_optimized_motion.py \
  --case-summary "Motion to seal sensitive personal information" \
  --case-insights-file writer_agents/outputs/case_insights.json \
  --enable-google-docs \
  --master-draft-mode
```

### Option 4: Explicit File Paths

If you need to override specific files:

```bash
python3 writer_agents/scripts/generate_optimized_motion.py \
  --case-summary "Motion to seal sensitive personal information" \
  --hk-statement-path case_law_data/tmp_corpus/Exhibit\ 2\ —\ Certified\ Statement\ of\ Claim\ \(Hong\ Kong,\ 2\ Jun\ 2025\).txt \
  --ogc-emails-path case_law_data/tmp_corpus/3\ Emails\ to\ Harvard_s\ OGC\ in\ 2025.txt \
  --enable-google-docs \
  --master-draft-mode
```

## Keeping Your Corpus Up-to-Date

### Adding New Documents

1. **Copy new documents** to your source directory (e.g., `C:\Users\User\OneDrive\Desktop\4` on Windows)
2. **Re-run ingestion**:
   ```bash
   python case_law_data/scripts/ingest_personal_corpus.py \
     --source-dir "C:\\Users\\User\\OneDrive\\Desktop\\4" \
     --output-dir case_law_data/tmp_corpus \
     --metadata case_law_data/results/personal_corpus_metadata.json
   ```

### Regenerating Features & Analysis

After adding documents, regenerate features and SHAP analysis:

```bash
# Extract features
python case_law_data/scripts/extract_personal_corpus_features.py \
  --corpus-dir case_law_data/tmp_corpus \
  --output case_law_data/results/personal_corpus_features.csv \
  --skip-existing

# Run SHAP analysis
python case_law_data/scripts/hk_deep_dive_analysis.py \
  --folder case_law_data/tmp_corpus \
  --model case_law_data/models/section_1782_discovery_model.cbm \
  --features-csv case_law_data/features/unified_features.csv \
  --output-dir case_law_data/results

# Update embeddings for semantic search
python case_law_data/scripts/create_personal_corpus_embeddings.py \
  --corpus-dir case_law_data/tmp_corpus \
  --output case_law_data/results/personal_corpus_embeddings.db \
  --faiss-index case_law_data/results/personal_corpus_embeddings.faiss
```

### Full Pipeline

Use the full pipeline script to do everything at once:

```bash
python case_law_data/scripts/run_full_personal_corpus_pipeline.py \
  --source-dir "C:\\Users\\User\\OneDrive\\Desktop\\4" \
  --skip-draft  # Skip motion generation, just update corpus
```

## How Facts Are Used

Once extracted, case facts are used throughout the workflow:

1. **Planning Phase**: Facts are injected into the planner's goal to ensure the draft addresses actual case circumstances
2. **Drafting Phase**: Facts are available in the SK context for plugins to reference
3. **Quality Gates**: The `fact_alignment` gate validates that drafts reference required facts (Harvard, OGC, HK filing, specific dates)
4. **Feature Plugins**: Privacy, safety, and retaliation plugins use actual case facts to provide specific recommendations
5. **Feature Orchestrator**: Weak feature payloads include fact prompts to guide improvements

## Fact Blocks

The following fact blocks are extracted and made available:

- `hk_allegation_defamation`: Defamatory statements published by Harvard Clubs
- `hk_allegation_ccp_family`: Xi Mingze photograph disclosure and safety risk
- `hk_allegation_competitor`: Business competitor motive
- `hk_retaliation_events`: Timeline of defamation and republication
- `privacy_leak_events`: Disclosure of personal information
- `safety_concerns`: Risk of political persecution and torture
- `ogc_email_1_threat`: First OGC email (April 7, 2025)
- `ogc_email_2_non_response`: Follow-up email (April 18, 2025)
- `ogc_email_3_meet_confer`: Meet-and-confer email (August 11, 2025)
- `harvard_retaliation_events`: Summary of Harvard Clubs' actions and OGC non-response
- `ogc_email_allegations`: Summary of all three OGC emails

## Integration with Processed Metadata

The workflow can optionally use processed metadata from:
- `case_law_data/results/personal_corpus_metadata.json` - Document metadata
- `case_law_data/results/personal_corpus_features.csv` - Extracted features
- `case_law_data/results/personal_corpus_aggregated_statistics.json` - SHAP summaries
- `case_law_data/results/personal_corpus_embeddings.{db,faiss}` - Semantic search indices

These are used to enrich case insights with feature analysis and SHAP importance scores.

## Troubleshooting

### Files Not Found

If auto-discovery fails:
1. Check that `case_law_data/tmp_corpus/` exists and contains your documents
2. Verify file names match the expected patterns
3. Use `--hk-statement-path` and `--ogc-emails-path` to specify files explicitly

### Outdated Facts

If the workflow uses old facts:
1. Re-run `build_case_insights.py` to regenerate insights
2. Or delete `writer_agents/outputs/case_insights.json` to force regeneration

### Missing Documents

If documents are missing from the corpus:
1. Check the source directory (e.g., `C:\Users\User\OneDrive\Desktop\4`)
2. Re-run `ingest_personal_corpus.py` to update the corpus
3. Verify files appear in `case_law_data/tmp_corpus/`

## Best Practices

1. **Keep corpus synchronized**: Re-run ingestion whenever you add new documents
2. **Use auto-discovery**: Let the workflow find files automatically unless you need to override
3. **Rebuild insights after updates**: Regenerate case insights if you update source documents
4. **Version control**: Consider committing `case_insights.json` to track fact extraction changes
5. **Regular updates**: Run the full pipeline periodically to keep features and embeddings current

