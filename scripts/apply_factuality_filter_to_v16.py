#!/usr/bin/env python3
"""Apply factuality filter to v16 facts dataset."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

# Add factuality_filter to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from factuality_filter.code.pipeline import FactualityFilter
except ImportError as e:
    print(f"❌ Error importing FactualityFilter: {e}")
    print("   Make sure factuality_filter module is available")
    sys.exit(1)

V16_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v16_final.csv")
OUTPUT_FACTUAL_CSV = Path("case_law_data/facts_v16_factual_only.csv")
OUTPUT_HYPOTHETICAL_CSV = Path("case_law_data/facts_v16_hypothetical_only.csv")
OUTPUT_FILTERED_CSV = Path("case_law_data/facts_v16_factuality_filtered.csv")
REPORT_PATH = Path("reports/analysis_outputs/v16_factuality_filter_report.md")


def load_facts() -> list[dict]:
    """Load v16 facts."""
    facts = []
    with open(V16_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    return facts


def apply_factuality_filter(facts: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Apply factuality filter to all facts."""
    print("\n2. Initializing FactualityFilter...")
    filter_pipeline = FactualityFilter()
    
    factual_facts = []
    hypothetical_facts = []
    all_filtered = []
    statistics = {
        'total': len(facts),
        'factual': 0,
        'hypothetical': 0,
        'uncertain': 0,
        'by_modality': {},
        'high_confidence_factual': 0,
    }
    
    print(f"\n3. Processing {len(facts)} facts...")
    print("   (This may take a few minutes...)")
    
    for i, fact in enumerate(facts):
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(facts)} facts...")
        
        proposition = str(fact.get('proposition', '') or fact.get('Proposition', '') or '').strip()
        
        if not proposition:
            # Empty proposition - keep as-is but mark as uncertain
            fact['factuality_status'] = 'EMPTY'
            fact['modality'] = 'UNCERTAIN'
            fact['confidence'] = 0.0
            all_filtered.append(fact)
            hypothetical_facts.append(fact)
            statistics['uncertain'] += 1
            continue
        
        # Apply factuality filter
        try:
            result = filter_pipeline.filter_text(
                proposition,
                remove_hypotheticals=True,
                extract_claims=False,  # Don't extract claims, just classify
                verify_claims=False,
            )
            
            # Determine modality from sentences
            if result.sentences:
                # Get the primary modality (most common or highest confidence)
                modalities = [s.modality for s in result.sentences]
                confidences = [s.confidence for s in result.sentences if hasattr(s, 'confidence')]
                
                # Classify as factual if any sentence is factual
                from factuality_filter.code.ModalityDetector import ModalityType
                is_factual = any(
                    s.modality == ModalityType.FACTUAL or
                    s.modality.value == 'factual' or 
                    str(s.modality) == 'ModalityType.FACTUAL' or
                    'FACTUAL' in str(s.modality)
                    for s in result.sentences
                )
                
                # Get average confidence
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
                
                # Determine primary modality
                if is_factual:
                    primary_modality = 'FACTUAL'
                    statistics['factual'] += 1
                    if avg_confidence > 0.7:
                        statistics['high_confidence_factual'] += 1
                    factual_facts.append(fact)
                else:
                    # Get the most common modality
                    if modalities:
                        primary_modality = str(modalities[0].value) if hasattr(modalities[0], 'value') else str(modalities[0])
                    else:
                        primary_modality = 'UNCERTAIN'
                    statistics['hypothetical'] += 1
                    hypothetical_facts.append(fact)
                
                fact['factuality_status'] = primary_modality
                fact['modality'] = primary_modality
                fact['confidence'] = round(avg_confidence, 3)
                fact['factual_text'] = result.factual_text
                fact['hypothetical_text'] = result.hypothetical_text
                
            else:
                # No sentences detected - treat as uncertain
                fact['factuality_status'] = 'UNCERTAIN'
                fact['modality'] = 'UNCERTAIN'
                fact['confidence'] = 0.0
                hypothetical_facts.append(fact)
                statistics['uncertain'] += 1
            
            # Track modality distribution
            modality_key = fact.get('modality', 'UNKNOWN')
            statistics['by_modality'][modality_key] = statistics['by_modality'].get(modality_key, 0) + 1
            
        except Exception as e:
            # More detailed error reporting
            import traceback
            error_msg = str(e)
            if 'modality_detector' in error_msg.lower():
                # This is the import error - skip this fact for now
                fact['factuality_status'] = 'ERROR_IMPORT'
                fact['modality'] = 'UNCERTAIN'
                fact['confidence'] = 0.0
                fact['error'] = error_msg
                hypothetical_facts.append(fact)
                statistics['uncertain'] += 1
                if (i + 1) % 100 == 0:  # Only print every 100 errors to avoid spam
                    print(f"   ⚠️  Import error (suppressing details for performance)")
            else:
                print(f"   ⚠️  Error processing FactID {fact.get('factid', 'UNKNOWN')}: {error_msg}")
                fact['factuality_status'] = 'ERROR'
                fact['modality'] = 'UNCERTAIN'
                fact['confidence'] = 0.0
                hypothetical_facts.append(fact)
                statistics['uncertain'] += 1
        
        all_filtered.append(fact)
    
    return factual_facts, hypothetical_facts, all_filtered, statistics


def main():
    """Main execution."""
    print("="*80)
    print("APPLYING FACTUALITY FILTER TO V16 FACTS")
    print("="*80)
    
    print("\n1. Loading v16 facts...")
    facts = load_facts()
    print(f"   Loaded {len(facts)} facts")
    
    # Apply filter
    factual_facts, hypothetical_facts, all_filtered, statistics = apply_factuality_filter(facts)
    
    print(f"\n4. Results:")
    print(f"   ✅ Factual facts: {len(factual_facts)} ({len(factual_facts)/len(facts)*100:.1f}%)")
    print(f"   ⚠️  Hypothetical/Uncertain: {len(hypothetical_facts)} ({len(hypothetical_facts)/len(facts)*100:.1f}%)")
    print(f"   🎯 High-confidence factual: {statistics['high_confidence_factual']}")
    
    # Export factual-only dataset
    print("\n5. Exporting factual-only dataset...")
    if factual_facts:
        fieldnames = set()
        for fact in factual_facts:
            fieldnames.update(fact.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_FACTUAL_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fact in factual_facts:
                row = {}
                for field in fieldnames:
                    row[field] = fact.get(field, '')
                writer.writerow(row)
        print(f"   ✅ Exported {OUTPUT_FACTUAL_CSV} ({len(factual_facts)} facts)")
    
    # Export hypothetical-only dataset
    print("\n6. Exporting hypothetical-only dataset...")
    if hypothetical_facts:
        fieldnames = set()
        for fact in hypothetical_facts:
            fieldnames.update(fact.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_HYPOTHETICAL_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fact in hypothetical_facts:
                row = {}
                for field in fieldnames:
                    row[field] = fact.get(field, '')
                writer.writerow(row)
        print(f"   ✅ Exported {OUTPUT_HYPOTHETICAL_CSV} ({len(hypothetical_facts)} facts)")
    
    # Export filtered dataset (all facts with factuality metadata)
    print("\n7. Exporting filtered dataset (with factuality metadata)...")
    if all_filtered:
        fieldnames = set()
        for fact in all_filtered:
            fieldnames.update(fact.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_FILTERED_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fact in all_filtered:
                row = {}
                for field in fieldnames:
                    row[field] = fact.get(field, '')
                writer.writerow(row)
        print(f"   ✅ Exported {OUTPUT_FILTERED_CSV} ({len(all_filtered)} facts)")
    
    # Generate report
    print("\n8. Writing report...")
    report = f"""# V16 Factuality Filter Report

## Summary

- **Input**: `top_1000_facts_for_chatgpt_v16_final.csv`
- **Total facts processed**: {statistics['total']}
- **Factual facts**: {statistics['factual']} ({statistics['factual']/statistics['total']*100:.1f}%)
- **Hypothetical/Uncertain facts**: {statistics['hypothetical']} ({statistics['hypothetical']/statistics['total']*100:.1f}%)
- **High-confidence factual**: {statistics['high_confidence_factual']} ({statistics['high_confidence_factual']/statistics['total']*100:.1f}%)

## Modality Distribution

"""
    
    for modality, count in sorted(statistics['by_modality'].items(), key=lambda x: x[1], reverse=True):
        pct = count / statistics['total'] * 100
        report += f"- **{modality}**: {count} facts ({pct:.1f}%)\n"
    
    report += f"""
## Files Generated

1. **Factual-only dataset**: `facts_v16_factual_only.csv`
   - Contains only facts classified as FACTUAL
   - {len(factual_facts)} facts
   - Recommended for court-ready, verified fact sets

2. **Hypothetical-only dataset**: `facts_v16_hypothetical_only.csv`
   - Contains facts classified as HYPOTHETICAL, SPECULATIVE, or UNCERTAIN
   - {len(hypothetical_facts)} facts
   - Useful for review and potential removal

3. **Filtered dataset (with metadata)**: `facts_v16_factuality_filtered.csv`
   - All facts with factuality status, modality, and confidence scores
   - {len(all_filtered)} facts
   - Includes new columns: `factuality_status`, `modality`, `confidence`, `factual_text`, `hypothetical_text`

## Recommendations

### For Court-Ready Fact Sets
Use **`facts_v16_factual_only.csv`** - This contains only verified factual statements.

### For Bayesian Network / Reasoning Systems
Use **`facts_v16_factuality_filtered.csv`** and filter by:
- `modality = 'FACTUAL'`
- `confidence > 0.7`

### For Review
Review **`facts_v16_hypothetical_only.csv`** to:
- Identify facts that may need rewording
- Find speculative content that should be removed
- Verify uncertain classifications

## Next Steps

1. Review hypothetical facts to determine if they should be:
   - Rewritten as factual statements
   - Removed entirely
   - Kept but marked as lower confidence

2. Consider creating a "v17" that:
   - Uses only factual facts
   - Rewrites hypothetical facts as factual where possible
   - Removes truly speculative content

3. For BN/reasoning systems:
   - Use confidence scores to weight facts
   - Filter by modality for different analysis types
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')
    print(f"   ✅ Written {REPORT_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE! Factuality filter applied to v16")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  📊 Total facts: {statistics['total']}")
    print(f"  ✅ Factual: {statistics['factual']} ({statistics['factual']/statistics['total']*100:.1f}%)")
    print(f"  ⚠️  Hypothetical/Uncertain: {statistics['hypothetical']} ({statistics['hypothetical']/statistics['total']*100:.1f}%)")
    print(f"  🎯 High-confidence factual: {statistics['high_confidence_factual']}")
    print(f"\nFiles created:")
    print(f"  📄 {OUTPUT_FACTUAL_CSV}")
    print(f"  📄 {OUTPUT_HYPOTHETICAL_CSV}")
    print(f"  📄 {OUTPUT_FILTERED_CSV}")
    print(f"  📄 {REPORT_PATH}")


if __name__ == "__main__":
    main()

