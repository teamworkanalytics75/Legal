#!/usr/bin/env python3
"""
CLI wrapper for applying factuality filter to fact datasets.

Usage:
    python scripts/filter_facts_cli.py <input_csv> [--output-dir OUTPUT_DIR] [--extract-claims] [--verify]
    python scripts/filter_facts_cli.py case_law_data/facts_v16_final.csv --extract-claims
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from factuality_filter.code.pipeline import FactualityFilter
    from factuality_filter.code.ModalityDetector import ModalityType
except ImportError as e:
    print(f"❌ Error importing FactualityFilter: {e}")
    print("   Make sure factuality_filter module is available")
    sys.exit(1)


def load_facts(input_csv: Path) -> list[dict]:
    """Load facts from CSV."""
    facts = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    return facts


def filter_facts(
    facts: list[dict],
    extract_claims: bool = False,
    verify_claims: bool = False,
    proposition_column: str = 'proposition'
) -> tuple[list[dict], list[dict], dict]:
    """Apply factuality filter to facts."""
    filter_pipeline = FactualityFilter()
    
    factual_facts = []
    hypothetical_facts = []
    statistics = {
        'total': len(facts),
        'factual': 0,
        'hypothetical': 0,
        'uncertain': 0,
        'by_modality': {},
        'high_confidence_factual': 0,
    }
    
    print(f"Processing {len(facts)} facts...")
    
    for i, fact in enumerate(facts):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(facts)} facts...")
        
        # Get proposition text
        proposition = str(fact.get(proposition_column, '') or fact.get('Proposition', '') or '').strip()
        
        if not proposition:
            fact['factuality_status'] = 'EMPTY'
            fact['modality'] = 'UNCERTAIN'
            fact['confidence'] = 0.0
            hypothetical_facts.append(fact)
            statistics['uncertain'] += 1
            continue
        
        # Apply filter
        try:
            result = filter_pipeline.filter_text(
                proposition,
                remove_hypotheticals=True,
                extract_claims=extract_claims,
                verify_claims=verify_claims,
            )
            
            if result.sentences:
                modalities = [s.modality for s in result.sentences]
                confidences = [s.confidence for s in result.sentences if hasattr(s, 'confidence')]
                
                # Check if factual
                is_factual = any(
                    s.modality == ModalityType.FACTUAL or
                    s.modality.value == 'factual' or 
                    'FACTUAL' in str(s.modality)
                    for s in result.sentences
                )
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
                
                if is_factual:
                    primary_modality = 'FACTUAL'
                    statistics['factual'] += 1
                    if avg_confidence > 0.7:
                        statistics['high_confidence_factual'] += 1
                    factual_facts.append(fact)
                else:
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
                
                # Add claims if extracted
                if extract_claims and result.claims:
                    fact['extracted_claims'] = ' | '.join([c.text for c in result.claims])
            else:
                fact['factuality_status'] = 'UNCERTAIN'
                fact['modality'] = 'UNCERTAIN'
                fact['confidence'] = 0.0
                hypothetical_facts.append(fact)
                statistics['uncertain'] += 1
            
            modality_key = fact.get('modality', 'UNKNOWN')
            statistics['by_modality'][modality_key] = statistics['by_modality'].get(modality_key, 0) + 1
            
        except Exception as e:
            fact['factuality_status'] = 'ERROR'
            fact['modality'] = 'UNCERTAIN'
            fact['confidence'] = 0.0
            fact['error'] = str(e)
            hypothetical_facts.append(fact)
            statistics['uncertain'] += 1
    
    return factual_facts, hypothetical_facts, statistics


def export_csv(facts: list[dict], output_path: Path):
    """Export facts to CSV."""
    if not facts:
        return
    
    fieldnames = set()
    for fact in facts:
        fieldnames.update(fact.keys())
    fieldnames = sorted(list(fieldnames))
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fact in facts:
            row = {}
            for field in fieldnames:
                row[field] = fact.get(field, '')
            writer.writerow(row)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Apply factuality filter to fact datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter v16 facts
  python scripts/filter_facts_cli.py case_law_data/facts_v16_final.csv
  
  # Filter with claim extraction
  python scripts/filter_facts_cli.py case_law_data/facts_v16_final.csv --extract-claims
  
  # Custom output directory
  python scripts/filter_facts_cli.py input.csv --output-dir outputs/
        """
    )
    
    parser.add_argument('input_csv', type=Path, help='Input CSV file with facts')
    parser.add_argument('--output-dir', type=Path, default=None, 
                       help='Output directory (default: same as input)')
    parser.add_argument('--extract-claims', action='store_true',
                       help='Extract structured claims from factual text')
    parser.add_argument('--verify', action='store_true',
                       help='Verify claims (requires knowledge graph)')
    parser.add_argument('--proposition-column', default='proposition',
                       help='Column name containing proposition text (default: proposition)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_csv.exists():
        print(f"❌ Error: Input file not found: {args.input_csv}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = args.input_csv.parent
    
    input_stem = args.input_csv.stem
    
    print("="*80)
    print("FACTUALITY FILTER CLI")
    print("="*80)
    print(f"\nInput: {args.input_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Extract claims: {args.extract_claims}")
    print(f"Verify claims: {args.verify}")
    print()
    
    # Load facts
    print("1. Loading facts...")
    facts = load_facts(args.input_csv)
    print(f"   Loaded {len(facts)} facts")
    
    # Apply filter
    print("\n2. Applying factuality filter...")
    factual_facts, hypothetical_facts, statistics = filter_facts(
        facts,
        extract_claims=args.extract_claims,
        verify_claims=args.verify,
        proposition_column=args.proposition_column
    )
    
    # Results
    print(f"\n3. Results:")
    print(f"   ✅ Factual: {statistics['factual']} ({statistics['factual']/statistics['total']*100:.1f}%)")
    print(f"   ⚠️  Hypothetical/Uncertain: {statistics['hypothetical']} ({statistics['hypothetical']/statistics['total']*100:.1f}%)")
    print(f"   🎯 High-confidence factual: {statistics['high_confidence_factual']}")
    
    # Export
    print(f"\n4. Exporting results...")
    
    factual_output = output_dir / f"{input_stem}_factual_only.csv"
    hypothetical_output = output_dir / f"{input_stem}_hypothetical_only.csv"
    filtered_output = output_dir / f"{input_stem}_filtered.csv"
    
    export_csv(factual_facts, factual_output)
    print(f"   ✅ Factual-only: {factual_output} ({len(factual_facts)} facts)")
    
    export_csv(hypothetical_facts, hypothetical_output)
    print(f"   ✅ Hypothetical-only: {hypothetical_output} ({len(hypothetical_facts)} facts)")
    
    export_csv(facts, filtered_output)  # All facts with metadata
    print(f"   ✅ Filtered (with metadata): {filtered_output} ({len(facts)} facts)")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE!")
    print(f"{'='*80}")
    print(f"\nFiles created:")
    print(f"  📄 {factual_output}")
    print(f"  📄 {hypothetical_output}")
    print(f"  📄 {filtered_output}")


if __name__ == "__main__":
    main()

