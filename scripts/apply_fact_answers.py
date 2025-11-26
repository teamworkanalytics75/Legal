#!/usr/bin/env python3
"""
Apply fact answers/clarifications to update the facts database.

Supports dictation-friendly input format - you can paste pages of answers at a time.

Usage:
    # From a text file with answers
    python scripts/apply_fact_answers.py --input answers.txt --dataset v16
    
    # Interactive mode - paste answers directly
    python scripts/apply_fact_answers.py --interactive --dataset v16
    
    # From stdin
    cat answers.txt | python scripts/apply_fact_answers.py --stdin --dataset v16
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Default dataset paths
DATASETS = {
    'v16': Path("case_law_data/top_1000_facts_for_chatgpt_v16_final.csv"),
    'v15': Path("case_law_data/top_1000_facts_for_chatgpt_v15_final.csv"),
    'v14': Path("case_law_data/top_1000_facts_for_chatgpt_v14_final.csv"),
}


def parse_answer_line(line: str) -> Dict[str, str] | None:
    """
    Parse a single answer line in dictation-friendly format.
    
    Supported formats:
    - "FactID 123: EventDate = 2025-04-07"
    - "FactID 123: Proposition = New proposition text here"
    - "FactID 123: ActorRole = Harvard"
    - "FactID 123: Subject = Plaintiff"
    - "123: EventDate = 2025-04-07"  (short form)
    - "Fact 123: EventDate is 2025-04-07"  (natural language)
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    # Pattern: FactID <id>: <field> = <value>
    patterns = [
        r'FactID\s+([A-Z0-9_]+)\s*:\s*(\w+)\s*=\s*(.+)$',
        r'Fact\s+([A-Z0-9_]+)\s*:\s*(\w+)\s*=\s*(.+)$',
        r'([A-Z0-9_]+)\s*:\s*(\w+)\s*=\s*(.+)$',
        r'FactID\s+([A-Z0-9_]+)\s*:\s*(\w+)\s+is\s+(.+)$',
        r'Fact\s+([A-Z0-9_]+)\s*:\s*(\w+)\s+is\s+(.+)$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            factid = match.group(1).strip()
            field = match.group(2).strip()
            value = match.group(3).strip()
            
            # Normalize field names
            field_map = {
                'eventdate': 'eventdate',
                'date': 'eventdate',
                'proposition': 'proposition',
                'prop': 'proposition',
                'actorrole': 'actorrole',
                'actor': 'actorrole',
                'role': 'actorrole',
                'subject': 'subject',
                'eventtype': 'eventtype',
                'type': 'eventtype',
                'eventlocation': 'eventlocation',
                'location': 'eventlocation',
                'safetyrisk': 'safetyrisk',
                'risk': 'safetyrisk',
                'publicexposure': 'publicexposure',
                'exposure': 'publicexposure',
            }
            
            field = field.lower()
            if field in field_map:
                field = field_map[field]
            else:
                # Try to match common variations
                for key, mapped in field_map.items():
                    if key in field or field in key:
                        field = mapped
                        break
            
            return {
                'factid': factid,
                'field': field,
                'value': value,
            }
    
    return None


def parse_answer_text(text: str) -> List[Dict[str, str]]:
    """
    Parse multi-line answer text into structured updates.
    
    Handles:
    - Multiple answers per line
    - Multi-line values (propositions)
    - Comments (lines starting with #)
    - Empty lines
    """
    answers = []
    current_answer = None
    current_value_lines = []
    
    for line in text.split('\n'):
        line = line.rstrip()
        
        # Check if this starts a new answer
        parsed = parse_answer_line(line)
        
        if parsed:
            # Save previous answer if exists
            if current_answer:
                current_answer['value'] = '\n'.join(current_value_lines).strip()
                answers.append(current_answer)
            
            # Start new answer
            current_answer = parsed
            current_value_lines = [parsed['value']]
        elif current_answer:
            # Continuation of current value (multi-line proposition)
            if line.strip() or current_value_lines:
                current_value_lines.append(line)
        elif line.strip() and not line.startswith('#'):
            # Try to parse as standalone line
            parsed = parse_answer_line(line)
            if parsed:
                answers.append(parsed)
    
    # Save last answer
    if current_answer:
        current_answer['value'] = '\n'.join(current_value_lines).strip()
        answers.append(current_answer)
    
    return answers


def load_facts(dataset_path: Path) -> List[Dict]:
    """Load facts from CSV."""
    facts = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    return facts


def apply_answers(facts: List[Dict], answers: List[Dict[str, str]]) -> Tuple[List[Dict], Dict]:
    """Apply answers to facts."""
    updated_facts = []
    changes = {
        'updated': 0,
        'not_found': 0,
        'by_field': {},
    }
    
    # Group answers by FactID
    answers_by_factid = {}
    for answer in answers:
        factid = answer['factid']
        if factid not in answers_by_factid:
            answers_by_factid[factid] = []
        answers_by_factid[factid].append(answer)
    
    # Apply updates
    for fact in facts:
        factid = str(fact.get('factid', '')).strip()
        
        if factid in answers_by_factid:
            updated = False
            for answer in answers_by_factid[factid]:
                field = answer['field']
                value = answer['value']
                
                # Update the field
                old_value = str(fact.get(field, '')).strip()
                fact[field] = value
                
                # Also update PropositionClean_v2 if updating proposition
                if field == 'proposition' and 'propositionclean_v2' in fact:
                    fact['propositionclean_v2'] = value
                
                if old_value != value:
                    updated = True
                    changes['by_field'][field] = changes['by_field'].get(field, 0) + 1
            
            if updated:
                changes['updated'] += 1
        else:
            # Check if any answer FactID might match (case-insensitive, partial)
            matched = False
            for answer_factid, answer_list in answers_by_factid.items():
                if factid.lower() == answer_factid.lower() or factid.endswith(answer_factid) or answer_factid.endswith(factid):
                    matched = True
                    for answer in answer_list:
                        field = answer['field']
                        value = answer['value']
                        fact[field] = value
                        if field == 'proposition' and 'propositionclean_v2' in fact:
                            fact['propositionclean_v2'] = value
                        changes['by_field'][field] = changes['by_field'].get(field, 0) + 1
                    changes['updated'] += 1
                    break
            
            if not matched:
                # Check if this is a new fact to add
                pass
        
        updated_facts.append(fact)
    
    # Check for answers that didn't match any facts
    matched_factids = set()
    for fact in facts:
        matched_factids.add(str(fact.get('factid', '')).strip())
    
    for answer_factid in answers_by_factid.keys():
        if answer_factid not in matched_factids:
            # Try case-insensitive match
            found = False
            for factid in matched_factids:
                if factid.lower() == answer_factid.lower():
                    found = True
                    break
            if not found:
                changes['not_found'] += 1
                print(f"   ⚠️  Warning: FactID '{answer_factid}' not found in dataset")
    
    return updated_facts, changes


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Apply fact answers/clarifications to update the facts database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Format Examples:

  FactID 123: EventDate = 2025-04-07
  FactID 456: Proposition = Harvard published Statement 1 in April 2019
  FactID 789: ActorRole = Harvard Office of General Counsel
  FactID 101: Subject = Plaintiff

  # Multi-line propositions:
  FactID 234: Proposition = This is a long proposition
  that spans multiple lines
  and continues here.

  # Short form:
  123: EventDate = 2025-04-07

  # Natural language:
  Fact 456: EventDate is 2025-04-07

Examples:
  # From file
  python scripts/apply_fact_answers.py --input answers.txt --dataset v16
  
  # Interactive (paste answers, then Ctrl+D or Ctrl+Z)
  python scripts/apply_fact_answers.py --interactive --dataset v16
  
  # From stdin
  cat answers.txt | python scripts/apply_fact_answers.py --stdin --dataset v16
        """
    )
    
    parser.add_argument('--input', type=Path, help='Input file with answers')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode (read from stdin)')
    parser.add_argument('--stdin', action='store_true', help='Read from stdin')
    parser.add_argument('--dataset', default='v16', choices=list(DATASETS.keys()),
                       help='Dataset version to update (default: v16)')
    parser.add_argument('--output', type=Path, help='Output CSV file (default: auto-generated)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without applying')
    
    args = parser.parse_args()
    
    # Get input text
    if args.input:
        if not args.input.exists():
            print(f"❌ Error: Input file not found: {args.input}")
            sys.exit(1)
        text = args.input.read_text(encoding='utf-8')
    elif args.interactive or args.stdin:
        print("Enter answers (one per line, or multi-line for propositions).")
        print("Press Ctrl+D (Linux/Mac) or Ctrl+Z (Windows) when done:")
        print()
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)
    
    # Parse answers
    print("="*80)
    print("APPLYING FACT ANSWERS")
    print("="*80)
    print(f"\n1. Parsing answers...")
    answers = parse_answer_text(text)
    print(f"   Parsed {len(answers)} answer(s)")
    
    if not answers:
        print("   ⚠️  No valid answers found in input")
        sys.exit(1)
    
    # Show parsed answers
    print("\n2. Parsed answers:")
    for answer in answers:
        print(f"   FactID {answer['factid']}: {answer['field']} = {answer['value'][:60]}...")
    
    # Load facts
    dataset_path = DATASETS[args.dataset]
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found: {dataset_path}")
        sys.exit(1)
    
    print(f"\n3. Loading dataset: {dataset_path}")
    facts = load_facts(dataset_path)
    print(f"   Loaded {len(facts)} facts")
    
    # Apply answers
    print(f"\n4. Applying answers...")
    if args.dry_run:
        print("   [DRY RUN - No changes will be saved]")
        # Just show what would change
        for answer in answers:
            factid = answer['factid']
            found = any(str(f.get('factid', '')).strip() == factid for f in facts)
            if found:
                print(f"   ✅ Would update FactID {factid}: {answer['field']}")
            else:
                print(f"   ⚠️  FactID {factid} not found")
    else:
        updated_facts, changes = apply_answers(facts, answers)
        
        print(f"   ✅ Updated {changes['updated']} fact(s)")
        if changes['not_found'] > 0:
            print(f"   ⚠️  {changes['not_found']} answer(s) didn't match any facts")
        
        if changes['by_field']:
            print(f"\n   Changes by field:")
            for field, count in sorted(changes['by_field'].items()):
                print(f"     {field}: {count}")
        
        # Export
        output_path = args.output or dataset_path.parent / f"{dataset_path.stem}_updated.csv"
        print(f"\n5. Exporting updated dataset...")
        
        if updated_facts:
            fieldnames = set()
            for fact in updated_facts:
                fieldnames.update(fact.keys())
            fieldnames = sorted(list(fieldnames))
            
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for fact in updated_facts:
                    row = {}
                    for field in fieldnames:
                        row[field] = fact.get(field, '')
                    writer.writerow(row)
        
        print(f"   ✅ Exported {output_path}")
        print(f"\n{'='*80}")
        print("✅ COMPLETE!")
        print(f"{'='*80}")
        print(f"\nUpdated dataset: {output_path}")
        print(f"Changes applied: {changes['updated']} fact(s)")


if __name__ == "__main__":
    main()

