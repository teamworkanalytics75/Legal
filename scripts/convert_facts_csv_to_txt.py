#!/usr/bin/env python3
"""Convert facts CSV to readable text file."""

from __future__ import annotations

import csv
from pathlib import Path

INPUT_CSV = Path("case_law_data/top_1000_facts_for_chatgpt_final.csv")
OUTPUT_TXT = Path("case_law_data/facts_final.txt")


def main():
    """Convert CSV to TXT."""
    print("="*80)
    print("CONVERTING FINAL FACTS TO TEXT FILE")
    print("="*80)
    print("(Final version after 25 rounds of refinement)")
    
    facts = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    
    print(f"\nLoaded {len(facts)} facts")
    
    # Generate text file
    lines = []
    lines.append("="*80)
    lines.append("FACT DATASET - FINAL VERSION - MASTER FACTS DATABASE")
    lines.append("(After 25 rounds of refinement: v16 → v25)")
    lines.append("="*80)
    lines.append("")
    lines.append(f"Total facts: {len(facts)}")
    lines.append("")
    lines.append("="*80)
    lines.append("")
    
    for idx, fact in enumerate(facts, 1):
        factid = str(fact.get('factid', '')).strip()
        proposition = str(fact.get('proposition', '')).strip()
        subject = str(fact.get('subject', '')).strip()
        actorrole = str(fact.get('actorrole', '')).strip()
        eventdate = str(fact.get('eventdate', '')).strip()
        eventlocation = str(fact.get('eventlocation', '')).strip()
        safetyrisk = str(fact.get('safetyrisk', '')).strip()
        publicexposure = str(fact.get('publicexposure', '')).strip()
        truthstatus = str(fact.get('truthstatus', '')).strip()
        evidencetype = str(fact.get('evidencetype', '')).strip()
        eventtype = str(fact.get('eventtype', '')).strip()
        
        # Clean up 'nan' values
        def clean_value(val):
            if val.lower() in ('nan', '', 'none'):
                return 'Unknown'
            return val
        
        subject = clean_value(subject)
        actorrole = clean_value(actorrole)
        eventdate = clean_value(eventdate)
        eventlocation = clean_value(eventlocation)
        safetyrisk = clean_value(safetyrisk)
        publicexposure = clean_value(publicexposure)
        truthstatus = clean_value(truthstatus)
        evidencetype = clean_value(evidencetype)
        eventtype = clean_value(eventtype)
        
        lines.append(f"FACT #{idx}")
        lines.append(f"FactID: {factid}")
        lines.append("")
        lines.append(f"PROPOSITION:")
        lines.append(f"{proposition}")
        lines.append("")
        lines.append(f"METADATA:")
        lines.append(f"  Subject: {subject}")
        lines.append(f"  ActorRole: {actorrole}")
        lines.append(f"  EventType: {eventtype}")
        lines.append(f"  EventDate: {eventdate}")
        lines.append(f"  EventLocation: {eventlocation}")
        lines.append(f"  TruthStatus: {truthstatus}")
        lines.append(f"  EvidenceType: {evidencetype}")
        lines.append(f"  SafetyRisk: {safetyrisk}")
        lines.append(f"  PublicExposure: {publicexposure}")
        
        # Add causal salience if present
        causal_score = fact.get('causal_salience_score', '')
        if causal_score and str(causal_score).strip() not in ('', 'nan', 'None'):
            lines.append(f"  CausalSalienceScore: {causal_score}")
            causal_reason = str(fact.get('causal_salience_reason', '')).strip()
            if causal_reason and causal_reason.lower() not in ('', 'nan', 'none'):
                lines.append(f"  CausalSalienceReason: {causal_reason}")
        
        lines.append("")
        lines.append("-"*80)
        lines.append("")
    
    OUTPUT_TXT.write_text("\n".join(lines), encoding='utf-8')
    print(f"\n✅ Exported {OUTPUT_TXT}")
    print(f"   Total facts: {len(facts)}")
    print(f"   File size: {OUTPUT_TXT.stat().st_size / 1024:.1f} KB")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

