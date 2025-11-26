#!/usr/bin/env python3
"""Generate v14 Top 100 highest salience facts."""

from __future__ import annotations

import csv
from pathlib import Path

FINAL_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_final.csv")
OUTPUT_CSV = Path("case_law_data/top_100_facts_final.csv")
OUTPUT_TXT = Path("case_law_data/top_100_facts_final.txt")
REPORT_PATH = Path("reports/analysis_outputs/final_top_100_report.md")


def load_facts() -> list[dict]:
    """Load final facts."""
    facts = []
    with open(FINAL_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    return facts


def get_salience_score(fact: dict) -> float:
    """Get salience score for sorting."""
    # Try new_salience_score first
    score = fact.get('new_salience_score', '')
    if score and str(score).strip() not in ('', 'nan', 'None'):
        try:
            return float(score)
        except (ValueError, TypeError):
            pass
    
    # Fall back to causal_salience_score
    score = fact.get('causal_salience_score', '')
    if score and str(score).strip() not in ('', 'nan', 'None'):
        try:
            return float(score)
        except (ValueError, TypeError):
            pass
    
    return 0.0


def main():
    """Main execution."""
    print("="*80)
    print("GENERATING FINAL TOP 100")
    print("(Final version after 25 rounds of refinement)")
    print("="*80)
    
    print("\n1. Loading final facts...")
    facts = load_facts()
    print(f"   Loaded {len(facts)} facts")
    
    print("\n2. Sorting by salience...")
    # Sort by salience score (highest first)
    facts_sorted = sorted(facts, key=get_salience_score, reverse=True)
    
    # Get top 100
    top_100 = facts_sorted[:100]
    print(f"   Selected top 100 facts")
    
    print("\n3. Exporting Top 100 CSV...")
    if top_100:
        fieldnames = set()
        for fact in top_100:
            fieldnames.update(fact.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fact in top_100:
                row = {}
                for field in fieldnames:
                    row[field] = fact.get(field, '')
                writer.writerow(row)
        print(f"   ✅ Exported {OUTPUT_CSV}")
    
    print("\n4. Exporting Top 100 TXT...")
    lines = []
    lines.append("="*80)
    lines.append("TOP 100 FACTS - FINAL VERSION")
    lines.append("(After 25 rounds of refinement: v16 → v25)")
    lines.append("="*80)
    lines.append("")
    lines.append("Highest causal salience facts from the master database")
    lines.append(f"Total facts in database: {len(facts)}")
    lines.append(f"Top 100 facts (highest salience)")
    lines.append("")
    lines.append("="*80)
    lines.append("")
    
    for idx, fact in enumerate(top_100, 1):
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
        salience_score = get_salience_score(fact)
        salience_reason = str(fact.get('causal_salience_reason', '')).strip()
        
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
        
        lines.append(f"RANK #{idx} (Salience: {salience_score:.3f})")
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
        if salience_reason and salience_reason.lower() not in ('', 'nan', 'none'):
            lines.append(f"  CausalSalienceReason: {salience_reason}")
        lines.append("")
        lines.append("-"*80)
        lines.append("")
    
    OUTPUT_TXT.write_text("\n".join(lines), encoding='utf-8')
    print(f"   ✅ Exported {OUTPUT_TXT}")
    
    print("\n5. Writing report...")
    report = f"""# Final Top 100 Facts Report

## Summary

- **Input**: `top_1000_facts_for_chatgpt_final.csv`
- **Output**: `top_100_facts_final.csv` and `top_100_facts_final.txt`
- **Total facts in database**: {len(facts)}
- **Top 100 facts selected**: {len(top_100)}

## Salience Distribution

- **Highest salience**: {get_salience_score(top_100[0]):.3f}
- **Lowest salience (top 100)**: {get_salience_score(top_100[-1]):.3f}

## Top 10 Facts by Salience

"""
    
    for idx, fact in enumerate(top_100[:10], 1):
        factid = fact.get('factid', '')
        prop = str(fact.get('proposition', ''))[:100]
        score = get_salience_score(fact)
        report += f"{idx}. **{factid}** (Score: {score:.3f})\n"
        report += f"   {prop}...\n\n"
    
    report += f"""
## Files Generated

- **CSV**: `top_100_facts_final.csv`
- **TXT**: `top_100_facts_final.txt`

## Usage

These are the 100 highest-salience facts from the final master database (after 25 rounds of refinement), ranked by causal importance to the legal case.
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')
    print(f"   ✅ Written {REPORT_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE! Final Top 100 generated")
    print(f"{'='*80}")
    print(f"\nFiles created:")
    print(f"  📄 {OUTPUT_CSV}")
    print(f"  📄 {OUTPUT_TXT}")
    print(f"  📄 {REPORT_PATH}")


if __name__ == "__main__":
    main()


