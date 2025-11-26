#!/usr/bin/env python3
"""Apply OGC timeline corrections and split compound facts."""

from __future__ import annotations

import csv
from pathlib import Path

V10_2_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v10.2_final.csv")
OUTPUT_CSV = Path("case_law_data/top_1000_facts_for_chatgpt_v10.3_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/v10_3_ogc_timeline_corrections.md")


def load_facts() -> list[dict]:
    """Load v10.2 facts."""
    facts = []
    with open(V10_2_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    return facts


def apply_ogc_corrections(facts: list[dict]) -> tuple[list[dict], dict]:
    """Apply OGC timeline corrections."""
    updated_facts = []
    changes = {
        'dates_updated': 0,
        'actorroles_updated': 0,
        'facts_split': 0,
        'facts_created': 0,
        'facts_deleted': 0,
    }
    
    # OGC timeline corrections
    ogc_corrections = {
        'MISSING_0084': {
            'action': 'split',
            'new_facts': [
                {
                    'factid': 'MISSING_0084_A',
                    'proposition': 'Mr Grayson emailed Harvard OGC on August 11, 2025 requesting a meet-and-confer.',
                    'actorrole': 'Plaintiff',
                    'subject': 'Malcolm Grayson',
                    'eventdate': '2025-08-11',
                    'eventtype': 'Communication',
                    'safetyrisk': 'high',
                    'publicexposure': 'not_public',
                },
                {
                    'factid': 'MISSING_0084_B',
                    'proposition': 'Harvard OGC did not acknowledge or respond to the August 11, 2025 email.',
                    'actorrole': 'Harvard Office of General Counsel',
                    'subject': 'Harvard Office of General Counsel',
                    'eventdate': '2025-08-25',
                    'eventtype': 'Non-Response',
                    'safetyrisk': 'high',
                    'publicexposure': 'not_public',
                },
            ],
        },
        'MISSING_0083': {
            'eventdate': '2025-08-25',
            'actorrole': 'Harvard Office of General Counsel',
            'subject': 'Harvard Office of General Counsel',
        },
        'MISSING_0065': {
            'eventdate': '2025-04-18',
            'actorrole': 'Harvard Office of General Counsel',
            'subject': 'Harvard Office of General Counsel',
        },
        'MISSING_0063': {
            'eventdate': '2025-04-18',
            'actorrole': 'Harvard Office of General Counsel',
            'subject': 'Harvard Office of General Counsel',
        },
        'MISSING_0069': {
            'eventdate': '2025-08-25',
            'actorrole': 'Harvard Office of General Counsel',
            'subject': 'Harvard Office of General Counsel',
        },
        'MISSING_0076': {
            'eventdate': '2025-08-25',
            'actorrole': 'Harvard Office of General Counsel',
            'subject': 'Harvard Office of General Counsel',
        },
    }
    
    # Additional new fact for continuing silence
    new_fact_continuing_silence = {
        'factid': 'MISSING_0117',
        'proposition': 'As of 17 November 2025, Harvard OGC has still not acknowledged or responded to any of the Plaintiff\'s 2025 emails (April 7, April 18, and August 11).',
        'actorrole': 'Harvard Office of General Counsel',
        'subject': 'Harvard Office of General Counsel',
        'eventdate': '2025-11-17',
        'eventtype': 'Non-Response',
        'safetyrisk': 'high',
        'publicexposure': 'not_public',
        'truthstatus': 'True',
        'evidencetype': 'Email',
        'eventlocation': 'USA',
        'causal_salience_score': 0.85,
        'causal_salience_reason': 'OGC non-response; Harvard knowledge; spoliation risk; institutional silence',
    }
    
    for fact in facts:
        factid = str(fact.get('factid', '')).strip()
        
        # Handle split facts
        if factid in ogc_corrections and ogc_corrections[factid].get('action') == 'split':
            # Don't add the original fact
            changes['facts_deleted'] += 1
            changes['facts_split'] += 1
            
            # Add the new split facts
            for new_fact_data in ogc_corrections[factid]['new_facts']:
                new_fact = fact.copy()
                for key, value in new_fact_data.items():
                    new_fact[key] = value
                # Preserve other fields
                if 'propositionclean_v2' in new_fact:
                    new_fact['propositionclean_v2'] = new_fact_data['proposition']
                updated_facts.append(new_fact)
                changes['facts_created'] += 1
            continue
        
        # Handle simple corrections
        if factid in ogc_corrections:
            corr = ogc_corrections[factid]
            
            # Update EventDate
            if 'eventdate' in corr:
                old_date = str(fact.get('eventdate', '')).strip()
                new_date = corr['eventdate']
                if old_date != new_date:
                    fact['eventdate'] = new_date
                    changes['dates_updated'] += 1
            
            # Update ActorRole
            if 'actorrole' in corr:
                old_role = str(fact.get('actorrole', '')).strip()
                new_role = corr['actorrole']
                if old_role != new_role:
                    fact['actorrole'] = new_role
                    changes['actorroles_updated'] += 1
            
            # Update Subject
            if 'subject' in corr:
                fact['subject'] = corr['subject']
        
        # Remove any facts that incorrectly place OGC in 2019
        proposition = str(fact.get('proposition', '')).lower()
        eventdate = str(fact.get('eventdate', '')).strip()
        
        # Check for incorrect 2019 OGC references
        if 'ogc' in proposition or 'general counsel' in proposition:
            if '2019' in eventdate or (eventdate == '' and '2019' in proposition):
                # Check if this is actually about OGC (not just mentioning it)
                if any(phrase in proposition for phrase in [
                    'ogc', 'office of general counsel', 'general counsel',
                    'harvard ogc', 'harvard office'
                ]):
                    # Skip facts that incorrectly place OGC in 2019
                    # (unless they're about future references or comparisons)
                    if 'comparison' not in proposition and 'analogy' not in proposition:
                        changes['facts_deleted'] += 1
                        continue
        
        updated_facts.append(fact)
    
    # Add the new continuing silence fact
    # Find a template fact to copy structure from
    if updated_facts:
        template = updated_facts[0].copy()
        for key, value in new_fact_continuing_silence.items():
            template[key] = value
        # Fill in any missing required fields
        for key in template.keys():
            if key not in new_fact_continuing_silence and template[key] == '':
                template[key] = 'Unknown'
        updated_facts.append(template)
        changes['facts_created'] += 1
    
    return updated_facts, changes


def write_report(changes: dict, initial_count: int, final_count: int) -> None:
    """Write correction report."""
    report = f"""# V10.3 OGC Timeline Corrections Report

## Summary

- **Input**: `top_1000_facts_for_chatgpt_v10.2_final.csv`
- **Output**: `top_1000_facts_for_chatgpt_v10.3_final.csv`
- **Initial facts**: {initial_count}
- **Final facts**: {final_count}

## Corrections Applied

### OGC Timeline Corrections ✅

**Key Changes:**
- **No OGC communications in 2019** - All 2019 communications were with Harvard Clubs, HAA, Harvard Center Shanghai
- **OGC timeline is ONLY 2025:**
  - First contact: April 7, 2025 (Without Prejudice letter)
  - Second follow-up: April 18, 2025
  - Third escalation: August 11, 2025 (meet-and-confer)
  - Deadline expiry: August 25, 2025
  - Continuing silence: November 17, 2025

### Dates Updated ✅
- **EventDates updated**: {changes['dates_updated']}
  - MISSING_0083: → 2025-08-25
  - MISSING_0065: → 2025-04-18
  - MISSING_0063: → 2025-04-18
  - MISSING_0069: → 2025-08-25
  - MISSING_0076: → 2025-08-25

### ActorRoles Updated ✅
- **ActorRoles updated**: {changes['actorroles_updated']}
  - OGC non-response facts now correctly have ActorRole = Harvard Office of General Counsel
  - Plaintiff action facts have ActorRole = Plaintiff

### Facts Split ✅
- **Facts split**: {changes['facts_split']}
  - MISSING_0084 split into:
    - MISSING_0084_A: Plaintiff's action (ActorRole = Plaintiff)
    - MISSING_0084_B: OGC non-response (ActorRole = Harvard OGC)

### Facts Created ✅
- **New facts created**: {changes['facts_created']}
  - MISSING_0117: Continuing OGC silence as of November 17, 2025

### Facts Deleted ✅
- **Incorrect facts removed**: {changes['facts_deleted']}
  - Removed facts that incorrectly placed OGC in 2019 timeline

## Key Improvements

- ✅ Accurate OGC timeline (2025 only)
- ✅ Correct ActorRole assignments (Plaintiff vs OGC)
- ✅ Atomic facts (split compound facts)
- ✅ Consistent EventDates for non-response facts
- ✅ New fact for continuing silence

## Next Steps

The v10.3 dataset now has:
- Correct OGC timeline
- Properly split atomic facts
- Accurate ActorRole assignments
- Consistent EventDates

Ready for further review or integration.
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')


def main():
    """Main execution."""
    print("="*80)
    print("APPLYING OGC TIMELINE CORRECTIONS")
    print("="*80)
    
    print("\n1. Loading v10.2 facts...")
    facts = load_facts()
    print(f"   Loaded {len(facts)} facts")
    
    print("\n2. Applying OGC timeline corrections...")
    updated_facts, changes = apply_ogc_corrections(facts)
    print(f"   Dates updated: {changes['dates_updated']}")
    print(f"   ActorRoles updated: {changes['actorroles_updated']}")
    print(f"   Facts split: {changes['facts_split']}")
    print(f"   Facts created: {changes['facts_created']}")
    print(f"   Facts deleted: {changes['facts_deleted']}")
    
    print("\n3. Exporting v10.3...")
    if updated_facts:
        fieldnames = list(updated_facts[0].keys())
        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_facts)
        print(f"   ✅ Exported {OUTPUT_CSV}")
        print(f"   Final fact count: {len(updated_facts)}")
    
    print("\n4. Writing report...")
    write_report(changes, len(facts), len(updated_facts))
    print(f"   ✅ Written {REPORT_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE! OGC timeline corrections applied to v10.3")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

