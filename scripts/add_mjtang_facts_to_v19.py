#!/usr/bin/env python3
"""
Add MJ Tang evidence facts to v18 to create v19.
These 3 facts provide critical evidence of Harvard's control over clubs
and the circular instruction chain in April 2019.
"""

import csv
from pathlib import Path

V18_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v18_final.csv")
V19_OUTPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v19_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/v19_mjtang_additions_report.md")

# New facts to add
NEW_FACTS = [
    {
        "factid": "V19_MJTANG_001",
        "proposition": "On April 2019, MJ Tang, then-President of the Harvard Club of Shanghai, told the plaintiff via WeChat that the defamation dispute was \"a university thing,\" indicating that Harvard University—not merely the clubs—was responsible for the 2019 statements.",
        "subject": "Harvard",
        "actorrole": "Harvard Club",
        "eventtype": "Communication",
        "eventdate": "2019-04-24",
        "eventlocation": "Shanghai / PRC",
        "truthstatus": "True",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.90",
        "causal_salience_reason": "Harvard-controlled actor present; admission of Harvard responsibility; communication event; contributes to showing Harvard → Club → Publication chain.",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V19_MJTANG_002",
        "proposition": "During the same April 2019 WeChat exchange, MJ Tang told the plaintiff she would forward his letter and information \"to the university,\" demonstrating that Harvard Club leadership understood the matter as requiring Harvard University review and action.",
        "subject": "Harvard",
        "actorrole": "Harvard Club",
        "eventtype": "Communication",
        "eventdate": "2019-04-24",
        "eventlocation": "Shanghai / PRC",
        "truthstatus": "True",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.92",
        "causal_salience_reason": "Harvard-controlled actor; acknowledgement of Harvard oversight; demonstrates club-to-university reporting path; strengthens Harvard → Club command structure evidence.",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V19_MJTANG_003",
        "proposition": "The WeChat exchange between the plaintiff and MJ Tang shows that Harvard Center Shanghai had already directed the plaintiff to speak with the Harvard Club, and that the Harvard Club then redirected the matter back to Harvard University, evidencing a circular instruction chain among Harvard entities in April 2019.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Communication",
        "eventdate": "2019-04-24",
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.94",
        "causal_salience_reason": "Exposes institutional coordination; demonstrates Harvard Center ↔ Harvard Club ↔ Harvard administrative loop; supports inference of Harvard-level authorization and knowledge.",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
]


def add_facts_to_v18():
    """Add new MJ Tang facts to v18 and create v19."""
    print("="*80)
    print("ADDING MJ TANG FACTS TO CREATE V19")
    print("="*80)
    print()
    
    # Load v18 facts
    print("1. Loading v18 facts...")
    facts = []
    fieldnames = None
    
    with open(V18_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            facts.append(row)
    
    print(f"   Loaded {len(facts)} facts from v18")
    
    # Add new facts
    print("\n2. Adding 3 new MJ Tang facts...")
    for new_fact in NEW_FACTS:
        # Ensure all fields are present
        fact_row = {}
        for field in fieldnames:
            fact_row[field] = new_fact.get(field.lower(), "")
        facts.append(fact_row)
        print(f"   ✅ Added {new_fact['factid']}")
    
    print(f"\n   Total facts in v19: {len(facts)}")
    
    # Write v19 CSV
    print("\n3. Writing v19 CSV...")
    with open(V19_OUTPUT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(facts)
    
    print(f"   ✅ Exported {V19_OUTPUT}")
    
    # Generate report
    print("\n4. Generating report...")
    report = f"""# V19 MJ Tang Facts Addition Report

## Summary

Added 3 new facts to v18 to create v19. These facts provide critical evidence of Harvard's control over clubs and the circular instruction chain in April 2019.

## Facts Added

### V19_MJTANG_001
- **Proposition**: MJ Tang told plaintiff the defamation dispute was "a university thing"
- **CausalSalienceScore**: 0.90
- **Key Evidence**: Admission of Harvard responsibility

### V19_MJTANG_002
- **Proposition**: MJ Tang said she would forward information "to the university"
- **CausalSalienceScore**: 0.92
- **Key Evidence**: Acknowledgement of Harvard oversight, club-to-university reporting path

### V19_MJTANG_003
- **Proposition**: Circular instruction chain (Harvard Center → Club → Harvard University)
- **CausalSalienceScore**: 0.94
- **Key Evidence**: Institutional coordination, Harvard-level authorization

## Impact

### Legal Theories Strengthened
- Non-delegable function (admissions)
- Harvard's actual knowledge
- Harvard's control over club actions
- Awareness of harm and risk
- Reasonable foreseeability
- Institutional coordination (key in 1782 and spoliation arguments)

### Causal DAG Improvements
- Unifies April 2019 communication nodes into a tight, documented, multi-actor chain
- Fills major missing link in v18
- Demonstrates Harvard Center ↔ Harvard Club ↔ Harvard University loop

## Statistics

- **Facts in v18**: {len(facts) - 3}
- **Facts added**: 3
- **Facts in v19**: {len(facts)}
- **New FactIDs**: V19_MJTANG_001, V19_MJTANG_002, V19_MJTANG_003

## Files Generated

- **V19 CSV**: `case_law_data/top_1000_facts_for_chatgpt_v19_final.csv`
- **Report**: `reports/analysis_outputs/v19_mjtang_additions_report.md`

## Next Steps

1. Generate v19 top 100 facts
2. Generate v19 causation nodes
3. Generate v19 human-readable TXT
4. Review v19 for any additional updates needed
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ Written {REPORT_PATH}")
    
    print()
    print("="*80)
    print("V19 CREATED SUCCESSFULLY")
    print("="*80)
    print()
    print(f"✅ Added 3 new MJ Tang facts")
    print(f"✅ Total facts in v19: {len(facts)}")
    print(f"✅ Output: {V19_OUTPUT}")
    print()
    print("New facts:")
    for fact in NEW_FACTS:
        print(f"  - {fact['factid']}: {fact['proposition'][:80]}...")
    print()


if __name__ == "__main__":
    add_facts_to_v18()

