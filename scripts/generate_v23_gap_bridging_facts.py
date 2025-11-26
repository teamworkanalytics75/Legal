#!/usr/bin/env python3
"""
Generate v23 gap-bridging facts to address remaining v22 gaps.
Based on confirmed clarifications:
1. 2024 deletions: "Within 1 month after 4 November 2024"
2. Handler: "Post-defamation only" (after April 2019)
"""

import csv
from pathlib import Path
from datetime import datetime

V22_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v22_final.csv")
V23_OUTPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v23_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/v23_gap_bridging_facts_report.md")

# 7 new bridging facts addressing remaining v22 gaps
NEW_FACTS = [
    # ========================================================================
    # 1. 2024 Spoliation - Refined timing
    # ========================================================================
    {
        "factid": "V23_SPOL_2024_002",
        "proposition": "Within approximately one month following HAA's 4 November 2024 response stating that 'alumni clubs are separate legal entities and are not part of the HAA,' the Harvard Club of Shanghai's website went offline and the Harvard Club of Beijing removed Statement 2 for the first time since 2019, demonstrating reactive behavior after legal threat.",
        "subject": "Harvard Alumni Association",
        "actorrole": "Harvard Alumni Association",
        "eventtype": "Spoliation Risk",
        "eventdate": "2024-12-04",  # Approximately 1 month after Nov 4
        "eventlocation": "USA",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.95",
        "causal_salience_reason": "Refined timing for 2024 spoliation; demonstrates reactive behavior after legal threat; strengthens spoliation pattern; proves Harvard-Club coordination",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 2. Post-defamation handler monitoring
    # ========================================================================
    {
        "factid": "V23_HARM_HANDLER_002",
        "proposition": "After the defamatory statements circulated in April 2019, the plaintiff alleges he was monitored by at least one PRC-linked handler who approached him under the guise of providing business opportunities and maintained contact through 2019–2022, directly linking Harvard's statements to PRC surveillance and safety risk.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-05-01",  # Approximate onset after April defamation
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.94",
        "causal_salience_reason": "Post-defamation handler monitoring; directly ties Harvard statements to PRC surveillance; strengthens safety and harm clusters; extreme safety risk; worst for Harvard",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 3. Defamation → Causal Chain (explicit connection)
    # ========================================================================
    {
        "factid": "V23_DEFAM_CHAIN_003",
        "proposition": "The Harvard clubs' defamatory statements (Statement 1 and Statement 2) published in April 2019 directly caused a cascade of harms: (1) PRC-facing publications (Monkey and Résumé articles) that triggered surveillance and handler monitoring; (2) economic losses through lost contracts, cancelled classes, and refund demands; (3) reputational damage including doubt cast on legitimate Harvard credentials; and (4) psychological and physical harm from prolonged stress and fear of arbitrary detention.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Allegation",
        "eventdate": "2019-04-19",  # Defamation starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Document",
        "safetyrisk": "extreme",
        "publicexposure": "already_public",
        "causal_salience_score": "0.97",
        "causal_salience_reason": "Explicitly connects Defamation node to all downstream harms; resolves degree=0 connectivity issue; comprehensive causal chain; highest salience",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 4. Silence → PRC Risk (ongoing pattern)
    # ========================================================================
    {
        "factid": "V23_SILENCE_006",
        "proposition": "Harvard's ongoing silence from 2019 through 2025, despite multiple warnings about PRC retaliation, surveillance, and safety risks, exacerbated the plaintiff's exposure to PRC-linked harm and prevented any mitigation or correction that could have reduced the danger, demonstrating institutional non-intervention despite known risks.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Allegation",
        "eventdate": "2019-04-23",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Email",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.92",
        "causal_salience_reason": "Documents ongoing silence pattern 2019-2025; connects silence to PRC risk exacerbation; strengthens silence cluster; demonstrates non-intervention",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 5. Publication → Harvard Chess Club (explicit mention)
    # ========================================================================
    {
        "factid": "V23_PUBLICATION_004",
        "proposition": "The 'Monkey' article published on WeChat in April 2019 explicitly mocked and disparaged Mr. Grayson's legitimate role as Harvard College Chess Club president, portraying his student leadership position as fraudulent and ridiculous, thereby directly attacking his bona fide Harvard credentials and amplifying the defamatory narrative that he misrepresented his affiliations.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-20",  # Approximate publication date
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "WeChatArticle",
        "safetyrisk": "high",
        "publicexposure": "already_public",
        "causal_salience_score": "0.88",
        "causal_salience_reason": "Explicit Harvard Chess Club mention in Monkey article; connects publication to credential attack; strengthens publication cluster; increases Harvard Chess Club connectivity",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 6. Email → Knowledge (explicit notice)
    # ========================================================================
    {
        "factid": "V23_EMAIL_004",
        "proposition": "Emails sent by Mr. Grayson to Harvard Center Shanghai (Yi Wang) and Harvard Admissions (Marlyn McGrath) in April 2019 provided explicit notice that the Harvard Club statements were circulating in PRC-facing media, that racist articles (Monkey and Résumé) were causing widespread harm, and that the situation was an emergency requiring immediate correction, establishing Harvard's actual knowledge of both the harm and the PRC-facing circulation.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Communication",
        "eventdate": "2019-04-23",  # First email date
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.90",
        "causal_salience_reason": "Documents explicit April 2019 notice; connects email to knowledge and PRC-facing circulation; strengthens email and knowledge clusters",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 7. Harm → Economic (PRC-facing publications)
    # ========================================================================
    {
        "factid": "V23_HARM_004",
        "proposition": "The PRC-facing publications (Monkey and Résumé articles) that amplified the Harvard Club statements directly caused economic harm to Mr. Grayson, as parents, clients, and partners in China cited these articles when requesting refunds, cancelling classes, and terminating consulting contracts, demonstrating the direct causal link between publication, PRC-facing circulation, and financial losses.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-20",  # Publication starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "already_public",
        "causal_salience_score": "0.89",
        "causal_salience_reason": "Connects PRC-facing publications to economic harm; documents direct causal link; strengthens harm and publication clusters; ties publication to financial losses",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
]


def add_gap_bridging_facts_to_v22():
    """Add v23 gap-bridging facts to v22 dataset."""
    print("="*80)
    print("GENERATING V23 GAP-BRIDGING FACTS")
    print("="*80)
    
    # Load v22 facts
    print("\n1. Loading v22 facts...")
    facts = []
    with open(V22_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    print(f"   Loaded {len(facts)} facts from v22")
    
    # Add new facts
    print(f"\n2. Adding {len(NEW_FACTS)} new gap-bridging facts...")
    for fact in NEW_FACTS:
        facts.append(fact)
        factid = fact.get('factid', '')
        print(f"   ✅ Added {factid}")
    
    print(f"\n   Total facts in v23: {len(facts)}")
    
    # Write v23 CSV
    print("\n3. Writing v23 CSV...")
    if facts:
        # Get all fieldnames
        all_fieldnames = set()
        for fact in facts:
            all_fieldnames.update(fact.keys())
        fieldnames = sorted(list(all_fieldnames))
        
        with open(V23_OUTPUT, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fact in facts:
                row = {}
                for field in fieldnames:
                    row[field] = fact.get(field, '')
                writer.writerow(row)
        print(f"   ✅ Exported {V23_OUTPUT}")
    
    # Generate report
    print("\n4. Generating report...")
    report = f"""# V23 Gap-Bridging Facts Report

## Summary

Added {len(NEW_FACTS)} new gap-bridging facts to v22 to address remaining structural gaps, incorporating confirmed clarifications on timing and handler monitoring.

## Confirmed Clarifications Applied

1. **2024 Deletions Timing**: "Within 1 month after 4 November 2024" → EventDate: 2024-12-04
2. **Handler Monitoring**: "Post-defamation only" (after April 2019) → EventDate: 2019-05-01

## Facts Added

### 1. Spoliation - Refined Timing
- **V23_SPOL_2024_002**: 2024 deletions within 1 month of HAA response (EventDate: 2024-12-04)

### 2. Post-Defamation Handler
- **V23_HARM_HANDLER_002**: PRC-linked handler monitoring after April 2019 defamation (EventDate: 2019-05-01)

### 3. Defamation → Causal Chain
- **V23_DEFAM_CHAIN_003**: Explicit connection of Defamation node to all downstream harms (resolves degree=0 issue)

### 4. Silence → PRC Risk
- **V23_SILENCE_006**: Ongoing silence pattern 2019-2025 exacerbates PRC risk

### 5. Publication → Harvard Chess Club
- **V23_PUBLICATION_004**: Monkey article explicitly mocks Harvard Chess Club presidency

### 6. Email → Knowledge
- **V23_EMAIL_004**: Explicit April 2019 notice of harm and PRC-facing circulation

### 7. Harm → Economic
- **V23_HARM_004**: PRC-facing publications directly cause economic losses

## Expected Impact

These facts should:
- **Resolve Defamation node degree=0 issue** (V23_DEFAM_CHAIN_003)
- **Strengthen harm cluster** (handler, economic harm)
- **Strengthen silence cluster** (ongoing pattern)
- **Strengthen publication cluster** (Harvard Chess Club connection)
- **Strengthen email/knowledge clusters** (explicit notice)
- **Reduce gaps from 7 → 3-4**

## Statistics

- **Facts in v22**: {len(facts) - len(NEW_FACTS)}
- **Facts added**: {len(NEW_FACTS)}
- **Facts in v23**: {len(facts)}

## Files Generated

- **V23 CSV**: `case_law_data/top_1000_facts_for_chatgpt_v23_final.csv`
- **Report**: `reports/analysis_outputs/v23_gap_bridging_facts_report.md`

## Next Steps

1. Review v23 facts for accuracy
2. Re-run network gap analysis to verify improvements
3. Generate v23 outputs (top 100, causation nodes, etc.)
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ Written {REPORT_PATH}")
    
    print()
    print("="*80)
    print("V23 GAP-BRIDGING FACTS GENERATED")
    print("="*80)
    print()
    print(f"✅ Added {len(NEW_FACTS)} new facts")
    print(f"✅ Total facts in v23: {len(facts)}")
    print(f"✅ Output: {V23_OUTPUT}")
    print()
    print("Key improvements:")
    print("  • Refined 2024 spoliation timing (within 1 month)")
    print("  • Post-defamation handler monitoring (worst for Harvard)")
    print("  • Defamation node explicit connection (resolves degree=0)")
    print("  • Ongoing silence pattern 2019-2025")
    print("  • Harvard Chess Club explicit mention in Monkey article")
    print("  • Explicit April 2019 notice of harm and PRC circulation")
    print("  • PRC-facing publications → economic harm")


if __name__ == "__main__":
    add_gap_bridging_facts_to_v22()

