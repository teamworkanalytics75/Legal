#!/usr/bin/env python3
"""
Generate v24 gap-bridging facts from 5 new answers.
Addresses:
1. Harvard Club knowledge of Xi slide (before April 19, 2019)
2. PRC handler timeline (2017 & post-defamation)
3. Surveillance indicators in PRC (2019-2022)
4. Largest economic losses after defamation
5. Statement 1 persistence through 2024-2025
"""

import csv
from pathlib import Path
from datetime import datetime

V23_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v23_final.csv")
V24_OUTPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v24_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/v24_gap_bridging_facts_report.md")

# 5 new facts based on user answers
NEW_FACTS = [
    # ========================================================================
    # 1. Harvard Club Knowledge of Xi Slide (Before April 19, 2019)
    # ========================================================================
    {
        "factid": "V24_KNOW_XI_001",
        "proposition": "Multiple members of the Harvard Club of Shanghai had firsthand knowledge of Mr. Grayson's Xi-related presentation materials before the April 19, 2019 publications. One such individual was Guan, a Harvard College graduate and member of the Harvard Club of Shanghai, who served as Mr. Grayson's translator during several public talks and appears in the same photograph that Yi Wang later confronted Mr. Grayson with. Because Guan translated these presentations in real time, he necessarily saw and understood the Xi Mingze slide and the political sensitivity of the content, establishing that Harvard-affiliated individuals had prior knowledge of the Xi slide before Statement 1 was issued.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Allegation",
        "eventdate": "2019-03-15",  # Before April 19, approximate date of talks with Guan
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.92",
        "causal_salience_reason": "Establishes Harvard Club knowledge of Xi slide before defamation; strengthens knowledge narrative; connects Harvard Club to foreseeability; high impact on knowledge cluster",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 2. PRC Handler Timeline - First Period (2017)
    # ========================================================================
    {
        "factid": "V24_HARM_HANDLER_2017_001",
        "proposition": "Beginning around January 2017 in Shanghai, during a year-long media tour, Mr. Grayson was approached by an individual who inserted himself into his professional activities, offered unsolicited 'connections,' and sought detailed information about his movements and work. At the time, Mr. Grayson did not understand this behavior as surveillance, but in retrospect, the pattern is consistent with PRC-linked monitoring of individuals engaged in public speaking and media activities.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2017-01-01",  # Approximate start of 2017 media tour
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.85",
        "causal_salience_reason": "Documents 2017 handler period; establishes pattern of PRC monitoring; connects to pre-defamation surveillance context",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 3. PRC Handler Timeline - Second Period (Post-Defamation, 2019-2022)
    # ========================================================================
    {
        "factid": "V24_HARM_HANDLER_2020_001",
        "proposition": "After Harvard-affiliated statements circulated in April 2019, Mr. Grayson largely remained indoors for several months. When he began re-engaging socially in approximately May 2020, again in Shanghai, he was approached by another individual who offered business opportunities and sought information about his activities. In retrospect, the patterns of questioning and contact are consistent with monitoring by a PRC-linked handler, lasting through the period in which he feared leaving China until 2022, directly linking Harvard's defamatory statements to post-defamation surveillance.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2020-05-01",  # Approximate onset of post-defamation handler contact
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.95",
        "causal_salience_reason": "Post-defamation handler monitoring; directly ties Harvard statements to PRC surveillance; strengthens safety harm narrative; extreme safety risk; worst for Harvard",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 4. Surveillance Indicators in PRC (2019-2022)
    # ========================================================================
    {
        "factid": "V24_HARM_SURVEILLANCE_001",
        "proposition": "The individual who approached Mr. Grayson in 2020 asked questions that were consistent with PRC information-gathering practices: targeted inquiries about his professional contacts, financial activities, social networks, and travel intentions. The pattern mirrored known forms of soft surveillance, and the timing—immediately following defamatory publications tied to politically sensitive material—aligns with how monitoring is typically initiated in similar cases, supporting the allegation that Harvard's statements triggered PRC-linked surveillance.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2020-05-01",  # Approximate onset
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.93",
        "causal_salience_reason": "Documents soft surveillance indicators; connects defamation to PRC monitoring patterns; strengthens safety harm narrative; extreme safety risk",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 5. Largest Economic Losses After Defamation
    # ========================================================================
    {
        "factid": "V24_HARM_ECON_003",
        "proposition": "Before the April 2019 defamation, Mr. Grayson routinely secured high-value consulting contracts, including multiple agreements worth approximately USD $50,000 each. After the Harvard Club statements and the resulting Monkey and Résumé articles circulated, he never again obtained a contract of comparable value. In addition, parents withdrew students and demanded refunds after reading the Monkey article, educational partners cancelled classes and speaking engagements, and investment opportunities he was actively pursuing in early 2019 collapsed once the defamatory narrative spread, representing a direct and measurable decline in professional viability.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.91",
        "causal_salience_reason": "Documents specific economic losses ($50K contracts); quantifies harm; connects defamation to measurable financial impact; strengthens harm narrative",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 6. Statement 1 Persistence Through 2024-2025
    # ========================================================================
    {
        "factid": "V24_PUB_PERSIST_001",
        "proposition": "Statement 1 was distributed in PDF form directly inside Harvard-affiliated WeChat groups in April 2019. Items shared in WeChat groups cannot be reliably deleted, and once a member has opened the PDF, it remains accessible in their personal cache and download history. Mr. Grayson informed Harvard of this persistence in 2019. During preparation for this litigation, he remained a member of the relevant Harvard alumni group and was able to re-download the same Statement 1 PDF in 2025, meaning that Statement 1 has been continuously available to hundreds or thousands of Harvard alumni and PRC-based readers from 2019 through 2025, regardless of whether website versions were later removed.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Publication",
        "eventdate": "2019-04-19",  # Initial publication
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Exhibit",
        "safetyrisk": "high",
        "publicexposure": "already_public",
        "causal_salience_score": "0.90",
        "causal_salience_reason": "Documents Statement 1 persistence through 2024-2025; establishes continuous availability; strengthens publication narrative; shows Harvard knowledge of persistence",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
]


def add_gap_bridging_facts_to_v23():
    """Add v24 gap-bridging facts to v23 dataset."""
    print("="*80)
    print("GENERATING V24 GAP-BRIDGING FACTS")
    print("="*80)
    
    # Load v23 facts
    print("\n1. Loading v23 facts...")
    facts = []
    with open(V23_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    print(f"   Loaded {len(facts)} facts from v23")
    
    # Add new facts
    print(f"\n2. Adding {len(NEW_FACTS)} new gap-bridging facts...")
    for fact in NEW_FACTS:
        facts.append(fact)
        factid = fact.get('factid', '')
        print(f"   ✅ Added {factid}")
    
    print(f"\n   Total facts in v24: {len(facts)}")
    
    # Write v24 CSV
    print("\n3. Writing v24 CSV...")
    if facts:
        # Get all fieldnames
        all_fieldnames = set()
        for fact in facts:
            all_fieldnames.update(fact.keys())
        fieldnames = sorted(list(all_fieldnames))
        
        with open(V24_OUTPUT, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fact in facts:
                row = {}
                for field in fieldnames:
                    row[field] = fact.get(field, '')
                writer.writerow(row)
        print(f"   ✅ Exported {V24_OUTPUT}")
    
    # Generate report
    print("\n4. Generating report...")
    report = f"""# V24 Gap-Bridging Facts Report

## Summary

Added {len(NEW_FACTS)} new gap-bridging facts to v23 based on 5 detailed answers addressing knowledge, surveillance, economic harm, and publication persistence.

## Facts Added

### 1. Harvard Club Knowledge of Xi Slide (Before April 19, 2019)
- **V24_KNOW_XI_001**: Guan (Harvard Club member, translator) had firsthand knowledge of Xi slide before Statement 1
- Establishes Harvard Club knowledge before defamation
- Strengthens knowledge and foreseeability narratives

### 2. PRC Handler Timeline - First Period (2017)
- **V24_HARM_HANDLER_2017_001**: 2017 handler during media tour in Shanghai
- Documents pre-defamation surveillance context
- Establishes pattern of PRC monitoring

### 3. PRC Handler Timeline - Second Period (Post-Defamation, 2019-2022)
- **V24_HARM_HANDLER_2020_001**: Post-defamation handler (May 2020 onward)
- Directly ties Harvard statements to PRC surveillance
- Strengthens safety harm narrative

### 4. Surveillance Indicators in PRC (2019-2022)
- **V24_HARM_SURVEILLANCE_001**: Soft surveillance patterns (professional contacts, financial activities, travel)
- Documents surveillance indicators consistent with PRC practices
- Connects defamation to monitoring patterns

### 5. Largest Economic Losses After Defamation
- **V24_HARM_ECON_003**: $50K contracts lost, refunds, cancelled opportunities
- Quantifies specific economic harm
- Documents measurable financial impact

### 6. Statement 1 Persistence Through 2024-2025
- **V24_PUB_PERSIST_001**: PDF in WeChat groups, continuously available 2019-2025
- Documents Statement 1 persistence despite website removals
- Establishes continuous availability to hundreds/thousands of readers

## Expected Impact

These facts should:
- **Strengthen knowledge narrative** (Harvard Club knew about Xi slide before defamation)
- **Strengthen harm narrative** (quantified economic losses, surveillance patterns)
- **Strengthen publication narrative** (Statement 1 persistence)
- **Strengthen safety narrative** (two-period handler timeline, surveillance indicators)
- **Connect defamation to concrete harms** (economic, surveillance, publication persistence)

## Statistics

- **Facts in v23**: {len(facts) - len(NEW_FACTS)}
- **Facts added**: {len(NEW_FACTS)}
- **Facts in v24**: {len(facts)}

## Files Generated

- **V24 CSV**: `case_law_data/top_1000_facts_for_chatgpt_v24_final.csv`
- **Report**: `reports/analysis_outputs/v24_gap_bridging_facts_report.md`

## Next Steps

1. Review v24 facts for accuracy
2. Re-run network gap analysis to verify improvements
3. Generate v24 outputs (top 100, causation nodes, etc.)
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ Written {REPORT_PATH}")
    
    print()
    print("="*80)
    print("V24 GAP-BRIDGING FACTS GENERATED")
    print("="*80)
    print()
    print(f"✅ Added {len(NEW_FACTS)} new facts")
    print(f"✅ Total facts in v24: {len(facts)}")
    print(f"✅ Output: {V24_OUTPUT}")
    print()
    print("Key additions:")
    print("  • Harvard Club knowledge of Xi slide (before defamation)")
    print("  • Two-period PRC handler timeline (2017 & post-defamation)")
    print("  • Surveillance indicators (soft surveillance patterns)")
    print("  • Quantified economic losses ($50K contracts)")
    print("  • Statement 1 persistence (2019-2025 via WeChat PDF)")


if __name__ == "__main__":
    add_gap_bridging_facts_to_v23()

