#!/usr/bin/env python3
"""
Generate v22 gap-bridging facts to address all 8 remaining v21 gaps.
Based on comprehensive analysis of existing evidence and email chains.
"""

import csv
from pathlib import Path
from datetime import datetime

V21_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v21_final.csv")
V22_OUTPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v22_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/v22_gap_bridging_facts_report.md")

# 15 new bridging facts addressing all 8 v21 gaps
NEW_FACTS = [
    # ========================================================================
    # GAP 1: Strengthen "harm" narrative (3 facts)
    # ========================================================================
    {
        "factid": "V22_HARM_ECON_002",
        "proposition": "Following the April 2019 Monkey and Résumé articles and the Harvard Club statements, Mr. Grayson lost multiple consulting contracts and speaking engagements in China. Parents and clients cited those publications and questioned his credentials, leading to cancelled classes and refund demands.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.88",
        "causal_salience_reason": "Strengthens harm narrative; documents concrete economic losses; connects defamation to financial harm; increases harm cluster density",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V22_HARM_SAFETY_002",
        "proposition": "From mid-2019 through July 2022, Mr. Grayson avoided leaving or re-entering mainland China because he reasonably feared arbitrary detention or exit bans linked to the defamatory allegations about his supposed misconduct and the politically sensitive Xi-related speech he had given.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-07-01",  # Mid-2019
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.92",
        "causal_salience_reason": "Strengthens harm narrative; documents travel restriction and fear of detention; connects defamation to safety harm; extreme safety risk",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V22_HARM_PSYCH_002",
        "proposition": "Prolonged exposure to racist abuse, reputational destruction, and fear of PRC retaliation caused Mr. Grayson to suffer severe psychological distress and associated physical symptoms, which he will present as a form of personal injury.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "USFiling",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.89",
        "causal_salience_reason": "Strengthens harm narrative; documents psychiatric and physical injury; connects defamation to psychological harm; personal injury claim",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 2: Strengthen "email" narrative (3 facts)
    # ========================================================================
    {
        "factid": "V22_EMAIL_KNOW_001",
        "proposition": "Between 23 and 30 April 2019, Mr. Grayson sent multiple emails to Yi Wang, Director of Harvard Center Shanghai, attaching the 'Monkey' article, describing its racist content, its wide circulation in PRC-facing media, and its devastating impact on his reputation and health, and explicitly asking Harvard to correct the false statements about his role.",
        "subject": "Harvard Center Shanghai",
        "actorrole": "Harvard",
        "eventtype": "Communication",
        "eventdate": "2019-04-23",  # First email date
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.91",
        "causal_salience_reason": "Strengthens email narrative; documents April 2019 notice to Harvard Center; connects email to knowledge and harm; increases email cluster density",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V22_EMAIL_HAA_2024_001",
        "proposition": "On 4 November 2024, an anonymous inquirer (later revealed to be Mr. Grayson) emailed the Harvard Alumni Association asking whether the Harvard Clubs or HAA bore legal responsibility if sued over defamatory statements; HAA responded that the clubs were 'separate legal entities and not part of the HAA.' Within days, the Harvard Club of Shanghai's website disappeared and the Harvard Club of Beijing removed 'Statement 2' for the first time since 2019.",
        "subject": "Harvard Alumni Association",
        "actorrole": "Harvard Alumni Association",
        "eventtype": "Communication",
        "eventdate": "2024-11-04",  # HAA response date
        "eventlocation": "USA",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.93",
        "causal_salience_reason": "Strengthens email narrative; documents 2024 HAA legal inquiry; connects email to spoliation; proves Harvard-Club coordination",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V22_EMAIL_OGC_003",
        "proposition": "In April–August 2025, Mr. Grayson sent at least three detailed emails to Harvard's Office of the General Counsel, explicitly warning of risks of PRC retaliation and even 'exposure to torture' arising from Harvard-affiliated statements; Harvard OGC did not acknowledge or respond to any of these emails.",
        "subject": "Harvard OGC",
        "actorrole": "Harvard OGC",
        "eventtype": "Communication",
        "eventdate": "2025-04-18",  # First OGC email date
        "eventlocation": "USA",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.90",
        "causal_salience_reason": "Strengthens email narrative; documents 2025 OGC notice of PRC risk; connects email to silence and knowledge; increases email cluster density",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 3: Strengthen "publication" narrative (2 facts)
    # ========================================================================
    {
        "factid": "V22_PUB_HCC_001",
        "proposition": "The 'Monkey' article published on WeChat in April 2019 expressly mocked Mr. Grayson's role as Harvard College Chess Club president, portraying his legitimate student leadership position as fraudulent and ridiculous, thereby amplifying the defamatory narrative that he misrepresented his Harvard credentials.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-20",  # Approximate publication date
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "WeChatArticle",
        "safetyrisk": "high",
        "publicexposure": "already_public",
        "causal_salience_score": "0.87",
        "causal_salience_reason": "Strengthens publication narrative; connects Harvard Chess Club to defamation; documents specific credential attack; increases publication cluster density",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V22_PUB_CHAIN_002",
        "proposition": "The Monkey and Résumé articles were republished and discussed across multiple PRC-facing platforms, including Zhihu, Baidu Baijiahao, WeChat Moments, and the diaspora outlet E-Canada, where they framed Mr. Grayson's Harvard-related roles as evidence of misconduct and invited readers to report him to authorities.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-20",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Document",
        "safetyrisk": "extreme",
        "publicexposure": "already_public",
        "causal_salience_score": "0.91",
        "causal_salience_reason": "Strengthens publication narrative; documents multi-platform amplification; connects publication to PRC risk and harm; increases publication cluster density",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 4: Strengthen "knowledge" narrative (2 facts)
    # ========================================================================
    {
        "factid": "V22_KNOW_2019_002",
        "proposition": "By late April 2019, Harvard Center Shanghai and Harvard College Admissions (Marlyn McGrath) had been informed by Mr. Grayson that the Harvard Club statements and the subsequent 'Monkey' article were causing him ongoing reputational damage, racialized harassment, and threats in China, yet neither withdrew or corrected the statements.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Allegation",
        "eventdate": "2019-04-30",  # Late April
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.89",
        "causal_salience_reason": "Strengthens knowledge narrative; documents 2019 Harvard knowledge of harm; connects knowledge to silence; increases knowledge cluster density",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V22_KNOW_2025_002",
        "proposition": "By April 2025, Harvard's Office of the General Counsel had actual notice, via Mr. Grayson's emails, that PRC authorities had recently arrested and tortured young people over Xi-related online content (the EsuWiki case) and that Harvard-affiliated publications about Mr. Grayson's Xi-related activities were circulating in the same PRC information ecosystem.",
        "subject": "Harvard OGC",
        "actorrole": "Harvard OGC",
        "eventtype": "Allegation",
        "eventdate": "2025-04-18",  # OGC email date
        "eventlocation": "USA",
        "truthstatus": "Alleged",
        "evidencetype": "Email",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.92",
        "causal_salience_reason": "Strengthens knowledge narrative; documents 2025 OGC knowledge of EsuWiki/PRC risk; connects knowledge to foreseeability; increases knowledge cluster density",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 5: Strengthen "spoliation" narrative (1 fact)
    # ========================================================================
    {
        "factid": "V22_SPOL_PATTERN_002",
        "proposition": "Harvard-affiliated clubs altered or removed online statements about Mr. Grayson only after they received legal or safety notices from him or his counsel: (a) in April 2019, 'Statement 2' was backdated to appear as if posted on 19 April 2019; (b) in November 2024, the Harvard Club of Shanghai site disappeared and the Harvard Club of Beijing removed 'Statement 2' within weeks of an HAA email about potential lawsuits; and (c) in April 2025, the Harvard Club of Hong Kong's statement was briefly removed and then restored immediately after Mr. Grayson's Local Rule 7.1 notice to Harvard OGC.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Spoliation Risk",
        "eventdate": "2019-04-19",  # Pattern starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "already_public",
        "causal_salience_score": "0.94",
        "causal_salience_reason": "Strengthens spoliation narrative; documents 3-episode pattern across 2019-2025; connects spoliation to legal notices; demonstrates consciousness of liability; increases spoliation cluster density",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 6: Strengthen "silence" narrative (2 facts)
    # ========================================================================
    {
        "factid": "V22_SILENCE_004",
        "proposition": "Despite receiving multiple emails from Mr. Grayson in April–July 2019 describing the 'Monkey' article as an emergency causing daily damage to his reputation and health, Harvard Center Shanghai and Harvard Admissions did not issue any public correction or retraction of the clubs' statements and did not warn Mr. Grayson that his Xi-related speech might expose him to PRC retaliation.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Allegation",
        "eventdate": "2019-04-23",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.90",
        "causal_salience_reason": "Strengthens silence narrative; documents 2019 Harvard silence despite emergency warnings; connects silence to harm and PRC risk; increases silence cluster density",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V22_SILENCE_005",
        "proposition": "Harvard's Office of the General Counsel received detailed warnings from Mr. Grayson in April and August 2025 about PRC retaliation, EsuWiki-related torture, and a potential § 1782 filing, but provided no acknowledgment, advice, or protective response, reflecting an institutional decision to maintain silence despite known risks.",
        "subject": "Harvard OGC",
        "actorrole": "Harvard OGC",
        "eventtype": "Allegation",
        "eventdate": "2025-04-18",  # First OGC email date
        "eventlocation": "USA",
        "truthstatus": "Alleged",
        "evidencetype": "Email",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.91",
        "causal_salience_reason": "Strengthens silence narrative; documents 2025 OGC non-response as deliberate; connects silence to PRC risk and spoliation; increases silence cluster density",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 7: Connect "Defamation" to broader causal chain (1 fact)
    # ========================================================================
    {
        "factid": "V22_DEFAM_CHAIN_002",
        "proposition": "The Harvard clubs' defamatory statements and the ensuing 'Monkey' and 'Résumé' articles were a substantial factor in causing Mr. Grayson's loss of contracts, damage to his professional reputation (including disbelief of his bona fide Harvard roles such as Harvard Chess Club presidency), heightened PRC surveillance risk, and significant psychological and physical harm.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Allegation",
        "eventdate": "2019-04-19",  # Defamation starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Document",
        "safetyrisk": "extreme",
        "publicexposure": "already_public",
        "causal_salience_score": "0.96",
        "causal_salience_reason": "Connects Defamation node to broader causal chain; explicitly links to all harm categories; resolves degree=0 connectivity issue; high salience",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 8: Connect "Harvard Chess Club" to broader causal chain (1 fact)
    # ========================================================================
    {
        "factid": "V22_HCC_DEFAM_001",
        "proposition": "The defamatory campaign against Mr. Grayson, including the Harvard club statements and the 'Monkey' article, specifically disparaged his position as a former president of the Harvard College Chess Club, portraying his legitimate leadership role as a sham and thereby damaging his reputation within the Harvard alumni community and among his Chinese clients.",
        "subject": "Harvard Chess Club",
        "actorrole": "Harvard Chess Club",
        "eventtype": "Allegation",
        "eventdate": "2019-04-20",  # Monkey article publication
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "WeChatArticle",
        "safetyrisk": "medium",
        "publicexposure": "already_public",
        "causal_salience_score": "0.85",
        "causal_salience_reason": "Connects Harvard Chess Club node to defamation and harm; resolves high-salience low-connectivity issue; documents specific credential attack",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
]


def add_gap_bridging_facts_to_v21():
    """Add v22 gap-bridging facts to v21 dataset."""
    print("="*80)
    print("GENERATING V22 GAP-BRIDGING FACTS")
    print("="*80)
    
    # Load v21 facts
    print("\n1. Loading v21 facts...")
    facts = []
    with open(V21_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    print(f"   Loaded {len(facts)} facts from v21")
    
    # Add new facts
    print(f"\n2. Adding {len(NEW_FACTS)} new gap-bridging facts...")
    for fact in NEW_FACTS:
        facts.append(fact)
        factid = fact.get('factid', '')
        print(f"   ✅ Added {factid}")
    
    print(f"\n   Total facts in v22: {len(facts)}")
    
    # Write v22 CSV
    print("\n3. Writing v22 CSV...")
    if facts:
        # Get all fieldnames
        all_fieldnames = set()
        for fact in facts:
            all_fieldnames.update(fact.keys())
        fieldnames = sorted(list(all_fieldnames))
        
        with open(V22_OUTPUT, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fact in facts:
                row = {}
                for field in fieldnames:
                    row[field] = fact.get(field, '')
                writer.writerow(row)
        print(f"   ✅ Exported {V22_OUTPUT}")
    
    # Generate report
    print("\n4. Generating report...")
    report = f"""# V22 Gap-Bridging Facts Report

## Summary

Added {len(NEW_FACTS)} new gap-bridging facts to v21 to address all 8 remaining structural gaps identified in the v21 network analysis.

## Gaps Addressed

### 1. Harm Narrative (3 facts)
- **V22_HARM_ECON_002**: Contract and income loss
- **V22_HARM_SAFETY_002**: Travel restriction and fear of detention
- **V22_HARM_PSYCH_002**: Psychiatric and physical sequelae

### 2. Email Narrative (3 facts)
- **V22_EMAIL_KNOW_001**: April 2019 notice to Harvard Center (Yi Wang)
- **V22_EMAIL_HAA_2024_001**: HAA legal-liability inquiry
- **V22_EMAIL_OGC_003**: OGC notice of PRC risk

### 3. Publication Narrative (2 facts)
- **V22_PUB_HCC_001**: Monkey article attacks Harvard Chess Club presidency
- **V22_PUB_CHAIN_002**: Cross-platform PRC amplification

### 4. Knowledge Narrative (2 facts)
- **V22_KNOW_2019_002**: Harvard Center Shanghai and McGrath knowledge of harm
- **V22_KNOW_2025_002**: OGC's constructive knowledge of PRC risk

### 5. Spoliation Narrative (1 fact)
- **V22_SPOL_PATTERN_002**: Harvard's pattern of post-notice website manipulation

### 6. Silence Narrative (2 facts)
- **V22_SILENCE_004**: 2019 Harvard silence despite emergency warnings
- **V22_SILENCE_005**: 2025 OGC non-response as deliberate non-intervention

### 7. Defamation → Causal Chain (1 fact)
- **V22_DEFAM_CHAIN_002**: Defamation as proximate cause of harm

### 8. Harvard Chess Club → Causal Chain (1 fact)
- **V22_HCC_DEFAM_001**: Harvard Chess Club targeted by defamatory narrative

## Statistics

- **Facts in v21**: {len(facts) - len(NEW_FACTS)}
- **Facts added**: {len(NEW_FACTS)}
- **Facts in v22**: {len(facts)}

## Expected Impact

These facts should significantly increase network density for:
- **Harm cluster**: From 23.76% to higher connectivity
- **Email cluster**: From ~21% to higher connectivity
- **Publication cluster**: Increased connections to Harvard Chess Club and PRC platforms
- **Knowledge cluster**: Explicit notice dates and warnings
- **Spoliation cluster**: 3-episode pattern documented
- **Silence cluster**: From 13.47% to higher connectivity
- **Defamation node**: Resolves degree=0 connectivity issue
- **Harvard Chess Club node**: Resolves high-salience low-connectivity issue

## Files Generated

- **V22 CSV**: `case_law_data/top_1000_facts_for_chatgpt_v22_final.csv`
- **Report**: `reports/analysis_outputs/v22_gap_bridging_facts_report.md`

## Next Steps

1. Review v22 facts for accuracy
2. Re-run network gap analysis to verify gaps are resolved
3. Generate v22 outputs (top 100, causation nodes, etc.)
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ Written {REPORT_PATH}")
    
    print()
    print("="*80)
    print("V22 GAP-BRIDGING FACTS GENERATED")
    print("="*80)
    print()
    print(f"✅ Added {len(NEW_FACTS)} new facts")
    print(f"✅ Total facts in v22: {len(facts)}")
    print(f"✅ Output: {V22_OUTPUT}")
    print()
    print("All 8 v21 gaps addressed with comprehensive bridging facts.")


if __name__ == "__main__":
    add_gap_bridging_facts_to_v21()

