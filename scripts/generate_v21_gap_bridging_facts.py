#!/usr/bin/env python3
"""
Generate v21 gap-bridging facts from user answers to v20 gap-bridging questions.
Addresses all 11 structural gaps identified in v20 network analysis.

Key boundaries:
- NO direct proof of OGC-EsuWiki discussion (use "should have" language)
- NO direct PRC-Harvard contact (use knowledge/silence links)
- NO formal EsuWiki case inclusion (use risk comparator/monitoring language)
- YES to foreseeable risk, knowledge, silence, harm connections
"""

import csv
from pathlib import Path
from datetime import datetime

V20_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v20_final.csv")
V21_OUTPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v21_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/v21_gap_bridging_facts_report.md")

# New gap-bridging facts addressing the 11 v20 gaps
NEW_FACTS = [
    # ========================================================================
    # GAP 1: EsuWiki ↔ Harvard OGC
    # ========================================================================
    {
        "factid": "V21_ESU_OGC_001",
        "proposition": "Harvard is a large international university with ongoing investments, collaborations, and institutional relationships in China, including research partnerships, alumni networks, and educational programs.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Context",
        "eventdate": "2019-01-01",  # Ongoing context
        "eventlocation": "China",
        "truthstatus": "True",
        "evidencetype": "Document",
        "safetyrisk": "low",
        "publicexposure": "already_public",
        "causal_salience_score": "0.75",
        "causal_salience_reason": "Establishes Harvard's China footprint; supports institutional knowledge of PRC environment; foundation for foreseeability",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_ESU_OGC_002",
        "proposition": "Plaintiff alleges that any reasonable risk-management function at Harvard, including the Office of General Counsel, should have evaluated the EsuWiki crackdown (Case 1902136) as part of its China risk assessment, given Harvard's institutional presence in China and its duty of care to affiliates traveling or working there. There is no direct evidence that OGC actually performed such an evaluation.",
        "subject": "Harvard OGC",
        "actorrole": "Harvard OGC",
        "eventtype": "Allegation",
        "eventdate": "2019-06-14",  # EsuWiki case opening
        "eventlocation": "USA",
        "truthstatus": "Alleged",
        "evidencetype": "Document",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.88",
        "causal_salience_reason": "Foreseeability argument; connects EsuWiki to Harvard OGC; supports duty of care; high impact on gap resolution",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 2: Plaintiff ↔ EsuWiki
    # ========================================================================
    {
        "factid": "V21_PLAINTIFF_ESU_001",
        "proposition": "Plaintiff uses the EsuWiki case (Case 1902136) as a comparative risk benchmark, arguing that the treatment of Niu Tengyu and other contributors—including arrest, detention, and torture for Xi-related content—demonstrates the foreseeable severity of retaliation against individuals whose speech or activities involve Xi Jinping's family in the PRC context.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Argument",
        "eventdate": "2019-06-14",  # EsuWiki case opening
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Document",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.92",
        "causal_salience_reason": "Risk comparator; connects Plaintiff to EsuWiki pattern; establishes foreseeability of harm; extreme safety risk",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_PLAINTIFF_ESU_002",
        "proposition": "Plaintiff alleges that, given the EsuWiki crackdown's scope, focus on Xi-related content, and the fact that his own activities involved a slide referencing Xi Mingze in a Harvard context, it is inconceivable that he was not at least monitored or investigated by PRC actors once his Xi slide and associated Harvard-related publicity existed in PRC-accessible media.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Allegation",
        "eventdate": "2019-04-19",  # After Statement 1 publication
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Document",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.90",
        "causal_salience_reason": "Connects Plaintiff to EsuWiki monitoring pattern; supports surveillance allegations; high impact on gap resolution",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 3: EsuWiki ↔ Harvard Club
    # ========================================================================
    {
        "factid": "V21_ESU_CLUB_001",
        "proposition": "Plaintiff alleges that Harvard Club statements about his Harvard affiliation and admissions-related activities were broadcast into the same Xi-sensitive PRC ecosystem—including WeChat, Zhihu, Baidu, and diaspora news outlets—that was under active scrutiny in the EsuWiki crackdown, making retaliation against him foreseeable given the PRC's demonstrated pattern of severe response to Xi-related content.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Allegation",
        "eventdate": "2019-04-19",  # Statement 1 publication
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Document",
        "safetyrisk": "extreme",
        "publicexposure": "already_public",
        "causal_salience_score": "0.91",
        "causal_salience_reason": "Connects Harvard Club to EsuWiki environment; establishes ecosystem overlap; supports foreseeability; high impact",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 4: PRC ↔ Harvard OGC
    # ========================================================================
    {
        "factid": "V21_PRC_OGC_001",
        "proposition": "In 2025 communications to Harvard OGC, Plaintiff explicitly warned about PRC retaliation risks, safety concerns, and the sensitivity of the Xi-related context, including references to the PRC's pattern of arbitrary enforcement and the danger of politically sensitive allegations.",
        "subject": "Harvard OGC",
        "actorrole": "Harvard OGC",
        "eventtype": "Communication",
        "eventdate": "2025-04-18",  # OGC email date
        "eventlocation": "USA",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.89",
        "causal_salience_reason": "Establishes OGC knowledge of PRC risk; connects PRC to OGC via knowledge; supports foreseeability",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_PRC_OGC_002",
        "proposition": "Plaintiff alleges that Harvard OGC's decision not to respond or engage with his detailed safety and PRC risk warnings left him exposed to ongoing PRC-related risk, especially given Harvard's institutional knowledge of China travel risks, GSS advisories, and the EsuWiki crackdown pattern.",
        "subject": "Harvard OGC",
        "actorrole": "Harvard OGC",
        "eventtype": "Allegation",
        "eventdate": "2025-04-18",  # After OGC non-response
        "eventlocation": "USA",
        "truthstatus": "Alleged",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.87",
        "causal_salience_reason": "Connects PRC risk to OGC silence; supports harm causation; strengthens silence narrative",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 5: Strengthening "silence" narrative
    # ========================================================================
    {
        "factid": "V21_SILENCE_001",
        "proposition": "In 2019, Plaintiff informed Harvard Center Shanghai Director Yi Wang that the situation was harming his mental health and requested help with restoring his life and safety; Harvard entities did not provide substantive relief, correction, or retraction of the defamatory statements.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Communication",
        "eventdate": "2019-04-23",  # CORR_2019_001 date
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.85",
        "causal_salience_reason": "Strengthens silence narrative; documents mental health harm; connects Harvard to non-response",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_SILENCE_002",
        "proposition": "Plaintiff used coded language in communications with HAA (e.g., 'please help me get my life back'), aware that his communications might be monitored in China, reflecting the severity of the situation and his need for Harvard's assistance; Harvard did not provide substantive response.",
        "subject": "Harvard Alumni Association",
        "actorrole": "Harvard Alumni Association",
        "eventtype": "Communication",
        "eventdate": "2019-05-20",  # CORR_2019_007 date
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.83",
        "causal_salience_reason": "Strengthens silence narrative; shows coded language due to monitoring concerns; connects HAA to silence",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_SILENCE_003",
        "proposition": "By 2025, Plaintiff sent three detailed emails to Harvard OGC about safety and PRC risk; OGC did not acknowledge or respond to these communications, despite their explicit warnings about retaliation and the sensitive Xi-related context.",
        "subject": "Harvard OGC",
        "actorrole": "Harvard OGC",
        "eventtype": "Communication",
        "eventdate": "2025-04-18",  # OGC email date
        "eventlocation": "USA",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.88",
        "causal_salience_reason": "Strengthens silence narrative; documents OGC non-response; connects 2025 communications to silence",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 6: Strengthening "harm" narrative
    # ========================================================================
    {
        "factid": "V21_HARM_ECON_001",
        "proposition": "Plaintiff suffered economic harm in the form of lost consulting work, cancelled classes, refund demands, and loss of speaking invitations as a result of the defamatory environment created by Harvard Club statements and subsequent racist publications (Monkey and Résumé articles).",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.86",
        "causal_salience_reason": "Strengthens harm narrative; documents economic losses; connects defamation to economic harm",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_HARM_CRED_001",
        "proposition": "The defamatory publications cast doubt on Plaintiff's legitimate credentials, including his role as a Harvard Admissions interviewer/UAC, his Harvard Chess Club membership, his Rockefeller Fellowship, and his memory sports achievements, undermining his professional legitimacy.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "already_public",
        "causal_salience_score": "0.84",
        "causal_salience_reason": "Strengthens harm narrative; documents credential damage; connects defamation to reputational harm",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_HARM_HANDLER_001",
        "proposition": "Plaintiff alleges that PRC-linked surveillance included a dedicated handler whose job was to watch him and 'look after' him after Harvard destroyed his reputation, which he will seek to prove through testimony and corroborating evidence.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.93",
        "causal_salience_reason": "Strengthens harm narrative; documents surveillance/handler; extreme safety risk; connects defamation to safety harm",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_HARM_PSYCH_001",
        "proposition": "Plaintiff alleges that sustained stress and fear caused psychiatric injury and physical illness, which he intends to characterize as personal injury in damages arguments, resulting from the defamatory environment, surveillance, and Harvard's failure to correct the record.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.87",
        "causal_salience_reason": "Strengthens harm narrative; documents psychiatric/physical injury; connects defamation to psychological harm",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 7: Strengthening "knowledge" narrative
    # ========================================================================
    {
        "factid": "V21_KNOW_2019_001",
        "proposition": "In April 2019, Plaintiff sent emails to Harvard Center Shanghai (Yi Wang) and Harvard Admissions (Marlyn McGrath) that put Harvard on notice of racialized harassment, including links to the Monkey article and explanations of how the author was misled by Harvard's statements, establishing Harvard's knowledge of the harm being caused.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Communication",
        "eventdate": "2019-04-23",  # CORR_2019_001 date
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.91",
        "causal_salience_reason": "Strengthens knowledge narrative; documents 2019 notice of racialized attacks; connects Harvard to knowledge of harm",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_KNOW_2025_001",
        "proposition": "In 2025, Plaintiff sent detailed emails to Harvard OGC explicitly warning about PRC retaliation risks and the sensitivity of the Xi-related context, putting OGC on notice of ongoing safety concerns and the need for institutional response.",
        "subject": "Harvard OGC",
        "actorrole": "Harvard OGC",
        "eventtype": "Communication",
        "eventdate": "2025-04-18",  # OGC email date
        "eventlocation": "USA",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.90",
        "causal_salience_reason": "Strengthens knowledge narrative; documents 2025 notice of PRC risk; connects OGC to knowledge of ongoing harm",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 8: Strengthening "email" narrative
    # ========================================================================
    {
        "factid": "V21_EMAIL_CHAIN_001",
        "proposition": "Plaintiff's email communications with Harvard entities escalated from 2019 warnings about racialized attacks and reputational harm (to Yi Wang, McGrath, HAA) to 2025 detailed safety and PRC risk warnings (to OGC), creating a documented chain of notice and non-response.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Communication",
        "eventdate": "2019-04-23",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.88",
        "causal_salience_reason": "Strengthens email narrative; documents escalating communication chain; connects multiple Harvard actors",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_EMAIL_MJ_001",
        "proposition": "MJ Tang, President of the Harvard Club of Shanghai, stated that she had received instructions from multiple people on how to handle Plaintiff if he contacted her, indicating central coordination from within the university, not independent club action.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Communication",
        "eventdate": "2019-05-01",  # Approximate
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.85",
        "causal_salience_reason": "Strengthens email narrative; documents central coordination; connects Harvard Club to Harvard control",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 9: Strengthening "publication" narrative
    # ========================================================================
    {
        "factid": "V21_PUB_AMPLIFY_001",
        "proposition": "Harvard Club statements about Plaintiff were republished and amplified across multiple PRC-accessible platforms including WeChat Moments, Baidu Baijiahao, Zhihu, Sohu, E-Canada, and diaspora news outlets, creating a widespread defamatory environment that linked Harvard's statements to racist and false narratives.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-20",  # After Statement 1
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Document",
        "safetyrisk": "extreme",
        "publicexposure": "already_public",
        "causal_salience_score": "0.92",
        "causal_salience_reason": "Strengthens publication narrative; documents amplification across platforms; connects Harvard statements to widespread harm",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # GAP 10: Strengthening "spoliation" narrative
    # ========================================================================
    # Note: V20_SPOL_001 and V20_SPOL_002 already cover the main spoliation facts
    # The comprehensive pattern fact (V21_SPOL_PATTERN_001) is added below in the 2024 section
    # This gap is now fully addressed by the multi-year pattern fact
    
    # ========================================================================
    # GAP 11: Defamation → broader causal chain
    # ========================================================================
    # Note: V20_HARM_001-003 already connect Defamation to harm
    # This adds explicit causation language
    {
        "factid": "V21_DEFAM_CHAIN_001",
        "proposition": "Plaintiff alleges that Harvard's defamatory statements about him directly caused a cascade of harms: (1) safety harm through surveillance, fear of detention, and inability to safely leave China until 2022; (2) reputational harm through lost contracts, cancelled classes, and doubt cast on legitimate credentials; and (3) psychological/physical harm through sustained stress, anxiety, and psychiatric injury.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Allegation",
        "eventdate": "2019-04-19",  # Defamation starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Document",
        "safetyrisk": "extreme",
        "publicexposure": "already_public",
        "causal_salience_score": "0.95",
        "causal_salience_reason": "Connects Defamation to broader causal chain; explicitly links to all three harm categories; high salience",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # CRITICAL 2024 SPOLIATION SEQUENCE
    # Fills 3 major gaps: Harvard-Club coordination, spoliation pattern, knowledge/liability
    # Based on: Anonymous HAA inquiry about legal liability (before Nov 4, 2024)
    # HAA response: "Alumni clubs are separate legal entities" (Nov 4, 2024)
    # Immediate website deletions: Shanghai website disappeared, Beijing Statement 2 removed
    # ========================================================================
    {
        "factid": "V21_SPOL_2024_001",
        "proposition": "In November 2024, after the plaintiff anonymously emailed the Harvard Alumni Association asking about Harvard's legal relationship to its alumni clubs in the event of a lawsuit, including whether clubs or HAA should be sued, whether clubs were legally responsible, and whether HAA takes legal responsibility, the Harvard Club of Shanghai's website disappeared completely and the Harvard Club of Beijing removed Statement 2 for the first time since 2019, despite having remained untouched for five years. This occurred within 24-48 hours after HAA's November 4, 2024 response (Berry from AAD Service Desk) stating that 'alumni clubs are separate legal entities and are not part of the HAA.'",
        "subject": "Harvard Alumni Association",
        "actorrole": "Harvard Alumni Association",
        "eventtype": "Spoliation Risk",
        "eventdate": "2024-11-05",  # Website deletions occurred within 24-48 hours after Nov 4 HAA response
        "eventlocation": "USA",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.96",
        "causal_salience_reason": "Critical spoliation evidence; proves Harvard-Club coordination; demonstrates consciousness of liability; extremely probative; fills 3 major network gaps; shows coordinated response to legal inquiry; within 24-48 hours timing shows reactive behavior",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_SPOL_2024_002",
        "proposition": "On November 4, 2024, HAA (Berry from AAD Service Desk) told the inquirer (plaintiff using a pseudonym) that 'alumni clubs are separate legal entities and are not part of the HAA,' implying Harvard itself would not be liable. Immediately after this response, at least two club websites (Shanghai and Beijing) altered or removed content, suggesting coordinated risk response rather than independent club action and contradicting the 'separate legal entities' claim.",
        "subject": "Harvard Alumni Association",
        "actorrole": "Harvard Alumni Association",
        "eventtype": "Spoliation Risk",
        "eventdate": "2024-11-04",  # HAA response date
        "eventlocation": "USA",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.94",
        "causal_salience_reason": "Proves Harvard-Club coordination; contradicts 'separate legal entities' claim; demonstrates organized response to legal inquiry; fills Harvard-Club control gap; shows structural inconsistency",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V21_SPOL_PATTERN_001",
        "proposition": "Across 2019-2025, the only times Harvard Club websites removed or altered defamatory statements were immediately after inquiries or warnings about legal liability: (1) 2019 publication anomalies and backdating after initial complaints; (2) November 2024 website deletions after anonymous legal inquiry to HAA about suing clubs; and (3) April 2025 HK site disappearance/reappearance after OGC notice. This pattern of reactive website manipulation around moments of legal scrutiny supports a spoliation theory and demonstrates consciousness of liability.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Spoliation Risk",
        "eventdate": "2019-04-19",  # Starting point of pattern
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "already_public",
        "causal_salience_score": "0.93",
        "causal_salience_reason": "Documents pattern across 3 separate years; strengthens spoliation theory; demonstrates consciousness of liability; connects all reactive website changes; multi-year consistency",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
]


def add_gap_bridging_facts_to_v20():
    """Add v21 gap-bridging facts to v20 dataset."""
    print("="*80)
    print("GENERATING V21 GAP-BRIDGING FACTS")
    print("="*80)
    
    # Load v20 facts
    print("\n1. Loading v20 facts...")
    facts = []
    with open(V20_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    print(f"   Loaded {len(facts)} facts from v20")
    
    # Add new facts
    print(f"\n2. Adding {len(NEW_FACTS)} new gap-bridging facts...")
    for fact in NEW_FACTS:
        facts.append(fact)
        factid = fact.get('factid', '')
        print(f"   ✅ Added {factid}")
    
    print(f"\n   Total facts in v21: {len(facts)}")
    
    # Write v21 CSV
    print("\n3. Writing v21 CSV...")
    if facts:
        # Get all fieldnames
        all_fieldnames = set()
        for fact in facts:
            all_fieldnames.update(fact.keys())
        fieldnames = sorted(list(all_fieldnames))
        
        with open(V21_OUTPUT, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fact in facts:
                row = {}
                for field in fieldnames:
                    row[field] = fact.get(field, '')
                writer.writerow(row)
        print(f"   ✅ Exported {V21_OUTPUT}")
    
    # Generate report
    print("\n4. Generating report...")
    report = f"""# V21 Gap-Bridging Facts Report

## Summary

Added {len(NEW_FACTS)} new gap-bridging facts to v20 to address all 11 structural gaps identified in the v20 network analysis.

## Gaps Addressed

### High Priority Gaps (4 gaps)

1. **EsuWiki ↔ Harvard OGC** (2 facts)
   - V21_ESU_OGC_001: Harvard's China footprint (True)
   - V21_ESU_OGC_002: OGC should have evaluated EsuWiki (Alleged)

2. **Plaintiff ↔ EsuWiki** (2 facts)
   - V21_PLAINTIFF_ESU_001: EsuWiki as risk comparator (Alleged)
   - V21_PLAINTIFF_ESU_002: Plaintiff likely monitored (Alleged)

3. **EsuWiki ↔ Harvard Club** (1 fact)
   - V21_ESU_CLUB_001: Same ecosystem overlap (Alleged)

4. **PRC ↔ Harvard OGC** (2 facts)
   - V21_PRC_OGC_001: OGC knew about PRC risk (True)
   - V21_PRC_OGC_002: OGC silence left Plaintiff exposed (Alleged)

### Medium Priority Gaps (7 gaps)

5. **Silence narrative** (3 facts)
   - V21_SILENCE_001: Mental health warning to Yi Wang (True)
   - V21_SILENCE_002: Coded language to HAA (True)
   - V21_SILENCE_003: OGC non-response in 2025 (True)

6. **Harm narrative** (4 facts)
   - V21_HARM_ECON_001: Economic losses (Alleged)
   - V21_HARM_CRED_001: Credential damage (Alleged)
   - V21_HARM_HANDLER_001: Handler surveillance (Alleged)
   - V21_HARM_PSYCH_001: Psychiatric/physical injury (Alleged)

7. **Knowledge narrative** (2 facts)
   - V21_KNOW_2019_001: 2019 notice of racialized attacks (True)
   - V21_KNOW_2025_001: 2025 notice of PRC risk (True)

8. **Email narrative** (2 facts)
   - V21_EMAIL_CHAIN_001: Escalating communication chain (True)
   - V21_EMAIL_MJ_001: MJ Tang central coordination (True)

9. **Publication narrative** (1 fact)
   - V21_PUB_AMPLIFY_001: Multi-platform amplification (True)

10. **Spoliation narrative** (1 fact)
    - V21_SPOL_PATTERN_001: Pattern of irregularities (Alleged)

11. **Defamation → causal chain** (1 fact)
    - V21_DEFAM_CHAIN_001: Explicit harm cascade (Alleged)

## Key Boundaries Respected

- **NO** direct proof of OGC-EsuWiki discussion (uses "should have" language)
- **NO** direct PRC-Harvard contact (uses knowledge/silence links)
- **NO** formal EsuWiki case inclusion (uses risk comparator/monitoring language)
- **YES** to foreseeable risk, knowledge, silence, harm connections

## Statistics

- **Facts in v20**: {len(facts) - len(NEW_FACTS)}
- **Facts added**: {len(NEW_FACTS)}
- **Facts in v21**: {len(facts)}

## Files Generated

- **V21 CSV**: `case_law_data/top_1000_facts_for_chatgpt_v21_final.csv`
- **Report**: `reports/analysis_outputs/v21_gap_bridging_facts_report.md`

## Next Steps

1. Review v21 facts for accuracy and completeness
2. Re-run network gap analysis to verify gaps are resolved
3. Generate v21 outputs (top 100, causation nodes, etc.)
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ Written {REPORT_PATH}")
    
    print()
    print("="*80)
    print("V21 GAP-BRIDGING FACTS GENERATED")
    print("="*80)
    print()
    print(f"✅ Added {len(NEW_FACTS)} new facts")
    print(f"✅ Total facts in v21: {len(facts)}")
    print(f"✅ Output: {V21_OUTPUT}")
    print()
    print("All 11 v20 gaps addressed with appropriate True/Alleged boundaries.")


if __name__ == "__main__":
    add_gap_bridging_facts_to_v20()

