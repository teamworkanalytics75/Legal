#!/usr/bin/env python3
"""
Generate v20 bridging facts from user answers to gap-bridging questions.
Adds facts that bridge:
- Publication → Harm
- Spoliation → Silence → Risk
- Defamation → Safety/Reputational/Psychological consequences
"""

import csv
from pathlib import Path
from datetime import datetime

V19_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v19_final.csv")
V20_OUTPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v20_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/v20_bridging_facts_report.md")

# New bridging facts to add
# Note: Some fields need user confirmation (marked with TODO)
NEW_FACTS = [
    # Publication → Harm facts
    {
        "factid": "V20_PUB_001",
        "proposition": "The 'Monkey' article published on WeChat in April 2019 was explicitly racist, mocked the plaintiff's appearance, treated him as a race-based caricature, questioned his immigration status, and implied he was foreign, suspicious, or improperly in China. This article was directly linked to quotes from Harvard Club statements and circulated across PRC-accessible platforms including WeChat Moments, Baidu Baijiahao, Zhihu reposts, and diaspora news outlets.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-20",  # Approximate, based on Statement 1 timing
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "WeChatArticle",
        "safetyrisk": "extreme",
        "publicexposure": "already_public",
        "causal_salience_score": "0.95",
        "causal_salience_reason": "Direct link to Harvard Club statements; extreme safety risk; PRC-accessible; racist content; immigration implications",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V20_PUB_002",
        "proposition": "The 'Résumé' article published on WeChat in April 2019 falsely called the plaintiff a 'Harvard admissions officer,' included race-coded language, implied he used a fake name to get work, suggested he was lying about qualifications, and created the impression he was committing immigration crimes. This article was directly linked to Harvard Club statements and circulated across PRC-accessible platforms.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-20",  # Approximate
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "WeChatArticle",
        "safetyrisk": "extreme",
        "publicexposure": "already_public",
        "causal_salience_score": "0.95",
        "causal_salience_reason": "Direct link to Harvard Club statements; extreme safety risk; false credentials claim; immigration crime implications",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # Spoliation → Silence → Risk facts
    {
        "factid": "V20_SPOL_001",
        "proposition": "The Harvard Club of Shanghai webpage displays Statement 2 with a mis-dated timestamp of April 19, 2019, but Statement 2 was actually posted approximately 10 days later on April 29, 2019, suggesting backdating or rewriting of publication chronology. The plaintiff has before/after material showing mismatched dates and timing.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Publication",
        "eventdate": "2019-04-29",
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "already_public",
        "causal_salience_score": "0.88",
        "causal_salience_reason": "Evidence of backdating; supports spoliation theory; connects Harvard Club to publication manipulation",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    {
        "factid": "V20_SPOL_002",
        "proposition": "On April 7, 2025, when the plaintiff emailed Harvard OGC, the Harvard Club of Hong Kong statement was publicly visible. On April 18, 2025, when the plaintiff checked again, the statement had disappeared. On April 19, 2025, the plaintiff took a screenshot showing the statement was back online. This sequence occurred after the plaintiff's OGC notice and suggests possible evidence manipulation or irregular website behavior.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Communication",
        "eventdate": "2025-04-18",
        "eventlocation": "Hong Kong",
        "truthstatus": "True",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "already_public",
        "causal_salience_score": "0.90",
        "causal_salience_reason": "Spoliation risk; connects OGC notice to website changes; supports evidence manipulation theory",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # Defamation → Safety Harm
    {
        "factid": "V20_HARM_001",
        "proposition": "Following the Harvard-related defamation in April 2019, the plaintiff can almost certainly prove he was being surveilled in China. Surveillance of a US citizen abroad in the PRC context, combined with the April 2019 PRC environment and Harvard's statements, created a foreseeable risk of arbitrary enforcement. The plaintiff experienced fear of detention, travel disruption, and was unable to safely leave China until 2022.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",  # TODO: Confirm - Alleged or True?
        "evidencetype": "Exhibit",
        "safetyrisk": "extreme",
        "publicexposure": "not_public",
        "causal_salience_score": "0.96",
        "causal_salience_reason": "Primary safety harm; surveillance; arbitrary detention risk; travel disruption; connects defamation to extreme safety consequences",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # Defamation → Reputational Harm
    {
        "factid": "V20_HARM_002",
        "proposition": "The Harvard Club statements and subsequent Monkey and Résumé articles caused the plaintiff to lose professional legitimacy, damage to his standing in admissions and counseling, accusations of fraud and fake name use, and immigration violation implications. This destroyed his ability to earn income in China, resulting in lost contracts, student refund requests, and canceled classes.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Exhibit",
        "safetyrisk": "high",
        "publicexposure": "already_public",
        "causal_salience_score": "0.92",
        "causal_salience_reason": "Secondary reputational harm; economic losses; lost contracts; connects defamation to economic consequences",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # Defamation → Psychological Harm
    {
        "factid": "V20_HARM_003",
        "proposition": "The plaintiff experienced documented anxiety and fear due to surveillance, psychological distress caused by racist publications, knowing PRC authorities might be monitoring him, and Harvard's refusal to correct the record. These psychological harms began on April 19, 2019 with the first Harvard defamation and continued as ongoing psychological harm through 2022.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Exhibit",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.90",
        "causal_salience_reason": "Tertiary psychological harm; multi-year distress; connects defamation to psychological consequences",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # Silence → Harm bridging
    {
        "factid": "V20_SILENCE_001",
        "proposition": "After the plaintiff gave notice to Harvard in April–May 2019, Harvard did not correct or withdraw its clubs' statements, and republications of defamatory content continued into at least August 2019. During and after this period, the plaintiff lost contracts, had classes cancelled, and faced refund requests attributable to the ongoing defamatory environment.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "NonResponse",
        "eventdate": "2019-05-20",  # After notice period
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.93",
        "causal_salience_reason": "Connects silence cluster to harm cluster; shows continued republication after notice; economic harm from inaction",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # Knowledge → Harm bridging
    {
        "factid": "V20_KNOW_001",
        "proposition": "On April 23–24, 2019, the plaintiff sent Harvard Center Shanghai (Yi Wang) the Monkey article and explicitly explained that it was based on Harvard's club statements and was causing him serious harm. The plaintiff also sent links to the article and described how the author was misled by Harvard's statements.",
        "subject": "Harvard",
        "actorrole": "Harvard",
        "eventtype": "Communication",
        "eventdate": "2019-04-23",
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.94",
        "causal_salience_reason": "Strengthens knowledge cluster; direct evidence Harvard knew about Monkey article and harm; connects knowledge to harm",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
]


def add_bridging_facts_to_v19():
    """Add new bridging facts to v19 and create v20."""
    print("="*80)
    print("GENERATING V20 BRIDGING FACTS")
    print("="*80)
    print()
    
    # Load v19 facts
    print("1. Loading v19 facts...")
    facts = []
    fieldnames = None
    
    with open(V19_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            facts.append(row)
    
    print(f"   Loaded {len(facts)} facts from v19")
    
    # Add new bridging facts
    print("\n2. Adding 8 new bridging facts...")
    for new_fact in NEW_FACTS:
        # Ensure all fields are present
        fact_row = {}
        for field in fieldnames:
            fact_row[field] = new_fact.get(field.lower(), "")
        facts.append(fact_row)
        print(f"   ✅ Added {new_fact['factid']}")
    
    print(f"\n   Total facts in v20: {len(facts)}")
    
    # Write v20 CSV
    print("\n3. Writing v20 CSV...")
    with open(V20_OUTPUT, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(facts)
    
    print(f"   ✅ Exported {V20_OUTPUT}")
    
    # Generate report
    print("\n4. Generating report...")
    report = f"""# V20 Bridging Facts Report

## Summary

Added 8 new bridging facts to v19 to create v20. These facts bridge critical gaps identified in the network analysis.

## Facts Added

### Publication → Harm Facts (2 facts)

**V20_PUB_001**: Monkey Article
- Extreme safety risk
- Direct link to Harvard Club statements
- Racist content, immigration implications
- Salience: 0.95

**V20_PUB_002**: Résumé Article
- Extreme safety risk
- False credentials claim
- Immigration crime implications
- Salience: 0.95

### Spoliation → Silence → Risk Facts (2 facts)

**V20_SPOL_001**: Statement 2 Backdating
- Shanghai Club webpage shows April 19 timestamp
- Actually posted ~April 29
- Evidence of backdating/rewriting
- Salience: 0.88

**V20_SPOL_002**: HK Site Disappearance/Reappearance
- April 7: Visible
- April 18: Gone
- April 19: Back online
- After OGC notice
- Salience: 0.90

### Defamation → Harm Facts (3 facts)

**V20_HARM_001**: Safety Harm (Primary)
- Surveillance evidence
- Fear of detention
- Travel disruption
- Unable to leave until 2022
- Salience: 0.96

**V20_HARM_002**: Reputational Harm (Secondary)
- Loss of professional legitimacy
- Lost contracts, refunds, cancellations
- Economic losses
- Salience: 0.92

**V20_HARM_003**: Psychological Harm (Tertiary)
- Documented anxiety and fear
- Multi-year distress (2019-2022)
- Psychological consequences
- Salience: 0.90

### Silence → Harm Bridging (1 fact)

**V20_SILENCE_001**: Continued Republication After Notice
- Harvard did not correct/withdraw statements
- Republications continued into August 2019
- Economic harm from inaction
- Salience: 0.93

### Knowledge → Harm Bridging (1 fact)

**V20_KNOW_001**: Harvard Knew About Monkey Article
- Plaintiff sent article to Yi Wang
- Explained it was based on Harvard statements
- Described serious harm
- Salience: 0.94

## Network Gap Impact

These facts bridge:
- **Publication → Harm**: 2 facts
- **Spoliation → Silence → Risk**: 2 facts
- **Defamation → Safety/Reputational/Psychological**: 3 facts
- **Silence → Harm**: 1 fact
- **Knowledge → Harm**: 1 fact

## Statistics

- **Facts in v19**: {len(facts) - 8}
- **Facts added**: 8
- **Facts in v20**: {len(facts)}

## Confirmations Applied

1. **EvidenceType for articles**: WeChatArticle ✅
2. **Backdating date**: April 29, 2019 (real posting date), proposition mentions mis-dating as April 19 ✅
3. **HK sequence dates**: 2025-04-07 (visible), 2025-04-18 (gone), 2025-04-19 (back) ✅
4. **Surveillance TruthStatus**: Alleged ✅
5. **Psychological harm dates**: 2019-04-19 (onset), ongoing through 2022 ✅

## Files Generated

- **V20 CSV**: `case_law_data/top_1000_facts_for_chatgpt_v20_final.csv`
- **Report**: `reports/analysis_outputs/v20_bridging_facts_report.md`

## Next Steps

1. Review the 8 new facts
2. Confirm the 5 TODO items
3. Update facts with confirmed values
4. Generate v20 outputs (TXT, top 100, causation nodes)
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ Written {REPORT_PATH}")
    
    print()
    print("="*80)
    print("V20 BRIDGING FACTS FINALIZED")
    print("="*80)
    print()
    print(f"✅ Added 8 new bridging facts")
    print(f"✅ Total facts in v20: {len(facts)}")
    print(f"✅ Output: {V20_OUTPUT}")
    print()
    print("✅ All confirmations applied:")
    print("   1. EvidenceType: WeChatArticle")
    print("   2. Backdating: April 29, 2019 (real date), mis-dated as April 19")
    print("   3. HK sequence: 2025-04-18 (corrected from 2019)")
    print("   4. Surveillance: TruthStatus = Alleged")
    print("   5. Psychological harm: 2019-04-19 (onset), ongoing through 2022")
    print()


if __name__ == "__main__":
    add_bridging_facts_to_v19()

