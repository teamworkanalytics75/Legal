#!/usr/bin/env python3
"""Apply clarification answers to update the fact dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

V10_1_INPUT = Path("case_law_data/top_100_facts_v10.1.csv")
V10_FULL_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v10_final.csv")
OUTPUT_CSV = Path("case_law_data/top_100_facts_v10.2.csv")
OUTPUT_FULL_CSV = Path("case_law_data/top_1000_facts_for_chatgpt_v10.2_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/v10_2_clarifications_applied_report.md")


def load_facts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both v10.1 top 100 and full v10 dataset."""
    top100 = pd.read_csv(V10_1_INPUT, encoding='utf-8', low_memory=False)
    full = pd.read_csv(V10_FULL_INPUT, encoding='utf-8', low_memory=False)
    return top100, full


def apply_clarifications(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply clarification answers to facts."""
    df = df.copy()
    changes = {
        'dates_added': 0,
        'propositions_expanded': 0,
        'causal_connections_added': 0,
        'actorroles_updated': 0,
        'locations_added': 0,
    }
    
    # Define clarifications as a dictionary mapping FactID to updates
    clarifications = {
        # KGFACT_010 - Add causal connection
        'KGFACT_010': {
            'proposition': "On 14 June 2019, China's Ministry of Public Security opened EsuWiki case 1902136. The temporal overlap between Harvard-affiliated clubs publishing Statement 1 in April 2019 (which entered Chinese social media ecosystems) and PRC escalation against politically sensitive material in June 2019 makes the crackdown foreseeably dangerous for anyone implicated in politically sensitive content.",
            'causal_connection': 'temporal_overlap',
        },
        
        # 815 - Expand proposition
        '815': {
            'proposition': 'Exposure to politically sensitive accusations in the PRC creates a credible risk of torture, as demonstrated by the EsuWiki case, where multiple young contributors including Niu Tengyu were tortured in custody.',
        },
        
        # 3911 - Add causal connection
        '3911': {
            'proposition': "Harvard's own Global Support Services (GSS) rates travel to China as an elevated risk because of strained U.S.–China relations, restrictions on political activism and the risk of arbitrary detention. This means Harvard had constructive knowledge that negative allegations made by its affiliated clubs, if circulated in China, could foreseeably endanger the plaintiff.",
            'causal_connection': 'constructive_knowledge',
        },
        
        # 268 - Add causal connection
        '268': {
            'proposition': "All interviewers in China are Harvard College graduates, and work with the admission office on a volunteer basis. Grayson 先生现在不是哈佛面试官，也从未在哈佛 招办工作过. By publicly stating in Chinese that the plaintiff was not a Harvard interviewer or officer, the Harvard Clubs created a reputational dispute that was then republished in PRC-facing media and WeChat channels. In the PRC context, allegations of falsifying Harvard affiliation can become politically sensitive, triggering censorship, doxxing, and arbitrary enforcement.",
            'causal_connection': 'reputational_dispute_to_prc_risk',
        },
        
        # 885 - Add date and causal connection
        '885': {
            'eventdate': '2012-01-01',
            'proposition': 'Media and historical context exhibits: 2012 news articles about the viral story of Xi Mingze, President Xi Jinping\'s daughter, studying at Harvard and the immediate censorship of that story inside China. The Xi Mingze enrollment story was censored in China because it touched on top-level political sensitivities. The plaintiff\'s Xi-slide references placed them in the same sensitivity category, meaning that Harvard-affiliated publications contradicting or embarrassing PRC narratives could produce political retaliation or surveillance.',
            'causal_connection': 'xi_mingze_sensitivity_category',
        },
        
        # 44 - Add causal connection
        '44': {
            'proposition': 'He alleges that Harvard-affiliated alumni clubs in Greater China including the Harvard Clubs of Beijing, Shanghai and Hong Kong published and republished defamatory statements about him. These statements were published directly into PRC-accessible ecosystems (WeChat, Baidu, Zhihu, diaspora media), which routinely feed into PRC state surveillance monitoring. Thus, the defamatory statements were automatically ingested into PRC information pipelines, leading to foreseeable harassment, monitoring, and potential detention risk.',
            'causal_connection': 'prc_information_pipeline_ingestion',
        },
        
        # 324 - Add date
        '324': {
            'eventdate': '2025-06-05',
        },
        
        # MISSING_0095 - Add date
        'MISSING_0095': {
            'eventdate': '2025-04-19',
        },
        
        # MISSING_0070 - Add date
        'MISSING_0070': {
            'eventdate': '2025-07-11',
        },
        
        # 2428 - Add dates (this might need to be split into two facts)
        '2428': {
            'eventdate': '2019-04-24',  # First notice
            'note': 'This fact may need to be split - refers to two different dates',
        },
        
        # MISSING_0002_REWRITE_1 - Add date
        'MISSING_0002_REWRITE_1': {
            'eventdate': '2019-06-14',
        },
        
        # 1520 - Add date
        '1520': {
            'eventdate': '2019-04-20',
        },
        
        # 798 - Add date and causal connection
        '798': {
            'eventdate': '2018-01-01',
            'proposition': 'Communications between Harvard Global Support Services and HAA or club officials addressing travel risks in China and Hong Kong and any warnings about advocacy or political activity during 2018–2019. Harvard GSS sent warnings about PRC political and travel risks. These warnings show Harvard knew that PRC retaliation against speech, credentials claims, or political controversy was a real possibility.',
            'causal_connection': 'harvard_knew_prc_retaliation_risk',
        },
        
        # 2260 - Add date
        '2260': {
            'eventdate': '2019-04-19',
        },
        
        # 816 - Add date
        '816': {
            'eventdate': '2019-04-19',
        },
        
        # 1543_S1 - Add date
        '1543_S1': {
            'eventdate': '2019-04-24',
        },
        
        # 291 - Expand proposition
        '291': {
            'proposition': 'His résumé appears to contain misrepresentations regarding Harvard affiliation, which contributed to the confusion surrounding his credentials and the subsequent defamatory statements.',
        },
        
        # 3321 - Add date
        '3321': {
            'eventdate': '2019-04-19',
        },
        
        # 4056 - Expand proposition
        '4056': {
            'proposition': 'The defamatory statements were published to third parties, satisfying the publication element of libel.',
        },
        
        # 1365 - Add date
        '1365': {
            'eventdate': '2020-01-01',
        },
        
        # 348 - Add date
        '348': {
            'eventdate': '2018-01-01',
        },
        
        # 3569 - Expand proposition
        '3569': {
            'proposition': 'The PRC poses a safety risk due to its history of arbitrary detention, retaliation for political speech, and surveillance of individuals connected to politically sensitive allegations.',
        },
        
        # MISSING_0067 - Update ActorRole
        'MISSING_0067': {
            'actorrole': 'Harvard Office of the General Counsel',
            'eventdate': 'Unknown',  # Contextual historical comparison only
        },
        
        # MISSING_0080 - Add date note
        'MISSING_0080': {
            'eventdate': 'Unknown',  # Refers to OGC practice, no specific event
        },
        
        # 3780 - Add causal connection
        '3780': {
            'proposition': "Harvard's internal travel-risk modelling indicated that PRC travel carries a risk of arbitrary detention. This shows Harvard had knowledge of the danger before its affiliates published statements about the plaintiff that circulated in China, thus increasing foreseeability.",
            'causal_connection': 'harvard_knowledge_before_publication',
        },
        
        # 1543_S2 - Expand proposition
        '1543_S2': {
            'proposition': 'These emails warned Harvard entities of serious safety risks arising from the circulation of false statements in PRC-facing media.',
        },
    }
    
    # Apply clarifications
    for idx, row in df.iterrows():
        factid = str(row.get('factid', '')).strip()
        
        if factid in clarifications:
            clar = clarifications[factid]
            
            # Update EventDate
            if 'eventdate' in clar:
                old_date = str(row.get('eventdate', '')).strip()
                new_date = clar['eventdate']
                if old_date.lower() in ('', 'nan', 'unknown', 'none') or old_date != new_date:
                    df.at[idx, 'eventdate'] = new_date
                    if new_date != 'Unknown':
                        changes['dates_added'] += 1
            
            # Update Proposition
            if 'proposition' in clar:
                old_prop = str(row.get('proposition', '')).strip()
                new_prop = clar['proposition']
                if old_prop != new_prop:
                    df.at[idx, 'proposition'] = new_prop
                    if 'propositionclean_v2' in df.columns:
                        df.at[idx, 'propositionclean_v2'] = new_prop
                    changes['propositions_expanded'] += 1
            
            # Update ActorRole
            if 'actorrole' in clar:
                old_role = str(row.get('actorrole', '')).strip()
                new_role = clar['actorrole']
                if old_role != new_role:
                    df.at[idx, 'actorrole'] = new_role
                    changes['actorroles_updated'] += 1
            
            # Update EventLocation if needed
            if 'eventlocation' in clar:
                df.at[idx, 'eventlocation'] = clar['eventlocation']
                changes['locations_added'] += 1
            
            # Note causal connections added
            if 'causal_connection' in clar:
                changes['causal_connections_added'] += 1
    
    return df, changes


def write_report(changes: dict, initial_count: int, final_count: int) -> None:
    """Write report of applied clarifications."""
    report = f"""# V10.2 Clarifications Applied Report

## Summary

- **Input**: `{V10_1_INPUT.name}`
- **Output**: `{OUTPUT_CSV.name}`
- **Initial facts**: {initial_count}
- **Final facts**: {final_count}

## Clarifications Applied

### Dates Added ✅
- **EventDates added/updated**: {changes['dates_added']}
- Facts now have specific dates extracted from user answers

### Propositions Expanded ✅
- **Propositions expanded**: {changes['propositions_expanded']}
- Incomplete facts now have full context and causal connections

### Causal Connections Added ✅
- **Causal connections added**: {changes['causal_connections_added']}
- Facts now explicitly state how Harvard actions relate to PRC outcomes

### ActorRoles Updated ✅
- **ActorRoles updated**: {changes['actorroles_updated']}
- Generic roles replaced with specific entities

### Locations Added ✅
- **EventLocations added**: {changes['locations_added']}

## Key Improvements

- ✅ Temporal connections made explicit (e.g., April 2019 → June 2019 EsuWiki case)
- ✅ Causal pathways clarified (e.g., Harvard GSS warnings → constructive knowledge)
- ✅ Missing dates filled from evidence
- ✅ Incomplete propositions expanded with full context
- ✅ Actor specificity improved

## Next Steps

The v10.2 dataset now has:
- More complete propositions
- Explicit causal connections
- Filled EventDates
- Improved ActorRole specificity

Ready for further review or BN/DAG integration.
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')


def main():
    """Main execution."""
    print("="*80)
    print("APPLYING CLARIFICATION ANSWERS")
    print("="*80)
    
    print("\n1. Loading facts...")
    top100, full = load_facts()
    print(f"   Loaded {len(top100)} top 100 facts")
    print(f"   Loaded {len(full)} full dataset facts")
    
    print("\n2. Applying clarifications to top 100...")
    top100_updated, changes = apply_clarifications(top100)
    print(f"   Dates added: {changes['dates_added']}")
    print(f"   Propositions expanded: {changes['propositions_expanded']}")
    print(f"   Causal connections added: {changes['causal_connections_added']}")
    print(f"   ActorRoles updated: {changes['actorroles_updated']}")
    
    print("\n3. Applying clarifications to full dataset...")
    full_updated, full_changes = apply_clarifications(full)
    print(f"   Dates added: {full_changes['dates_added']}")
    print(f"   Propositions expanded: {full_changes['propositions_expanded']}")
    print(f"   Causal connections added: {full_changes['causal_connections_added']}")
    print(f"   ActorRoles updated: {full_changes['actorroles_updated']}")
    
    print("\n4. Exporting updated datasets...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    top100_updated.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"   ✅ Exported {OUTPUT_CSV}")
    
    OUTPUT_FULL_CSV.parent.mkdir(parents=True, exist_ok=True)
    full_updated.to_csv(OUTPUT_FULL_CSV, index=False, encoding='utf-8')
    print(f"   ✅ Exported {OUTPUT_FULL_CSV}")
    
    print("\n5. Writing report...")
    write_report(changes, len(top100), len(top100_updated))
    print(f"   ✅ Written {REPORT_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE! Clarifications applied to v10.2")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

