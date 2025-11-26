#!/usr/bin/env python3
"""
Generate v25 gap-bridging facts from 9 refined answers.
Addresses remaining v24 network gaps:
- Knowledge narrative (Harvard alumni knowledge before April 19)
- Spoliation narrative (Statement 2 date falsification, website deletion timing)
- Harm narrative (refined economic harm, comprehensive credential targeting)
- Publication narrative (Statement 1 accessibility proof, credential targeting)
- Email narrative (EsuWiki connection to OGC emails)
- Defamation connectivity (comprehensive credential targeting)
"""

import csv
from pathlib import Path
from datetime import datetime

V24_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v24_final.csv")
V25_OUTPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v25_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/v25_gap_bridging_facts_report.md")

# New facts based on refined answers
NEW_FACTS = [
    # ========================================================================
    # 1. Harvard Alumni Knowledge Before April 19, 2019 (Refined)
    # ========================================================================
    {
        "factid": "V25_KNOW_ALUMNI_001",
        "proposition": "Multiple Harvard alumni in China, including Guan and others who attended Mr. Grayson's 2017 lectures, were aware of the Xi slide and his human-rights content years before Harvard published the defamatory statements in April 2019. While no officer-level Harvard administrators or staff were informed before April 19, 2019, there was clear actual knowledge among Harvard-connected alumni, establishing that Harvard-affiliated individuals had prior knowledge of the politically sensitive content before the defamation.",
        "subject": "Harvard Alumni",
        "actorrole": "Harvard Club",
        "eventtype": "Allegation",
        "eventdate": "2017-01-01",  # Approximate start of 2017 tour
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.88",
        "causal_salience_reason": "Establishes broader Harvard alumni knowledge before defamation; strengthens knowledge narrative; connects to foreseeability",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 2. Statement 2 Date Falsification
    # ========================================================================
    {
        "factid": "V25_SPOL_DATE_001",
        "proposition": "Statement 2 was never published with its true date. Harvard replaced the body text of Statement 1 but kept the original April 19, 2019 date stamp, so Statement 2 effectively adopted a false April 19 timestamp. Mr. Grayson has screenshots proving that Statement 2 never carried its true publication date, establishing a pattern of date manipulation and falsification in Harvard's publication record.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Spoliation",
        "eventdate": "2019-04-29",  # Approximate actual publication date
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "already_public",
        "causal_salience_score": "0.87",
        "causal_salience_reason": "Documents Statement 2 date falsification; strengthens spoliation narrative; establishes pattern of manipulation",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 3. Website Deletion Timing (November 2024 - January 2025)
    # ========================================================================
    {
        "factid": "V25_SPOL_DELETE_001",
        "proposition": "The Harvard Club of Shanghai website and Statement 2 on the Beijing site were visible before Mr. Grayson's November 2024 HAA inquiry and were gone before his April 2025 emails to Harvard. Based on the behavior pattern and timing, the deletions occurred within the first two months after the November 2024 inquiry, directly linking the website removals to Harvard's awareness of potential legal liability.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Spoliation",
        "eventdate": "2024-12-01",  # Approximate midpoint of deletion window
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "already_public",
        "causal_salience_score": "0.86",
        "causal_salience_reason": "Documents website deletion timing; links deletions to HAA inquiry; strengthens spoliation narrative",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 4. Refined Economic Harm (Comprehensive)
    # ========================================================================
    {
        "factid": "V25_HARM_ECON_004",
        "proposition": "Following the April 2019 defamation, Mr. Grayson experienced direct, immediate, and measurable economic losses: parents of students contacted him after the defamation, citing online articles, expressing concern, and canceling programs with some requesting refunds; business partners cancelled scheduled classes and speaking events; before the defamation, he had already signed multiple $50,000-range deals, but after Harvard defamed him, he never secured another deal of that size again; he was also seeking investment at the time, and the controversy destroyed those opportunities, showing a clear and documented economic collapse following Harvard's actions.",
        "subject": "Plaintiff",
        "actorrole": "Plaintiff",
        "eventtype": "Harm",
        "eventdate": "2019-04-19",  # Starting point
        "eventlocation": "PRC",
        "truthstatus": "Alleged",
        "evidencetype": "Exhibit",
        "safetyrisk": "medium",
        "publicexposure": "not_public",
        "causal_salience_score": "0.92",
        "causal_salience_reason": "Documents comprehensive economic harm; quantifies losses ($50K contracts, refunds, cancelled events, investment opportunities); strengthens harm narrative",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 5. EsuWiki Connection to OGC Emails
    # ========================================================================
    {
        "factid": "V25_EMAIL_ESU_001",
        "proposition": "In his April 2025 emails to Harvard OGC, Mr. Grayson warned Harvard about political retaliation, torture exposure, and PRC sensitivity. While he did not reference EsuWiki explicitly, if Harvard already knew about his Xi slide before receiving his Hong Kong statement of claim, then they would have understood the EsuWiki arrests might have been connected to his Harvard affiliations as early as his April 7 email to them, meaning Harvard was on notice of the political danger even without explicit EsuWiki references.",
        "subject": "Harvard OGC",
        "actorrole": "Harvard OGC",
        "eventtype": "Communication",
        "eventdate": "2025-04-07",  # First OGC email date
        "eventlocation": "USA",
        "truthstatus": "Alleged",
        "evidencetype": "Email",
        "safetyrisk": "high",
        "publicexposure": "not_public",
        "causal_salience_score": "0.89",
        "causal_salience_reason": "Connects OGC emails to EsuWiki context; establishes Harvard knowledge of political danger; strengthens email and knowledge narratives",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 6. Statement 1 Accessibility Proof (Refined)
    # ========================================================================
    {
        "factid": "V25_PUB_PERSIST_002",
        "proposition": "Statement 1 was posted as a PDF in the Harvard Club WeChat groups. Items posted in those groups cannot be removed retroactively. Mr. Grayson told McGrath this in 2019. To prepare for this lawsuit, he re-entered the same WeChat group in 2025, located the original PDF, and downloaded it again, proving that Statement 1 remained available to any member who had previously opened it, including hundreds or thousands of Harvard alumni, regardless of whether website versions were later removed.",
        "subject": "Harvard Club",
        "actorrole": "Harvard Club",
        "eventtype": "Publication",
        "eventdate": "2019-04-19",  # Initial publication
        "eventlocation": "PRC",
        "truthstatus": "True",
        "evidencetype": "Exhibit",
        "safetyrisk": "high",
        "publicexposure": "already_public",
        "causal_salience_score": "0.91",
        "causal_salience_reason": "Documents Statement 1 persistence with proof (2025 re-download); establishes continuous availability; strengthens publication narrative; shows Harvard knowledge of persistence",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 7. Comprehensive Credential Targeting by Monkey/Résumé Articles
    # ========================================================================
    {
        "factid": "V25_DEFAM_CRED_001",
        "proposition": "The Monkey and Résumé articles did not merely mock a single credential like Mr. Grayson's Harvard Chess Club position—they systematically attacked and delegitimized nearly every Harvard credential he possesses, including his academic degrees (dual concentration in Philosophy and Religion, minor in Psychology), Rockefeller honors and fellowships, admissions roles (interviewer, UAC membership), chess leadership (Harvard Chess Club president), memory scholarship (U.S. top 20 memory expert), Associated Press scholar designation, and professional achievements, using Harvard's own defamatory statements as their foundation to create a false insinuation that all such claims were deceptions.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-19",  # Approximate publication date
        "eventlocation": "PRC",
        "truthstatus": "HostileFalseClaim",
        "evidencetype": "Document",
        "safetyrisk": "high",
        "publicexposure": "already_public",
        "causal_salience_score": "0.94",
        "causal_salience_reason": "Documents comprehensive credential targeting; connects defamation to publication amplification; strengthens publication and harm narratives; worst for Harvard",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 8. Academic Credentials Targeted
    # ========================================================================
    {
        "factid": "V25_DEFAM_ACAD_001",
        "proposition": "The Monkey and Résumé articles misrepresented or ridiculed Mr. Grayson's academic qualifications, including his dual concentration (double major) in Philosophy and Religion, his minor in Psychology, and his status as a Harvard graduate, framing these legitimate achievements as unbelievable, inflated, or fraudulent by pairing them with the defamatory Harvard Club statements to imply they were part of a fabricated persona.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-19",
        "eventlocation": "PRC",
        "truthstatus": "HostileFalseClaim",
        "evidencetype": "Document",
        "safetyrisk": "high",
        "publicexposure": "already_public",
        "causal_salience_score": "0.88",
        "causal_salience_reason": "Documents academic credential targeting; connects to defamation narrative; strengthens publication and harm narratives",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 9. Rockefeller and Fellowship Credentials Targeted
    # ========================================================================
    {
        "factid": "V25_DEFAM_ROCK_001",
        "proposition": "The Monkey and Résumé articles mocked or cast doubt on Mr. Grayson's Rockefeller fellowship and honors, Rockefeller awards, and affiliation with Rockefeller programs or recognitions, listing these legitimate achievements in a sarcastic manner and pairing them with Harvard Club statements to imply they were fabricated or exaggerated, thereby undermining his fellowship-based credentials.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-19",
        "eventlocation": "PRC",
        "truthstatus": "HostileFalseClaim",
        "evidencetype": "Document",
        "safetyrisk": "high",
        "publicexposure": "already_public",
        "causal_salience_score": "0.87",
        "causal_salience_reason": "Documents Rockefeller credential targeting; connects to defamation narrative; strengthens publication and harm narratives",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 10. Admissions Roles Targeted
    # ========================================================================
    {
        "factid": "V25_DEFAM_ADM_001",
        "proposition": "Both the Monkey and Résumé articles repeatedly and aggressively attacked Mr. Grayson's admissions interviewer status, his Harvard Undergraduate Admissions Committee (UAC) membership, and his volunteer role serving Harvard admissions in China, contrasting these real credentials with the defamatory Harvard Club statements to create the false insinuation that he never held any admissions-related role at all and that all such claims were deceptions.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-19",
        "eventlocation": "PRC",
        "truthstatus": "HostileFalseClaim",
        "evidencetype": "Document",
        "safetyrisk": "high",
        "publicexposure": "already_public",
        "causal_salience_score": "0.90",
        "causal_salience_reason": "Documents admissions role targeting; directly connects to Harvard Club statements; strengthens publication and harm narratives",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
    
    # ========================================================================
    # 11. Professional and Memory Credentials Targeted
    # ========================================================================
    {
        "factid": "V25_DEFAM_PROF_001",
        "proposition": "The Monkey and Résumé articles lumped together Mr. Grayson's memory championship standing, U.S. 'top 20 memory expert' recognition, Rockefeller essay award, Associated Press scholar designation, and language proficiency, presenting them as absurd, fabricated, or inherently suspicious, specifically in reliance on Harvard Club statements as 'evidence' that his credentials were fraudulent, thereby attacking his professional and fellowship-based achievements.",
        "subject": "Third-Party Publisher",
        "actorrole": "Third-Party Publisher",
        "eventtype": "Publication",
        "eventdate": "2019-04-19",
        "eventlocation": "PRC",
        "truthstatus": "HostileFalseClaim",
        "evidencetype": "Document",
        "safetyrisk": "high",
        "publicexposure": "already_public",
        "causal_salience_score": "0.88",
        "causal_salience_reason": "Documents professional credential targeting; connects to defamation narrative; strengthens publication and harm narratives",
        "classificationfixed_v3": "",
        "manual_salience_tier": "",
        "new_salience_score": "",
    },
]


def add_gap_bridging_facts_to_v24():
    """Add v25 gap-bridging facts to v24 dataset."""
    print("="*80)
    print("GENERATING V25 GAP-BRIDGING FACTS")
    print("="*80)
    
    # Load v24 facts
    print("\n1. Loading v24 facts...")
    facts = []
    with open(V24_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    print(f"   Loaded {len(facts)} facts from v24")
    
    # Add new facts
    print(f"\n2. Adding {len(NEW_FACTS)} new gap-bridging facts...")
    for fact in NEW_FACTS:
        facts.append(fact)
        factid = fact.get('factid', '')
        print(f"   ✅ Added {factid}")
    
    print(f"\n   Total facts in v25: {len(facts)}")
    
    # Write v25 CSV
    print("\n3. Writing v25 CSV...")
    if facts:
        # Get all fieldnames
        all_fieldnames = set()
        for fact in facts:
            all_fieldnames.update(fact.keys())
        fieldnames = sorted(list(all_fieldnames))
        
        with open(V25_OUTPUT, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for fact in facts:
                row = {}
                for field in fieldnames:
                    row[field] = fact.get(field, '')
                writer.writerow(row)
        print(f"   ✅ Exported {V25_OUTPUT}")
    
    # Generate report
    print("\n4. Generating report...")
    report = f"""# V25 Gap-Bridging Facts Report

## Summary

Added {len(NEW_FACTS)} new gap-bridging facts to v24 based on 9 refined answers addressing remaining network gaps.

## Facts Added

### 1. Knowledge Narrative
- **V25_KNOW_ALUMNI_001**: Broader Harvard alumni knowledge before April 19, 2019
- Establishes multiple alumni awareness of Xi slide before defamation

### 2. Spoliation Narrative
- **V25_SPOL_DATE_001**: Statement 2 date falsification (never published with true date)
- **V25_SPOL_DELETE_001**: Website deletion timing (November 2024 - January 2025)
- Documents pattern of date manipulation and reactive deletions

### 3. Harm Narrative
- **V25_HARM_ECON_004**: Comprehensive economic harm (refined with specific details)
- Documents $50K contracts, refunds, cancelled events, investment opportunities

### 4. Email Narrative
- **V25_EMAIL_ESU_001**: EsuWiki connection to OGC emails
- Establishes Harvard knowledge of political danger through EsuWiki context

### 5. Publication Narrative
- **V25_PUB_PERSIST_002**: Statement 1 accessibility proof (2025 re-download)
- Documents continuous availability with concrete evidence

### 6. Defamation Connectivity (Comprehensive Credential Targeting)
- **V25_DEFAM_CRED_001**: Comprehensive credential targeting summary
- **V25_DEFAM_ACAD_001**: Academic credentials targeted
- **V25_DEFAM_ROCK_001**: Rockefeller credentials targeted
- **V25_DEFAM_ADM_001**: Admissions roles targeted
- **V25_DEFAM_PROF_001**: Professional and memory credentials targeted

## Expected Impact

These facts should:
- **Strengthen knowledge narrative** (broader alumni knowledge before defamation)
- **Strengthen spoliation narrative** (date falsification, deletion timing)
- **Strengthen harm narrative** (comprehensive economic harm, credential targeting)
- **Strengthen email narrative** (EsuWiki connection to OGC emails)
- **Strengthen publication narrative** (Statement 1 proof, comprehensive credential targeting)
- **Connect defamation node** (comprehensive credential targeting directly links to defamation)

## Statistics

- **Facts in v24**: {len(facts) - len(NEW_FACTS)}
- **Facts added**: {len(NEW_FACTS)}
- **Facts in v25**: {len(facts)}

## Files Generated

- **V25 CSV**: `case_law_data/top_1000_facts_for_chatgpt_v25_final.csv`
- **Report**: `reports/analysis_outputs/v25_gap_bridging_facts_report.md`

## Next Steps

1. Review v25 facts for accuracy
2. Re-run network gap analysis to verify improvements
3. Generate v25 outputs (top 100, causation nodes, etc.)
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✅ Written {REPORT_PATH}")
    
    print()
    print("="*80)
    print("V25 GAP-BRIDGING FACTS GENERATED")
    print("="*80)
    print()
    print(f"✅ Added {len(NEW_FACTS)} new facts")
    print(f"✅ Total facts in v25: {len(facts)}")
    print(f"✅ Output: {V25_OUTPUT}")
    print()
    print("Key additions:")
    print("  • Broader Harvard alumni knowledge before defamation")
    print("  • Statement 2 date falsification")
    print("  • Website deletion timing (Nov 2024 - Jan 2025)")
    print("  • Comprehensive economic harm (refined)")
    print("  • EsuWiki connection to OGC emails")
    print("  • Statement 1 accessibility proof (2025 re-download)")
    print("  • Comprehensive credential targeting (5 facts covering all credential types)")


if __name__ == "__main__":
    add_gap_bridging_facts_to_v24()

