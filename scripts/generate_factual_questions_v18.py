#!/usr/bin/env python3
"""
Generate specific factual questions and fact-check tasks for v18 dataset.
Organized by the 6 critical categories identified in the factual audit.
"""

from pathlib import Path

OUTPUT_QUESTIONS_CSV = Path("case_law_data/v18_factual_questions.csv")
OUTPUT_QUESTIONS_TXT = Path("case_law_data/v18_factual_questions.txt")
OUTPUT_REPORT = Path("reports/analysis_outputs/v18_factual_questions_report.md")

# Fact-check tasks organized by category
FACT_CHECK_TASKS = {
    "A. Harvard's Knowledge and Foreseeability": {
        "priority": "CRITICAL",
        "description": "Core to establishing foreseeability. Only 3 'Harvard Knowledge' nodes found in causation report.",
        "tasks": [
            {
                "task_id": "A1",
                "question": "What is the exact content of Harvard GSS risk advisories regarding PRC travel?",
                "specific_questions": [
                    "Did GSS explicitly warn about arbitrary detention in China?",
                    "Did GSS warn about exit bans?",
                    "Did GSS warn about retaliation for political speech?",
                    "Did GSS warn about risks of referencing sensitive political topics + personal background?",
                    "What is the exact wording of these warnings?",
                ],
                "expected_sources": [
                    "GSS email archives (2018-2019)",
                    "GSS travel advisory documents",
                    "GSS distribution lists showing who received warnings",
                    "HAA/club email chains referencing GSS warnings",
                ],
                "related_factids": ["798", "281", "3780", "3910"],
                "validation_criteria": [
                    "Primary source document (email, PDF, official advisory)",
                    "Date stamp showing when warning was issued",
                    "Distribution list or CC list showing recipients",
                    "Exact quoted text from warning",
                ],
            },
            {
                "task_id": "A2",
                "question": "Who at Harvard actually received or reviewed GSS warnings and risk materials?",
                "specific_questions": [
                    "Which named Harvard decision-makers received GSS warnings?",
                    "Did Yi Wang receive GSS warnings?",
                    "Did MJ Tang receive GSS warnings?",
                    "Did HAA staff receive GSS warnings?",
                    "Did OGC lawyers receive GSS warnings?",
                    "Can this be proven via email headers, CC lists, meeting notes, or GSS distribution lists?",
                ],
                "expected_sources": [
                    "Email headers showing recipients",
                    "CC/BCC lists from GSS emails",
                    "Meeting notes or minutes referencing GSS warnings",
                    "GSS distribution lists or mailing lists",
                    "Witness statements from recipients",
                ],
                "related_factids": ["798", "281", "3780"],
                "validation_criteria": [
                    "Email metadata showing recipient list",
                    "Documented acknowledgment of receipt",
                    "Evidence of review (read receipts, replies, meeting notes)",
                ],
            },
            {
                "task_id": "A3",
                "question": "What did Harvard know about Peter Humphrey's case and when?",
                "specific_questions": [
                    "When did Harvard become aware of Peter Humphrey's 2013-2015 PRC imprisonment?",
                    "Which Harvard officials were aware of this case?",
                    "Was this case discussed in Harvard internal communications?",
                    "Was this case referenced in GSS warnings or risk assessments?",
                ],
                "expected_sources": [
                    "Harvard internal emails referencing Humphrey case",
                    "GSS risk assessments mentioning Humphrey",
                    "Fairbank Center communications about Humphrey",
                    "Meeting notes or reports discussing Humphrey case",
                ],
                "related_factids": ["2539"],
                "validation_criteria": [
                    "Dated internal communication referencing Humphrey",
                    "Evidence of awareness before April 2019",
                ],
            },
            {
                "task_id": "A4",
                "question": "What did Harvard know about Xi Mingze censorship incidents?",
                "specific_questions": [
                    "When did Harvard become aware of PRC censorship of Xi Mingze/Harvard enrollment stories?",
                    "Which Harvard officials were aware of this censorship?",
                    "Was this discussed in Harvard internal communications?",
                    "Was this referenced in risk assessments?",
                ],
                "expected_sources": [
                    "Harvard internal emails about Xi Mingze censorship",
                    "Risk assessments mentioning Xi Mingze sensitivity",
                    "Media monitoring reports",
                ],
                "related_factids": ["885"],
                "validation_criteria": [
                    "Dated communication showing awareness",
                    "Evidence of awareness before April 2019",
                ],
            },
        ],
    },
    "B. Harvard's Control Over Clubs (Non-Delegable Duty)": {
        "priority": "CRITICAL",
        "description": "ActorRole normalization cleaned labels, but underlying control/responsibility questions remain unresolved.",
        "tasks": [
            {
                "task_id": "B1",
                "question": "What is the textual proof of governance relationships between Harvard, HAA, and clubs?",
                "specific_questions": [
                    "Do bylaws show clubs operate 'under the auspices' of HAA?",
                    "Do affiliation agreements show HAA or Harvard can direct content of club statements?",
                    "Do branding rules show Harvard control over club communications?",
                    "Do oversight provisions show admissions activity is a non-delegable function of Harvard College?",
                ],
                "expected_sources": [
                    "HAA bylaws",
                    "Club affiliation agreements",
                    "Harvard branding guidelines",
                    "Harvard admissions policies",
                    "Club charter documents",
                ],
                "related_factids": ["480", "1269", "44"],
                "validation_criteria": [
                    "Primary source document (bylaw, agreement, policy)",
                    "Exact quoted text showing control/responsibility",
                    "Date of document (to show it was in effect in 2019)",
                ],
            },
            {
                "task_id": "B2",
                "question": "Who actually instructed the clubs in 2019 to publish Statement 1/Statement 2?",
                "specific_questions": [
                    "Did Cambridge-based actors draft Statement 1?",
                    "Did Cambridge-based actors approve Statement 1?",
                    "Did Cambridge-based actors order Statement 1?",
                    "Are there specific emails or instructions showing university direction?",
                    "Who sent the instructions? (Yi Wang? MJ Tang? HAA staff? OGC?)",
                    "When were instructions sent?",
                ],
                "expected_sources": [
                    "Emails from Cambridge to clubs",
                    "Meeting notes showing instructions",
                    "Draft versions of Statement 1 with Cambridge edits",
                    "Approval chains showing Cambridge sign-off",
                ],
                "related_factids": ["2260", "816", "212", "KGFACT_015"],
                "validation_criteria": [
                    "Dated email or document showing instruction",
                    "Clear sender (Cambridge-based actor)",
                    "Clear recipient (club)",
                    "Exact instruction text",
                ],
            },
            {
                "task_id": "B3",
                "question": "What is the evidence of Harvard's non-delegable duty regarding admissions?",
                "specific_questions": [
                    "Do Harvard policies state that admissions is a non-delegable function?",
                    "Do policies show Harvard retains control over admissions-related communications?",
                    "Are there examples of Harvard directing or controlling club admissions activities?",
                ],
                "expected_sources": [
                    "Harvard admissions policies",
                    "Harvard governance documents",
                    "Examples of Harvard directing club admissions activities",
                ],
                "related_factids": ["1269", "268", "1831"],
                "validation_criteria": [
                    "Primary source policy document",
                    "Exact quoted text",
                    "Date showing policy was in effect in 2019",
                ],
            },
        ],
    },
    "C. OGC Non-Response & Litigation Hold": {
        "priority": "HIGH",
        "description": "10 OGC Non-Response nodes found, but hard factual questions about receipt and litigation hold remain.",
        "tasks": [
            {
                "task_id": "C1",
                "question": "Do we have proof that OGC actually received the April 7, April 18, and August 11, 2025 emails?",
                "specific_questions": [
                    "Are there server-side evidence (bounce logs, delivery receipts)?",
                    "Are there OGC disclaimers or auto-replies?",
                    "If not, are we inferring receipt from circumstantial clues?",
                    "Do we have technical logs showing delivery?",
                ],
                "expected_sources": [
                    "Email server logs",
                    "Delivery receipts",
                    "Read receipts",
                    "Bounce logs (if emails bounced)",
                    "OGC auto-reply or out-of-office messages",
                ],
                "related_factids": ["CORR_2025_001", "CORR_2025_002", "CORR_2025_003", "MISSING_0084"],
                "validation_criteria": [
                    "Technical log showing delivery",
                    "Receipt confirmation",
                    "Evidence of non-bounce",
                ],
            },
            {
                "task_id": "C2",
                "question": "Is there evidence that Harvard did NOT issue a litigation hold in 2019 or 2025?",
                "specific_questions": [
                    "Did Harvard issue a litigation hold in 2019?",
                    "Did Harvard issue a litigation hold in 2025?",
                    "If issued, was it broad or narrow?",
                    "Did it apply to web pages only, or also to internal emails?",
                    "Is there evidence of spoliation (deletion of relevant materials)?",
                ],
                "expected_sources": [
                    "Litigation hold notices (if issued)",
                    "Evidence of absence of litigation hold",
                    "Evidence of document deletion or spoliation",
                    "Discovery responses showing what was preserved",
                ],
                "related_factids": ["MISSING_0067", "MISSING_0069"],
                "validation_criteria": [
                    "Documented litigation hold (if exists)",
                    "Evidence of absence (if no hold issued)",
                    "Evidence of spoliation (if materials deleted)",
                ],
            },
            {
                "task_id": "C3",
                "question": "What is the evidence of OGC's duty to investigate, request, and preserve?",
                "specific_questions": [
                    "Do Harvard policies require OGC to investigate complaints?",
                    "Do policies require OGC to preserve evidence when litigation is foreseeable?",
                    "What is the standard of care for OGC in similar situations?",
                ],
                "expected_sources": [
                    "Harvard OGC policies",
                    "Legal standards for litigation hold",
                    "Case law on OGC duties",
                ],
                "related_factids": ["MISSING_0067", "MISSING_0069"],
                "validation_criteria": [
                    "Primary source policy or legal standard",
                    "Exact quoted text",
                ],
            },
        ],
    },
    "D. PRC Response and EsuWiki Linkage": {
        "priority": "CRITICAL",
        "description": "Only 3 PRC Response nodes found, but theory depends on Harvard publicity → PRC monitoring → EsuWiki → arrests → torture chain.",
        "tasks": [
            {
                "task_id": "D1",
                "question": "What is the exact EsuWiki timeline and are all dates tied to specific, citable sources?",
                "specific_questions": [
                    "Is case-opening date (2019-06-14) tied to a specific source?",
                    "What are the exact arrest dates?",
                    "What are the exact charge descriptions?",
                    "Are these facts in v18 always labelled TruthStatus = True where they should be?",
                ],
                "expected_sources": [
                    "PRC court documents",
                    "PRC Ministry of Public Security announcements",
                    "Media reports (with source verification)",
                    "EsuWiki case file documents",
                ],
                "related_factids": ["KGFACT_010", "KGFACT_011", "KGFACT_012", "815"],
                "validation_criteria": [
                    "Primary source document (court doc, official announcement)",
                    "Date stamp",
                    "Exact quoted text",
                ],
            },
            {
                "task_id": "D2",
                "question": "Is there any direct evidence of PRC reliance on Harvard-related material?",
                "specific_questions": [
                    "Do any PRC or state-media documents explicitly reference Harvard?",
                    "Do any PRC documents reference Harvard clubs?",
                    "Do any PRC documents reference Xi Mingze at Harvard?",
                    "Do any PRC documents reference plaintiff's name?",
                    "Or is the linkage entirely inferential (timing + topic overlap)?",
                ],
                "expected_sources": [
                    "PRC court documents",
                    "PRC state media articles",
                    "PRC security service documents",
                    "EsuWiki case file",
                ],
                "related_factids": ["KGFACT_010", "KGFACT_011", "KGFACT_012", "KGFACT_015"],
                "validation_criteria": [
                    "Primary source document explicitly referencing Harvard/plaintiff",
                    "Or clear statement that linkage is inferential",
                ],
            },
            {
                "task_id": "D3",
                "question": "What is the evidence connecting Harvard publications to EsuWiki investigation?",
                "specific_questions": [
                    "What is the temporal connection (April 2019 Harvard publications → June 2019 EsuWiki opening)?",
                    "What is the topical connection (Xi Mingze references in both)?",
                    "What is the evidence of PRC monitoring of Harvard-related content?",
                ],
                "expected_sources": [
                    "Timeline analysis",
                    "Content analysis showing Xi Mingze references",
                    "Evidence of PRC monitoring (if available)",
                ],
                "related_factids": ["KGFACT_015", "885", "1636"],
                "validation_criteria": [
                    "Clear temporal connection",
                    "Clear topical connection",
                    "Evidence of monitoring (if available)",
                ],
            },
        ],
    },
    "E. Third-Party Amplifiers": {
        "priority": "HIGH",
        "description": "19 Public Exposure nodes found, but each major amplifier needs verification.",
        "tasks": [
            {
                "task_id": "E1",
                "question": "For each major amplifier (Zhihu, E-Canada, Sohu, WeChat, Baidu), can we verify the publication?",
                "specific_questions": [
                    "Do we have the URL or archived copy for each amplifier?",
                    "What is the exact publication date/time for each?",
                    "Does the text actually reproduce relevant parts of Statement 1/Statement 2?",
                    "Does it mischaracterize plaintiff in the way alleged?",
                ],
                "expected_sources": [
                    "URLs or archived copies of amplifier articles",
                    "Screenshots with timestamps",
                    "Wayback Machine archives",
                    "Original source documents",
                ],
                "related_factids": [
                    "KGFACT_001",  # Zhihu
                    "KGFACT_002",  # E-Canada
                    "KGFACT_003",  # Sohu
                    "KGFACT_005",  # WeChat Monkey
                    "KGFACT_006",  # WeChat Résumé
                    "KGFACT_004",  # Baidu
                ],
                "validation_criteria": [
                    "URL or archived copy",
                    "Date stamp",
                    "Exact quoted text from amplifier",
                    "Comparison showing reproduction of Statement 1/2",
                ],
            },
            {
                "task_id": "E2",
                "question": "For each amplifier, can we show it was accessible inside the PRC?",
                "specific_questions": [
                    "Was Zhihu accessible in PRC in April 2019?",
                    "Was E-Canada accessible in PRC?",
                    "Was Sohu accessible in PRC?",
                    "Was WeChat accessible in PRC?",
                    "Was Baidu accessible in PRC?",
                    "Does this matter for 'PRC visibility' tags?",
                ],
                "expected_sources": [
                    "Geographic accessibility data",
                    "Platform documentation",
                    "User location data (if available)",
                ],
                "related_factids": [
                    "KGFACT_001", "KGFACT_002", "KGFACT_003",
                    "KGFACT_004", "KGFACT_005", "KGFACT_006",
                ],
                "validation_criteria": [
                    "Evidence of PRC accessibility",
                    "Date showing accessibility in April 2019",
                ],
            },
            {
                "task_id": "E3",
                "question": "What is the evidence of amplification and republication?",
                "specific_questions": [
                    "How many times was Statement 1 republished?",
                    "What was the reach/audience of each amplifier?",
                    "What is the evidence of harm from amplification?",
                ],
                "expected_sources": [
                    "Republication counts",
                    "Platform analytics (if available)",
                    "Evidence of harm (harassment, monitoring, etc.)",
                ],
                "related_factids": ["2612", "KGFACT_001", "KGFACT_002", "KGFACT_003"],
                "validation_criteria": [
                    "Quantified republication data",
                    "Evidence of reach",
                    "Evidence of harm",
                ],
            },
        ],
    },
    "F. Plaintiff's Own Status and History": {
        "priority": "MEDIUM",
        "description": "High-salience facts about plaintiff's corrections and disclosures need evidentiary basis.",
        "tasks": [
            {
                "task_id": "F1",
                "question": "What is the primary-source proof of plaintiff's Harvard roles?",
                "specific_questions": [
                    "Do we have Harvard records showing plaintiff was Alumni interviewer 2011-2015?",
                    "Do we have records showing plaintiff was on Undergraduate Admissions Council?",
                    "Do we have Harvard emails or Handbook excerpts allowing 'affiliated with Admissions Office' description?",
                ],
                "expected_sources": [
                    "Harvard records (interviewer assignments, UAC membership)",
                    "Harvard emails confirming roles",
                    "Harvard Handbook excerpts",
                    "Official Harvard communications",
                ],
                "related_factids": ["CORR_2014_001", "CORR_2017_001", "268", "1831"],
                "validation_criteria": [
                    "Primary source Harvard document",
                    "Date showing role was active",
                    "Exact description of role",
                ],
            },
            {
                "task_id": "F2",
                "question": "For each 'hostile' résumé or ad, can we show the misrepresentation?",
                "specific_questions": [
                    "What were the versions plaintiff approved?",
                    "What were the versions they published?",
                    "What is the delta (where they upgraded 'interviewer' → 'officer')?",
                ],
                "expected_sources": [
                    "Approved versions of résumés/ads",
                    "Published versions of résumés/ads",
                    "Comparison showing misrepresentation",
                ],
                "related_factids": ["291", "1667", "1822", "1636"],
                "validation_criteria": [
                    "Side-by-side comparison",
                    "Clear evidence of misrepresentation",
                    "Date of misrepresentation",
                ],
            },
            {
                "task_id": "F3",
                "question": "What is the evidence of plaintiff's disability disclosure in 2014?",
                "specific_questions": [
                    "When exactly was disability disclosed?",
                    "To whom was it disclosed?",
                    "What was the exact content of disclosure?",
                ],
                "expected_sources": [
                    "2014 email or communication",
                    "Harvard records of disclosure",
                ],
                "related_factids": ["CORR_2014_001"],
                "validation_criteria": [
                    "Dated communication",
                    "Exact text of disclosure",
                ],
            },
        ],
    },
}


def generate_questions_csv():
    """Generate CSV of all factual questions."""
    rows = []
    for category, category_data in FACT_CHECK_TASKS.items():
        for task in category_data["tasks"]:
            for specific_q in task["specific_questions"]:
                rows.append({
                    "category": category,
                    "priority": category_data["priority"],
                    "task_id": task["task_id"],
                    "main_question": task["question"],
                    "specific_question": specific_q,
                    "related_factids": "; ".join(task["related_factids"]),
                    "expected_sources": "; ".join(task["expected_sources"]),
                })
    return rows


def generate_questions_txt():
    """Generate human-readable text file of all questions."""
    lines = []
    lines.append("="*80)
    lines.append("V18 FACTUAL QUESTIONS - FACT-CHECK TASKS")
    lines.append("="*80)
    lines.append("")
    lines.append("This document lists all factual questions that need to be answered")
    lines.append("before v18 can be considered a 'final truth table' for litigation.")
    lines.append("")
    lines.append("="*80)
    lines.append("")
    
    for category, category_data in FACT_CHECK_TASKS.items():
        lines.append(f"{category}")
        lines.append(f"Priority: {category_data['priority']}")
        lines.append(f"Description: {category_data['description']}")
        lines.append("")
        lines.append("-"*80)
        lines.append("")
        
        for task in category_data["tasks"]:
            lines.append(f"Task {task['task_id']}: {task['question']}")
            lines.append("")
            lines.append("Specific Questions:")
            for i, sq in enumerate(task["specific_questions"], 1):
                lines.append(f"  {i}. {sq}")
            lines.append("")
            lines.append("Expected Sources:")
            for source in task["expected_sources"]:
                lines.append(f"  - {source}")
            lines.append("")
            lines.append("Related FactIDs:")
            lines.append(f"  {', '.join(task['related_factids'])}")
            lines.append("")
            lines.append("Validation Criteria:")
            for criterion in task["validation_criteria"]:
                lines.append(f"  - {criterion}")
            lines.append("")
            lines.append("-"*80)
            lines.append("")
    
    return "\n".join(lines)


def generate_report():
    """Generate summary report."""
    total_tasks = sum(len(cat["tasks"]) for cat in FACT_CHECK_TASKS.values())
    total_questions = sum(
        len(task["specific_questions"])
        for cat in FACT_CHECK_TASKS.values()
        for task in cat["tasks"]
    )
    
    report = f"""# V18 Factual Questions Report

## Summary

This report identifies **{total_tasks} fact-check tasks** across **6 critical categories**, containing **{total_questions} specific questions** that need to be answered before v18 can be considered a "final truth table" for litigation.

## Categories and Priorities

"""
    
    for category, category_data in FACT_CHECK_TASKS.items():
        report += f"### {category}\n\n"
        report += f"- **Priority**: {category_data['priority']}\n"
        report += f"- **Tasks**: {len(category_data['tasks'])}\n"
        report += f"- **Description**: {category_data['description']}\n\n"
    
    report += """## Task Breakdown

"""
    
    for category, category_data in FACT_CHECK_TASKS.items():
        report += f"### {category}\n\n"
        for task in category_data["tasks"]:
            report += f"**{task['task_id']}**: {task['question']}\n"
            report += f"- Specific questions: {len(task['specific_questions'])}\n"
            report += f"- Related FactIDs: {', '.join(task['related_factids'])}\n\n"
    
    report += """## Files Generated

- **Questions CSV**: `case_law_data/v18_factual_questions.csv`
- **Questions TXT**: `case_law_data/v18_factual_questions.txt`
- **Report**: `reports/analysis_outputs/v18_factual_questions_report.md`

## Next Steps

1. Review the factual questions by category
2. Prioritize based on litigation needs
3. Gather source documents for each category
4. Answer questions systematically
5. Update v18 facts with verified information
6. Generate v19 with fact-checked data

## Notes

- Questions are organized by category for systematic review
- Each task includes expected sources and validation criteria
- Related FactIDs help identify which facts need updating
- Priority levels guide which categories to tackle first
"""
    
    return report


def main():
    """Generate factual questions documents."""
    print("="*80)
    print("GENERATING V18 FACTUAL QUESTIONS")
    print("="*80)
    print()
    
    # Generate CSV
    print("1. Generating questions CSV...")
    rows = generate_questions_csv()
    import csv
    with open(OUTPUT_QUESTIONS_CSV, 'w', encoding='utf-8', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"   ✅ Exported {OUTPUT_QUESTIONS_CSV}")
    print(f"   Total questions: {len(rows)}")
    
    # Generate TXT
    print("\n2. Generating questions TXT...")
    txt_content = generate_questions_txt()
    OUTPUT_QUESTIONS_TXT.write_text(txt_content, encoding='utf-8')
    print(f"   ✅ Exported {OUTPUT_QUESTIONS_TXT}")
    
    # Generate report
    print("\n3. Generating report...")
    report = generate_report()
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report, encoding='utf-8')
    print(f"   ✅ Written {OUTPUT_REPORT}")
    
    # Summary
    total_tasks = sum(len(cat["tasks"]) for cat in FACT_CHECK_TASKS.values())
    total_questions = sum(
        len(task["specific_questions"])
        for cat in FACT_CHECK_TASKS.values()
        for task in cat["tasks"]
    )
    
    print()
    print("="*80)
    print("FACTUAL QUESTIONS GENERATED")
    print("="*80)
    print()
    print(f"✅ {total_tasks} fact-check tasks across 6 categories")
    print(f"✅ {total_questions} specific questions")
    print()
    print("Categories:")
    for category, category_data in FACT_CHECK_TASKS.items():
        print(f"  - {category}: {len(category_data['tasks'])} tasks ({category_data['priority']} priority)")
    print()
    print("Files:")
    print(f"  📄 {OUTPUT_QUESTIONS_CSV}")
    print(f"  📄 {OUTPUT_QUESTIONS_TXT}")
    print(f"  📄 {OUTPUT_REPORT}")
    print()


if __name__ == "__main__":
    main()

