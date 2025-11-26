#!/usr/bin/env python3
"""Extract the next batch of unanswered clarification questions."""

from __future__ import annotations

import csv
from pathlib import Path

INPUT_CSV = Path("case_law_data/clarification_questions.csv")
OUTPUT_TXT = Path("case_law_data/clarification_questions_batch_2.txt")
OUTPUT_CSV = Path("case_law_data/clarification_questions_batch_2.csv")

# Facts already answered in batch 1
ANSWERED_FACTIDS = {
    'KGFACT_010', '815', '3911', '268', '885', '324', 'MISSING_0095', 
    'MISSING_0070', '2428', 'MISSING_0002_REWRITE_1', '1520', '798', 
    '2260', '816', '1543_S1', '291', '3321', '4056', '1365', '348', 
    '3569', 'MISSING_0067', 'MISSING_0080', '3780', '1543_S2'
}


def main():
    """Extract next batch of questions."""
    print("="*80)
    print("EXTRACTING NEXT BATCH OF CLARIFICATION QUESTIONS")
    print("="*80)
    
    # Load all questions
    questions = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    
    print(f"\nTotal questions: {len(questions)}")
    
    # Filter out already answered questions
    remaining = [q for q in questions if str(q.get('factid', '')).strip() not in ANSWERED_FACTIDS]
    print(f"Already answered: {len(ANSWERED_FACTIDS)}")
    print(f"Remaining: {len(remaining)}")
    
    # Sort by priority_score (highest first)
    def get_priority_score(q):
        try:
            return float(q.get('priority_score', 0))
        except (ValueError, TypeError):
            return 0.0
    
    remaining.sort(key=get_priority_score, reverse=True)
    
    # Take next 100
    batch_2 = remaining[:100]
    print(f"\nExtracting next 100 questions...")
    
    # Write CSV
    if batch_2:
        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=batch_2[0].keys())
            writer.writeheader()
            writer.writerows(batch_2)
        print(f"✅ CSV written: {OUTPUT_CSV}")
    
    # Write formatted text file
    lines = []
    lines.append("="*80)
    lines.append("CLARIFICATION QUESTIONS - BATCH 2 (Next 100)")
    lines.append("="*80)
    lines.append("")
    lines.append("Instructions: Answer each question to clarify the corresponding fact.")
    lines.append("Your answers will be used to improve fact accuracy and completeness.")
    lines.append("")
    lines.append("="*80)
    lines.append("")
    
    for row in batch_2:
        factid = str(row.get('factid', '')).strip()
        proposition = str(row.get('proposition', '')).strip()
        issue_type = str(row.get('issue_type', '')).strip()
        priority = str(row.get('priority', '')).strip()
        question = str(row.get('question', '')).strip()
        context = str(row.get('context', '')).strip()
        actorrole = str(row.get('current_actorrole', '')).strip()
        eventdate = str(row.get('current_eventdate', '')).strip()
        safetyrisk = str(row.get('current_safetyrisk', '')).strip()
        publicexposure = str(row.get('current_publicexposure', '')).strip()
        
        # Clean up 'nan' values
        if actorrole.lower() == 'nan':
            actorrole = 'Unknown'
        if eventdate.lower() in ('nan', ''):
            eventdate = 'Unknown'
        if safetyrisk.lower() == 'nan':
            safetyrisk = 'Unknown'
        if publicexposure.lower() == 'nan':
            publicexposure = 'Unknown'
        
        lines.append(f"FACT ID: {factid}")
        lines.append(f"PROPOSITION: {proposition[:200]}{'...' if len(proposition) > 200 else ''}")
        lines.append(f"PRIORITY: {priority.upper()}")
        lines.append(f"ISSUE TYPE: {issue_type}")
        lines.append("")
        lines.append(f"Q: {question}")
        lines.append(f"   Context: {context[:150]}{'...' if len(context) > 150 else ''}")
        lines.append(f"   Current ActorRole: {actorrole}")
        lines.append(f"   Current EventDate: {eventdate}")
        lines.append(f"   Current SafetyRisk: {safetyrisk}")
        lines.append(f"   Current PublicExposure: {publicexposure}")
        lines.append("")
        lines.append("   A: [Your answer here]")
        lines.append("")
        lines.append("")
        lines.append("-"*80)
        lines.append("")
    
    OUTPUT_TXT.write_text("\n".join(lines), encoding='utf-8')
    print(f"✅ Text file written: {OUTPUT_TXT}")
    
    # Summary
    print(f"\n{'='*80}")
    print("BATCH 2 SUMMARY")
    print(f"{'='*80}")
    print(f"Total questions: {len(batch_2)}")
    
    # Count issue types
    issue_types = {}
    priorities = {}
    for row in batch_2:
        it = str(row.get('issue_type', '')).strip()
        p = str(row.get('priority', '')).strip()
        issue_types[it] = issue_types.get(it, 0) + 1
        priorities[p] = priorities.get(p, 0) + 1
    
    print(f"\nIssue types:")
    for issue_type, count in sorted(issue_types.items(), key=lambda x: -x[1]):
        print(f"  - {issue_type}: {count}")
    print(f"\nPriority distribution:")
    for priority, count in sorted(priorities.items(), key=lambda x: -x[1]):
        print(f"  - {priority}: {count}")
    print(f"\n{'='*80}")
    print(f"✅ Files ready:")
    print(f"   📄 Text: {OUTPUT_TXT}")
    print(f"   📊 CSV: {OUTPUT_CSV}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
