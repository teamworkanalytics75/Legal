#!/usr/bin/env python3
"""Check which clarification questions remain unanswered."""

from __future__ import annotations

import csv
import re
from pathlib import Path

V16_DATASET = Path("case_law_data/top_1000_facts_for_chatgpt_v16_final.csv")
CRUCIAL_QUESTIONS = Path("case_law_data/crucial_questions_v15.txt")


def load_facts() -> dict:
    """Load v16 facts."""
    facts = {}
    if not V16_DATASET.exists():
        return facts
    
    with open(V16_DATASET, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            factid = str(row.get('factid', '')).strip()
            facts[factid] = row
    return facts


def extract_questions_from_text(text: str) -> list[dict]:
    """Extract questions from the crucial questions text file."""
    questions = []
    
    # Pattern to match question blocks
    pattern = r'### Question #(\d+).*?\*\*FactID\*\*:\s*([A-Z0-9_]+).*?\*\*Type\*\*:\s*(\w+).*?\*\*Question\*\*:\s*(.*?)(?=\*\*Context\*\*|###|$)'
    
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        q_num = match.group(1)
        factid = match.group(2)
        q_type = match.group(3)
        question = match.group(4).strip()
        
        questions.append({
            'number': int(q_num),
            'factid': factid,
            'type': q_type,
            'question': question,
        })
    
    return questions


def check_if_answered(fact: dict, q_type: str) -> bool:
    """Check if a question has been answered based on fact data."""
    if not fact:
        return False
    
    # Check based on question type
    if 'date' in q_type.lower():
        eventdate = str(fact.get('eventdate', '')).strip()
        return eventdate and eventdate.lower() not in ('', 'unknown', 'nan', 'none')
    
    elif 'causal' in q_type.lower():
        proposition = str(fact.get('proposition', '')).strip()
        # Check if proposition contains causal language
        causal_indicators = ['because', 'therefore', 'thus', 'caused', 'led to', 'resulted in', 
                           'foreseeable', 'foreseeably', 'temporal overlap', 'constructive knowledge']
        return any(indicator in proposition.lower() for indicator in causal_indicators)
    
    elif 'proposition' in q_type.lower() or 'incomplete' in q_type.lower():
        proposition = str(fact.get('proposition', '')).strip()
        return len(proposition) > 50  # Reasonable length
    
    elif 'actor' in q_type.lower() or 'subject' in q_type.lower():
        actorrole = str(fact.get('actorrole', '')).strip()
        subject = str(fact.get('subject', '')).strip()
        return actorrole or subject
    
    return False


def main():
    """Main execution."""
    print("="*80)
    print("CHECKING UNANSWERED CLARIFICATION QUESTIONS")
    print("="*80)
    
    # Load facts
    print("\n1. Loading v16 facts...")
    facts = load_facts()
    print(f"   Loaded {len(facts)} facts")
    
    # Load questions
    if not CRUCIAL_QUESTIONS.exists():
        print(f"\n❌ Error: Questions file not found: {CRUCIAL_QUESTIONS}")
        return
    
    print(f"\n2. Loading questions from {CRUCIAL_QUESTIONS.name}...")
    text = CRUCIAL_QUESTIONS.read_text(encoding='utf-8')
    questions = extract_questions_from_text(text)
    print(f"   Found {len(questions)} questions")
    
    # Check each question
    print(f"\n3. Checking answer status...")
    answered = []
    unanswered = []
    
    for q in questions:
        factid = q['factid']
        fact = facts.get(factid)
        is_answered = check_if_answered(fact, q['type'])
        
        if is_answered:
            answered.append(q)
        else:
            unanswered.append(q)
    
    # Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\n✅ Answered: {len(answered)}/{len(questions)} ({len(answered)/len(questions)*100:.1f}%)")
    print(f"⚠️  Unanswered: {len(unanswered)}/{len(questions)} ({len(unanswered)/len(questions)*100:.1f}%)")
    
    if unanswered:
        print(f"\n{'='*80}")
        print("UNANSWERED QUESTIONS")
        print(f"{'='*80}\n")
        
        # Group by priority/type
        by_type = {}
        for q in unanswered:
            q_type = q['type']
            if q_type not in by_type:
                by_type[q_type] = []
            by_type[q_type].append(q)
        
        for q_type, qs in sorted(by_type.items(), key=lambda x: -len(x[1])):
            print(f"\n{q_type.upper()} ({len(qs)} questions):")
            for q in qs[:10]:  # Show first 10 of each type
                factid = q['factid']
                fact = facts.get(factid)
                status = "❌ Missing" if not fact else "⚠️  Incomplete"
                print(f"  {status} - Question #{q['number']}: FactID {factid}")
                print(f"    {q['question'][:80]}...")
            if len(qs) > 10:
                print(f"  ... and {len(qs) - 10} more")
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"\nYou have {len(unanswered)} unanswered questions remaining.")
        print(f"\nTo answer them:")
        print(f"  1. Review: {CRUCIAL_QUESTIONS}")
        print(f"  2. Create answers file using the template:")
        print(f"     cp case_law_data/answers_template.txt my_answers.txt")
        print(f"  3. Add your answers in dictation-friendly format")
        print(f"  4. Apply: python scripts/apply_fact_answers.py --input my_answers.txt --dataset v16")
    else:
        print(f"\n🎉 All questions have been answered!")


if __name__ == "__main__":
    main()

