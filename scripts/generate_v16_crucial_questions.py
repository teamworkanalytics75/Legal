#!/usr/bin/env python3
"""Generate crucial clarification questions for v16 dataset (updated)."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from collections import defaultdict

# Use the updated v16 dataset
V16_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v16_final_updated.csv")
OUTPUT_CSV = Path("case_law_data/crucial_questions_v16.csv")
OUTPUT_TXT = Path("case_law_data/crucial_questions_v16.txt")
REPORT_PATH = Path("reports/analysis_outputs/crucial_questions_v16_report.md")


def load_facts() -> list[dict]:
    """Load v16 facts."""
    facts = []
    input_path = V16_INPUT
    if not input_path.exists():
        # Fallback to original v16 if updated doesn't exist
        input_path = Path("case_law_data/top_1000_facts_for_chatgpt_v16_final.csv")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    return facts


def detect_crucial_issues(fact: dict) -> list[dict]:
    """Detect crucial issues that need clarification."""
    issues = []
    factid = str(fact.get('factid', '')).strip()
    prop = str(fact.get('proposition', '')).strip()
    salience = float(fact.get('causal_salience_score', 0) or 0)
    
    # Only flag high-salience facts (top 20%) or critical issues
    is_high_salience = salience >= 0.7
    
    # Issue 1: Missing EventDate in high-salience facts
    eventdate = str(fact.get('eventdate', '')).strip()
    if is_high_salience and (not eventdate or eventdate.lower() in ('', 'nan', 'unknown', 'none')):
        # Check if date is mentioned in proposition
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
            r'\b(apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}',
            r'\d{4}-\d{2}-\d{2}',
        ]
        has_date_in_text = any(re.search(pattern, prop, re.IGNORECASE) for pattern in date_patterns)
        
        if has_date_in_text:
            issues.append({
                'type': 'date_extraction_high_priority',
                'priority': 'critical',
                'question': f"Fact {factid} (Salience: {salience:.2f}) mentions a date in the text but EventDate is missing. What is the exact date (YYYY-MM-DD format)?",
                'context': prop[:200],
                'factid': factid,
                'salience': salience,
            })
        else:
            issues.append({
                'type': 'missing_date_high_priority',
                'priority': 'high',
                'question': f"Fact {factid} (Salience: {salience:.2f}) has no EventDate. When did this event occur?",
                'context': prop[:200],
                'factid': factid,
                'salience': salience,
            })
    
    # Issue 2: Incomplete or truncated propositions in high-salience facts
    if is_high_salience:
        if len(prop) < 30:
            issues.append({
                'type': 'incomplete_proposition_high_priority',
                'priority': 'high',
                'question': f"Fact {factid} (Salience: {salience:.2f}) is very short ({len(prop)} chars). Can you expand this into a complete, factual proposition?",
                'context': prop,
                'factid': factid,
                'salience': salience,
            })
        elif prop.endswith('...') or prop.endswith('…'):
            issues.append({
                'type': 'truncated_proposition',
                'priority': 'high',
                'question': f"Fact {factid} (Salience: {salience:.2f}) appears truncated. Can you provide the complete proposition?",
                'context': prop,
                'factid': factid,
                'salience': salience,
            })
    
    # Issue 3: Ambiguous ActorRole/Subject in high-salience facts
    if is_high_salience:
        actorrole = str(fact.get('actorrole', '')).strip().lower()
        subject = str(fact.get('subject', '')).strip().lower()
        
        if actorrole in ('', 'nan', 'unknown', 'none') or subject in ('', 'nan', 'unknown', 'none'):
            issues.append({
                'type': 'missing_actor_subject',
                'priority': 'high',
                'question': f"Fact {factid} (Salience: {salience:.2f}) has missing or ambiguous ActorRole/Subject. Who is the primary actor and subject?",
                'context': prop[:200],
                'factid': factid,
                'salience': salience,
            })
    
    # Issue 4: Missing explicit causal connection in high-salience facts
    if is_high_salience and 'harvard' in prop.lower() and ('prc' in prop.lower() or 'china' in prop.lower()):
        # Check for explicit causal language
        explicit_causal = [
            'foreseeable', 'foreseeably', 'constructive knowledge', 'should have known',
            'knew or should have known', 'temporal overlap', 'caused', 'led to',
            'resulted in', 'therefore', 'thus', 'because', 'which means', 'this means',
            'demonstrates', 'shows', 'indicates', 'strengthens', 'increases risk',
            'exposes to', 'creates risk', 'makes', 'establishes'
        ]
        
        has_explicit = any(phrase in prop.lower() for phrase in explicit_causal)
        
        if not has_explicit:
            issues.append({
                'type': 'missing_explicit_causal_connection',
                'priority': 'critical',
                'question': f"Fact {factid} (Salience: {salience:.2f}) mentions both Harvard and PRC/China but lacks explicit causal language. Can you add explicit causal connection language (e.g., 'foreseeably', 'should have known', 'led to', 'resulted in')?",
                'context': prop[:200],
                'factid': factid,
                'salience': salience,
            })
    
    # Issue 5: Vague language in high-salience facts
    if is_high_salience:
        vague_patterns = [
            r'\b(some|several|many|various|certain)\b',
            r'\b(approximately|about|around|roughly)\b',
            r'\b(may|might|could|possibly|perhaps)\b',
        ]
        if any(re.search(pattern, prop, re.IGNORECASE) for pattern in vague_patterns):
            issues.append({
                'type': 'vague_language',
                'priority': 'medium',
                'question': f"Fact {factid} (Salience: {salience:.2f}) contains vague language. Can you make this more specific and factual?",
                'context': prop[:200],
                'factid': factid,
                'salience': salience,
            })
    
    return issues


def prioritize_questions(questions: list[dict]) -> list[dict]:
    """Prioritize questions by importance."""
    priority_order = {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}
    
    for q in questions:
        priority_score = priority_order.get(q.get('priority', 'low'), 0)
        salience = q.get('salience', 0)
        # Combined score: priority weight + salience
        q['combined_score'] = priority_score * 10 + salience
    
    return sorted(questions, key=lambda x: x.get('combined_score', 0), reverse=True)


def main():
    """Main execution."""
    print("="*80)
    print("GENERATING CRUCIAL QUESTIONS FOR V16 (UPDATED)")
    print("="*80)
    
    print(f"\n1. Loading v16 facts from: {V16_INPUT}")
    facts = load_facts()
    print(f"   Loaded {len(facts)} facts")
    
    print("\n2. Analyzing for crucial issues...")
    all_questions = []
    for fact in facts:
        issues = detect_crucial_issues(fact)
        all_questions.extend(issues)
    
    print(f"   Found {len(all_questions)} crucial questions")
    
    print("\n3. Prioritizing questions...")
    prioritized = prioritize_questions(all_questions)
    
    # Group by type
    by_type = defaultdict(list)
    for q in prioritized:
        by_type[q['type']].append(q)
    
    print("\n4. Question breakdown by type:")
    for qtype, qlist in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"   {qtype}: {len(qlist)}")
    
    # Get top 20 most crucial
    top_20 = prioritized[:20]
    
    print("\n5. Exporting crucial questions CSV...")
    if prioritized:
        fieldnames = ['factid', 'type', 'priority', 'question', 'context', 'salience', 'combined_score']
        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for q in prioritized:
                writer.writerow({
                    'factid': q.get('factid', ''),
                    'type': q.get('type', ''),
                    'priority': q.get('priority', ''),
                    'question': q.get('question', ''),
                    'context': q.get('context', ''),
                    'salience': q.get('salience', 0),
                    'combined_score': q.get('combined_score', 0),
                })
        print(f"   ✅ Exported {OUTPUT_CSV}")
    
    print("\n6. Exporting crucial questions TXT...")
    lines = []
    lines.append("="*80)
    lines.append("CRUCIAL CLARIFICATION QUESTIONS - V16 (UPDATED)")
    lines.append("="*80)
    lines.append("")
    lines.append(f"Total questions: {len(prioritized)}")
    lines.append(f"Top 20 most crucial questions shown below")
    lines.append("")
    lines.append("="*80)
    lines.append("")
    
    lines.append("## TOP 20 MOST CRUCIAL QUESTIONS")
    lines.append("")
    for idx, q in enumerate(top_20, 1):
        lines.append(f"### Question #{idx} (Priority: {q['priority'].upper()}, Salience: {q['salience']:.2f})")
        lines.append(f"**FactID**: {q['factid']}")
        lines.append(f"**Type**: {q['type']}")
        lines.append("")
        lines.append(f"**Question**:")
        lines.append(q['question'])
        lines.append("")
        lines.append(f"**Context**:")
        lines.append(q['context'])
        lines.append("")
        lines.append("-"*80)
        lines.append("")
    
    if len(prioritized) > 20:
        lines.append(f"\n... and {len(prioritized) - 20} more questions (see CSV for full list)")
    
    OUTPUT_TXT.write_text("\n".join(lines), encoding='utf-8')
    print(f"   ✅ Exported {OUTPUT_TXT}")
    
    print("\n7. Writing report...")
    report = f"""# Crucial Questions Report (V16 Updated)

## Summary

- **Input**: `{V16_INPUT.name}`
- **Total facts analyzed**: {len(facts)}
- **Total crucial questions generated**: {len(prioritized)}

## Priority Breakdown

"""
    
    by_priority = defaultdict(int)
    for q in prioritized:
        by_priority[q['priority']] += 1
    
    for priority in ['critical', 'high', 'medium', 'low']:
        count = by_priority.get(priority, 0)
        if count > 0:
            report += f"- **{priority.upper()}**: {count}\n"
    
    report += f"""
## Issue Type Breakdown

"""
    for qtype, qlist in sorted(by_type.items(), key=lambda x: -len(x[1])):
        report += f"- **{qtype}**: {len(qlist)}\n"
    
    report += f"""
## Top 10 Most Crucial Questions

"""
    for idx, q in enumerate(prioritized[:10], 1):
        report += f"{idx}. **FactID {q['factid']}** ({q['type']}, Priority: {q['priority']}, Salience: {q['salience']:.2f})\n"
        report += f"   Q: {q['question']}\n\n"
    
    report += f"""
## Files Generated

- **CSV**: `crucial_questions_v16.csv` (all {len(prioritized)} questions)
- **TXT**: `crucial_questions_v16.txt` (top 20 questions)

## Next Steps

1. Review the top 20 questions in the TXT file
2. Answer the critical and high-priority questions first
3. Use the fact answers script to apply your answers:
   ```bash
   python scripts/apply_fact_answers.py --input answers.txt --dataset v16
   ```
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')
    print(f"   ✅ Written {REPORT_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE! Crucial questions generated for v16")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  - Total questions: {len(prioritized)}")
    print(f"  - Critical: {by_priority.get('critical', 0)}")
    print(f"  - High: {by_priority.get('high', 0)}")
    print(f"  - Medium: {by_priority.get('medium', 0)}")
    print(f"\nFiles created:")
    print(f"  📄 {OUTPUT_CSV}")
    print(f"  📄 {OUTPUT_TXT}")
    print(f"  📄 {REPORT_PATH}")
    print(f"\n💡 Review the top 20 questions in: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()

