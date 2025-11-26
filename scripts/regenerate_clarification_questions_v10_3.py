#!/usr/bin/env python3
"""Regenerate clarification questions based on v10.3, excluding already-answered facts."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

V10_3_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v10.3_final.csv")
OUTPUT_CSV = Path("case_law_data/clarification_questions_v10_3.csv")
OUTPUT_TXT = Path("case_law_data/clarification_questions_v10_3.txt")
REPORT_PATH = Path("reports/analysis_outputs/clarification_questions_v10_3_report.md")

# Facts that have already been answered (exclude from question generation)
ANSWERED_FACTIDS = {
    # Batch 1 answers
    'KGFACT_010', '815', '3911', '268', '885', '324', 'MISSING_0095', 
    'MISSING_0070', '2428', 'MISSING_0002_REWRITE_1', '1520', '798', 
    '2260', '816', '1543_S1', '291', '3321', '4056', '1365', '348', 
    '3569', 'MISSING_0067', 'MISSING_0080', '3780', '1543_S2',
    # OGC timeline corrections
    'MISSING_0084_A', 'MISSING_0084_B', 'MISSING_0083', 'MISSING_0065',
    'MISSING_0063', 'MISSING_0069', 'MISSING_0076', 'MISSING_0117',
    'OGC_001', 'OGC_002', 'OGC_003', 'OGC_004', 'OGC_005', 'OGC_006', 'OGC_007',
}


def load_facts() -> list[dict]:
    """Load v10.3 facts."""
    facts = []
    with open(V10_3_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    return facts


def detect_clarity_issues(row: dict) -> list[dict]:
    """Detect clarity issues in a fact and generate questions."""
    issues = []
    prop = str(row.get('proposition', '')).strip()
    factid = str(row.get('factid', '')).strip()
    
    # Skip already-answered facts
    if factid in ANSWERED_FACTIDS:
        return []
    
    # Issue 1: Missing or vague EventDate
    eventdate = str(row.get('eventdate', '')).strip()
    if not eventdate or eventdate.lower() in ('', 'nan', 'unknown', 'none'):
        # Check if date is mentioned in proposition
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',  # Month day
            r'\b(apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}',  # Abbreviated months
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD format
        ]
        has_date_in_text = any(re.search(pattern, prop, re.IGNORECASE) for pattern in date_patterns)
        
        if has_date_in_text:
            issues.append({
                'type': 'date_extraction',
                'priority': 'high',
                'question': f"Fact {factid} mentions a date in the text but EventDate is missing. What is the exact date (YYYY-MM-DD format)?",
                'context': prop[:200],
            })
        else:
            issues.append({
                'type': 'missing_date',
                'priority': 'medium',
                'question': f"Fact {factid} has no EventDate. When did this event occur? (If unknown, specify 'unknown' or approximate timeframe)",
                'context': prop[:200],
            })
    
    # Issue 2: Ambiguous ActorRole
    actorrole = str(row.get('actorrole', '')).strip().lower()
    prop_lower = prop.lower()
    
    if actorrole in ('', 'nan', 'unknown', 'none'):
        # Check if we can infer from proposition
        if 'ogc' in prop_lower or 'general counsel' in prop_lower:
            issues.append({
                'type': 'actor_specificity',
                'priority': 'high',
                'question': f"Fact {factid} mentions OGC/General Counsel but ActorRole is '{row.get('actorrole', '')}'. Should this be 'Harvard Office of General Counsel'?",
                'context': prop[:200],
            })
        elif 'harvard club' in prop_lower:
            issues.append({
                'type': 'actor_specificity',
                'priority': 'high',
                'question': f"Fact {factid} mentions Harvard Club but ActorRole is '{row.get('actorrole', '')}'. Which specific club? (Harvard Club of Hong Kong/Beijing/Shanghai)",
                'context': prop[:200],
            })
        elif 'wechat' in prop_lower or 'zhihu' in prop_lower or 'baidu' in prop_lower:
            issues.append({
                'type': 'actor_specificity',
                'priority': 'medium',
                'question': f"Fact {factid} mentions a platform (WeChat/Zhihu/Baidu) but ActorRole is '{row.get('actorrole', '')}'. Should this be 'Third-Party Publisher' or 'Platform'?",
                'context': prop[:200],
            })
    
    # Issue 3: Vague or incomplete proposition
    if len(prop) < 30:
        issues.append({
            'type': 'incomplete_proposition',
            'priority': 'high',
            'question': f"Fact {factid} is very short: '{prop}'. Can you provide more context or expand this into a complete factual statement?",
            'context': prop,
        })
    
    # Issue 4: Missing Subject
    subject = str(row.get('subject', '')).strip()
    if not subject or subject.lower() in ('', 'nan', 'unknown', 'none'):
        issues.append({
            'type': 'missing_subject',
            'priority': 'medium',
            'question': f"Fact {factid} has no Subject. Who or what is the primary subject/actor of this fact?",
            'context': prop[:200],
        })
    
    # Issue 5: Missing EventLocation
    eventlocation = str(row.get('eventlocation', '')).strip()
    if not eventlocation or eventlocation.lower() in ('', 'nan', 'unknown', 'none'):
        # Check if location is mentioned in proposition
        location_keywords = ['china', 'prc', 'hong kong', 'beijing', 'shanghai', 'usa', 'united states']
        has_location = any(keyword in prop_lower for keyword in location_keywords)
        
        if has_location:
            issues.append({
                'type': 'location_extraction',
                'priority': 'medium',
                'question': f"Fact {factid} mentions a location but EventLocation is missing. What is the specific location? (PRC, Hong Kong, USA, etc.)",
                'context': prop[:200],
            })
    
    # Issue 6: Vague language
    vague_patterns = [
        r'\b(some|many|several|various|certain|vague|unclear)\b',
        r'\b(about|approximately|around|roughly)\b',
        r'\b(etc\.|\.\.\.|and so on)\b',
    ]
    has_vague = any(re.search(pattern, prop, re.IGNORECASE) for pattern in vague_patterns)
    
    if has_vague and len(prop) > 50:  # Only flag if it's substantial enough to be vague
        issues.append({
            'type': 'vague_language',
            'priority': 'medium',
            'question': f"Fact {factid} contains vague language. Can you make this more specific and concrete?",
            'context': prop[:200],
        })
    
    # Issue 7: Missing causal connection
    if 'harvard' in prop_lower and ('prc' in prop_lower or 'china' in prop_lower or 'chinese' in prop_lower):
        # Check if causal connection is explicit
        causal_verbs = ['caused', 'led to', 'resulted in', 'triggered', 'provoked', 'preceded', 'followed', 'makes', 'creates', 'produces', 'foreseeably']
        has_causal_verb = any(verb in prop_lower for verb in causal_verbs)
        
        if not has_causal_verb:
            issues.append({
                'type': 'missing_causal_connection',
                'priority': 'high',
                'question': f"Fact {factid} mentions both Harvard and PRC/China but doesn't explicitly state the causal connection. How does Harvard's action relate to the PRC outcome?",
                'context': prop[:200],
            })
    
    # Issue 8: Incomplete OGC non-response facts (should be rare now, but check)
    if 'ogc' in prop_lower or 'general counsel' in prop_lower:
        if 'did not respond' in prop_lower or 'no response' in prop_lower or 'silence' in prop_lower:
            # Check if date is specified
            if not eventdate or eventdate.lower() in ('', 'nan', 'unknown', 'none'):
                issues.append({
                    'type': 'ogc_date_missing',
                    'priority': 'high',
                    'question': f"Fact {factid} is an OGC non-response fact but EventDate is missing. When did this non-response occur? (This is critical for the timeline)",
                    'context': prop[:200],
                })
    
    return issues


def calculate_priority_score(issue: dict, row: dict) -> float:
    """Calculate priority score for an issue."""
    base_scores = {
        'high': 5.0,
        'medium': 3.0,
        'low': 1.0,
    }
    
    score = base_scores.get(issue['priority'], 1.0)
    
    # Boost for high-risk facts
    safetyrisk = str(row.get('safetyrisk', '')).strip().lower()
    if safetyrisk in ('extreme', 'high'):
        score += 0.5
    
    # Boost for public exposure facts
    publicexposure = str(row.get('publicexposure', '')).strip().lower()
    if publicexposure == 'already_public':
        score += 0.3
    
    # Boost for causal salience
    causal_score = row.get('causal_salience_score', 0)
    if isinstance(causal_score, (int, float)) and causal_score > 0.8:
        score += 0.2
    
    return score


def generate_questions(facts: list[dict]) -> list[dict]:
    """Generate clarification questions for all facts."""
    all_questions = []
    
    for row in facts:
        issues = detect_clarity_issues(row)
        
        for issue in issues:
            priority_score = calculate_priority_score(issue, row)
            
            question_row = {
                'factid': row.get('factid', ''),
                'proposition': str(row.get('proposition', ''))[:200],
                'issue_type': issue['type'],
                'priority': issue['priority'],
                'priority_score': priority_score,
                'question': issue['question'],
                'context': issue['context'],
                'current_actorrole': row.get('actorrole', ''),
                'current_eventdate': row.get('eventdate', ''),
                'current_safetyrisk': row.get('safetyrisk', ''),
                'current_publicexposure': row.get('publicexposure', ''),
            }
            all_questions.append(question_row)
    
    # Sort by priority score
    all_questions.sort(key=lambda x: x['priority_score'], reverse=True)
    
    return all_questions


def export_questions(questions: list[dict]) -> None:
    """Export questions to CSV and TXT."""
    # Export CSV
    if questions:
        fieldnames = list(questions[0].keys())
        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(questions)
    
    # Export TXT
    lines = []
    lines.append("="*80)
    lines.append("CLARIFICATION QUESTIONS FOR FACT DATASET (v10.3)")
    lines.append("="*80)
    lines.append("")
    lines.append("Instructions: Answer each question to clarify the corresponding fact.")
    lines.append("Your answers will be used to improve fact accuracy and completeness.")
    lines.append("")
    lines.append(f"Total questions: {len(questions)}")
    lines.append("")
    lines.append("="*80)
    lines.append("")
    
    for idx, q in enumerate(questions, 1):
        factid = str(q.get('factid', '')).strip()
        proposition = str(q.get('proposition', '')).strip()
        issue_type = str(q.get('issue_type', '')).strip()
        priority = str(q.get('priority', '')).strip()
        question_text = str(q.get('question', '')).strip()
        context = str(q.get('context', '')).strip()
        actorrole = str(q.get('current_actorrole', '')).strip()
        eventdate = str(q.get('current_eventdate', '')).strip()
        safetyrisk = str(q.get('current_safetyrisk', '')).strip()
        publicexposure = str(q.get('current_publicexposure', '')).strip()
        
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
        lines.append(f"PROPOSITION: {proposition}")
        lines.append(f"PRIORITY: {priority.upper()}")
        lines.append(f"ISSUE TYPE: {issue_type}")
        lines.append("")
        lines.append(f"Q: {question_text}")
        lines.append(f"   Context: {context}")
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


def write_report(questions: list[dict], total_facts: int) -> None:
    """Write report of generated questions."""
    high_priority = [q for q in questions if q.get('priority') == 'high']
    medium_priority = [q for q in questions if q.get('priority') == 'medium']
    low_priority = [q for q in questions if q.get('priority') == 'low']
    
    # Count by issue type
    issue_types = {}
    for q in questions:
        it = q.get('issue_type', '')
        issue_types[it] = issue_types.get(it, 0) + 1
    
    report = f"""# Clarification Questions Report (v10.3)

## Summary

- **Input**: `top_1000_facts_for_chatgpt_v10.3_final.csv`
- **Total facts analyzed**: {total_facts}
- **Facts already answered**: {len(ANSWERED_FACTIDS)}
- **Total questions generated**: {len(questions)}

## Priority Breakdown

- **High priority**: {len(high_priority)}
- **Medium priority**: {len(medium_priority)}
- **Low priority**: {len(low_priority)}

## Issue Type Breakdown

"""
    
    for issue_type, count in sorted(issue_types.items(), key=lambda x: -x[1]):
        report += f"- **{issue_type}**: {count}\n"
    
    report += f"""
## Top 20 Highest Priority Questions

"""
    
    for idx, q in enumerate(questions[:20], 1):
        report += f"{idx}. **{q.get('factid')}** - {q.get('issue_type')} (Priority: {q.get('priority')}, Score: {q.get('priority_score', 0):.2f})\n"
        report += f"   Q: {q.get('question', '')[:100]}...\n\n"
    
    report += f"""
## Files Generated

- **CSV**: `{OUTPUT_CSV.name}`
- **TXT**: `{OUTPUT_TXT.name}`

## Next Steps

1. Review the top priority questions
2. Answer them comprehensively (ChatGPT's long answers work great!)
3. Apply answers to create v10.4
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')


def main():
    """Main execution."""
    print("="*80)
    print("REGENERATING CLARIFICATION QUESTIONS (v10.3)")
    print("="*80)
    
    print("\n1. Loading v10.3 facts...")
    facts = load_facts()
    print(f"   Loaded {len(facts)} facts")
    print(f"   Excluding {len(ANSWERED_FACTIDS)} already-answered facts")
    
    print("\n2. Analyzing facts for clarity issues...")
    questions = generate_questions(facts)
    print(f"   Generated {len(questions)} clarification questions")
    
    print("\n3. Priority breakdown:")
    high = len([q for q in questions if q.get('priority') == 'high'])
    medium = len([q for q in questions if q.get('priority') == 'medium'])
    low = len([q for q in questions if q.get('priority') == 'low'])
    print(f"   High priority: {high}")
    print(f"   Medium priority: {medium}")
    print(f"   Low priority: {low}")
    
    print("\n4. Issue type breakdown:")
    issue_types = {}
    for q in questions:
        it = q.get('issue_type', '')
        issue_types[it] = issue_types.get(it, 0) + 1
    for issue_type, count in sorted(issue_types.items(), key=lambda x: -x[1])[:10]:
        print(f"   {issue_type}: {count}")
    
    print("\n5. Exporting questions...")
    export_questions(questions)
    print(f"   ✅ Exported {OUTPUT_CSV}")
    print(f"   ✅ Exported {OUTPUT_TXT}")
    
    print("\n6. Writing report...")
    write_report(questions, len(facts))
    print(f"   ✅ Written {REPORT_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE! Review questions in:")
    print(f"   {OUTPUT_TXT}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

