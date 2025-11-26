#!/usr/bin/env python3
"""Generate clarification questions for unclear facts in the dataset."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

V10_1_INPUT = Path("case_law_data/top_100_facts_v10.1.csv")
OUTPUT_CSV = Path("case_law_data/clarification_questions.csv")
OUTPUT_TXT = Path("case_law_data/clarification_questions.txt")
REPORT_PATH = Path("reports/analysis_outputs/clarification_questions_report.md")


def load_facts() -> pd.DataFrame:
    """Load facts dataset."""
    if not V10_1_INPUT.exists():
        # Fallback to v10 if v10.1 doesn't exist
        fallback = Path("case_law_data/top_1000_facts_for_chatgpt_v10_final.csv")
        if fallback.exists():
            return pd.read_csv(fallback, encoding='utf-8', low_memory=False)
        raise FileNotFoundError(f"Input file not found: {V10_1_INPUT}")
    return pd.read_csv(V10_1_INPUT, encoding='utf-8', low_memory=False)


def detect_clarity_issues(row: pd.Series) -> list[dict]:
    """Detect clarity issues in a fact and generate questions."""
    issues = []
    prop = str(row.get('proposition', '')).strip()
    factid = str(row.get('factid', '')).strip()
    
    # Issue 1: Missing or vague EventDate
    eventdate = str(row.get('eventdate', '')).strip()
    if not eventdate or eventdate.lower() in ('', 'nan', 'unknown', 'none'):
        # Check if date is mentioned in proposition
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',  # Month day
            r'\b(apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}',  # Abbreviated months
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
    
    if actorrole in ('', 'nan', 'unknown', 'none') or actorrole == 'harvard':
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
    
    # Issue 5: Ambiguous EventLocation
    location = str(row.get('eventlocation', '')).strip()
    if not location or location.lower() in ('', 'nan', 'unknown', 'none'):
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
    
    # Issue 6: Unclear SafetyRisk classification
    safetyrisk = str(row.get('safetyrisk', '')).strip().lower()
    prop_lower = prop.lower()
    
    if safetyrisk in ('', 'nan', 'none'):
        # Check if it should have a risk classification
        risk_keywords = ['torture', 'arrest', 'detention', 'harm', 'threat', 'risk', 'danger', 'persecution', 'doxxing', 'esuwiki']
        has_risk_indicators = any(keyword in prop_lower for keyword in risk_keywords)
        
        if has_risk_indicators:
            issues.append({
                'type': 'missing_safety_risk',
                'priority': 'high',
                'question': f"Fact {factid} mentions risk indicators (torture/arrest/harm/etc.) but SafetyRisk is 'none'. What should the SafetyRisk be? (extreme/high/medium/none)",
                'context': prop[:200],
            })
    
    # Issue 7: Unclear PublicExposure
    publicexposure = str(row.get('publicexposure', '')).strip().lower()
    
    if publicexposure in ('', 'nan', 'unknown', 'none'):
        # Check if it's about publication
        pub_keywords = ['published', 'article', 'wechat', 'zhihu', 'baidu', 'sohu', 'circulated', 'shared', 'posted']
        has_pub_indicators = any(keyword in prop_lower for keyword in pub_keywords)
        
        if has_pub_indicators:
            issues.append({
                'type': 'missing_public_exposure',
                'priority': 'medium',
                'question': f"Fact {factid} mentions publication/sharing but PublicExposure is missing. Should this be 'already_public', 'partially_public', or 'not_public'?",
                'context': prop[:200],
            })
    
    # Issue 8: Ambiguous or vague language
    vague_patterns = [
        r'\b(some|certain|various|several|many|few)\b',
        r'\b(possibly|perhaps|maybe|might|could|may)\b',
        r'\b(around|approximately|about|roughly)\b',
        r'\b(related to|involving|concerning|regarding)\b.*\b(things|matters|issues|stuff)\b',
    ]
    
    for pattern in vague_patterns:
        if re.search(pattern, prop_lower):
            issues.append({
                'type': 'vague_language',
                'priority': 'medium',
                'question': f"Fact {factid} contains vague language. Can you make this more specific and concrete?",
                'context': prop[:200],
            })
            break
    
    # Issue 9: Missing causal connection
    if 'harvard' in prop_lower and ('prc' in prop_lower or 'china' in prop_lower):
        # Check if causal connection is explicit
        causal_verbs = ['caused', 'led to', 'resulted in', 'triggered', 'provoked', 'preceded', 'followed']
        has_causal_verb = any(verb in prop_lower for verb in causal_verbs)
        
        if not has_causal_verb:
            issues.append({
                'type': 'missing_causal_connection',
                'priority': 'high',
                'question': f"Fact {factid} mentions both Harvard and PRC/China but doesn't explicitly state the causal connection. How does Harvard's action relate to the PRC outcome?",
                'context': prop[:200],
            })
    
    # Issue 10: Incomplete OGC non-response facts
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


def prioritize_questions(issues: list[dict], row: pd.Series) -> list[dict]:
    """Prioritize questions based on fact importance and issue type."""
    priority_weights = {
        'high': 3,
        'medium': 2,
        'low': 1,
    }
    
    # Get fact salience score
    salience = float(row.get('new_salience_score', row.get('causal_salience_score', 0.0))) if pd.notna(row.get('new_salience_score', row.get('causal_salience_score', 0.0))) else 0.0
    
    for issue in issues:
        priority_weight = priority_weights.get(issue['priority'], 1)
        issue['priority_score'] = priority_weight * (1.0 + salience)
    
    # Sort by priority score
    issues.sort(key=lambda x: x['priority_score'], reverse=True)
    
    return issues


def generate_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Generate clarification questions for all facts."""
    all_questions = []
    
    for idx, row in df.iterrows():
        issues = detect_clarity_issues(row)
        prioritized = prioritize_questions(issues, row)
        
        for issue in prioritized:
            question_row = {
                'factid': row.get('factid', ''),
                'proposition': str(row.get('proposition', ''))[:200],
                'issue_type': issue['type'],
                'priority': issue['priority'],
                'priority_score': issue['priority_score'],
                'question': issue['question'],
                'context': issue['context'],
                'current_actorrole': row.get('actorrole', ''),
                'current_eventdate': row.get('eventdate', ''),
                'current_safetyrisk': row.get('safetyrisk', ''),
                'current_publicexposure': row.get('publicexposure', ''),
            }
            all_questions.append(question_row)
    
    questions_df = pd.DataFrame(all_questions)
    
    # Sort by priority score
    questions_df = questions_df.sort_values('priority_score', ascending=False)
    
    return questions_df


def export_questions(questions_df: pd.DataFrame) -> None:
    """Export questions to CSV and TXT."""
    # Export CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    questions_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    # Export TXT (human-readable format)
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CLARIFICATION QUESTIONS FOR FACT DATASET\n")
        f.write("="*80 + "\n\n")
        f.write("Instructions: Answer each question to clarify the corresponding fact.\n")
        f.write("Your answers will be used to improve fact accuracy and completeness.\n\n")
        f.write("="*80 + "\n\n")
        
        current_factid = None
        for _, row in questions_df.iterrows():
            factid = str(row['factid'])
            
            # Group questions by fact
            if factid != current_factid:
                if current_factid is not None:
                    f.write("\n" + "-"*80 + "\n\n")
                current_factid = factid
                f.write(f"FACT ID: {factid}\n")
                f.write(f"PROPOSITION: {row['proposition']}\n")
                f.write(f"PRIORITY: {row['priority'].upper()}\n")
                f.write(f"ISSUE TYPE: {row['issue_type']}\n\n")
            
            f.write(f"Q: {row['question']}\n")
            f.write(f"   Context: {row['context']}\n")
            if row['current_actorrole']:
                f.write(f"   Current ActorRole: {row['current_actorrole']}\n")
            if row['current_eventdate']:
                f.write(f"   Current EventDate: {row['current_eventdate']}\n")
            if row['current_safetyrisk']:
                f.write(f"   Current SafetyRisk: {row['current_safetyrisk']}\n")
            if row['current_publicexposure']:
                f.write(f"   Current PublicExposure: {row['current_publicexposure']}\n")
            f.write(f"\n   A: [Your answer here]\n\n")


def write_report(questions_df: pd.DataFrame) -> None:
    """Write summary report."""
    total_questions = len(questions_df)
    high_priority = len(questions_df[questions_df['priority'] == 'high'])
    medium_priority = len(questions_df[questions_df['priority'] == 'medium'])
    low_priority = len(questions_df[questions_df['priority'] == 'low'])
    
    issue_type_counts = questions_df['issue_type'].value_counts().to_dict()
    
    report = f"""# Clarification Questions Report

## Summary

- **Total questions generated**: {total_questions}
- **High priority**: {high_priority}
- **Medium priority**: {medium_priority}
- **Low priority**: {low_priority}

## Issue Type Distribution

"""
    
    for issue_type, count in sorted(issue_type_counts.items(), key=lambda x: x[1], reverse=True):
        report += f"- **{issue_type.replace('_', ' ').title()}**: {count} questions\n"
    
    report += f"""
## Top 10 Highest Priority Questions

"""
    
    for idx, row in questions_df.head(10).iterrows():
        report += f"### {row['factid']} - {row['issue_type'].replace('_', ' ').title()}\n"
        report += f"**Priority**: {row['priority'].upper()}\n\n"
        report += f"**Question**: {row['question']}\n\n"
        report += f"**Context**: {row['context']}\n\n"
        report += "---\n\n"
    
    report += f"""
## Files Created

- `{OUTPUT_CSV.name}` - All questions in CSV format
- `{OUTPUT_TXT.name}` - Questions in human-readable format for answering

## Next Steps

1. Review the questions in `{OUTPUT_TXT.name}`
2. Answer each question with specific, factual information
3. Use your answers to update the fact dataset
4. Re-run fact extraction/validation with clarified information
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')


def main():
    """Main execution."""
    print("="*80)
    print("GENERATING CLARIFICATION QUESTIONS")
    print("="*80)
    
    print("\n1. Loading facts...")
    df = load_facts()
    print(f"   Loaded {len(df)} facts")
    
    print("\n2. Analyzing facts for clarity issues...")
    questions_df = generate_questions(df)
    print(f"   Generated {len(questions_df)} clarification questions")
    
    print("\n3. Priority breakdown:")
    print(f"   High priority: {len(questions_df[questions_df['priority'] == 'high'])}")
    print(f"   Medium priority: {len(questions_df[questions_df['priority'] == 'medium'])}")
    print(f"   Low priority: {len(questions_df[questions_df['priority'] == 'low'])}")
    
    print("\n4. Issue type breakdown:")
    for issue_type, count in questions_df['issue_type'].value_counts().head(10).items():
        print(f"   {issue_type}: {count}")
    
    print("\n5. Exporting questions...")
    export_questions(questions_df)
    print(f"   ✅ Exported {OUTPUT_CSV}")
    print(f"   ✅ Exported {OUTPUT_TXT}")
    
    print("\n6. Writing report...")
    write_report(questions_df)
    print(f"   ✅ Written {REPORT_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE! Review questions in:")
    print(f"   {OUTPUT_TXT}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

