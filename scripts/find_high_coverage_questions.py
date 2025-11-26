#!/usr/bin/env python3
"""Find questions with highest coverage - answering them helps answer many related questions."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

INPUT_CSV = Path("case_law_data/clarification_questions_batch_2.csv")
FULL_CSV = Path("case_law_data/clarification_questions.csv")
OUTPUT_TXT = Path("case_law_data/top_10_high_coverage_questions.txt")
OUTPUT_CSV = Path("case_law_data/top_10_high_coverage_questions.csv")


def extract_topics(proposition: str, factid: str) -> set[str]:
    """Extract topic keywords from proposition and factid."""
    topics = set()
    text = (proposition + " " + factid).lower()
    
    # Key topic patterns
    topic_patterns = {
        'ogc': ['ogc', 'general counsel', 'office of general counsel'],
        'esuwiki': ['esuwiki', 'esu wiki', '恶俗维基'],
        'statement_1': ['statement 1', 'statement one'],
        'statement_2': ['statement 2', 'statement two'],
        'xi_mingze': ['xi mingze', 'xi mingze', 'mingze'],
        'harvard_club': ['harvard club', 'harvard clubs'],
        'prc_risk': ['prc', 'china', 'chinese', 'travel risk', 'safety risk'],
        'april_2019': ['april 2019', '2019-04'],
        'april_2025': ['april 2025', '2025-04'],
        'gss': ['gss', 'global support services'],
        'hong_kong': ['hong kong', 'hkcoc'],
        'wechat': ['wechat', 'wechat'],
        'resume': ['resume', 'résumé', 'credentials'],
        'torture': ['torture', 'detention'],
        'silence': ['silence', 'no response', 'non-response'],
    }
    
    for topic, patterns in topic_patterns.items():
        if any(pattern in text for pattern in patterns):
            topics.add(topic)
    
    return topics


def calculate_coverage_score(question_row: dict, all_questions: list[dict]) -> tuple[int, list[str]]:
    """Calculate how many other questions this question helps answer."""
    coverage_count = 0
    covered_factids = []
    
    q_factid = str(question_row.get('factid', '')).strip()
    q_proposition = str(question_row.get('proposition', '')).strip().lower()
    q_issue_type = str(question_row.get('issue_type', '')).strip()
    q_topics = extract_topics(q_proposition, q_factid)
    
    # Count related questions
    for other_q in all_questions:
        other_factid = str(other_q.get('factid', '')).strip()
        other_proposition = str(other_q.get('proposition', '')).strip().lower()
        other_topics = extract_topics(other_proposition, other_factid)
        
        # Skip self
        if other_factid == q_factid and str(other_q.get('issue_type', '')).strip() == q_issue_type:
            continue
        
        # Same factid = definitely related
        if other_factid == q_factid:
            coverage_count += 1
            if other_factid not in covered_factids:
                covered_factids.append(other_factid)
            continue
        
        # Overlapping topics = likely related
        if q_topics & other_topics:
            # Check for specific relationships
            is_related = False
            
            # Date-related: if one asks for date, answering helps with other date questions in same timeframe
            if q_issue_type == 'date_extraction' and 'date' in str(other_q.get('issue_type', '')).lower():
                # Check if same time period mentioned
                if any(period in q_proposition and period in other_proposition 
                       for period in ['2019', '2020', '2025', '2018', 'april', 'may', 'june']):
                    is_related = True
            
            # OGC questions are all related
            if 'ogc' in q_topics and 'ogc' in other_topics:
                is_related = True
            
            # EsuWiki questions are all related
            if 'esuwiki' in q_topics and 'esuwiki' in other_topics:
                is_related = True
            
            # Statement 1/2 questions are related
            if ('statement_1' in q_topics and 'statement_1' in other_topics) or \
               ('statement_2' in q_topics and 'statement_2' in other_topics):
                is_related = True
            
            # Xi Mingze questions are related
            if 'xi_mingze' in q_topics and 'xi_mingze' in other_topics:
                is_related = True
            
            # PRC risk questions are related
            if 'prc_risk' in q_topics and 'prc_risk' in other_topics:
                is_related = True
            
            if is_related:
                coverage_count += 1
                if other_factid not in covered_factids:
                    covered_factids.append(other_factid)
    
    return coverage_count, covered_factids


def main():
    """Find top 10 high-coverage questions."""
    print("="*80)
    print("FINDING HIGH-COVERAGE QUESTIONS")
    print("="*80)
    
    # Load all questions (batch 2 + already answered to get full context)
    all_questions = []
    for csv_file in [FULL_CSV, INPUT_CSV]:
        if csv_file.exists():
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Deduplicate
                    key = (str(row.get('factid', '')).strip(), str(row.get('issue_type', '')).strip())
                    if not any((str(q.get('factid', '')).strip(), str(q.get('issue_type', '')).strip()) == key 
                              for q in all_questions):
                        all_questions.append(row)
    
    print(f"\nTotal questions loaded: {len(all_questions)}")
    
    # Load batch 2 questions (the ones we want to rank)
    batch_2_questions = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_2_questions.append(row)
    
    print(f"Batch 2 questions to analyze: {len(batch_2_questions)}")
    
    # Calculate coverage for each question
    print("\nCalculating coverage scores...")
    question_scores = []
    
    for q in batch_2_questions:
        coverage_count, covered_factids = calculate_coverage_score(q, all_questions)
        priority_score = float(q.get('priority_score', 0))
        
        # Combined score: coverage * 2 + priority (coverage is more important)
        combined_score = (coverage_count * 2) + priority_score
        
        question_scores.append({
            'question': q,
            'coverage_count': coverage_count,
            'covered_factids': covered_factids,
            'priority_score': priority_score,
            'combined_score': combined_score,
        })
    
    # Sort by combined score
    question_scores.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Deduplicate by factid - keep the question with highest coverage for each fact
    seen_factids = {}
    deduplicated = []
    for item in question_scores:
        factid = str(item['question'].get('factid', '')).strip()
        if factid not in seen_factids or item['coverage_count'] > seen_factids[factid]['coverage_count']:
            seen_factids[factid] = item
    
    deduplicated = list(seen_factids.values())
    deduplicated.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Get top 10
    top_10 = deduplicated[:10]
    
    print(f"\nTop 10 high-coverage questions:")
    print("-"*80)
    
    # Write output
    lines = []
    lines.append("="*80)
    lines.append("TOP 10 HIGH-COVERAGE QUESTIONS")
    lines.append("="*80)
    lines.append("")
    lines.append("These questions, when answered comprehensively, will help answer")
    lines.append("the greatest number of related or sub-questions.")
    lines.append("")
    lines.append("="*80)
    lines.append("")
    
    top_10_data = []
    
    for idx, item in enumerate(top_10, 1):
        q = item['question']
        factid = str(q.get('factid', '')).strip()
        proposition = str(q.get('proposition', '')).strip()
        issue_type = str(q.get('issue_type', '')).strip()
        priority = str(q.get('priority', '')).strip()
        question_text = str(q.get('question', '')).strip()
        context = str(q.get('context', '')).strip()
        
        coverage = item['coverage_count']
        covered = item['covered_factids']
        
        lines.append(f"#{idx} - COVERAGE SCORE: {coverage} related questions")
        lines.append(f"FACT ID: {factid}")
        lines.append(f"ISSUE TYPE: {issue_type}")
        lines.append(f"PRIORITY: {priority.upper()}")
        lines.append("")
        lines.append(f"PROPOSITION: {proposition[:300]}{'...' if len(proposition) > 300 else ''}")
        lines.append("")
        lines.append(f"Q: {question_text}")
        lines.append(f"   Context: {context[:200]}{'...' if len(context) > 200 else ''}")
        lines.append("")
        lines.append(f"   📊 This question helps answer {coverage} related questions")
        if covered:
            lines.append(f"   📋 Related FactIDs: {', '.join(covered[:10])}{'...' if len(covered) > 10 else ''}")
        lines.append("")
        lines.append("   A: [Your comprehensive answer here - ChatGPT's long answers work great!]")
        lines.append("")
        lines.append("-"*80)
        lines.append("")
        
        top_10_data.append({
            'rank': idx,
            'factid': factid,
            'issue_type': issue_type,
            'priority': priority,
            'coverage_count': coverage,
            'question': question_text,
            'proposition': proposition[:200],
        })
    
    OUTPUT_TXT.write_text("\n".join(lines), encoding='utf-8')
    print(f"✅ Text file written: {OUTPUT_TXT}")
    
    # Write CSV
    if top_10_data:
        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=top_10_data[0].keys())
            writer.writeheader()
            writer.writerows(top_10_data)
        print(f"✅ CSV written: {OUTPUT_CSV}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("TOP 10 SUMMARY")
    print(f"{'='*80}")
    for idx, item in enumerate(top_10, 1):
        q = item['question']
        factid = str(q.get('factid', '')).strip()
        coverage = item['coverage_count']
        print(f"{idx:2d}. {factid:30s} - Coverage: {coverage:2d} questions")
    print(f"{'='*80}")
    print(f"\n✅ Files ready:")
    print(f"   📄 Text: {OUTPUT_TXT}")
    print(f"   📊 CSV: {OUTPUT_CSV}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

