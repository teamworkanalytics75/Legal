#!/usr/bin/env python3
"""
Analyze top 1000 facts CSV for the issues ChatGPT identified.

Checks for:
1. Fragment facts (garbage rows)
2. Duplicates and near-duplicates
3. Missing critical negative facts (OGC non-response)
4. Missing publication/republication facts
5. Missing EsuWiki causation chain facts
6. Missing/incorrect Subject fields
7. Improper "Alleged" vs "True" labels
8. Safety_risk = "none" on high-risk facts
9. Public_exposure = "not_public" on public facts
10. Incorrect Actor Role classification
"""

import pandas as pd
import re
from pathlib import Path
from collections import Counter

def analyze_fragments(df):
    """Check for fragment facts (garbage rows)."""
    print("\n" + "="*80)
    print("1. FRAGMENT FACTS ANALYSIS")
    print("="*80)
    
    fragment_patterns = [
        r'^">\s*to\s*:\s*"',
        r'^"Notify you of my"',
        r'^"issued a pdf statement \("statement"',
        r'^"javier guzman <\["',
        r'^"Any such disclosure \."',
        r'^"The document refers to\s*$',  # Empty after prefix
    ]
    
    short_facts = df[df['Proposition'].str.len() < 30].copy()
    print(f"\nFacts under 30 characters: {len(short_facts)}")
    
    # Check if they have verbs or proper nouns
    verbs = ['is', 'was', 'were', 'has', 'have', 'alleges', 'claims', 'said', 'stated']
    proper_nouns = ['Harvard', 'Wang', 'Plaintiff', 'PRC', 'OGC', 'China', 'Beijing']
    
    valid_short = []
    invalid_short = []
    
    for idx, row in short_facts.iterrows():
        prop = str(row['Proposition']).lower()
        has_verb = any(v in prop for v in verbs)
        has_proper = any(pn.lower() in prop for pn in proper_nouns)
        
        if has_verb or has_proper:
            valid_short.append(idx)
        else:
            invalid_short.append(idx)
    
    print(f"  Valid short facts (have verb/proper noun): {len(valid_short)}")
    print(f"  Invalid short facts (fragments): {len(invalid_short)}")
    
    if invalid_short:
        print(f"\n  Sample invalid fragments:")
        for idx in invalid_short[:10]:
            print(f"    - {df.loc[idx, 'Proposition'][:100]}")
    
    return invalid_short

def analyze_duplicates(df):
    """Check for exact and semantic duplicates."""
    print("\n" + "="*80)
    print("2. DUPLICATES ANALYSIS")
    print("="*80)
    
    # Exact duplicates
    exact_dupes = df[df.duplicated(subset=['Proposition'], keep=False)]
    print(f"\nExact duplicate propositions: {len(exact_dupes)} rows")
    print(f"  Unique duplicate propositions: {exact_dupes['Proposition'].nunique()}")
    
    # Show most common duplicates
    prop_counts = df['Proposition'].value_counts()
    duplicates = prop_counts[prop_counts > 1]
    if len(duplicates) > 0:
        print(f"\n  Most duplicated propositions:")
        for prop, count in duplicates.head(5).items():
            print(f"    [{count}x] {str(prop)[:100]}...")
    
    return exact_dupes

def check_critical_facts(df):
    """Check for missing critical negative facts."""
    print("\n" + "="*80)
    print("3. CRITICAL NEGATIVE FACTS CHECK")
    print("="*80)
    
    ogc_patterns = [
        r'OGC.*did not respond',
        r'OGC.*did not acknowledge',
        r'No reply.*OGC',
        r'OGC.*never responded',
        r'OGC.*failed to respond',
        r'non.?response',
        r'no response',
    ]
    
    found = []
    for pattern in ogc_patterns:
        matches = df[df['Proposition'].str.contains(pattern, case=False, na=False)]
        if len(matches) > 0:
            found.extend(matches.index.tolist())
    
    print(f"\nOGC non-response facts found: {len(set(found))}")
    if len(set(found)) == 0:
        print("  ❌ MISSING: No OGC non-response facts found!")
    else:
        print("  ✅ Found OGC non-response facts:")
        for idx in list(set(found))[:5]:
            print(f"    - {df.loc[idx, 'Proposition'][:150]}")
    
    return found

def check_publication_facts(df):
    """Check for publication/republication facts."""
    print("\n" + "="*80)
    print("4. PUBLICATION FACTS CHECK")
    print("="*80)
    
    keywords = {
        'Statement 1': ['statement 1', 'statement one'],
        'Statement 2': ['statement 2', 'statement two'],
        'Monkey Article': ['monkey', 'resume article'],
        'WeChat': ['wechat', 'weixin'],
        'Zhihu': ['zhihu'],
    }
    
    for key, patterns in keywords.items():
        matches = df[df['Proposition'].str.contains('|'.join(patterns), case=False, na=False)]
        print(f"\n{key} mentions: {len(matches)}")
        if len(matches) == 0:
            print(f"  ❌ MISSING: No {key} facts found!")
        else:
            print(f"  ✅ Found {len(matches)} facts mentioning {key}")

def check_subject_fields(df):
    """Check Subject field quality."""
    print("\n" + "="*80)
    print("6. SUBJECT FIELD ANALYSIS")
    print("="*80)
    
    empty_subjects = df[df['Subject'].isna() | (df['Subject'] == '')]
    print(f"\nEmpty Subject fields: {len(empty_subjects)} / {len(df)} ({len(empty_subjects)/len(df)*100:.1f}%)")
    
    if len(empty_subjects) > len(df) * 0.5:
        print("  ❌ PROBLEM: More than 50% of facts have empty Subject fields!")
    
    # Check for noise in Subject
    non_empty = df[df['Subject'].notna() & (df['Subject'] != '')]
    noise_patterns = ['yesHarvard', 'HarvardHarvard', 'Unknown', 'UnknownUnknown']
    noise_count = 0
    for idx, row in non_empty.iterrows():
        subj = str(row['Subject'])
        if any(pattern.lower() in subj.lower() for pattern in noise_patterns):
            noise_count += 1
    
    print(f"  Subject fields with noise: {noise_count}")
    
    return empty_subjects

def check_truth_labels(df):
    """Check "Alleged" vs "True" distribution."""
    print("\n" + "="*80)
    print("7. TRUTH STATUS ANALYSIS")
    print("="*80)
    
    truth_counts = df['TruthStatus'].value_counts()
    print(f"\nTruthStatus distribution:")
    for status, count in truth_counts.items():
        pct = count / len(df) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")
    
    true_count = truth_counts.get('True', 0)
    if true_count < len(df) * 0.1:
        print(f"\n  ❌ PROBLEM: Only {true_count} facts marked as 'True' (<10%)")
        print("    Many facts should be True (dates, emails, publications, OGC non-responses)")

def check_safety_risk(df):
    """Check SafetyRisk classification."""
    print("\n" + "="*80)
    print("8. SAFETY RISK ANALYSIS")
    print("="*80)
    
    risk_counts = df['SafetyRisk'].value_counts()
    print(f"\nSafetyRisk distribution:")
    for risk, count in risk_counts.items():
        pct = count / len(df) * 100
        print(f"  {risk}: {count} ({pct:.1f}%)")
    
    # Check high-risk keywords
    high_risk_keywords = ['torture', 'detention', 'arrest', 'surveillance', 'doxxing', 'harassment', 'persecution']
    high_risk_facts = df[df['Proposition'].str.contains('|'.join(high_risk_keywords), case=False, na=False)]
    
    none_on_high_risk = high_risk_facts[high_risk_facts['SafetyRisk'] == 'none']
    print(f"\nHigh-risk keyword facts: {len(high_risk_facts)}")
    print(f"  Facts with SafetyRisk='none' but high-risk keywords: {len(none_on_high_risk)}")
    
    if len(none_on_high_risk) > 0:
        print(f"  ❌ PROBLEM: {len(none_on_high_risk)} high-risk facts incorrectly labeled as 'none'")

def check_public_exposure(df):
    """Check PublicExposure classification."""
    print("\n" + "="*80)
    print("9. PUBLIC EXPOSURE ANALYSIS")
    print("="*80)
    
    exposure_counts = df['PublicExposure'].value_counts()
    print(f"\nPublicExposure distribution:")
    for exp, count in exposure_counts.items():
        pct = count / len(df) * 100
        print(f"  {exp}: {count} ({pct:.1f}%)")
    
    # Check public keywords
    public_keywords = ['statement 1', 'statement 2', 'wechat', 'zhihu', 'published', 'article', 'post']
    public_facts = df[df['Proposition'].str.contains('|'.join(public_keywords), case=False, na=False)]
    
    not_public_on_public = public_facts[public_facts['PublicExposure'] == 'not_public']
    print(f"\nPublic keyword facts: {len(public_facts)}")
    print(f"  Facts with PublicExposure='not_public' but public keywords: {len(not_public_on_public)}")
    
    if len(not_public_on_public) > len(public_facts) * 0.5:
        print(f"  ❌ PROBLEM: {len(not_public_on_public)} public facts incorrectly labeled as 'not_public'")

def check_actor_roles(df):
    """Check ActorRole classification."""
    print("\n" + "="*80)
    print("10. ACTOR ROLE ANALYSIS")
    print("="*80)
    
    role_counts = df['ActorRole'].value_counts()
    print(f"\nActorRole distribution:")
    for role, count in role_counts.items():
        pct = count / len(df) * 100
        print(f"  {role}: {count} ({pct:.1f}%)")
    
    # Check for common misclassifications
    harvard_clubs = df[df['Proposition'].str.contains('harvard.*club', case=False, na=False)]
    clubs_other = harvard_clubs[harvard_clubs['ActorRole'] == 'Other']
    print(f"\nHarvard Club mentions: {len(harvard_clubs)}")
    print(f"  Misclassified as 'Other': {len(clubs_other)}")

def main():
    csv_path = Path("case_law_data/top_1000_facts_for_chatgpt.csv")
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return
    
    print("="*80)
    print("ANALYZING TOP 1000 FACTS CSV")
    print("="*80)
    print(f"\nFile: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} facts")
    print(f"Columns: {list(df.columns)}")
    
    # Run all analyses
    invalid_fragments = analyze_fragments(df)
    duplicates = analyze_duplicates(df)
    ogc_facts = check_critical_facts(df)
    check_publication_facts(df)
    empty_subjects = check_subject_fields(df)
    check_truth_labels(df)
    check_safety_risk(df)
    check_public_exposure(df)
    check_actor_roles(df)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal facts: {len(df)}")
    print(f"Invalid fragments: {len(invalid_fragments)}")
    print(f"Duplicate rows: {len(duplicates)}")
    print(f"OGC non-response facts: {len(set(ogc_facts))}")
    print(f"Empty Subject fields: {len(empty_subjects)} ({len(empty_subjects)/len(df)*100:.1f}%)")
    
    print("\n✅ Analysis complete!")
    print(f"\nFile location: {csv_path.absolute()}")

if __name__ == "__main__":
    main()

