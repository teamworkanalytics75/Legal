#!/usr/bin/env python3
"""
Analyze Denial Reasons in 1782 Cases
"""

import json
import re

def analyze_denial_reasons():
    """Analyze the specific reasons for denial in 1782 cases."""

    # Load analysis results
    with open("data/case_law/simple_pdf_analysis.json", 'r') as f:
        results = json.load(f)

    # Get denied cases
    denied_cases = [r for r in results if r['outcome']['outcome'] == 'denied']

    print("🚫 DENIAL REASONS ANALYSIS")
    print("="*50)
    print(f"📊 Total Denied Cases: {len(denied_cases)}")

    # Analyze each denied case
    denial_patterns = {
        'intel_factors': {},
        'citation_patterns': {},
        'text_length_analysis': {},
        'specific_reasons': []
    }

    print(f"\n📋 DENIED CASES BREAKDOWN:")
    print("-" * 50)

    for i, case in enumerate(denied_cases, 1):
        print(f"\n{i}. {case['case_name']}")
        print(f"   Text Length: {case['text_length']:,} characters")
        print(f"   Intel Factors: {sum(case['intel_factors'].values())}/4")
        print(f"   Section 1782 Mentions: {case['section_1782_mentions']}")
        print(f"   Citations: {len(case['citations'])}")

        # Analyze Intel factors in denied cases
        for factor, present in case['intel_factors'].items():
            if factor not in denial_patterns['intel_factors']:
                denial_patterns['intel_factors'][factor] = {'present': 0, 'absent': 0}

            if present:
                denial_patterns['intel_factors'][factor]['present'] += 1
            else:
                denial_patterns['intel_factors'][factor]['absent'] += 1

    # Intel Factor Analysis for Denied Cases
    print(f"\n🔍 INTEL FACTORS IN DENIED CASES:")
    print("-" * 50)

    for factor, stats in denial_patterns['intel_factors'].items():
        total = stats['present'] + stats['absent']
        present_rate = stats['present'] / total if total > 0 else 0
        absent_rate = stats['absent'] / total if total > 0 else 0

        print(f"{factor}:")
        print(f"  Present: {stats['present']}/{total} ({present_rate:.1%})")
        print(f"  Absent: {stats['absent']}/{total} ({absent_rate:.1%})")

        # Identify patterns
        if absent_rate > 0.5:
            print(f"  ⚠️  HIGH ABSENCE RATE - This factor missing in {absent_rate:.1%} of denied cases")

    # Compare with successful cases
    successful_cases = [r for r in results if r['outcome']['outcome'] == 'granted']

    print(f"\n📊 DENIED vs SUCCESSFUL COMPARISON:")
    print("-" * 50)

    for factor in ['factor_1', 'factor_2', 'factor_3', 'factor_4']:
        denied_present = sum(1 for r in denied_cases if r['intel_factors'].get(factor, False))
        successful_present = sum(1 for r in successful_cases if r['intel_factors'].get(factor, False))

        denied_rate = denied_present / len(denied_cases) if denied_cases else 0
        successful_rate = successful_present / len(successful_cases) if successful_cases else 0

        print(f"{factor}:")
        print(f"  Denied Cases: {denied_present}/{len(denied_cases)} ({denied_rate:.1%})")
        print(f"  Successful Cases: {successful_present}/{len(successful_cases)} ({successful_rate:.1%})")

        if successful_rate > denied_rate:
            print(f"  ✅ Factor more common in successful cases (+{successful_rate - denied_rate:.1%})")
        elif denied_rate > successful_rate:
            print(f"  ❌ Factor more common in denied cases (+{denied_rate - successful_rate:.1%})")

    # Text Length Analysis
    denied_lengths = [r['text_length'] for r in denied_cases]
    successful_lengths = [r['text_length'] for r in successful_cases]

    print(f"\n📄 TEXT LENGTH ANALYSIS:")
    print("-" * 50)
    print(f"Denied Cases Average: {sum(denied_lengths)/len(denied_lengths):,.0f} characters")
    print(f"Successful Cases Average: {sum(successful_lengths)/len(successful_lengths):,.0f} characters")

    if sum(successful_lengths)/len(successful_lengths) > sum(denied_lengths)/len(denied_lengths):
        print("✅ Successful cases tend to be longer (more comprehensive)")
    else:
        print("❌ Denied cases tend to be longer")

    # Citation Analysis
    denied_citations = []
    successful_citations = []

    for r in denied_cases:
        for citation in r['citations']:
            denied_citations.append(citation['pattern'])

    for r in successful_cases:
        for citation in r['citations']:
            successful_citations.append(citation['pattern'])

    print(f"\n📚 CITATION ANALYSIS:")
    print("-" * 50)
    print(f"Denied Cases Citations: {len(denied_citations)} total")
    print(f"Successful Cases Citations: {len(successful_citations)} total")

    # Section 1782 Mentions
    denied_1782_mentions = sum(r['section_1782_mentions'] for r in denied_cases)
    successful_1782_mentions = sum(r['section_1782_mentions'] for r in successful_cases)

    print(f"\n⚖️ SECTION 1782 MENTIONS:")
    print("-" * 50)
    print(f"Denied Cases: {denied_1782_mentions} total ({denied_1782_mentions/len(denied_cases):.1f} per case)")
    print(f"Successful Cases: {successful_1782_mentions} total ({successful_1782_mentions/len(successful_cases):.1f} per case)")

    # Key Insights
    print(f"\n💡 KEY DENIAL INSIGHTS:")
    print("-" * 50)

    # Factor 2 analysis
    factor_2_denied = sum(1 for r in denied_cases if r['intel_factors'].get('factor_2', False))
    factor_2_successful = sum(1 for r in successful_cases if r['intel_factors'].get('factor_2', False))

    print(f"1. Factor 2 (Receptivity):")
    print(f"   - Denied cases: {factor_2_denied}/{len(denied_cases)} ({factor_2_denied/len(denied_cases):.1%})")
    print(f"   - Successful cases: {factor_2_successful}/{len(successful_cases)} ({factor_2_successful/len(successful_cases):.1%})")

    if factor_2_successful/len(successful_cases) > factor_2_denied/len(denied_cases):
        print(f"   ✅ Factor 2 is MORE COMMON in successful cases")
    else:
        print(f"   ❌ Factor 2 is LESS COMMON in successful cases")

    # Factor 4 analysis
    factor_4_denied = sum(1 for r in denied_cases if r['intel_factors'].get('factor_4', False))
    factor_4_successful = sum(1 for r in successful_cases if r['intel_factors'].get('factor_4', False))

    print(f"\n2. Factor 4 (Unduly Intrusive):")
    print(f"   - Denied cases: {factor_4_denied}/{len(denied_cases)} ({factor_4_denied/len(denied_cases):.1%})")
    print(f"   - Successful cases: {factor_4_successful}/{len(successful_cases)} ({factor_4_successful/len(successful_cases):.1%})")

    if factor_4_denied/len(denied_cases) > factor_4_successful/len(successful_cases):
        print(f"   ⚠️  Factor 4 is MORE COMMON in denied cases - suggests burden concerns")
    else:
        print(f"   ✅ Factor 4 is LESS COMMON in denied cases")

    # Citation analysis
    intel_cited_denied = sum(1 for r in denied_cases if any(c['pattern'] == 'Intel\\s+Corp' for c in r['citations']))
    intel_cited_successful = sum(1 for r in successful_cases if any(c['pattern'] == 'Intel\\s+Corp' for c in r['citations']))

    print(f"\n3. Intel Corp Citation:")
    print(f"   - Denied cases: {intel_cited_denied}/{len(denied_cases)} ({intel_cited_denied/len(denied_cases):.1%})")
    print(f"   - Successful cases: {intel_cited_successful}/{len(successful_cases)} ({intel_cited_successful/len(successful_cases):.1%})")

    if intel_cited_successful/len(successful_cases) > intel_cited_denied/len(denied_cases):
        print(f"   ✅ Intel citation is MORE COMMON in successful cases")
    else:
        print(f"   ❌ Intel citation is LESS COMMON in successful cases")

    print(f"\n🎯 MAJOR DENIAL REASONS IDENTIFIED:")
    print("-" * 50)

    # Identify the most common patterns in denied cases
    reasons = []

    # Check for missing Factor 2
    if factor_2_denied/len(denied_cases) < factor_2_successful/len(successful_cases):
        reasons.append("Missing Factor 2 (Receptivity) analysis")

    # Check for Factor 4 issues
    if factor_4_denied/len(denied_cases) > factor_4_successful/len(successful_cases):
        reasons.append("Factor 4 (Burden) concerns not adequately addressed")

    # Check for missing Intel citation
    if intel_cited_denied/len(denied_cases) < intel_cited_successful/len(successful_cases):
        reasons.append("Missing or insufficient Intel Corp citation")

    # Check for shorter documents
    if sum(denied_lengths)/len(denied_lengths) < sum(successful_lengths)/len(successful_lengths):
        reasons.append("Insufficient analysis/documentation")

    for i, reason in enumerate(reasons, 1):
        print(f"{i}. {reason}")

    if not reasons:
        print("No clear patterns identified - need deeper text analysis")

if __name__ == "__main__":
    analyze_denial_reasons()
