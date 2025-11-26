#!/usr/bin/env python3
"""
PDF-Based 1782 Case Analysis
Analyze the 13 downloaded 1782 PDFs for patterns and insights
"""

import json
import pandas as pd
from pathlib import Path
import re
import sys
import os

def load_pdf_cases():
    """Load the 13 PDF cases with their metadata."""

    # Load alignment data
    with open("data/case_law/json_pdf_alignment.json", 'r') as f:
        alignment_data = json.load(f)

    # Filter for successful PDF downloads
    pdf_cases = []
    for cluster_id, case_info in alignment_data.items():
        if case_info.get('download_status') == 'success' and case_info.get('pdf_path'):
            pdf_cases.append({
                'cluster_id': cluster_id,
                'case_name': case_info['case_name'],
                'pdf_path': case_info['pdf_path'],
                'json_path': case_info['json_path'],
                'file_size': case_info.get('file_size', 0)
            })

    return pd.DataFrame(pdf_cases)

def extract_pdf_text(pdf_path):
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")

        return '\n\n'.join(text_parts)

    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def analyze_pdf_texts(pdf_texts):
    """Run ML analysis on PDF texts."""

    analysis_results = {}

    for cluster_id, case_data in pdf_texts.items():
        print(f"Analyzing: {case_data['case_name']}")

        text = case_data['text']

        # Basic analysis
        analysis = {
            'cluster_id': cluster_id,
            'case_name': case_data['case_name'],
            'text_length': len(text),
            'word_count': len(text.split()),
            'page_count': text.count('--- Page'),
            'intel_factors': {},
            'outcome': {},
            'citations': [],
            'statutory_prereqs': {}
        }

        # Intel factor detection
        intel_patterns = {
            'factor_1': [r'foreign\s+participant', r'person\s+from\s+whom\s+discovery'],
            'factor_2': [r'receptiv(?:ity|e)', r'foreign\s+tribunal'],
            'factor_3': [r'circumvent(?:ion|ing)', r'evasion\s+of\s+foreign'],
            'factor_4': [r'unduly\s+intrusive', r'excessive\s+burden']
        }

        for factor, patterns in intel_patterns.items():
            factor_found = False
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    factor_found = True
                    break
            analysis['intel_factors'][factor] = factor_found

        # Outcome detection
        outcome_patterns = {
            'granted': [r'granted', r'permitted', r'allowed'],
            'denied': [r'denied', r'rejected', r'disallowed'],
            'partial': [r'partial', r'limited', r'restricted']
        }

        outcome_scores = {}
        for outcome, patterns in outcome_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            outcome_scores[outcome] = score

        # Determine outcome
        if outcome_scores['granted'] > outcome_scores['denied']:
            analysis['outcome'] = {'outcome': 'granted', 'confidence': 0.8}
        elif outcome_scores['denied'] > outcome_scores['granted']:
            analysis['outcome'] = {'outcome': 'denied', 'confidence': 0.8}
        else:
            analysis['outcome'] = {'outcome': 'unclear', 'confidence': 0.5}

        # Citation detection
        citation_patterns = [
            r'Intel\s+Corp',
            r'ZF\s+Automotive',
            r'AlixPartners',
            r'28\s+U\.S\.C\.\s+§\s*1782'
        ]

        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                analysis['citations'].append({
                    'pattern': pattern,
                    'count': len(matches),
                    'mentions': matches
                })

        analysis_results[cluster_id] = analysis

    return analysis_results

def generate_insights(analysis_results):
    """Generate insights from PDF analysis."""

    insights = {
        'total_cases': len(analysis_results),
        'outcome_distribution': {},
        'intel_factor_stats': {},
        'citation_stats': {},
        'text_stats': {},
        'success_patterns': []
    }

    # Outcome distribution
    for case_id, analysis in analysis_results.items():
        outcome = analysis['outcome']['outcome']
        insights['outcome_distribution'][outcome] = insights['outcome_distribution'].get(outcome, 0) + 1

    # Intel factor stats
    for factor in ['factor_1', 'factor_2', 'factor_3', 'factor_4']:
        detected = sum(1 for analysis in analysis_results.values()
                      if analysis['intel_factors'].get(factor, False))
        insights['intel_factor_stats'][factor] = {
            'detected': detected,
            'total': len(analysis_results),
            'detection_rate': detected / len(analysis_results)
        }

    # Citation stats
    all_citations = []
    for analysis in analysis_results.values():
        for citation in analysis['citations']:
            all_citations.append(citation['pattern'])

    citation_counts = {}
    for citation in all_citations:
        citation_counts[citation] = citation_counts.get(citation, 0) + 1

    insights['citation_stats'] = citation_counts

    # Text stats
    text_lengths = [analysis['text_length'] for analysis in analysis_results.values()]
    insights['text_stats'] = {
        'avg_length': sum(text_lengths) / len(text_lengths),
        'min_length': min(text_lengths),
        'max_length': max(text_lengths)
    }

    # Success patterns
    successful_cases = [analysis for analysis in analysis_results.values()
                       if analysis['outcome']['outcome'] == 'granted']

    if successful_cases:
        success_factors = {}
        for factor in ['factor_1', 'factor_2', 'factor_3', 'factor_4']:
            detected_in_success = sum(1 for case in successful_cases
                                    if case['intel_factors'].get(factor, False))
            success_factors[factor] = detected_in_success / len(successful_cases)

        insights['success_patterns'] = success_factors

    return insights

def run_pdf_analysis():
    """Complete PDF analysis script"""
    print("🚀 Starting PDF-based 1782 analysis...")
    print("="*60)

    # 1. Load PDF cases
    print("📄 Loading PDF cases...")
    pdf_cases_df = load_pdf_cases()
    print(f"✅ Loaded {len(pdf_cases_df)} PDF cases")

    for _, case in pdf_cases_df.iterrows():
        print(f"  - {case['case_name']}")

    # 2. Extract text
    print(f"\n📖 Extracting text from PDFs...")
    pdf_texts = {}
    for _, case in pdf_cases_df.iterrows():
        print(f"  Extracting: {case['case_name']}")
        text = extract_pdf_text(case['pdf_path'])
        if text:
            pdf_texts[case['cluster_id']] = {
                'case_name': case['case_name'],
                'text': text,
                'text_length': len(text),
                'pdf_path': case['pdf_path']
            }
            print(f"    ✅ Extracted {len(text)} characters")
        else:
            print(f"    ❌ Failed to extract text")

    print(f"\n✅ Successfully extracted text from {len(pdf_texts)} PDFs")

    # 3. Analyze
    print(f"\n🧠 Running ML analysis on PDF texts...")
    analysis_results = analyze_pdf_texts(pdf_texts)

    # 4. Generate insights
    print(f"\n📊 Generating insights...")
    insights = generate_insights(analysis_results)

    # 5. Save results
    output_file = "data/case_law/pdf_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'analysis_results': analysis_results,
            'insights': insights
        }, f, indent=2)

    print(f"\n💾 Results saved to {output_file}")

    # 6. Print summary
    print("\n" + "="*60)
    print("📈 1782 PDF ANALYSIS RESULTS")
    print("="*60)
    print(f"Total cases analyzed: {insights['total_cases']}")
    print(f"Average text length: {insights['text_stats']['avg_length']:.0f} characters")
    print(f"Text range: {insights['text_stats']['min_length']} - {insights['text_stats']['max_length']} characters")

    print("\n📊 Outcome Distribution:")
    for outcome, count in insights['outcome_distribution'].items():
        percentage = (count / insights['total_cases']) * 100
        print(f"  {outcome}: {count} cases ({percentage:.1f}%)")

    print("\n🔍 Intel Factor Detection Rates:")
    for factor, stats in insights['intel_factor_stats'].items():
        print(f"  {factor}: {stats['detected']}/{stats['total']} ({stats['detection_rate']:.1%})")

    print("\n📚 Citation Patterns:")
    for citation, count in insights['citation_stats'].items():
        print(f"  {citation}: {count} mentions")

    if insights['success_patterns']:
        print("\n🎯 Success Patterns:")
        for factor, rate in insights['success_patterns'].items():
            print(f"  {factor} in successful cases: {rate:.1%}")

    print("\n✅ Analysis complete!")
    return insights

if __name__ == "__main__":
    insights = run_pdf_analysis()
