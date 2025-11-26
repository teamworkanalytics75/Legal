#!/usr/bin/env python3
"""
Simple PDF Analysis - Process one PDF at a time
"""

import json
import pandas as pd
from pathlib import Path
import re

def load_pdf_cases():
    """Load the 13 PDF cases with their metadata."""
    with open("data/case_law/json_pdf_alignment.json", 'r') as f:
        alignment_data = json.load(f)

    pdf_cases = []
    for cluster_id, case_info in alignment_data.items():
        if case_info.get('download_status') == 'success' and case_info.get('pdf_path'):
            pdf_cases.append({
                'cluster_id': cluster_id,
                'case_name': case_info['case_name'],
                'pdf_path': case_info['pdf_path'],
                'file_size': case_info.get('file_size', 0)
            })

    return pdf_cases

def analyze_single_pdf(pdf_path, case_name):
    """Analyze a single PDF file."""
    try:
        import pdfplumber

        print(f"Processing: {case_name}")

        # Extract text
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        full_text = '\n'.join(text_parts)

        if len(full_text) < 100:
            print(f"  ❌ Text too short: {len(full_text)} chars")
            return None

        print(f"  ✅ Extracted {len(full_text)} characters")

        # Basic analysis
        analysis = {
            'case_name': case_name,
            'text_length': len(full_text),
            'word_count': len(full_text.split()),
            'page_count': len(text_parts),
            'intel_factors': {},
            'outcome': {},
            'citations': [],
            'section_1782_mentions': 0
        }

        # Count 1782 mentions
        section_1782_patterns = [
            r'28\s+U\.S\.C\.\s+§\s*1782',
            r'28\s+USC\s+1782',
            r'section\s+1782',
            r'§\s*1782'
        ]

        for pattern in section_1782_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            analysis['section_1782_mentions'] += len(matches)

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
                if re.search(pattern, full_text, re.IGNORECASE):
                    factor_found = True
                    break
            analysis['intel_factors'][factor] = factor_found

        # Outcome detection
        granted_patterns = [r'granted', r'permitted', r'allowed', r'approved']
        denied_patterns = [r'denied', r'rejected', r'disallowed', r'dismissed']

        granted_count = sum(len(re.findall(pattern, full_text, re.IGNORECASE)) for pattern in granted_patterns)
        denied_count = sum(len(re.findall(pattern, full_text, re.IGNORECASE)) for pattern in denied_patterns)

        if granted_count > denied_count:
            analysis['outcome'] = {'outcome': 'granted', 'confidence': 0.8, 'granted_count': granted_count, 'denied_count': denied_count}
        elif denied_count > granted_count:
            analysis['outcome'] = {'outcome': 'denied', 'confidence': 0.8, 'granted_count': granted_count, 'denied_count': denied_count}
        else:
            analysis['outcome'] = {'outcome': 'unclear', 'confidence': 0.5, 'granted_count': granted_count, 'denied_count': denied_count}

        # Citation detection
        citation_patterns = [
            r'Intel\s+Corp',
            r'ZF\s+Automotive',
            r'AlixPartners',
            r'Euromepa'
        ]

        for pattern in citation_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                analysis['citations'].append({
                    'pattern': pattern,
                    'count': len(matches)
                })

        return analysis

    except Exception as e:
        print(f"  ❌ Error processing {pdf_path}: {e}")
        return None

def main():
    print("🚀 Starting Simple PDF Analysis...")
    print("="*50)

    # Load PDF cases
    pdf_cases = load_pdf_cases()
    print(f"📄 Found {len(pdf_cases)} PDF cases")

    # Process each PDF
    results = []
    for i, case in enumerate(pdf_cases, 1):
        print(f"\n[{i}/{len(pdf_cases)}] Processing: {case['case_name']}")

        analysis = analyze_single_pdf(case['pdf_path'], case['case_name'])
        if analysis:
            analysis['cluster_id'] = case['cluster_id']
            analysis['file_size'] = case['file_size']
            results.append(analysis)

    # Generate summary
    print(f"\n📊 ANALYSIS SUMMARY")
    print("="*50)
    print(f"Successfully processed: {len(results)}/{len(pdf_cases)} PDFs")

    if results:
        # Outcome distribution
        outcomes = [r['outcome']['outcome'] for r in results]
        outcome_counts = {}
        for outcome in outcomes:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        print(f"\n📈 Outcome Distribution:")
        for outcome, count in outcome_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {outcome}: {count} cases ({percentage:.1f}%)")

        # Intel factor stats
        print(f"\n🔍 Intel Factor Detection:")
        for factor in ['factor_1', 'factor_2', 'factor_3', 'factor_4']:
            detected = sum(1 for r in results if r['intel_factors'].get(factor, False))
            print(f"  {factor}: {detected}/{len(results)} ({detected/len(results):.1%})")

        # Citation stats
        all_citations = []
        for r in results:
            for citation in r['citations']:
                all_citations.append(citation['pattern'])

        citation_counts = {}
        for citation in all_citations:
            citation_counts[citation] = citation_counts.get(citation, 0) + 1

        if citation_counts:
            print(f"\n📚 Citation Patterns:")
            for citation, count in citation_counts.items():
                print(f"  {citation}: {count} mentions")

        # Text stats
        text_lengths = [r['text_length'] for r in results]
        print(f"\n📄 Text Statistics:")
        print(f"  Average length: {sum(text_lengths)/len(text_lengths):.0f} characters")
        print(f"  Range: {min(text_lengths)} - {max(text_lengths)} characters")

        # 1782 mentions
        section_1782_total = sum(r['section_1782_mentions'] for r in results)
        print(f"\n⚖️ Section 1782 Mentions: {section_1782_total} total")

    # Save results
    with open("data/case_law/simple_pdf_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to simple_pdf_analysis.json")
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
