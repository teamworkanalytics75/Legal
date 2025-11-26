#!/usr/bin/env python3
"""
Analyze Unique Cases and Document Types
Counts unique cases and identifies which have original petitions.
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any

def extract_case_info(filename: str) -> Dict[str, str]:
    """Extract case information from filename."""
    # Pattern: 120-mc-00032_doc11_Order_on_MC-Application_Pursuant_to_281782.txt
    # Or: 120-mc-00032_doc1_Petition_Other.txt

    parts = filename.replace('.txt', '').split('_')
    if len(parts) >= 2:
        case_id = parts[0]  # e.g., "120-mc-00032"
        doc_num = parts[1]  # e.g., "doc11"
        doc_type = '_'.join(parts[2:]) if len(parts) > 2 else "unknown"

        return {
            'case_id': case_id,
            'doc_number': doc_num,
            'doc_type': doc_type,
            'filename': filename
        }
    return {
        'case_id': 'unknown',
        'doc_number': 'unknown',
        'doc_type': 'unknown',
        'filename': filename
    }

def is_original_petition(doc_type: str) -> bool:
    """Check if document appears to be an original petition."""
    doc_lower = doc_type.lower()

    # Keywords that suggest original petitions
    petition_keywords = [
        'petition',
        'application',
        'motion for discovery',
        'miscellaneous relief',
        'ex parte application',
        'misc case',
        'discovery',
        'initiating document',
        'miscellaneous case opening',
        'attorney civil case opening'
    ]

    # Check if it's a doc1 (first document) or contains petition keywords
    return any(keyword in doc_lower for keyword in petition_keywords)

def is_court_opinion(doc_type: str) -> bool:
    """Check if document appears to be a court opinion/order."""
    doc_lower = doc_type.lower()

    opinion_keywords = [
        'order',
        'memorandum opinion',
        'memorandum and order',
        'report and recommendation',
        'judgment',
        'opinion'
    ]

    return any(keyword in doc_lower for keyword in opinion_keywords)

def analyze_cases():
    """Analyze all extracted text files to count unique cases."""

    text_dir = Path("data/case_law/extracted_text")
    if not text_dir.exists():
        print("❌ Text directory not found!")
        return

    text_files = list(text_dir.glob("*.txt"))
    print(f"📁 Found {len(text_files)} text files")

    # Group by case
    cases = defaultdict(list)
    case_stats = defaultdict(lambda: {
        'total_docs': 0,
        'has_petition': False,
        'has_opinion': False,
        'petition_types': [],
        'opinion_types': [],
        'all_doc_types': []
    })

    for text_file in text_files:
        case_info = extract_case_info(text_file.name)
        case_id = case_info['case_id']

        cases[case_id].append(case_info)
        case_stats[case_id]['total_docs'] += 1
        case_stats[case_id]['all_doc_types'].append(case_info['doc_type'])

        # Check for original petition
        if is_original_petition(case_info['doc_type']):
            case_stats[case_id]['has_petition'] = True
            case_stats[case_id]['petition_types'].append(case_info['doc_type'])

        # Check for court opinion
        if is_court_opinion(case_info['doc_type']):
            case_stats[case_id]['has_opinion'] = True
            case_stats[case_id]['opinion_types'].append(case_info['doc_type'])

    # Summary statistics
    total_cases = len(cases)
    cases_with_petitions = sum(1 for stats in case_stats.values() if stats['has_petition'])
    cases_with_opinions = sum(1 for stats in case_stats.values() if stats['has_opinion'])
    cases_with_both = sum(1 for stats in case_stats.values() if stats['has_petition'] and stats['has_opinion'])

    print("\n" + "=" * 60)
    print("📊 UNIQUE CASES ANALYSIS")
    print("=" * 60)

    print(f"\n📈 OVERVIEW:")
    print(f"  Total unique cases: {total_cases}")
    print(f"  Cases with original petitions: {cases_with_petitions}")
    print(f"  Cases with court opinions: {cases_with_opinions}")
    print(f"  Cases with both petitions AND opinions: {cases_with_both}")

    print(f"\n📊 DOCUMENT COVERAGE:")
    print(f"  Cases with petitions only: {cases_with_petitions - cases_with_both}")
    print(f"  Cases with opinions only: {cases_with_opinions - cases_with_both}")
    print(f"  Cases with both: {cases_with_both}")
    print(f"  Cases with neither: {total_cases - cases_with_petitions - cases_with_opinions + cases_with_both}")

    # Document type analysis
    all_doc_types = []
    for stats in case_stats.values():
        all_doc_types.extend(stats['all_doc_types'])

    doc_type_counts = Counter(all_doc_types)

    print(f"\n🔍 TOP DOCUMENT TYPES:")
    for doc_type, count in doc_type_counts.most_common(10):
        print(f"  {doc_type}: {count} documents")

    # Cases with petitions
    print(f"\n📋 CASES WITH ORIGINAL PETITIONS:")
    petition_cases = [(case_id, stats) for case_id, stats in case_stats.items() if stats['has_petition']]
    petition_cases.sort(key=lambda x: x[0])

    for i, (case_id, stats) in enumerate(petition_cases[:10]):  # Show first 10
        petition_types = ', '.join(set(stats['petition_types']))
        print(f"  {i+1:2d}. {case_id} ({stats['total_docs']} docs) - {petition_types}")

    if len(petition_cases) > 10:
        print(f"  ... and {len(petition_cases) - 10} more cases")

    # Save detailed results
    results = {
        'summary': {
            'total_cases': total_cases,
            'cases_with_petitions': cases_with_petitions,
            'cases_with_opinions': cases_with_opinions,
            'cases_with_both': cases_with_both,
            'total_documents': len(text_files)
        },
        'case_details': dict(case_stats),
        'document_type_counts': dict(doc_type_counts)
    }

    output_file = Path("data/case_law/analysis_results/unique_cases_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")
    print(f"\n🎯 ANSWER: {cases_with_petitions} unique cases have at least 1 original petition document!")

if __name__ == "__main__":
    analyze_cases()
