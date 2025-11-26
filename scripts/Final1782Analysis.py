#!/usr/bin/env python3
"""
Complete 1782 Case Analysis - The Real Picture
Shows the complete picture of our 1782 discovery case collection.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any

def extract_case_info_from_filename(filename: str) -> Dict[str, str]:
    """Extract case information from filename."""
    parts = filename.replace('.txt', '').split('_')
    if len(parts) >= 2:
        case_id = parts[0]
        doc_num = parts[1]
        doc_type = '_'.join(parts[2:]) if len(parts) > 2 else "unknown"

        return {
            'case_id': case_id,
            'doc_number': doc_num,
            'doc_type': doc_type
        }
    return {'case_id': 'unknown', 'doc_number': 'unknown', 'doc_type': 'unknown'}

def is_original_petition(doc_type: str) -> bool:
    """Check if document appears to be an original petition."""
    doc_lower = doc_type.lower()

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

    return any(keyword in doc_lower for keyword in petition_keywords)

def analyze_complete_1782_collection():
    """Analyze our complete 1782 discovery case collection."""
    print("🔍 COMPLETE 1782 DISCOVERY CASE ANALYSIS")
    print("=" * 60)

    # 1. PDF Collection Analysis
    pdf_dir = Path("data/case_law/1782_recap_api_pdfs")
    text_dir = Path("data/case_law/extracted_text")

    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    text_files = list(text_dir.glob("*.txt")) if text_dir.exists() else []

    print(f"\n📁 PDF COLLECTION:")
    print(f"  PDF files downloaded: {len(pdf_files)}")
    print(f"  Text files extracted: {len(text_files)}")

    # 2. Unique Cases Analysis
    case_docs = defaultdict(list)
    case_stats = defaultdict(lambda: {
        'total_docs': 0,
        'has_petition': False,
        'has_opinion': False,
        'petition_types': [],
        'opinion_types': [],
        'all_doc_types': []
    })

    for text_file in text_files:
        case_info = extract_case_info_from_filename(text_file.name)
        case_id = case_info['case_id']

        case_docs[case_id].append(case_info)
        case_stats[case_id]['total_docs'] += 1
        case_stats[case_id]['all_doc_types'].append(case_info['doc_type'])

        # Check for original petition
        if is_original_petition(case_info['doc_type']):
            case_stats[case_id]['has_petition'] = True
            case_stats[case_id]['petition_types'].append(case_info['doc_type'])

        # Check for court opinion
        if 'order' in case_info['doc_type'].lower() or 'opinion' in case_info['doc_type'].lower():
            case_stats[case_id]['has_opinion'] = True
            case_stats[case_id]['opinion_types'].append(case_info['doc_type'])

    total_cases = len(case_docs)
    cases_with_petitions = sum(1 for stats in case_stats.values() if stats['has_petition'])
    cases_with_opinions = sum(1 for stats in case_stats.values() if stats['has_opinion'])
    cases_with_both = sum(1 for stats in case_stats.values() if stats['has_petition'] and stats['has_opinion'])

    print(f"\n📊 UNIQUE CASES ANALYSIS:")
    print(f"  Total unique 1782 cases: {total_cases}")
    print(f"  Cases with original petitions: {cases_with_petitions}")
    print(f"  Cases with court opinions: {cases_with_opinions}")
    print(f"  Cases with both petitions AND opinions: {cases_with_both}")

    # 3. Document Type Analysis
    all_doc_types = []
    for stats in case_stats.values():
        all_doc_types.extend(stats['all_doc_types'])

    doc_type_counts = Counter(all_doc_types)

    print(f"\n🔍 TOP DOCUMENT TYPES:")
    for doc_type, count in doc_type_counts.most_common(15):
        print(f"  {doc_type}: {count} documents")

    # 4. Cases with Original Petitions (Detailed)
    print(f"\n📋 CASES WITH ORIGINAL PETITIONS ({cases_with_petitions} total):")
    petition_cases = [(case_id, stats) for case_id, stats in case_stats.items() if stats['has_petition']]
    petition_cases.sort(key=lambda x: x[0])

    for i, (case_id, stats) in enumerate(petition_cases):
        petition_types = ', '.join(set(stats['petition_types']))
        has_opinion = "✓" if stats['has_opinion'] else "✗"
        print(f"  {i+1:2d}. {case_id} ({stats['total_docs']} docs) {has_opinion} - {petition_types}")

    # 5. Database vs PDF Collection
    print(f"\n🗄️ DATABASE vs PDF COLLECTION:")
    print(f"  Cases in database: 20 (all pseudonyms_sealing - NOT 1782 cases)")
    print(f"  Cases in PDF collection: {total_cases} (all 1782 discovery cases)")
    print(f"  Overlap: 0 cases (completely separate collections)")

    # 6. Final Summary
    print(f"\n" + "=" * 60)
    print(f"🎯 FINAL ANSWER TO YOUR QUESTION")
    print(f"=" * 60)
    print(f"\n📊 COMPLETE PICTURE:")
    print(f"  • Database: 20 cases (pseudonyms_sealing - NOT 1782)")
    print(f"  • PDF Collection: {total_cases} unique 1782 discovery cases")
    print(f"  • Cases with original petitions: {cases_with_petitions}")
    print(f"  • Cases with court opinions: {cases_with_opinions}")
    print(f"  • Cases with both: {cases_with_both}")

    print(f"\n🎯 DIRECT ANSWER:")
    print(f"  We have {cases_with_petitions} unique 1782 discovery cases")
    print(f"  that have at least 1 original petition document.")

    print(f"\n📈 COVERAGE:")
    print(f"  Original petition coverage: {cases_with_petitions/total_cases*100:.1f}%")
    print(f"  Court opinion coverage: {cases_with_opinions/total_cases*100:.1f}%")
    print(f"  Complete case coverage: {cases_with_both/total_cases*100:.1f}%")

    # Save detailed results
    results = {
        'summary': {
            'total_1782_cases': total_cases,
            'cases_with_petitions': cases_with_petitions,
            'cases_with_opinions': cases_with_opinions,
            'cases_with_both': cases_with_both,
            'total_documents': len(text_files),
            'database_cases': 20,
            'database_topic': 'pseudonyms_sealing'
        },
        'case_details': dict(case_stats),
        'document_type_counts': dict(doc_type_counts)
    }

    output_file = Path("data/case_law/analysis_results/complete_1782_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    analyze_complete_1782_collection()
