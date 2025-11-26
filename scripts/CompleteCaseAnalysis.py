#!/usr/bin/env python3
"""
Complete Case Analysis - Database vs PDF Collection
Analyzes all cases in database vs PDFs downloaded to get complete picture.
"""

import json
import mysql.connector
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any

def get_database_connection():
    """Connect to MySQL database."""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='thetimeisN0w!',
            database='lawsuit_docs'
        )
        return connection
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def analyze_database_cases():
    """Analyze all cases in the database."""
    conn = get_database_connection()
    if not conn:
        return None

    cursor = conn.cursor()

    # Get all cases
    cursor.execute("SELECT id, case_name, citation, court, date_filed, topic, download_url FROM case_law")
    all_cases = cursor.fetchall()

    # Get 1782 cases specifically
    cursor.execute("SELECT id, case_name, citation, court, date_filed, topic, download_url FROM case_law WHERE topic = '1782_discovery'")
    discovery_cases = cursor.fetchall()

    conn.close()

    return {
        'all_cases': all_cases,
        'discovery_cases': discovery_cases
    }

def analyze_pdf_collection():
    """Analyze the PDF collection we downloaded."""
    pdf_dir = Path("data/case_law/1782_recap_api_pdfs")
    text_dir = Path("data/case_law/extracted_text")

    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    text_files = list(text_dir.glob("*.txt")) if text_dir.exists() else []

    # Extract case IDs from filenames
    pdf_case_ids = set()
    text_case_ids = set()

    for pdf_file in pdf_files:
        case_id = pdf_file.stem.split('_')[0]
        pdf_case_ids.add(case_id)

    for text_file in text_files:
        case_id = text_file.stem.split('_')[0]
        text_case_ids.add(case_id)

    return {
        'pdf_files': len(pdf_files),
        'text_files': len(text_files),
        'pdf_case_ids': pdf_case_ids,
        'text_case_ids': text_case_ids
    }

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

def analyze_complete_picture():
    """Analyze the complete picture of all cases."""
    print("🔍 Analyzing Complete Case Picture...")
    print("=" * 60)

    # 1. Database analysis
    print("\n📊 DATABASE ANALYSIS:")
    db_data = analyze_database_cases()
    if not db_data:
        print("❌ Could not connect to database")
        return

    all_cases = db_data['all_cases']
    discovery_cases = db_data['discovery_cases']

    print(f"  Total cases in database: {len(all_cases)}")
    print(f"  1782 discovery cases in database: {len(discovery_cases)}")

    # 2. PDF collection analysis
    print("\n📁 PDF COLLECTION ANALYSIS:")
    pdf_data = analyze_pdf_collection()

    print(f"  PDF files downloaded: {pdf_data['pdf_files']}")
    print(f"  Text files extracted: {pdf_data['text_files']}")
    print(f"  Unique cases with PDFs: {len(pdf_data['pdf_case_ids'])}")
    print(f"  Unique cases with text: {len(pdf_data['text_case_ids'])}")

    # 3. Cross-reference analysis
    print("\n🔗 CROSS-REFERENCE ANALYSIS:")

    # Cases in database but not in PDF collection
    db_case_names = {case[1] for case in discovery_cases}
    pdf_case_names = set()

    # Try to match PDF case IDs to database case names
    # This is approximate since we'd need better matching logic
    print(f"  Cases in database: {len(db_case_names)}")
    print(f"  Cases with PDFs: {len(pdf_data['pdf_case_ids'])}")

    # 4. Original petition analysis
    print("\n📋 ORIGINAL PETITION ANALYSIS:")

    text_dir = Path("data/case_law/extracted_text")
    if text_dir.exists():
        text_files = list(text_dir.glob("*.txt"))

        # Group by case and check for petitions
        cases_with_petitions = set()
        case_doc_counts = defaultdict(int)

        for text_file in text_files:
            case_info = extract_case_info_from_filename(text_file.name)
            case_id = case_info['case_id']
            case_doc_counts[case_id] += 1

            if is_original_petition(case_info['doc_type']):
                cases_with_petitions.add(case_id)

        print(f"  Cases with original petitions: {len(cases_with_petitions)}")
        print(f"  Total unique cases in PDF collection: {len(case_doc_counts)}")

        # Show some examples
        print(f"\n📋 SAMPLE CASES WITH PETITIONS:")
        sample_cases = sorted(list(cases_with_petitions))[:10]
        for i, case_id in enumerate(sample_cases):
            doc_count = case_doc_counts[case_id]
            print(f"  {i+1:2d}. {case_id} ({doc_count} documents)")

    # 5. Summary
    print("\n" + "=" * 60)
    print("📊 COMPLETE SUMMARY")
    print("=" * 60)

    if text_dir.exists():
        text_files = list(text_dir.glob("*.txt"))
        cases_with_petitions = set()

        for text_file in text_files:
            case_info = extract_case_info_from_filename(text_file.name)
            if is_original_petition(case_info['doc_type']):
                cases_with_petitions.add(case_info['case_id'])

        print(f"\n🎯 FINAL ANSWER:")
        print(f"  Total 1782 cases in database: {len(discovery_cases)}")
        print(f"  Cases with downloaded PDFs: {len(pdf_data['pdf_case_ids'])}")
        print(f"  Cases with original petitions: {len(cases_with_petitions)}")
        print(f"\n📈 COVERAGE:")
        print(f"  PDF coverage: {len(pdf_data['pdf_case_ids'])/len(discovery_cases)*100:.1f}%")
        print(f"  Original petition coverage: {len(cases_with_petitions)/len(discovery_cases)*100:.1f}%")

if __name__ == "__main__":
    analyze_complete_picture()
