#!/usr/bin/env python3
"""
Simple Search for User's Arguments
==================================

Searches for the user's specific arguments about McGrath's PowerPoint knowledge.
"""

import sqlite3
from pathlib import Path


def search_simple(keyword: str, limit: int = 5):
    """Simple search function."""
    lawsuit_db_path = r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"

    if not Path(lawsuit_db_path).exists():
        print(f"Database not found: {lawsuit_db_path}")
        return []

    conn = sqlite3.connect(lawsuit_db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT rowid, SUBSTR(content, 1, 500) as preview, LENGTH(content) as doc_length
        FROM cleaned_documents
        WHERE LOWER(content) LIKE ?
        ORDER BY doc_length DESC
        LIMIT ?
    """, (f'%{keyword.lower()}%', limit))

    results = cursor.fetchall()
    conn.close()

    return results


def main():
    """Search for user's specific arguments."""
    print("SEARCHING FOR USER'S SPECIFIC ARGUMENTS")
    print("=" * 50)

    # Search terms for user's arguments
    search_terms = [
        "alumni",
        "club",
        "three year",
        "same talk",
        "same slide",
        "sendelta",
        "yi wang",
        "confront",
        "fresh graduate",
        "ccp",
        "pressure",
        "powerpoint",
        "ppt",
        "slide",
        "xi mingze"
    ]

    all_results = {}

    for term in search_terms:
        print(f"\nSearching for '{term}'...")
        results = search_simple(term, limit=3)

        if results:
            print(f"Found {len(results)} documents:")
            all_results[term] = []

            for i, (doc_id, preview, length) in enumerate(results, 1):
                # Clean preview of Unicode characters
                clean_preview = preview.encode('ascii', 'ignore').decode('ascii')
                print(f"  [{i}] Doc #{doc_id} ({length:,} chars)")
                print(f"      {clean_preview[:200]}...")

                all_results[term].append({
                    "doc_id": doc_id,
                    "preview": clean_preview,
                    "length": length
                })
        else:
            print(f"No documents found for '{term}'")
            all_results[term] = []

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY OF USER'S ARGUMENTS FOUND:")
    print("=" * 50)

    total_found = 0
    for term, results in all_results.items():
        if results:
            print(f"{term}: {len(results)} documents")
            total_found += len(results)
        else:
            print(f"{term}: 0 documents")

    print(f"\nTotal documents found: {total_found}")

    # Key findings
    print("\nKEY FINDINGS:")
    if all_results.get("alumni") or all_results.get("club"):
        print("✓ Found evidence of alumni/club network")
    if all_results.get("three year") or all_results.get("same talk"):
        print("✓ Found evidence of repeated presentations")
    if all_results.get("sendelta"):
        print("✓ Found evidence of Sendelta investigation")
    if all_results.get("yi wang") or all_results.get("confront"):
        print("✓ Found evidence of Yi Wang confrontation")
    if all_results.get("fresh graduate"):
        print("✓ Found evidence of fresh graduate status")
    if all_results.get("ccp") or all_results.get("pressure"):
        print("✓ Found evidence of CCP pressure")
    if all_results.get("powerpoint") or all_results.get("ppt") or all_results.get("slide"):
        print("✓ Found evidence of PowerPoint presentations")
    if all_results.get("xi mingze"):
        print("✓ Found evidence of Xi Mingze references")


if __name__ == "__main__":
    main()
