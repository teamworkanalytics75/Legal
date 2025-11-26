#!/usr/bin/env python3
"""
Non-interactive McGrath Evidence Search
=======================================

Searches the lawsuit database for McGrath-related evidence without user input.
"""

import sqlite3
from pathlib import Path

def search_mcgrath_evidence():
    """Search for McGrath-related evidence in the lawsuit database."""
    print("Searching lawsuit database for McGrath evidence...")

    lawsuit_db_path = r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"

    if not Path(lawsuit_db_path).exists():
        print(f"Database not found: {lawsuit_db_path}")
        return

    conn = sqlite3.connect(lawsuit_db_path)
    cursor = conn.cursor()

    # Get database stats
    cursor.execute("SELECT COUNT(*) FROM cleaned_documents")
    total_docs = cursor.fetchone()[0]
    print(f"Total documents in database: {total_docs}")

    # Search for McGrath
    print("\nSearching for 'McGrath'...")
    cursor.execute("""
        SELECT rowid, SUBSTR(content, 1, 500) as preview, LENGTH(content) as doc_length
        FROM cleaned_documents
        WHERE LOWER(content) LIKE '%mcgrath%'
        ORDER BY doc_length DESC
        LIMIT 10
    """)

    mcgrath_results = cursor.fetchall()
    print(f"Found {len(mcgrath_results)} documents containing 'McGrath':")

    for i, (doc_id, preview, length) in enumerate(mcgrath_results, 1):
        print(f"\n[{i}] Document #{doc_id} ({length:,} chars)")
        print(f"Preview: {preview.strip().encode('ascii', 'ignore').decode('ascii')}...")

    # Search for Malcolm Grayson
    print("\n" + "="*50)
    print("Searching for 'Malcolm Grayson'...")
    cursor.execute("""
        SELECT rowid, SUBSTR(content, 1, 500) as preview, LENGTH(content) as doc_length
        FROM cleaned_documents
        WHERE LOWER(content) LIKE '%malcolm%grayson%'
        ORDER BY doc_length DESC
        LIMIT 10
    """)

    grayson_results = cursor.fetchall()
    print(f"Found {len(grayson_results)} documents containing 'Malcolm Grayson':")

    for i, (doc_id, preview, length) in enumerate(grayson_results, 1):
        print(f"\n[{i}] Document #{doc_id} ({length:,} chars)")
        print(f"Preview: {preview.strip().encode('ascii', 'ignore').decode('ascii')}...")

    # Search for Xi Mingze
    print("\n" + "="*50)
    print("Searching for 'Xi Mingze'...")
    cursor.execute("""
        SELECT rowid, SUBSTR(content, 1, 500) as preview, LENGTH(content) as doc_length
        FROM cleaned_documents
        WHERE LOWER(content) LIKE '%xi%mingze%'
        ORDER BY doc_length DESC
        LIMIT 10
    """)

    xi_mingze_results = cursor.fetchall()
    print(f"Found {len(xi_mingze_results)} documents containing 'Xi Mingze':")

    for i, (doc_id, preview, length) in enumerate(xi_mingze_results, 1):
        print(f"\n[{i}] Document #{doc_id} ({length:,} chars)")
        print(f"Preview: {preview.strip().encode('ascii', 'ignore').decode('ascii')}...")

    # Search for April 2019
    print("\n" + "="*50)
    print("Searching for 'April 2019'...")
    cursor.execute("""
        SELECT rowid, SUBSTR(content, 1, 500) as preview, LENGTH(content) as doc_length
        FROM cleaned_documents
        WHERE LOWER(content) LIKE '%april%2019%'
        ORDER BY doc_length DESC
        LIMIT 10
    """)

    april_results = cursor.fetchall()
    print(f"Found {len(april_results)} documents containing 'April 2019':")

    for i, (doc_id, preview, length) in enumerate(april_results, 1):
        print(f"\n[{i}] Document #{doc_id} ({length:,} chars)")
        print(f"Preview: {preview.strip().encode('ascii', 'ignore').decode('ascii')}...")

    # Search for Statement 1
    print("\n" + "="*50)
    print("Searching for 'Statement 1'...")
    cursor.execute("""
        SELECT rowid, SUBSTR(content, 1, 500) as preview, LENGTH(content) as doc_length
        FROM cleaned_documents
        WHERE LOWER(content) LIKE '%statement%1%'
        ORDER BY doc_length DESC
        LIMIT 10
    """)

    statement_results = cursor.fetchall()
    print(f"Found {len(statement_results)} documents containing 'Statement 1':")

    for i, (doc_id, preview, length) in enumerate(statement_results, 1):
        print(f"\n[{i}] Document #{doc_id} ({length:,} chars)")
        print(f"Preview: {preview.strip().encode('ascii', 'ignore').decode('ascii')}...")

    conn.close()

    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"McGrath documents: {len(mcgrath_results)}")
    print(f"Malcolm Grayson documents: {len(grayson_results)}")
    print(f"Xi Mingze documents: {len(xi_mingze_results)}")
    print(f"April 2019 documents: {len(april_results)}")
    print(f"Statement 1 documents: {len(statement_results)}")

if __name__ == "__main__":
    search_mcgrath_evidence()
