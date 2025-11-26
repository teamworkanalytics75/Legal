#!/usr/bin/env python3
"""Check lawsuit database structure."""

import sqlite3
import json
from pathlib import Path

def check_lawsuit_db():
    """Check the lawsuit database structure."""
    db_path = r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"

    if not Path(db_path).exists():
        print(f"[ERROR] Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    print(f"[OK] Tables: {tables}")

    # Check document count
    if 'cleaned_documents' in tables:
        cur.execute("SELECT COUNT(*) FROM cleaned_documents")
        doc_count = cur.fetchone()[0]
        print(f"[OK] Documents: {doc_count}")

        # Check schema
        cur.execute("PRAGMA table_info(cleaned_documents)")
        schema = cur.fetchall()
        print(f"[OK] Schema: {[col[1] for col in schema]}")

        # Sample document
        cur.execute("SELECT id, LENGTH(content) FROM cleaned_documents LIMIT 1")
        sample = cur.fetchone()
        if sample:
            print(f"[OK] Sample: ID {sample[0]} ({sample[1]} chars)")

    conn.close()

def check_analysis_results():
    """Check analysis results directory."""
    analysis_dir = Path("analysis_outputs")
    if analysis_dir.exists():
        files = list(analysis_dir.glob("*.json"))
        print(f"[OK] Analysis files: {len(files)}")
        for f in files:
            print(f"   - {f.name}")
    else:
        print("[ERROR] analysis_outputs/ not found")

if __name__ == "__main__":
    print("[INFO] Checking lawsuit database...")
    check_lawsuit_db()
    print("[INFO] Checking analysis results...")
    check_analysis_results()
