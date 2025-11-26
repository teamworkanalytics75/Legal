#!/usr/bin/env python3
"""Analyze unified_corpus to understand what it contains."""

import sqlite3
from pathlib import Path
from collections import defaultdict

WORKSPACE_ROOT = Path(__file__).parent
unified_db = WORKSPACE_ROOT / "case_law_data" / "unified_corpus.db"

print("=" * 80)
print("ANALYZING UNIFIED_CORPUS CONTENTS")
print("=" * 80)
print()

if not unified_db.exists():
    print(f"[ERROR] unified_corpus.db not found")
    exit(1)

conn = sqlite3.connect(str(unified_db))
cursor = conn.cursor()

# Get table structure
cursor.execute("PRAGMA table_info(cases)")
columns = [row[1] for row in cursor.fetchall()]
print(f"Columns: {', '.join(columns)}")
print()

# Total cases
cursor.execute("""
    SELECT COUNT(*)
    FROM cases
    WHERE cleaned_text IS NOT NULL
      AND cleaned_text != ''
      AND case_name IS NOT NULL
""")
total = cursor.fetchone()[0]
print(f"Total cases: {total:,}")
print()

# Check corpus_type distribution
if 'corpus_type' in columns:
    cursor.execute("""
        SELECT corpus_type, COUNT(*) as count
        FROM cases
        WHERE cleaned_text IS NOT NULL
          AND cleaned_text != ''
          AND case_name IS NOT NULL
        GROUP BY corpus_type
        ORDER BY count DESC
    """)
    print("Corpus type distribution:")
    for ctype, count in cursor.fetchall():
        print(f"  {ctype or '(null)'}: {count:,}")
    print()

# Check for 1782 cases
cursor.execute("""
    SELECT COUNT(*)
    FROM cases
    WHERE cleaned_text IS NOT NULL
      AND cleaned_text != ''
      AND case_name IS NOT NULL
      AND (
        corpus_type = 'section_1782_discovery'
        OR cleaned_text LIKE '%section 1782%'
        OR cleaned_text LIKE '%28 U.S.C. 1782%'
        OR cleaned_text LIKE '%U.S.C. § 1782%'
      )
""")
count_1782 = cursor.fetchone()[0]
print(f"Section 1782 cases: {count_1782:,}")
print()

# Court distribution
cursor.execute("""
    SELECT court, COUNT(*) as count
    FROM cases
    WHERE cleaned_text IS NOT NULL
      AND cleaned_text != ''
      AND case_name IS NOT NULL
    GROUP BY court
    ORDER BY count DESC
    LIMIT 20
""")
print("Top 20 courts:")
for court, count in cursor.fetchall():
    print(f"  {court[:70]}: {count:,}")
print()

# Check for MA federal courts
cursor.execute("""
    SELECT COUNT(*)
    FROM cases
    WHERE cleaned_text IS NOT NULL
      AND cleaned_text != ''
      AND case_name IS NOT NULL
      AND (
        court LIKE '%Massachusetts%'
        OR court LIKE '%Mass.%'
        OR court LIKE '%First Circuit%'
      )
""")
count_ma = cursor.fetchone()[0]
print(f"MA-related cases: {count_ma:,}")
print()

# Check for Harvard mentions
cursor.execute("""
    SELECT COUNT(*)
    FROM cases
    WHERE cleaned_text IS NOT NULL
      AND cleaned_text != ''
      AND case_name IS NOT NULL
      AND (
        case_name LIKE '%Harvard%'
        OR cleaned_text LIKE '%Harvard%'
      )
""")
count_harvard = cursor.fetchone()[0]
print(f"Harvard-related cases: {count_harvard:,}")
print()

# Check for China mentions
cursor.execute("""
    SELECT COUNT(*)
    FROM cases
    WHERE cleaned_text IS NOT NULL
      AND cleaned_text != ''
      AND case_name IS NOT NULL
      AND (
        case_name LIKE '%China%'
        OR cleaned_text LIKE '%China%'
        OR cleaned_text LIKE '%Chinese%'
      )
""")
count_china = cursor.fetchone()[0]
print(f"China-related cases: {count_china:,}")
print()

# Appellate/Supreme courts
cursor.execute("""
    SELECT COUNT(*)
    FROM cases
    WHERE cleaned_text IS NOT NULL
      AND cleaned_text != ''
      AND case_name IS NOT NULL
      AND (
        court LIKE '%Supreme Court%'
        OR court LIKE '%Court of Appeals%'
        OR court LIKE '%Circuit%'
      )
""")
count_appellate = cursor.fetchone()[0]
print(f"Appellate/Supreme cases: {count_appellate:,}")
print()

conn.close()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

