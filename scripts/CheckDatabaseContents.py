#!/usr/bin/env python3
"""
Check Database Contents
See what's actually in the database and understand the case distribution.
"""

import mysql.connector
from collections import Counter

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

def analyze_database_contents():
    """Analyze what's actually in the database."""
    conn = get_database_connection()
    if not conn:
        return

    cursor = conn.cursor()

    print("🔍 DATABASE CONTENTS ANALYSIS")
    print("=" * 50)

    # Get all cases with their topics
    cursor.execute("SELECT id, case_name, topic, citation, court FROM case_law ORDER BY id")
    all_cases = cursor.fetchall()

    print(f"\n📊 TOTAL CASES IN DATABASE: {len(all_cases)}")

    # Analyze topics
    topics = [case[2] for case in all_cases if case[2]]
    topic_counts = Counter(topics)

    print(f"\n📋 TOPICS IN DATABASE:")
    for topic, count in topic_counts.most_common():
        print(f"  {topic}: {count} cases")

    # Show all cases
    print(f"\n📄 ALL CASES:")
    for i, (case_id, case_name, topic, citation, court) in enumerate(all_cases):
        print(f"  {i+1:2d}. [{topic}] {case_name}")
        if citation:
            print(f"      Citation: {citation}")
        if court:
            print(f"      Court: {court}")
        print()

    # Check for 1782-related cases regardless of topic
    print(f"\n🔍 SEARCHING FOR 1782-RELATED CASES:")
    cursor.execute("SELECT id, case_name, topic, citation FROM case_law WHERE case_name LIKE '%1782%' OR citation LIKE '%1782%' OR opinion_text LIKE '%1782%'")
    discovery_cases = cursor.fetchall()

    print(f"  Found {len(discovery_cases)} cases with '1782' in name/citation/text:")
    for case_id, case_name, topic, citation in discovery_cases:
        print(f"    - [{topic}] {case_name}")
        if citation:
            print(f"      Citation: {citation}")

    conn.close()

if __name__ == "__main__":
    analyze_database_contents()
