#!/usr/bin/env python3
"""Check progress of fact extraction script."""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent

print("\n" + "="*80)
print("FACT EXTRACTION PROGRESS CHECKER")
print("="*80)
print()

# Find extraction process using ps
try:
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True,
        text=True,
        timeout=5
    )
    extraction_lines = [line for line in result.stdout.split('\n') if 'extract_facts_ml_enhanced' in line and 'grep' not in line]
    
    if extraction_lines:
        print("✅ Extraction process found:")
        for line in extraction_lines:
            parts = line.split()
            if len(parts) >= 2:
                pid = parts[1]
                cpu = parts[2] if len(parts) > 2 else "?"
                mem = parts[3] if len(parts) > 3 else "?"
                print(f"   PID: {pid}")
                print(f"   CPU: {cpu}%")
                print(f"   Memory: {mem}%")
                print()
                print(f"   💡 To kill: kill {pid}")
                print(f"   💡 To restart with fast mode:")
                print(f"      python writer_agents/scripts/extract_facts_ml_enhanced.py --method fast --verbose")
    else:
        print("❌ No extraction process found")
        print("\n💡 If extraction was running, it may have:")
        print("   - Completed successfully")
        print("   - Crashed (check logs)")
        print("   - Been killed")
except Exception as e:
    print(f"⚠️  Could not check processes: {e}")
    print("   Try: ps aux | grep extract_facts_ml_enhanced")

# Check database for progress
db_path = REPO_ROOT / "case_law_data" / "lawsuit_facts_database.db"
if db_path.exists():
    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fact_registry")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"📊 Database contains {count:,} facts")
    except Exception as e:
        print(f"⚠️  Could not read database: {e}")
else:
    print("⚠️  Database not found yet")

print("\n" + "="*80)
print("\n💡 If stuck, try:")
print("   1. Kill process: kill <PID>")
print("   2. Restart with fast mode: python writer_agents/scripts/extract_facts_ml_enhanced.py --method fast")
print("   3. Or skip QA: python writer_agents/scripts/extract_facts_ml_enhanced.py --method ner,temporal,openie")
print()

