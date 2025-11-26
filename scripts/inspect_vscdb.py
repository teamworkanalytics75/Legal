#!/usr/bin/env python3
"""Inspect VSCode database structure"""
import sqlite3
import json
import os
from pathlib import Path

# Find latest workspace state
workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
vscdb_files = list(workspace_dir.glob("*/state.vscdb"))
if not vscdb_files:
    print("No databases found")
    exit(1)

# Use the most recently modified one
latest_db = max(vscdb_files, key=lambda p: p.stat().st_mtime)
print(f"Inspecting: {latest_db}")

try:
    conn = sqlite3.connect(str(latest_db))
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\nTables found: {tables}")

    # Inspect each table
    for table in tables:
        print(f"\n--- Table: {table} ---")
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        print("Columns:", columns)

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"Row count: {count}")

        # Sample data
        cursor.execute(f"SELECT * FROM {table} LIMIT 3")
        rows = cursor.fetchall()
        if rows:
            print("Sample rows:")
            for i, row in enumerate(rows):
                print(f"  Row {i}: {row[:2]}...")  # First 2 columns only

    conn.close()
except Exception as e:
    print(f"Error: {e}")

