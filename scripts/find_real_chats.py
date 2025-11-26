#!/usr/bin/env python3
"""Find where chats are actually stored"""
import sqlite3
import json
import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

print(f"Examining: {db_path}\n")

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get ALL keys and sample their content
cursor.execute("SELECT key, value FROM ItemTable")
rows = cursor.fetchall()

print(f"Total keys: {len(rows)}\n")

# Look for keys with 'composerChatViewPane' - these likely have chat data
for key, value in rows[:100]:  # Check first 100
    if 'composerChatViewPane' in key and 'numberOfVisibleViews' not in key:
        print(f"\nExamining: {key}")
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                print(f"  Type: dict, keys: {list(data.keys())[:10]}")
                # Check for common chat fields
                if any(field in data for field in ['messages', 'title', 'createdAt', 'updatedAt', 'conversationId']):
                    print(f"  >>> This looks like chat data! <<<")
                    print(f"  Full keys: {list(data.keys())}")
            elif isinstance(data, list):
                print(f"  Type: list, length: {len(data)}")
                if data and isinstance(data[0], dict):
                    print(f"  First item keys: {list(data[0].keys())}")
        except Exception as e:
            print(f"  Failed to parse: {e}")

conn.close()

