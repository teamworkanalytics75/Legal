#!/usr/bin/env python3
"""Examine chat data in VSCode database"""
import sqlite3
import json
import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
vscdb_files = list(workspace_dir.glob("*/state.vscdb"))
latest_db = max(vscdb_files, key=lambda p: p.stat().st_mtime)
print(f"Examining: {latest_db}\n")

conn = sqlite3.connect(str(latest_db))
cursor = conn.cursor()

# Get all keys that might contain chat data
cursor.execute("SELECT key FROM ItemTable WHERE key LIKE '%chat%' OR key LIKE '%agent%' OR key LIKE '%session%'")
chat_keys = [row[0] for row in cursor.fetchall()]
print(f"Chat-related keys: {chat_keys}\n")

# Also check for keys with UUIDs (likely individual chats)
cursor.execute("SELECT key FROM ItemTable WHERE key LIKE '%-%-%-%-%'")
uuid_keys = [row[0] for row in cursor.fetchall()]
print(f"Found {len(uuid_keys)} keys with UUID pattern")

# Examine the value of these keys
for key in uuid_keys[:5]:  # First 5
    cursor.execute("SELECT value FROM ItemTable WHERE key=?", (key,))
    result = cursor.fetchone()
    if result:
        value = result[0]
        try:
            # Try to decode as JSON
            decoded = json.loads(value)
            print(f"\nKey: {key}")
            print(f"Type: {type(decoded)}")
            if isinstance(decoded, dict):
                print(f"Keys: {list(decoded.keys())[:10]}")
            elif isinstance(decoded, list):
                print(f"Length: {len(decoded)}")
                if decoded:
                    print(f"First item keys: {list(decoded[0].keys()) if isinstance(decoded[0], dict) else 'N/A'}")
        except:
            print(f"\nKey: {key} (not JSON)")
            print(f"Value (first 200 chars): {str(value)[:200]}")

conn.close()

