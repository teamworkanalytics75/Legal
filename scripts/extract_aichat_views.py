#!/usr/bin/env python3
"""Extract chat data from aichat.view keys"""
import sqlite3
import json
import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

print(f"Extracting from: {db_path}\n")

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get all aichat.view keys
cursor.execute("SELECT key, value FROM ItemTable WHERE key LIKE 'workbench.panel.aichat.view.%'")
rows = cursor.fetchall()

print(f"Found {len(rows)} aichat views\n")

chats_found = 0

for key, value in rows:
    try:
        data = json.loads(value)
        if isinstance(data, dict):
            print(f"\nChat: {key}")
            print(f"  Keys: {list(data.keys())}")

            # Look for title, messages, etc.
            if 'title' in data:
                print(f"  Title: {data['title']}")
            if 'messages' in data:
                messages = data['messages']
                print(f"  Messages: {len(messages)}")
                if messages and isinstance(messages[0], dict):
                    print(f"  First message role: {messages[0].get('role', 'unknown')}")
            if 'createdAt' in data:
                print(f"  Created: {data['createdAt']}")

            chats_found += 1

    except Exception as e:
        print(f"  Failed to parse {key}: {e}")

print(f"\n\nTotal chats extracted: {chats_found}")

conn.close()

