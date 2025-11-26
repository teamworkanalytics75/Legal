#!/usr/bin/env python3
"""Extract actual chat data from database"""
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
vscdb_files = list(workspace_dir.glob("*/state.vscdb"))

print(f"Searching {len(vscdb_files)} databases...\n")

total_chats = 0

for db_path in vscdb_files[:3]:  # Check first 3 databases
    print(f"\n--- {db_path.name} ---")

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check for interactive.sessions
        cursor.execute("SELECT value FROM ItemTable WHERE key='interactive.sessions'")
        result = cursor.fetchone()

        if result:
            value = result[0]
            try:
                sessions = json.loads(value)
                print(f"Found {len(sessions)} interactive sessions")

                for i, session in enumerate(sessions):
                    print(f"\n  Session {i+1}:")
                    if isinstance(session, dict):
                        print(f"    Keys: {list(session.keys())}")
                        # Look for messages
                        if 'messages' in session:
                            print(f"    Messages: {len(session['messages'])}")
                        if 'title' in session:
                            print(f"    Title: {session['title']}")
                    total_chats += 1
            except Exception as e:
                print(f"Failed to parse: {e}")

        # Also check each UUID key individually
        cursor.execute("SELECT key FROM ItemTable WHERE key NOT LIKE '%.numberOfVisibleViews' AND key LIKE '%-%-%-%-%'")
        uuid_keys = [row[0] for row in cursor.fetchall()]

        for key in uuid_keys:
            cursor.execute("SELECT value FROM ItemTable WHERE key=?", (key,))
            result = cursor.fetchone()
            if result:
                value = result[0]
                try:
                    data = json.loads(value)
                    # Check if this looks like chat data
                    if isinstance(data, dict) and 'messages' in data:
                        print(f"\n  Found chat in key: {key}")
                        print(f"    Messages: {len(data['messages'])}")
                        total_chats += 1
                except:
                    pass

        conn.close()

    except Exception as e:
        print(f"Error with {db_path}: {e}")

print(f"\n\nTotal chats found: {total_chats}")

