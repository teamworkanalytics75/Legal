import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Check history.entries
cursor.execute("SELECT key, value FROM ItemTable WHERE key='history.entries'")
result = cursor.fetchone()

if result:
    data = json.loads(result[1])
    print(f"History entries found: {len(data) if isinstance(data, list) else 'Not a list'}")

    if isinstance(data, list):
        for i, entry in enumerate(data[:5]):
            print(f"\nEntry {i+1}:")
            if isinstance(entry, dict):
                print(f"  Keys: {list(entry.keys())[:10]}")
                if 'source' in entry:
                    print(f"  Source: {entry['source']}")
                if 'body' in entry:
                    print(f"  Body: {str(entry['body'])[:100]}")

conn.close()

