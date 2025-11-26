import sqlite3
import json
import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print(f"Checking: {db_path}\n")

# Check aiService.generations
print("=== aiService.generations ===")
cursor.execute("SELECT value FROM ItemTable WHERE key='aiService.generations'")
result = cursor.fetchone()
if result:
    data = json.loads(result[0])
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:10]}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")
        if data:
            print(f"First item type: {type(data[0])}")
            print(f"First item keys: {list(data[0].keys())[:10] if isinstance(data[0], dict) else 'N/A'}")

# Check aiService.prompts
print("\n=== aiService.prompts ===")
cursor.execute("SELECT value FROM ItemTable WHERE key='aiService.prompts'")
result = cursor.fetchone()
if result:
    data = json.loads(result[0])
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:10]}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")

conn.close()

