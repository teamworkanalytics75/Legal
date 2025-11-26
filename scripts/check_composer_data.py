import sqlite3
import json
import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

print(f"Checking: {db_path}\n")

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Check composer.composerData
print("=== composer.composerData ===")
cursor.execute("SELECT value FROM ItemTable WHERE key='composer.composerData'")
result = cursor.fetchone()
if result:
    data = json.loads(result[0])
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")

# Check workbench.backgroundComposer.workspacePersistentData
print("\n=== workbench.backgroundComposer.workspacePersistentData ===")
cursor.execute("SELECT value FROM ItemTable WHERE key='workbench.backgroundComposer.workspacePersistentData'")
result = cursor.fetchone()
if result:
    data = json.loads(result[0])
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for key in data.keys():
            print(f"  {key}: {type(data[key])}")
            if isinstance(data[key], list):
                print(f"    Length: {len(data[key])}")
                if data[key] and isinstance(data[key][0], dict):
                    print(f"    First item keys: {list(data[key][0].keys())}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")

conn.close()

