import sqlite3
import json
import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()
cursor.execute("SELECT value FROM ItemTable WHERE key='composer.composerData'")
result = cursor.fetchone()
data = json.loads(result[0])

print(f"Selected composers: {data.get('selectedComposerIds', [])}")
all_composers = data.get('allComposers', [])
print(f"\nAll Composers: {len(all_composers)}\n")

for i, comp_data in enumerate(all_composers):
    print(f"\nComposer {i+1}:")
    print(f"  Type: {type(comp_data)}")
    if isinstance(comp_data, dict):
        print(f"  Keys: {list(comp_data.keys())[:20]}")
        if 'messages' in comp_data:
            print(f"  Messages: {len(comp_data['messages'])}")
        if 'title' in comp_data:
            print(f"  Title: {comp_data['title']}")
        if 'createdAt' in comp_data:
            print(f"  Created: {comp_data['createdAt']}")
        if 'id' in comp_data:
            print(f"  ID: {comp_data['id']}")

conn.close()

