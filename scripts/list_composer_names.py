import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()
cursor.execute("SELECT value FROM ItemTable WHERE key='composer.composerData'")
result = cursor.fetchone()
data = json.loads(result[0])

all_composers = data.get('allComposers', [])

print(f"Found {len(all_composers)} composers:\n")

for i, comp in enumerate(all_composers):
    if isinstance(comp, dict):
        name = comp.get('name', 'Unnamed')
        created = comp.get('createdAt', 0)
        date_str = datetime.fromtimestamp(created / 1000).strftime('%Y-%m-%d %H:%M')
        print(f"{i+1}. {name[:60]}")
        print(f"   Created: {date_str}")
        print()

conn.close()

