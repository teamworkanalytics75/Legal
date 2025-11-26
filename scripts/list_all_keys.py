import sqlite3
import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()
cursor.execute("SELECT key FROM ItemTable")
keys = [row[0] for row in cursor.fetchall()]

print("All keys:")
print("\n".join(sorted(keys)))

conn.close()

