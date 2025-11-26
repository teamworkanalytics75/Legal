import sqlite3
import json
import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

cursor.execute("SELECT value FROM ItemTable WHERE key='aiService.prompts'")
result = cursor.fetchone()
data = json.loads(result[0])

print(f"Found {len(data)} prompts\n")

for i, prompt in enumerate(data[:5]):
    print(f"\nPrompt {i+1}:")
    if isinstance(prompt, dict):
        print(f"  Keys: {list(prompt.keys())[:15]}")

        # Check for message-like fields
        for key in ['text', 'content', 'body', 'message', 'role']:
            if key in prompt:
                val = prompt[key]
                if isinstance(val, str) and len(val) > 20:
                    print(f"  {key}: {val[:100]}...")
                else:
                    print(f"  {key}: {val}")

        # Show timestamp if exists
        for key in ['timestamp', 'createdAt', 'time', 'date']:
            if key in prompt:
                print(f"  {key}: {prompt[key]}")

conn.close()

