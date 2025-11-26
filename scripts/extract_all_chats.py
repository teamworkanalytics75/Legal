import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print(f"Extracting from: {db_path}\n")

# Get composer metadata
cursor.execute("SELECT value FROM ItemTable WHERE key='composer.composerData'")
composer_data = json.loads(cursor.fetchone()[0])
all_composers = composer_data.get('allComposers', [])

# Get prompts
cursor.execute("SELECT value FROM ItemTable WHERE key='aiService.prompts'")
prompts = json.loads(cursor.fetchone()[0])

print(f"Found {len(all_composers)} composers and {len(prompts)} prompts\n")

# Create mapping
composer_map = {}
for comp in all_composers:
    if isinstance(comp, dict):
        comp_id = comp.get('composerId')
        name = comp.get('name', 'Unnamed')
        composer_map[comp_id] = name

# Extract conversations from prompts
cursor_chats = []
codex_chats = []

for prompt in prompts:
    if isinstance(prompt, dict) and 'text' in prompt:
        text = prompt['text']
        command = prompt.get('commandType', '')

        # Detect Codex vs Cursor
        if 'codex' in text.lower() or 'Read this chat with codex' in text:
            codex_chats.append(text[:200])
        else:
            cursor_chats.append(text[:200])

print(f"\nCursor chats: {len(cursor_chats)}")
print(f"Codex chats: {len(codex_chats)}")
print(f"\nComposer titles: {len(all_composers)}")
for comp in all_composers:
    if isinstance(comp, dict):
        print(f"  - {comp.get('name', 'Unnamed')}")

conn.close()

