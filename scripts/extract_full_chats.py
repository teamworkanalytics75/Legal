import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get composer metadata
cursor.execute("SELECT value FROM ItemTable WHERE key='composer.composerData'")
composer_data = json.loads(cursor.fetchone()[0])
all_composers = composer_data.get('allComposers', [])

print(f"Found {len(all_composers)} chats\n")

chats_with_messages = 0

for i, comp in enumerate(all_composers):
    if not isinstance(comp, dict):
        continue

    comp_id = comp.get('composerId')
    name = comp.get('name', 'Unnamed')
    created = comp.get('createdAt', 0)
    date_str = datetime.fromtimestamp(created / 1000).strftime('%Y-%m-%d %H:%M')

    print(f"\n{i+1}. {name}")
    print(f"   ID: {comp_id}")
    print(f"   Created: {date_str}")

    # Look for messages in composerChatViewPane keys
    cursor.execute("""
        SELECT key, value FROM ItemTable
        WHERE key LIKE ? OR key LIKE ?
    """, (f'workbench.panel.composerChatViewPane.{comp_id}', f'workbench.panel.aichat.{comp_id}'))

    results = cursor.fetchall()
    if results:
        for key, value in results:
            data = json.loads(value)
            # Search for messages in nested structure
            def find_messages(obj, depth=0):
                if depth > 5:  # Prevent infinite recursion
                    return []
                messages = []
                if isinstance(obj, dict):
                    if 'messages' in obj and isinstance(obj['messages'], list):
                        return obj['messages']
                    for k, v in obj.items():
                        msgs = find_messages(v, depth+1)
                        if msgs:
                            return msgs
                elif isinstance(obj, list):
                    for item in obj:
                        msgs = find_messages(item, depth+1)
                        if msgs:
                            return msgs
                return messages

            messages = find_messages(data)
            if messages:
                print(f"   ✓ Found {len(messages)} messages")
                for msg in messages[:2]:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'unknown')
                        content = str(msg.get('content', ''))[:100]
                        print(f"      {role}: {content}")
                chats_with_messages += 1
                break

print(f"\n\nTotal chats with messages: {chats_with_messages}/{len(all_composers)}")

conn.close()

