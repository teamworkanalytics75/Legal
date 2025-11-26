import json
from pathlib import Path

data = json.load(open('data/cursor_chats.json'))
chats = data['chats']

cursor_chats = [c for c in chats if c['agent_type'] == 'cursor']
codex_chats = [c for c in chats if c['agent_type'] == 'codex']

print(f"Total: {len(chats)}")
print(f"Cursor: {len(cursor_chats)}")
print(f"Codex: {len(codex_chats)}")

print(f"\nSample Codex chats:")
for i, chat in enumerate(codex_chats[:5], 1):
    print(f"\n{i}. {chat['content'][:200]}")

