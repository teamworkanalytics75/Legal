import sqlite3
import json
import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
db_path = max(workspace_dir.glob("*/state.vscdb"), key=lambda p: p.stat().st_mtime)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get ALL keys and find ones that might contain messages
cursor.execute("SELECT key, value FROM ItemTable")
all_rows = cursor.fetchall()

print(f"Total keys: {len(all_rows)}\n")

for key, value in all_rows:
    try:
        data = json.loads(value)

        # Look for messages recursively
        def find_messages(obj, depth=0, path=""):
            if depth > 8:  # Prevent too deep recursion
                return None

            if isinstance(obj, dict):
                # Check for obvious message fields
                if 'role' in obj and 'content' in obj:
                    return obj
                if 'text' in obj and 'user' in str(obj).lower():
                    return obj

                # Recursively search
                for k, v in obj.items():
                    result = find_messages(v, depth+1, f"{path}.{k}")
                    if result:
                        return result

            elif isinstance(obj, list) and len(obj) > 0:
                # Check first item
                if isinstance(obj[0], dict) and 'role' in obj[0]:
                    return obj[:3]  # Return first 3 items
                # Recursively search items
                for i, item in enumerate(obj[:5]):  # Check first 5 items
                    result = find_messages(item, depth+1, f"{path}[{i}]")
                    if result:
                        return result
            return None

        messages = find_messages(data)
        if messages:
            print(f"\n🔍 FOUND MESSAGES IN: {key}")
            print(f"   Type: {type(messages)}")
            if isinstance(messages, list):
                print(f"   Count: {len(messages)} messages")
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'unknown')
                        content = str(msg.get('content', 'N/A'))[:80]
                        print(f"      {role}: {content}")
            elif isinstance(messages, dict):
                role = messages.get('role', 'unknown')
                content = str(messages.get('content', 'N/A'))[:80]
                print(f"      Single message - {role}: {content}")

    except Exception as e:
        pass

conn.close()

